#!/usr/bin/env python3
"""
prepare_ax650_export.py

AX650/Pulsar 向けに Irodori-TTS / Irodori-TTS-Lite の一部モジュールを
固定 shape の ONNX として切り出すための前処理スクリプト。

狙い:
- 実行時は uv を使う
- Lite の patch を export-safe 設定で適用
- Triton fused INT4 ではなく、ONNX export しやすい fp16/torch module 形へ戻す
- CPU/NPU 分割のため、submodule 単位で ONNX を吐く
- Pulsar へ渡す前の shape / 入出力名 / calibration seed を揃える

注意:
- Irodori-TTS 本体の runtime/module 名は更新され得るため、loader と submodule path は CLI で指定する。
- まず --list-modules で runtime/model の構造を見て、--submodule を決める。
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.table import Table

console = Console()


def import_symbol(spec: str) -> Any:
    """
    "package.module:object.attr" 形式で import する。
    """
    if ":" not in spec:
        raise ValueError(f"--loader must be like 'pkg.mod:callable', got: {spec}")
    mod_name, attr_path = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj: Any = mod
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def get_attr_path(obj: Any, path: str) -> Any:
    """
    "a.b.c" 形式で属性を辿る。list/Sequential 用に数字indexも対応。
    """
    cur = obj
    if not path:
        return cur
    for part in path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def maybe_apply_irodori_lite_patch(args: argparse.Namespace) -> None:
    """
    Irodori-TTS-Lite が入っている場合だけ export-safe 設定で patch する。
    use_fused=False:
      Triton FusedInt4Linear は AX650/ONNX へ直接は持ち込まない。
    force_fp16=True:
      export時のdtypeを抑える。
    pack_rtn_extras=False:
      text embedding / encoder extras を packed on-the-fly のままにせず、
      できるだけ通常torch opへ戻す。
    """
    if args.no_lite_patch:
        console.print("[yellow]skip irodori_tts_lite.patch() by --no-lite-patch[/yellow]")
        return

    try:
        import irodori_tts_lite  # type: ignore
    except Exception as e:
        console.print(f"[yellow]irodori_tts_lite not importable, skip patch: {e}[/yellow]")
        return

    kwargs = dict(
        use_fused=False,
        force_fp16=True,
        disable_eager=False,
        pack_rtn_extras=False,
    )
    if args.duration_donor:
        kwargs["duration_donor"] = args.duration_donor

    console.print(f"[cyan]apply irodori_tts_lite.configure({kwargs})[/cyan]")
    irodori_tts_lite.configure(**kwargs)
    irodori_tts_lite.patch()


def parse_loader_kwargs(raw: list[str]) -> dict[str, Any]:
    """
    --loader-kw key=json_value を dict にする。
    例:
      --loader-kw device='"cuda"'
      --loader-kw dtype='"float16"'
      --loader-kw use_duration_predictor=true
    """
    out: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--loader-kw expects key=json_value, got: {item}")
        k, v = item.split("=", 1)
        out[k] = json.loads(v)
    return out


def load_root(args: argparse.Namespace) -> Any:
    maybe_apply_irodori_lite_patch(args)

    loader = import_symbol(args.loader)
    kwargs = parse_loader_kwargs(args.loader_kw)
    loader_args = [json.loads(x) for x in args.loader_arg]

    console.print(f"[cyan]load root using {args.loader}[/cyan]")
    console.print(f"[dim]args={loader_args}, kwargs={kwargs}[/dim]")
    root = loader(*loader_args, **kwargs)
    return root


def list_named_modules(root: Any, max_rows: int = 300) -> None:
    """
    torch.nn.Module を含む root から named_modules を表示。
    runtimeが直接Moduleでない場合は、よくある属性名を探す。
    """
    candidates: list[tuple[str, Any]] = [("", root)]
    for name in ["model", "tts", "net", "module", "runtime", "inferencer"]:
        if hasattr(root, name):
            candidates.append((name, getattr(root, name)))

    table = Table(title="Torch modules")
    table.add_column("path", overflow="fold")
    table.add_column("type", overflow="fold")

    seen = set()
    rows = 0
    for prefix, obj in candidates:
        if not isinstance(obj, torch.nn.Module):
            continue
        for name, mod in obj.named_modules():
            path = ".".join(p for p in [prefix, name] if p)
            if path in seen:
                continue
            seen.add(path)
            table.add_row(path or "<root>", mod.__class__.__name__)
            rows += 1
            if rows >= max_rows:
                table.add_row("...", f"truncated at {max_rows} rows")
                console.print(table)
                return

    if rows == 0:
        console.print("[red]No torch.nn.Module found. Try setting --root-path or inspect your loader output.[/red]")
    else:
        console.print(table)


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "int64": torch.int64,
        "long": torch.long,
        "int32": torch.int32,
        "bool": torch.bool,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


@dataclass
class InputSpec:
    name: str
    shape: list[int]
    dtype: str = "float16"
    kind: str = "randn"  # randn | zeros | ones | arange


def load_input_specs(path: Path) -> list[InputSpec]:
    data = yaml.safe_load(path.read_text())
    specs = []
    for item in data["inputs"]:
        specs.append(InputSpec(**item))
    return specs


def make_dummy_inputs(specs: list[InputSpec], device: str) -> tuple[torch.Tensor, ...]:
    tensors = []
    for spec in specs:
        dtype = dtype_from_name(spec.dtype)
        shape = tuple(spec.shape)

        if spec.kind == "randn":
            x = torch.randn(shape, dtype=torch.float32)
            if dtype in (torch.float16, torch.bfloat16, torch.float32):
                x = x.to(dtype)
            else:
                x = x.to(dtype)
        elif spec.kind == "zeros":
            x = torch.zeros(shape, dtype=dtype)
        elif spec.kind == "ones":
            x = torch.ones(shape, dtype=dtype)
        elif spec.kind == "arange":
            n = int(np.prod(shape))
            x = torch.arange(n, dtype=dtype).reshape(shape)
        else:
            raise ValueError(f"Unsupported input kind: {spec.kind}")

        tensors.append(x.to(device))
    return tuple(tensors)


def export_onnx(args: argparse.Namespace) -> None:
    root = load_root(args)

    if args.root_path:
        root = get_attr_path(root, args.root_path)

    if args.list_modules:
        list_named_modules(root)
        return

    if not args.submodule:
        raise SystemExit("--submodule is required unless --list-modules is used")

    module = get_attr_path(root, args.submodule)
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"--submodule {args.submodule} is not torch.nn.Module: {type(module)}")

    module.eval()

    # AX650/Pulsar向けの初手は fp16 固定を推奨。
    if args.force_fp16:
        module = module.half()

    device = args.device
    module = module.to(device)

    specs = load_input_specs(Path(args.input_spec))
    dummy_inputs = make_dummy_inputs(specs, device=device)
    input_names = [s.name for s in specs]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]export {args.submodule} -> {out_path}[/cyan]")
    console.print(f"[dim]input_names={input_names}[/dim]")

    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy_inputs,
            str(out_path),
            input_names=input_names,
            output_names=args.output_names,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,  # まず固定shapeで行く
        )

    console.print("[green]ONNX export done[/green]")

    if args.simplify:
        import onnx
        from onnxsim import simplify

        console.print("[cyan]simplify ONNX[/cyan]")
        model = onnx.load(str(out_path))
        sim_model, ok = simplify(model)
        if not ok:
            raise RuntimeError("onnxsim simplify failed")
        onnx.save(sim_model, str(out_path))
        console.print("[green]ONNX simplify done[/green]")

    meta_path = out_path.with_suffix(".export_meta.json")
    meta = {
        "submodule": args.submodule,
        "root_path": args.root_path,
        "opset": args.opset,
        "fixed_shape": True,
        "inputs": [s.__dict__ for s in specs],
        "outputs": args.output_names,
        "notes": [
            "Use Netron to inspect unsupported ops before Pulsar build.",
            "Keep CPU-side loop/control outside this ONNX.",
            "For AX650, start with batch=1 and fixed text_len/latent_len.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    console.print(f"[green]wrote {meta_path}[/green]")


def write_pulsar_stub(args: argparse.Namespace) -> None:
    """
    Pulsar buildの雛形だけ生成。
    実際のPulsar設定項目は環境のPulsar2バージョンに合わせて調整する。
    """
    onnx_path = Path(args.out)
    stub = {
        "model": str(onnx_path),
        "target_hardware": "AX650",
        "strategy": "fixed-shape module export",
        "recommended": {
            "npu_mode": "NPU1 or NPU2/NPU3 after profiling",
            "quantization": "start fp16/bf16 or int8; do not feed Triton packed int4 directly",
            "calibration": "use real TTS activations, not random Gaussian, for DiT quality",
        },
        "input_shapes": {
            spec.name: spec.shape for spec in load_input_specs(Path(args.input_spec))
        },
    }
    path = onnx_path.with_suffix(".pulsar_stub.yaml")
    path.write_text(yaml.safe_dump(stub, sort_keys=False, allow_unicode=True))
    console.print(f"[green]wrote {path}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loader", required=True, help="e.g. irodori_tts.inference_runtime:InferenceRuntime.from_key")
    parser.add_argument("--loader-arg", action="append", default=[], help="JSON positional arg, repeatable")
    parser.add_argument("--loader-kw", action="append", default=[], help="key=json_value, repeatable")
    parser.add_argument("--root-path", default="", help="optional attr path after loader result")
    parser.add_argument("--submodule", default="", help="module attr path under root/root-path")
    parser.add_argument("--input-spec", default="configs/dit_step_256.yaml")
    parser.add_argument("--out", default="build/dit_step_256.onnx")
    parser.add_argument("--output-names", nargs="+", default=["out"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-fp16", action="store_true", default=True)
    parser.add_argument("--no-force-fp16", dest="force_fp16", action="store_false")
    parser.add_argument("--simplify", action="store_true", default=True)
    parser.add_argument("--no-simplify", dest="simplify", action="store_false")
    parser.add_argument("--list-modules", action="store_true")
    parser.add_argument("--no-lite-patch", action="store_true")
    parser.add_argument("--duration-donor", default="")
    parser.add_argument("--write-pulsar-stub", action="store_true", default=True)
    args = parser.parse_args()

    export_onnx(args)
    if not args.list_modules and args.write_pulsar_stub:
        write_pulsar_stub(args)


if __name__ == "__main__":
    main()
