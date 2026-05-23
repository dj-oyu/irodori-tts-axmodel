#!/usr/bin/env python3
"""
inspect_model.py

Irodori-TTS-500M-v3 の実体を読み、AX650/AX8850 export 用に必要な
「本物の」情報を JSON にダンプする。

ダンプ内容:
- model_cfg (latent_dim / model_dim / num_layers / patch sizes など実値)
- top-level named_children と代表 submodule の構造
- forward hook で捕捉した各 export 対象モジュールの実 I/O shape
  (TextEncoder / ReferenceLatentEncoder / DiffusionBlock /
   DurationPredictor / forward_with_encoded_conditions)

placeholder の configs/*.yaml を実値で置き換えるための土台。
Irodori-TTS の uv env から実行する想定:

  cd ../../Irodori-TTS
  uv run python ../irodori-tts-axmodel/irodori_ax650_preprocess/scripts/inspect_model.py \
      --hf-checkpoint Aratako/Irodori-TTS-500M-v3 \
      --out /path/to/irodori-tts-axmodel/build/model_introspection.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
)


def _shape_of(x):
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (list, tuple)):
        return [_shape_of(v) for v in x]
    if isinstance(x, dict):
        return {k: _shape_of(v) for k, v in x.items()}
    return repr(type(x).__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-500M-v3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-steps", type=int, default=4)
    args = ap.parse_args()

    ckpt = hf_hub_download(repo_id=args.hf_checkpoint, filename="model.safetensors")
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=ckpt,
            model_device=args.device,
            codec_device=args.device,
        )
    )
    model = runtime.model

    import dataclasses

    report: dict = {}
    cfg = runtime.model_cfg
    if dataclasses.is_dataclass(cfg):
        report["model_cfg"] = dataclasses.asdict(cfg)
    else:
        report["model_cfg"] = {
            k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))
        }
    # 派生プロパティも明示記録
    for prop in ["patched_latent_dim", "speaker_patched_latent_dim", "use_speaker_condition"]:
        if hasattr(cfg, prop):
            report["model_cfg"][prop] = getattr(cfg, prop)
    report["model_dtype"] = str(next(model.parameters()).dtype)

    # 上位構造
    report["named_children"] = {
        name: child.__class__.__name__ for name, child in model.named_children()
    }

    # 代表 submodule (DiffusionBlock 1個) の構造
    def module_tree(mod, max_depth=2, _depth=0):
        out = {"type": mod.__class__.__name__}
        if _depth >= max_depth:
            return out
        children = {
            n: module_tree(c, max_depth, _depth + 1) for n, c in mod.named_children()
        }
        if children:
            out["children"] = children
        return out

    # DiffusionBlock のリストを探す
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "DiffusionBlock":
            report["diffusion_block_path"] = name
            report["diffusion_block_tree"] = module_tree(mod, max_depth=3)
            break

    # 実 I/O shape を hook で捕捉
    captured: dict = {}
    handles = []
    target_classes = {
        "TextEncoder",
        "ReferenceLatentEncoder",
        "DiffusionBlock",
        "DurationPredictor",
    }

    def make_hook(key):
        def hook(module, inputs, output):
            if key in captured:
                return
            captured[key] = {
                "class": module.__class__.__name__,
                "inputs": [_shape_of(a) for a in inputs],
                "output": _shape_of(output),
            }
        return hook

    seen_classes = set()
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__
        if cls in target_classes and cls not in seen_classes:
            handles.append(mod.register_forward_hook(make_hook(f"{cls}@{name}")))
            seen_classes.add(cls)

    # forward_with_encoded_conditions の入力 shape も捕捉
    orig_fwec = model.forward_with_encoded_conditions

    def wrapped_fwec(*a, **kw):
        if "forward_with_encoded_conditions" not in captured:
            captured["forward_with_encoded_conditions"] = {
                "args": [_shape_of(x) for x in a],
                "kwargs": {k: _shape_of(v) for k, v in kw.items()},
            }
        return orig_fwec(*a, **kw)

    model.forward_with_encoded_conditions = wrapped_fwec

    # 短い synth を 1 回流して hook を発火させる
    runtime.synthesize(
        SamplingRequest(
            text="こんにちは、テストです。",
            no_ref=True,
            num_steps=args.num_steps,
        )
    )

    for h in handles:
        h.remove()
    report["captured_io"] = captured

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {out_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
