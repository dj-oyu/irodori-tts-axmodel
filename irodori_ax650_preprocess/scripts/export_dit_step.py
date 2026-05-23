#!/usr/bin/env python3
"""
export_dit_step.py

Irodori-TTS-500M-v3 の DiT 1-step (forward_with_encoded_conditions) を
固定 shape ONNX へ切り出し、PyTorch 出力と数値検証する。

設計（FINDINGS.md）:
- Option B（既定）: context_kv_cache を使わず毎 step 再計算。入力 6 個で Pulsar2 親和性が高い。
- runtime(inference_runtime) は import しない。codec/dacvae/silentcipher を経由すると
  Irodori env の protobuf 3.19.6 ピンと onnx(>=4) が衝突するため、`irodori_tts.model` だけを使う。
- 実 conditioning は capture_dit_inputs.py が保存した .ref.pt（実 synth 由来）を読む。

前提:
- capture_dit_inputs.py を Irodori env で先に実行し inputs/ref を保存済み。
- model_cfg は build/model_introspection.json（inspect_model.py 出力）から読む。

実行（preprocess env, 例）:
  cd /path/to/irodori-tts-axmodel/irodori_ax650_preprocess
  PYTHONPATH=/path/to/Irodori-TTS uv run python scripts/export_dit_step.py \
    --weights ~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v3/snapshots/*/model.safetensors \
    --model-cfg-json ../build/model_introspection.json \
    --inputs ../build/dit_step_b1_fp32.ref.pt \
    --out ../build/dit_step_b1_fp32.onnx
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file

from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT


class DiTStepWrapperNoKV(nn.Module):
    """Option B: KV cache 入力なし。text/speaker_state から毎回 KV を再計算する DiT 1-step。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x_t, t, text_state, text_mask, speaker_state, speaker_mask):
        return self.model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=None,
            caption_mask=None,
            latent_mask=None,
            context_kv_cache=None,
        )


def build_model(weights: str, cfg_json: str) -> TextToLatentRFDiT:
    cfg_all = json.loads(Path(cfg_json).read_text())["model_cfg"]
    field_names = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg_kwargs = {k: v for k, v in cfg_all.items() if k in field_names}
    cfg = ModelConfig(**cfg_kwargs)
    model = TextToLatentRFDiT(cfg)
    state = load_safetensors_file(weights, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="model.safetensors path (glob 可)")
    ap.add_argument("--model-cfg-json", required=True)
    ap.add_argument("--inputs", required=True, help="capture_dit_inputs.py の .ref.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no-dynamo", action="store_true")
    ap.add_argument("--dynamic", action="store_true",
                    help="batch と latent 長を動的軸に（full-loop 検証用。NPU 用固定 shape とは別）")
    args = ap.parse_args()

    weights = sorted(glob.glob(args.weights))
    if not weights:
        raise FileNotFoundError(f"weights not found: {args.weights}")
    model = build_model(weights[0], args.model_cfg_json)

    # export-safe 化: RoPE 実数化 + SDPA bool→additive mask（IsNaN 除去）。数値等価。
    from rope_export_patch import apply_export_patches

    apply_export_patches(model)

    wrapper = DiTStepWrapperNoKV(model).eval()

    blob = torch.load(args.inputs, map_location="cpu", weights_only=False)
    ins = blob["inputs"]
    names = ["x_t", "t", "text_state", "text_mask", "speaker_state", "speaker_mask"]
    dummy = tuple(ins[n] for n in names)

    if args.fp16:
        model.half()
        dummy = tuple(d.half() if d.is_floating_point() else d for d in dummy)

    print("[shapes]")
    for n, d in zip(names, dummy):
        print(f"  {n}: {tuple(d.shape)} {d.dtype}")

    with torch.no_grad():
        ref = wrapper(*dummy)
    print(f"[ref] v_pred {tuple(ref.shape)} {ref.dtype}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # dynamic: batch(B) と latent 長(T) を Dim 記号で動的化。
    # 注: legacy dynamic_axes だと torch.export が内部 reshape の B を example 値に specialize
    #     してしまう（B=1 が焼き付く）ため、dynamic_shapes + torch.export.Dim を使う。
    dynamic_shapes = None
    if args.dynamic:
        from torch.export import Dim

        B = Dim("B", min=1, max=8)
        T = Dim("T", min=1, max=4096)
        # forward(x_t, t, text_state, text_mask, speaker_state, speaker_mask) の各引数に対応。
        dynamic_shapes = (
            {0: B, 1: T},  # x_t
            {0: B},        # t
            {0: B},        # text_state
            {0: B},        # text_mask
            {0: B},        # speaker_state
            {0: B},        # speaker_mask
        )

    exported = False
    if not args.no_dynamo:
        try:
            print("[export] dynamo exporter ...")
            with torch.no_grad():
                torch.onnx.export(
                    wrapper, dummy, str(out_path),
                    input_names=names, output_names=["v_pred"],
                    opset_version=args.opset, dynamo=True, dynamic_shapes=dynamic_shapes,
                )
            exported = True
            print("[export] dynamo OK")
        except Exception as e:
            print(f"[export] dynamo failed: {type(e).__name__}: {e}")

    if not exported:
        print("[export] torchscript exporter (dynamic 非対応, 固定 shape) ...")
        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy, str(out_path),
                input_names=names, output_names=["v_pred"],
                opset_version=args.opset, do_constant_folding=True,
            )
        print("[export] torchscript OK")

    # 数値検証 (onnxruntime)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feed = {}
    for n, d in zip(names, dummy):
        arr = d.cpu().numpy()
        if d.dtype == torch.bool:
            arr = arr.astype(np.bool_)
        feed[n] = arr
    ort_out = sess.run(["v_pred"], feed)[0]
    ref_np = ref.cpu().float().numpy()
    ort_np = ort_out.astype(np.float32)
    max_abs = float(np.max(np.abs(ref_np - ort_np)))
    denom = float(np.max(np.abs(ref_np))) + 1e-9
    print(f"[validate] max_abs_err={max_abs:.3e}  rel={max_abs/denom:.3e}  out_range=±{denom:.3f}")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
