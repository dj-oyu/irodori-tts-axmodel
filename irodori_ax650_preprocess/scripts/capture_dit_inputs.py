#!/usr/bin/env python3
"""
capture_dit_inputs.py

実 synth を 1 回流して DiT 1-step (forward_with_encoded_conditions) の実引数と
実出力(v_pred) を .ref.pt に保存する。export_dit_step.py の入力 / 検証基準になる。

runtime(inference_runtime) を使うので **Irodori-TTS の uv env** で実行する:
  cd /path/to/Irodori-TTS
  PYTHONPATH=/path/to/Irodori-TTS uv run python \
    /path/to/irodori-tts-axmodel/irodori_ax650_preprocess/scripts/capture_dit_inputs.py \
    --out /path/to/irodori-tts-axmodel/build/dit_step_b1_fp32.ref.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-500M-v3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text", default="こんにちは、テストです。")
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1, help="保存する batch 数（CFG 実行は B=3, 既定は先頭1件）")
    args = ap.parse_args()

    ckpt = hf_hub_download(repo_id=args.hf_checkpoint, filename="model.safetensors")
    runtime = InferenceRuntime.from_key(
        RuntimeKey(checkpoint=ckpt, model_device=args.device, codec_device=args.device)
    )
    model = runtime.model
    captured: dict = {}
    orig = model.forward_with_encoded_conditions

    def wrapped(*a, **kw):
        out = orig(*a, **kw)
        if "kwargs" not in captured:
            captured["kwargs"] = kw
            captured["out"] = out.detach()
        return out

    model.forward_with_encoded_conditions = wrapped
    try:
        runtime.synthesize(SamplingRequest(text=args.text, no_ref=True, num_steps=args.num_steps))
    finally:
        model.forward_with_encoded_conditions = orig

    kw = captured["kwargs"]
    b = args.batch
    names = ["x_t", "t", "text_state", "text_mask", "speaker_state", "speaker_mask"]
    inputs = {n: kw[n][:b].contiguous().cpu() for n in names}
    v_pred = captured["out"][:b].contiguous().cpu()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"inputs": inputs, "v_pred": v_pred}, str(out_path))
    print("[saved]", out_path)
    for n, t in inputs.items():
        print(f"  {n}: {tuple(t.shape)} {t.dtype}")
    print(f"  v_pred: {tuple(v_pred.shape)} {v_pred.dtype}  range=±{v_pred.abs().max().item():.3f}")


if __name__ == "__main__":
    main()
