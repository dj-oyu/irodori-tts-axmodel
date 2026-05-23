#!/usr/bin/env python3
"""
export_text_encoder.py

TextEncoder (TextToLatentRFDiT.text_encoder) を固定 shape ONNX 化する。
CPU オフロード削減（#1）の一環。DiT と同型（RoPE + SelfAttention/SDPA + RMSNorm）なので
既存 export patch（apply_export_patches）でそのまま行ける。

I/O: input_ids[1,256] int64, mask[1,256] bool -> text_state[1,256,512]

実行（preprocess env）:
  cd /path/to/irodori-tts-axmodel/irodori_ax650_preprocess
  PYTHONPATH=/path/to/Irodori-TTS uv run python scripts/export_text_encoder.py \
    --weights ~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v3/snapshots/*/model.safetensors \
    --model-cfg-json ../build/model_introspection.json --out ../build/text_encoder.onnx
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


class TextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder: nn.Module):
        super().__init__()
        self.enc = text_encoder

    def forward(self, input_ids, mask):
        return self.enc(input_ids, mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model-cfg-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg_all = json.loads(Path(args.model_cfg_json).read_text())["model_cfg"]
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_all.items() if k in fields})
    model = TextToLatentRFDiT(cfg)
    w = sorted(glob.glob(args.weights))[0]
    miss, unexp = model.load_state_dict(load_safetensors_file(w, device="cpu"), strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)}")

    from rope_export_patch import apply_export_patches

    apply_export_patches(model)
    model.eval()
    wrapper = TextEncoderWrapper(model.text_encoder).eval()

    input_ids = torch.arange(args.seq, dtype=torch.int64).unsqueeze(0) % cfg.text_vocab_size
    mask = torch.ones(1, args.seq, dtype=torch.bool)
    dummy = (input_ids, mask)

    with torch.no_grad():
        ref = wrapper(*dummy)
    print(f"[ref] text_state {tuple(ref.shape)} {ref.dtype}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(out_path),
            input_names=["input_ids", "mask"], output_names=["text_state"],
            opset_version=args.opset, dynamo=True,
        )
    print("[export] OK")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feed = {"input_ids": input_ids.numpy(), "mask": mask.numpy()}
    feed = {n: feed[n] for n in [i.name for i in sess.get_inputs()]}
    out = sess.run(["text_state"], feed)[0]
    err = float(np.max(np.abs(ref.numpy() - out)))
    print(f"[validate] max_abs_err={err:.3e}")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
