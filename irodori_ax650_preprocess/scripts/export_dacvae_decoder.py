#!/usr/bin/env python3
"""
export_dacvae_decoder.py

DACVAE decoder (codec.model.decode) を固定 shape ONNX 化する。
CPU オフロード削減（#1）の本命。A55 で最重量の conv vocoder を NPU へ。

注意:
- DACVAE は dacvae package（Irodori env のみ）に依存するため **Irodori env で実行**。
  ただし torch.onnx.export は onnx(protobuf>=4) を要求し Irodori env の pin(3.19.6) と衝突するので、
  `uv run --with onnx --with onnxscript --with "protobuf>=4.25"` で実行する。
- weight_norm(parametrize, 62 個) を fold してから export（remove_parametrizations）。
- Snake1d は sin ベース → ONNX Sin に出る。Pulsar2 の対応は build で要確認。

実行:
  cd /path/to/Irodori-TTS
  PYTHONPATH=/path/to/Irodori-TTS uv run --with onnx --with onnxscript --with "protobuf>=4.25" \
    python /path/to/irodori-tts-axmodel/irodori_ax650_preprocess/scripts/export_dacvae_decoder.py \
    --out /path/to/irodori-tts-axmodel/build/dacvae_decoder.onnx --latent-len 119
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from irodori_tts.codec import DACVAECodec


class DecoderWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, z):  # z: (B, 32, T) -> (B, 1, samples)
        return self.model.decode(z)


def fold_weight_norm(m: nn.Module) -> int:
    n = 0
    for _, mod in m.named_modules():
        ph = getattr(mod, "parametrizations", None)
        if ph is not None and "weight" in ph:
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent-len", type=int, default=119)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    codec = DACVAECodec.load(
        repo_id=args.codec_repo, device="cpu", dtype=torch.float32,
        deterministic_encode=True, deterministic_decode=True,
    )
    m = codec.model.eval()
    folded = fold_weight_norm(m)
    print(f"[fold] removed weight_norm from {folded} modules")

    wrapper = DecoderWrapper(m).eval()
    z = torch.randn(1, args.latent_dim, args.latent_len, dtype=torch.float32)
    with torch.no_grad():
        ref = wrapper(z)
    print(f"[ref] decode {tuple(z.shape)} -> {tuple(ref.shape)} {ref.dtype}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (z,), str(out_path),
            input_names=["z"], output_names=["audio"],
            opset_version=args.opset, dynamo=True,
        )
    print("[export] OK")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    out = sess.run(["audio"], {"z": z.numpy()})[0]
    err = float(np.max(np.abs(ref.numpy() - out)))
    print(f"[validate] max_abs_err={err:.3e}")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
