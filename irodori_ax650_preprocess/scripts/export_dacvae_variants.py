#!/usr/bin/env python3
"""
Export DACVAE decoder variants for Pulsar2 tiler/broadcast experiments.

Variants:
  --variant expand : Snake monkeypatched so alpha is pre-expanded to (1,C,T)
                     (no runtime (1,C,1)x(1,C,T) broadcast). Tests the small-T
                     "broadcast dim 2: T vs 1536" build failure.
  --variant plain  : unchanged Snake (baseline, same as production export).

Run in Irodori venv with onnx extras:
  cd /home/exe/ai/Irodori-TTS
  PYTHONPATH=/home/exe/ai/Irodori-TTS .venv/bin/python ... is NOT enough (needs onnx>=4).
  Use: uv run --with onnx --with onnxscript --with "protobuf>=4.25" python <this> ...
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import dacvae.nn.layers as Lmod
from irodori_tts.codec import DACVAECodec


def patch_snake_expand():
    """Replace Snake1d.forward to pre-expand alpha to full time length (no broadcast)."""
    def forward(self, x):
        # x: (B, C, T). alpha: (1, C, 1). Pre-expand alpha to (B, C, T) explicitly.
        shape = x.shape
        x2 = x.reshape(shape[0], shape[1], -1)
        T = x2.shape[-1]
        a = self.alpha.expand(x2.shape[0], self.alpha.shape[1], T)  # (B,C,T)
        recip = (self.alpha + 1e-9).reciprocal().expand(x2.shape[0], self.alpha.shape[1], T)
        x2 = x2 + recip * torch.sin(a * x2).pow(2)
        return x2.reshape(shape)
    Lmod.Snake1d.forward = forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent-len", type=int, default=32)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--variant", choices=["expand", "plain", "cos", "mulmul"], default="expand")
    args = ap.parse_args()

    if args.variant == "expand":
        patch_snake_expand()
        print("[patch] Snake1d -> pre-expanded alpha (no broadcast)")
    elif args.variant == "cos":
        def forward_cos(self, x):
            # sin^2(a x) = (1 - cos(2 a x)) / 2  -- avoids Sin+Pow, no fused-Snake pattern
            shape = x.shape
            x2 = x.reshape(shape[0], shape[1], -1)
            recip = (self.alpha + 1e-9).reciprocal()
            x2 = x2 + recip * 0.5 * (1.0 - torch.cos(2.0 * self.alpha * x2))
            return x2.reshape(shape)
        Lmod.Snake1d.forward = forward_cos
        print("[patch] Snake1d -> cos identity (1-cos(2ax))/2")
    elif args.variant == "mulmul":
        def forward_mm(self, x):
            shape = x.shape
            x2 = x.reshape(shape[0], shape[1], -1)
            recip = (self.alpha + 1e-9).reciprocal()
            s = torch.sin(self.alpha * x2)
            x2 = x2 + recip * (s * s)
            return x2.reshape(shape)
        Lmod.Snake1d.forward = forward_mm
        print("[patch] Snake1d -> sin*sin (no Pow)")

    codec = DACVAECodec.load(
        repo_id=args.codec_repo, device="cpu", dtype=torch.float32,
        deterministic_encode=True, deterministic_decode=True,
    )
    m = codec.model.eval()

    # fold old-style weight_norm hooks (the production fold_weight_norm is a no-op
    # because dacvae uses torch.nn.utils.weight_norm, not parametrize).
    n_folded = 0
    for mod in m.modules():
        if hasattr(mod, "weight_g") and hasattr(mod, "weight_v"):
            try:
                torch.nn.utils.remove_weight_norm(mod)
                n_folded += 1
            except Exception:
                pass
    print(f"[fold] removed weight_norm from {n_folded} modules (hook-based)")

    class W(nn.Module):
        def __init__(self, model): super().__init__(); self.model = model
        def forward(self, z): return self.model.decode(z)

    wrapper = W(m).eval()
    z = torch.randn(1, args.latent_dim, args.latent_len, dtype=torch.float32)
    with torch.no_grad():
        ref = wrapper(z)
    print(f"[ref] decode {tuple(z.shape)} -> {tuple(ref.shape)}")

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, (z,), str(out_path),
                          input_names=["z"], output_names=["audio"],
                          opset_version=args.opset, dynamo=True)
    print("[export] OK")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    out = sess.run(["audio"], {"z": z.numpy()})[0]
    err = float(np.max(np.abs(ref.numpy() - out)))
    print(f"[validate] max_abs_err={err:.3e}  wrote {out_path}")


if __name__ == "__main__":
    main()
