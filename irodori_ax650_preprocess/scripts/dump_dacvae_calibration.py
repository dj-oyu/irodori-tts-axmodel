#!/usr/bin/env python3
"""
dump_dacvae_calibration.py

DACVAE decoder の Pulsar2 PTQ 用 **実 activation calibration** を作る。
入力は decoder への latent `z` (1,32,T)。実 synth を回して decode 直前の z を捕捉し、
入力ごと（z 1 本）の Numpy calib tar.gz を書く。ランダム z だと量子化品質が落ちる
（DiT と同じ。研究ビルドの cos 0.843 はランダム calib が一因）。

固定 shape 制約: 出荷 axmodel は b1・固定 T（既定 T=119）。**T=119 を出す発話**で集める必要があるため、
既定テキスト "こんにちは、テストです。"（T=119）を seed 違いで複数回流して多様性を出す。

Irodori env で実行（runtime + dacvae が要る）:
  cd /home/exe/ai/Irodori-TTS
  PYTHONPATH=/home/exe/ai/Irodori-TTS uv run python \
    /home/exe/ai/irodori-tts-axmodel/irodori_ax650_preprocess/scripts/dump_dacvae_calibration.py \
    --out-dir /home/exe/ai/irodori-tts-axmodel/build/calib_dacvae --num-synth 32 --latent-len 119
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest


def _find_codec(rt):
    """runtime から DACVAECodec（.model.decode を持つ）を探す。"""
    for attr in ("codec", "_codec", "vae", "dacvae"):
        c = getattr(rt, attr, None)
        if c is not None and hasattr(c, "model") and hasattr(c.model, "decode"):
            return c
    # fallback: 属性総当り
    for name in dir(rt):
        c = getattr(rt, name, None)
        if hasattr(c, "model") and hasattr(getattr(c, "model"), "decode"):
            return c
    raise RuntimeError("codec(.model.decode) が runtime 上に見つからない。実構造を確認のこと。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-500M-v3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text", default="こんにちは、テストです。")
    ap.add_argument("--num-synth", type=int, default=32, help="seed を変えて流す回数（= calib サンプル数）")
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--latent-len", type=int, default=119, help="採用する z の T（axmodel 固定長と一致必須）")
    args = ap.parse_args()

    ckpt = hf_hub_download(repo_id=args.hf_checkpoint, filename="model.safetensors")
    rt = InferenceRuntime.from_key(
        RuntimeKey(checkpoint=ckpt, model_device=args.device, codec_device=args.device)
    )
    codec = _find_codec(rt)
    orig = codec.model.decode
    captured: list[np.ndarray] = []

    def hook(z, *a, **kw):
        captured.append(z[:1].detach().float().cpu().numpy())  # (1,32,T)
        return orig(z, *a, **kw)

    codec.model.decode = hook
    try:
        for s in range(args.num_synth):
            rt.synthesize(SamplingRequest(text=args.text, no_ref=True,
                                          num_steps=args.num_steps, seed=s))
    finally:
        codec.model.decode = orig

    # T が一致するものだけ採用（duration は seed 不変のはずだが安全側で filter）
    samples = [z for z in captured if z.shape[-1] == args.latent_len]
    dropped = len(captured) - len(samples)
    print(f"[calib] captured {len(captured)}, kept {len(samples)} (T=={args.latent_len}), dropped {dropped}")
    if not samples:
        raise RuntimeError(f"T={args.latent_len} の z が 0 件。--latent-len を実 T に合わせるか text を変える。")
    print(f"  z shape {samples[0].shape} {samples[0].dtype}")

    out_dir = Path(args.out_dir)
    d = out_dir / "z"
    d.mkdir(parents=True, exist_ok=True)
    for j, z in enumerate(samples):
        np.save(d / f"{j:04d}.npy", z.astype(np.float32))
    tar_path = out_dir / "z.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for j in range(len(samples)):
            tf.add(d / f"{j:04d}.npy", arcname=f"{j:04d}.npy")
    print(f"[done] wrote {tar_path} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
