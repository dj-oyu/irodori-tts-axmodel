#!/usr/bin/env python3
"""
Stage C (CPU, onnxruntime, no root): DACVAE decode latent -> wav.

  python3 e2e_demo/c_decode.py --latent /tmp/e2e_latent.npy --out /tmp/e2e_out.wav
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="build/dacvae_decoder.onnx")
    ap.add_argument("--latent", default="/tmp/e2e_latent.npy")
    ap.add_argument("--out", default="/tmp/e2e_out.wav")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    z = np.load(args.latent).astype(np.float32)
    print(f"[in] {in_name}={z.shape}")

    t0 = time.time()
    audio = sess.run(None, {in_name: z})[0]  # (1, 1, samples)
    wav = np.asarray(audio).reshape(-1)
    sf.write(args.out, wav, args.sample_rate)
    print(f"[saved] {args.out}  samples={wav.shape[0]}  dur={wav.shape[0]/args.sample_rate:.2f}s  "
          f"decode={time.time()-t0:.1f}s  range=±{np.abs(wav).max():.3f}")


if __name__ == "__main__":
    main()
