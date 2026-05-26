#!/usr/bin/env python3
"""dacvae on-device check: NPU axmodel smoke + fp32-onnx vs NPU equivalence @ T=119.

Memory-light (no torch / no 2GB weights). Feeds the SAME z=(1,32,119) to:
  - build/axmodel_dacvae_b0/compiled.axmodel  (NPU, axengine, root)
  - build/dacvae_decoder.onnx                 (fp32 CPU ref, onnxruntime)
and reports finiteness, range, and SNR/corr between the two waveforms. This is a
vocoder-only fp32-vs-NPU equivalence probe (the one Ph2-flavored check not blocked
by the missing build-host sim outputs). Random z is an in-distribution-ish proxy;
treat SNR as wiring/sanity, not a perceptual quality verdict.
"""
from __future__ import annotations
import argparse, time
import numpy as np

def snr_db(ref, test):
    ref = ref.reshape(-1); test = test.reshape(-1)
    n = min(len(ref), len(test)); ref, test = ref[:n], test[:n]
    noise = test - ref
    p = float(np.sum(ref**2)); e = float(np.sum(noise**2))
    return 10*np.log10(p/e) if e > 0 else float("inf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axmodel", default="build/axmodel_dacvae_b0/compiled.axmodel")
    ap.add_argument("--onnx", default="build/dacvae_decoder.onnx")
    ap.add_argument("--T", type=int, default=119)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/tmp/dacvae_equiv")
    args = ap.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    z = rng.standard_normal((1, 32, args.T)).astype(np.float32)
    np.save(f"{args.out_dir}/z.npy", z)
    print(f"[z] {z.shape} std={z.std():.3f}")

    # NPU
    import axengine as axe
    sess = axe.InferenceSession(args.axmodel)
    zin = sess.get_inputs()[0].name
    t0 = time.time()
    a_npu = sess.run(None, {zin: z})[0]
    npu_ms = (time.time()-t0)*1000
    a_npu = np.asarray(a_npu)
    print(f"[npu] in={zin} out={a_npu.shape} finite={np.isfinite(a_npu).all()} "
          f"range=±{np.abs(a_npu).max():.3f} std={a_npu.std():.4f} {npu_ms:.0f}ms")
    np.save(f"{args.out_dir}/audio_npu.npy", a_npu)

    # fp32 onnx ref
    a_onnx = None
    try:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 8
        osess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
        oin = osess.get_inputs()[0].name
        t0 = time.time()
        a_onnx = np.asarray(osess.run(None, {oin: z})[0])
        onnx_ms = (time.time()-t0)*1000
        print(f"[onnx] in={oin} out={a_onnx.shape} finite={np.isfinite(a_onnx).all()} "
              f"range=±{np.abs(a_onnx).max():.3f} std={a_onnx.std():.4f} {onnx_ms:.0f}ms")
        np.save(f"{args.out_dir}/audio_onnx.npy", a_onnx)
    except Exception as e:
        print(f"[onnx] FAILED to run fp32 ref: {e}")

    if a_onnx is not None:
        rn, tn = a_onnx.reshape(-1), a_npu.reshape(-1)
        n = min(len(rn), len(tn))
        if len(rn) != len(tn):
            print(f"[WARN] length mismatch onnx={len(rn)} npu={len(tn)} -> comparing first {n}")
        corr = float(np.corrcoef(rn[:n], tn[:n])[0,1])
        print(f"[equiv] SNR={snr_db(a_onnx, a_npu):.2f}dB corr={corr:.4f}  "
              f"(sanity: vocoder single-pass should be high; §6 expects ~6dB-ish on real calib)")
    print("[done]")

if __name__ == "__main__":
    main()
