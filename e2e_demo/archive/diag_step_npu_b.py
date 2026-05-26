#!/usr/bin/env python3
"""
Option-B single-step NPU compare: feed the SAME (x_t,t,text_state,text_mask,
speaker_state,speaker_mask) from diag_step_torch_b.py into axmodel_b1 and compare
v_npu vs v_torch (cosine + layout hypotheses). Confirms whether the new axmodel is
numerically correct.

Run as root:
  sudo bash -c 'export PYTHONPATH=/home/admin-user/.local/lib/python3.10/site-packages; \
    cd /home/admin-user/github/irodori-tts-axmodel && python3 e2e_demo/diag_step_npu_b.py'
"""
from __future__ import annotations
import argparse
import numpy as np
import axengine as axe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axmodel", default="build/axmodel_b1/compiled.axmodel")
    ap.add_argument("--ref", default="/tmp/step_ref_b.npz")
    args = ap.parse_args()

    ref = np.load(args.ref)
    sess = axe.InferenceSession(args.axmodel)
    print("=== axmodel inputs ===")
    for i in sess.get_inputs():
        print(f"  {i.name}: shape={i.shape} dtype={i.dtype}")

    feed = {}
    for i in sess.get_inputs():
        n = i.name
        if n not in ref.files:
            raise SystemExit(f"axmodel input '{n}' not in ref npz keys {list(ref.files)}")
        arr = ref[n]
        # masks stored uint8; pass through. floats -> float32.
        feed[n] = arr if arr.dtype == np.uint8 else arr.astype(np.float32)

    v_npu = sess.run(None, feed)[0].astype(np.float32)
    np.save("/tmp/v_npu_b.npy", v_npu)
    v_torch = ref["v_torch"].astype(np.float32)

    def cos(a, b):
        a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print(f"v_torch std={v_torch.std():.4f} range=[{v_torch.min():.3f},{v_torch.max():.3f}] shape={v_torch.shape}")
    print(f"v_npu   std={v_npu.std():.4f} range=[{v_npu.min():.3f},{v_npu.max():.3f}] shape={v_npu.shape}")
    print("--- layout hypotheses (cosine vs v_torch) ---")
    print(f"identity              : {cos(v_npu, v_torch):.4f}")
    print(f"swap last two (0,2,1) : {cos(v_npu.transpose(0,2,1), v_torch):.4f}")
    print(f"view(1,32,119).T      : {cos(v_npu.reshape(1,32,119).transpose(0,2,1), v_torch):.4f}")


if __name__ == "__main__":
    main()
