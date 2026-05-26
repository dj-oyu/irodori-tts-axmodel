#!/usr/bin/env python3
"""
Single-step NPU comparison: run the axmodel on the SAME (x_t, t, cond KV, masks)
that diag_step_torch.py used, and compare v_npu vs v_torch.

Run as root:
  sudo bash -c 'export PYTHONPATH=/home/admin-user/.local/lib/python3.10/site-packages; \
    cd /home/admin-user/github/irodori-tts-axmodel && python3 e2e_demo/diag_step_npu.py'
"""
from __future__ import annotations
import argparse
import numpy as np
import axengine as axe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axmodel", default="build/axmodel_kv_b1/compiled.axmodel")
    ap.add_argument("--cond", default="/tmp/e2e_cond.npz")
    ap.add_argument("--ref", default="/tmp/step_ref.npz")
    args = ap.parse_args()

    cond = np.load(args.cond)
    ref = np.load(args.ref)
    sess = axe.InferenceSession(args.axmodel)
    kv_names = [i.name for i in sess.get_inputs() if i.name not in ("x_t", "t", "text_mask", "speaker_mask")]

    feed = {"x_t": ref["x_t"].astype(np.float32), "t": ref["t"].astype(np.float32),
            "text_mask": cond["text_mask"], "speaker_mask": cond["speaker_mask"]}
    for n in kv_names:
        feed[n] = cond[f"cond_{n}"].astype(np.float32)

    v_npu = sess.run(None, feed)[0].astype(np.float32)
    np.save("/tmp/v_npu.npy", v_npu)
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
    print(f"view(1,119,32) raw    : {cos(v_npu.reshape(1,119,32), v_torch):.4f}")
    print(f"reverse seq           : {cos(v_npu[:, ::-1, :], v_torch):.4f}")
    print(f"reverse latent dim    : {cos(v_npu[:, :, ::-1], v_torch):.4f}")


if __name__ == "__main__":
    main()
