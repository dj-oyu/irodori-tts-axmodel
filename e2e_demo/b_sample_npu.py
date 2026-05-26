#!/usr/bin/env python3
"""
Stage B (NPU, axengine, ROOT): Euler RF sampling with CFG using the DiT axmodel.

Mirrors irodori_tts.rf.sample_euler_rf_cfg (independent CFG, branches text+speaker):
  while cfg_min_t <= t <= cfg_max_t:
    v = v_cond + cfg_text*(v_cond - v_text) + cfg_spk*(v_cond - v_spk)
  else:
    v = v_cond
  x_t += v * (t_next - t)

Three KV caches (cond / text-uncond / spk-uncond) come from stage A in /tmp/e2e_cond.npz.
The b1 axmodel is called once per active branch per step (so up to 3x while 0.5<=t<=1.0).

Run as root (axmodel needs /dev/mem); PYTHONPATH to axengine --user site:
  sudo bash -c 'export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages; \
    cd <repo> && python3 e2e_demo/b_sample_npu.py'
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import axengine as axe


def sway_schedule(num_steps: int, sway_coeff: float = -1.0, init_scale: float = 0.999) -> np.ndarray:
    u = np.linspace(0.0, 1.0, num_steps + 1)
    u = u + sway_coeff * (np.cos(0.5 * math.pi * u) + u - 1.0)
    u = np.clip(u, 0.0, 1.0)
    t_schedule = (1.0 - u) * init_scale
    assert np.all(t_schedule[:-1] > t_schedule[1:]), "t_schedule must be strictly decreasing"
    return t_schedule.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axmodel", default="build/axmodel_kv_b1/compiled.axmodel")
    ap.add_argument("--cond", default="/tmp/e2e_cond.npz")
    ap.add_argument("--out", default="/tmp/e2e_latent.npy")
    ap.add_argument("--seq-len", type=int, default=119)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--sway-coeff", type=float, default=-1.0)
    ap.add_argument("--cfg-text", type=float, default=3.0)
    ap.add_argument("--cfg-spk", type=float, default=5.0)
    ap.add_argument("--cfg-min-t", type=float, default=0.5)
    ap.add_argument("--cfg-max-t", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cond = np.load(args.cond)
    sess = axe.InferenceSession(args.axmodel)
    kv_names = [i.name for i in sess.get_inputs() if i.name not in ("x_t", "t", "text_mask", "speaker_mask")]

    text_mask = cond["text_mask"]
    speaker_mask = cond["speaker_mask"]
    zero_text_mask = np.zeros_like(text_mask)
    zero_speaker_mask = np.zeros_like(speaker_mask)

    def branch_feed(branch: str, tmask: np.ndarray, smask: np.ndarray) -> dict:
        feed = {"text_mask": tmask, "speaker_mask": smask}
        for n in kv_names:  # n like "k_text_0"; stored as "<branch>_k_text_0"
            feed[n] = cond[f"{branch}_{n}"].astype(np.float32)
        return feed
    # static per-branch feeds (KV + masks); x_t/t added per step.
    feeds = {
        "cond": branch_feed("cond", text_mask, speaker_mask),
        "text": branch_feed("text", zero_text_mask, speaker_mask),
        "spk":  branch_feed("spk", text_mask, zero_speaker_mask),
    }

    t_schedule = sway_schedule(args.num_steps, args.sway_coeff)
    rng = np.random.default_rng(args.seed)
    x_t = rng.standard_normal((1, args.seq_len, args.latent_dim)).astype(np.float32)

    def run(branch: str, x: np.ndarray, t: float) -> np.ndarray:
        f = dict(feeds[branch])
        f["x_t"] = x
        f["t"] = np.array([t], dtype=np.float32)
        return sess.run(None, f)[0]

    total_ms = 0.0
    for i in range(args.num_steps):
        t = float(t_schedule[i])
        t_next = float(t_schedule[i + 1])
        s = time.time()
        if args.cfg_min_t <= t <= args.cfg_max_t:
            v_cond = run("cond", x_t, t)
            v_text = run("text", x_t, t)
            v_spk = run("spk", x_t, t)
            v = v_cond + args.cfg_text * (v_cond - v_text) + args.cfg_spk * (v_cond - v_spk)
            calls = 3
        else:
            v = run("cond", x_t, t)
            calls = 1
        dt_ms = (time.time() - s) * 1000.0
        total_ms += dt_ms
        x_t = x_t + v * (t_next - t)
        print(f"  step {i}: t={t:.4f}->{t_next:.4f} cfg={'on' if calls==3 else 'off'} "
              f"({calls} call) {dt_ms:.1f}ms")

    z = np.transpose(x_t, (0, 2, 1)).astype(np.float32)  # (1, latent_dim, T) for DACVAE
    np.save(args.out, z)
    print(f"[saved] {args.out} z={z.shape} std={z.std():.3f} | total NPU {total_ms:.0f}ms")


if __name__ == "__main__":
    main()
