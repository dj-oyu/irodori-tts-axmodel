#!/usr/bin/env python3
"""DiT (kv_long_lm_cosu16) NPU smoke + runtime latent_mask functional test.

Memory-light: feeds zeros for the 48 KV inputs (no torch / no Stage A), random x_t,
to confirm the 53-input axmodel EXECUTES on the NPU (no SEGV / CPU fallback) and to
measure per-call latency for the deploy budget. Then runs twice — full latent_mask vs
partial (valid only [:T_valid]) — to prove latent_mask is *live* (changes output),
not merely present as a graph input. This is wiring/liveness, NOT numerical correctness
(no fp32 reference, since the 2GB torch load fails the memory gate tonight).
"""
from __future__ import annotations
import argparse, time
import numpy as np
import axengine as axe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axmodel", default="build/axmodel_kv_long_lm_cosu16/compiled.axmodel")
    ap.add_argument("--t", type=float, default=0.8)
    ap.add_argument("--t-valid", type=int, default=119, help="valid latent frames for partial-mask run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sess = axe.InferenceSession(args.axmodel)
    ins = {i.name: (tuple(i.shape), str(i.dtype)) for i in sess.get_inputs()}
    T = ins["x_t"][0][1]
    print(f"[model] {len(ins)} inputs, x_t T={T}")

    rng = np.random.default_rng(args.seed)
    x_t = rng.standard_normal(ins["x_t"][0]).astype(np.float32)

    def base_feed(latent_valid):
        feed = {}
        for name, (shape, dt) in ins.items():
            if name == "x_t":
                feed[name] = x_t
            elif name == "t":
                feed[name] = np.array([args.t], np.float32)
            elif name == "latent_mask":
                m = np.zeros(shape, np.uint8); m[:, :latent_valid] = 1
                feed[name] = m
            elif name in ("text_mask", "speaker_mask"):
                feed[name] = np.ones(shape, np.uint8)
            else:  # KV caches -> zeros
                feed[name] = np.zeros(shape, np.float32)
        return feed

    # smoke + latency (full mask)
    full = base_feed(T)
    lat = []
    for _ in range(3):
        s = time.time(); v_full = sess.run(None, full)[0]; lat.append((time.time()-s)*1000)
    v_full = np.asarray(v_full)
    print(f"[smoke] v_pred={v_full.shape} finite={np.isfinite(v_full).all()} "
          f"std={v_full.std():.4f} latency={min(lat):.0f}/{np.mean(lat):.0f}ms (min/avg of 3)")

    # latent_mask liveness: full vs partial
    part = base_feed(args.t_valid)
    v_part = np.asarray(sess.run(None, part)[0])
    diff = np.abs(v_full - v_part)
    # split valid [:t_valid] vs masked [t_valid:] regions along T (axis=1)
    valid_d = float(diff[:, :args.t_valid, :].mean())
    masked_d = float(diff[:, args.t_valid:, :].mean())
    changed = float(diff.max())
    print(f"[latent_mask] full-vs-partial(valid={args.t_valid}/{T}): "
          f"max|Δ|={changed:.4f} mean|Δ|valid={valid_d:.4f} mean|Δ|masked={masked_d:.4f}")
    if changed > 1e-6:
        print("  [OK] latent_mask is LIVE (changes v_pred) — not a pruned/dead input")
    else:
        print("  [WARN] latent_mask changed nothing — possibly dead/pruned at runtime")
    print("[done]")

if __name__ == "__main__":
    main()
