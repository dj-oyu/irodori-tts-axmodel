#!/usr/bin/env python3
"""Bench Stage B (DiT NPU) + Stage C (dacvae NPU): statistically reliable per-phase
timing. Loads each axmodel once, then runs many iterations over the cond_*.npz set.

Records: model load times; per-sampling-iteration total; EVERY individual DiT sess.run
latency (large n); per-dacvae-decode latency. Emits mean/std/median/min/max/p95/CV +
timings.json. DiT/dacvae are fixed-shape so timing is ~text-independent (a finding).
"""
from __future__ import annotations
import argparse, glob, json, math, time
import numpy as np, axengine as axe

def sway_schedule(num_steps, sway_coeff=-1.0, init_scale=0.999):
    u = np.linspace(0.0, 1.0, num_steps + 1)
    u = u + sway_coeff * (np.cos(0.5 * math.pi * u) + u - 1.0)
    u = np.clip(u, 0.0, 1.0)
    t = (1.0 - u) * init_scale
    return t.astype(np.float32)

def stats(a):
    a = np.asarray(a, float)
    return {"n": int(a.size), "mean": round(float(a.mean()), 2),
            "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 2),
            "median": round(float(np.median(a)), 2), "min": round(float(a.min()), 2),
            "max": round(float(a.max()), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "cv_pct": round(float(a.std(ddof=1) / a.mean() * 100) if a.size > 1 and a.mean() else 0.0, 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", default="build/axmodel_kv_long_lm_cosu16/compiled.axmodel")
    ap.add_argument("--dacvae", default="build/axmodel_dacvae_b0/compiled.axmodel")
    ap.add_argument("--cond-dir", default="/tmp/bench_cond")
    ap.add_argument("--reps", type=int, default=2, help="passes over the cond set")
    ap.add_argument("--num-steps", type=int, default=32)
    ap.add_argument("--t-valid", type=int, default=119)
    ap.add_argument("--cfg-text", type=float, default=3.0)
    ap.add_argument("--cfg-spk", type=float, default=5.0)
    ap.add_argument("--cfg-min-t", type=float, default=0.5)
    ap.add_argument("--cfg-max-t", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="/tmp/bench_cond/npu_timings.json")
    args = ap.parse_args()

    t0 = time.time(); dit = axe.InferenceSession(args.dit); dit_load = time.time() - t0
    t0 = time.time(); dac = axe.InferenceSession(args.dacvae); dac_load = time.time() - t0
    ish = {i.name: tuple(i.shape) for i in dit.get_inputs()}
    T = ish["x_t"][1]; D = ish["x_t"][2]
    reserved = ("x_t", "t", "text_mask", "speaker_mask", "latent_mask")
    kv_names = [n for n in ish if n not in reserved]
    zin = dac.get_inputs()[0].name
    latent_mask = np.zeros((1, T), np.uint8); latent_mask[:, :args.t_valid] = 1
    t_sched = sway_schedule(args.num_steps)
    conds = sorted(glob.glob(f"{args.cond_dir}/cond_*.npz"))
    print(f"[load] dit={dit_load:.2f}s dacvae={dac_load:.2f}s | T={T} conds={len(conds)} reps={args.reps}")

    call_ms, sampl_s, dac_ms, per_text = [], [], [], {}
    for rep in range(args.reps):
        for cp in conds:
            cond = np.load(cp)
            tm, sm = cond["text_mask"], cond["speaker_mask"]
            ztm, zsm = np.zeros_like(tm), np.zeros_like(sm)
            def bf(branch, t_, s_):
                f = {"text_mask": t_, "speaker_mask": s_, "latent_mask": latent_mask}
                for n in kv_names: f[n] = cond[f"{branch}_{n}"].astype(np.float32)
                return f
            feeds = {"cond": bf("cond", tm, sm), "text": bf("text", ztm, sm), "spk": bf("spk", tm, zsm)}
            rng = np.random.default_rng(args.seed)
            x_t = rng.standard_normal((1, T, D)).astype(np.float32)
            def run(b, x, t):
                f = dict(feeds[b]); f["x_t"] = x; f["t"] = np.array([t], np.float32)
                s = time.time(); v = dit.run(None, f)[0]; call_ms.append((time.time()-s)*1000); return v
            s_all = time.time()
            for i in range(args.num_steps):
                t = float(t_sched[i]); tn = float(t_sched[i+1])
                if args.cfg_min_t <= t <= args.cfg_max_t:
                    vc = run("cond", x_t, t); vt = run("text", x_t, t); vs = run("spk", x_t, t)
                    v = vc + args.cfg_text*(vc-vt) + args.cfg_spk*(vc-vs)
                else:
                    v = run("cond", x_t, t)
                x_t = x_t + v*(tn-t)
            samp = time.time() - s_all; sampl_s.append(samp)
            z = np.ascontiguousarray(np.transpose(x_t, (0,2,1))[:, :, :args.t_valid])
            s = time.time(); dac.run(None, {zin: z}); dms = (time.time()-s)*1000; dac_ms.append(dms)
            per_text.setdefault(cp.split('/')[-1], []).append(round(samp, 2))
            print(f"  rep{rep} {cp.split('/')[-1]}: sampling={samp:.2f}s dacvae={dms:.0f}ms")

    result = {"phase": "npu_dit_dacvae", "num_steps": args.num_steps, "T_max": T,
              "dit_load_s": round(dit_load, 2), "dacvae_load_s": round(dac_load, 2),
              "dit_sampling_s": stats(sampl_s), "dit_per_call_ms": stats(call_ms),
              "dacvae_decode_ms": stats(dac_ms), "per_text_sampling_s": per_text}
    import os; os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    open(args.out_json, "w").write(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n=== STATS ===")
    print(f"DiT sampling/utterance (s): {result['dit_sampling_s']}")
    print(f"DiT per-call (ms):          {result['dit_per_call_ms']}")
    print(f"dacvae decode (ms):         {result['dacvae_decode_ms']}")

if __name__ == "__main__":
    main()
