#!/usr/bin/env python3
"""Step-count reduction study (B2): how few diffusion steps keep quality vs the
32-step production setting. num_steps is a pure CPU loop param — no axmodel rebuild.

Single-process: loads DiT(allfcu16_npu3) + dacvae(b0) axengine sessions ONCE, then
loops cond(texts) × N. Mirrors e2e_npu CFG sampling (53-input + latent_mask, trim).

Metrics vs the SAME text's N=32 output (the on-device reference — no fp32 ground truth,
so this is "divergence from the 32-step production setting on the same quantized DiT"):
  - latent cosine + relative L2 (same-trajectory check; necessary not sufficient)
  - log-magnitude STFT L1 (the under-integration/roughness metric; centroid/rolloff miss it)
  - clipping rate, range, std
Also: determinism check (N=32 twice, bitwise) to establish the reference noise floor.

  sudo -n PYTHONPATH=$HOME/.local/lib/python3.10/site-packages /usr/bin/python3.10 \
    e2e_demo/step_sweep.py --cond-dir /tmp/sweep_cond --out-dir /tmp/sweep
"""
from __future__ import annotations
import argparse, math, time, wave, glob, os
import numpy as np
import axengine as axe

STEPS = [8, 10, 12, 16, 20, 24, 32]  # 32 first computed as ref; N=4 dropped (always garbage)


def sway_schedule(num_steps, sway_coeff=-1.0, init_scale=0.999):
    u = np.linspace(0.0, 1.0, num_steps + 1)
    u = u + sway_coeff * (np.cos(0.5 * math.pi * u) + u - 1.0)
    u = np.clip(u, 0.0, 1.0)
    t = (1.0 - u) * init_scale
    assert np.all(t[:-1] > t[1:])
    return t.astype(np.float32)


def write_wav(path, wav, sr):
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


def log_stft(wav, n_fft=1024, hop=256):
    w = np.hanning(n_fft)
    n = (len(wav) - n_fft) // hop
    frames = np.stack([wav[i*hop:i*hop+n_fft] * w for i in range(n)])
    return np.log(np.abs(np.fft.rfft(frames, axis=1)) + 1e-6)


def sample(dit, dac, cond, kv_names, T, latent_dim, t_valid, N, cfg_text, cfg_spk,
           cfg_min_t, cfg_max_t, seed):
    text_mask = cond["text_mask"]; speaker_mask = cond["speaker_mask"]
    zero_tm = np.zeros_like(text_mask); zero_sm = np.zeros_like(speaker_mask)
    latent_mask = np.zeros((1, T), np.uint8); latent_mask[:, :t_valid] = 1
    def feed(branch, tm, sm):
        f = {"text_mask": tm, "speaker_mask": sm, "latent_mask": latent_mask}
        for n in kv_names:
            f[n] = cond[f"{branch}_{n}"].astype(np.float32)
        return f
    feeds = {"cond": feed("cond", text_mask, speaker_mask),
             "text": feed("text", zero_tm, speaker_mask),
             "spk":  feed("spk", text_mask, zero_sm)}
    t_sched = sway_schedule(N)
    rng = np.random.default_rng(seed)
    x_t = rng.standard_normal((1, T, latent_dim)).astype(np.float32)
    def run(branch, x, t):
        f = dict(feeds[branch]); f["x_t"] = x; f["t"] = np.array([t], np.float32)
        return dit.run(None, f)[0]
    tot_ms = 0.0; calls = 0
    for i in range(N):
        t = float(t_sched[i]); tn = float(t_sched[i+1]); s = time.time()
        if cfg_min_t <= t <= cfg_max_t:
            vc = run("cond", x_t, t); vt = run("text", x_t, t); vs = run("spk", x_t, t)
            v = vc + cfg_text*(vc - vt) + cfg_spk*(vc - vs); c = 3
        else:
            v = run("cond", x_t, t); c = 1
        tot_ms += (time.time()-s)*1000; calls += c
        x_t = x_t + v * (tn - t)
    z = np.ascontiguousarray(np.transpose(x_t, (0, 2, 1))[:, :, :t_valid]).astype(np.float32)
    s = time.time(); audio = np.asarray(dac.run(None, {dac.get_inputs()[0].name: z})[0]); dac_ms = (time.time()-s)*1000
    return z, audio.reshape(-1), tot_ms, calls, dac_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", default="build/axmodel_kv_long_lm_allfcu16_npu3/compiled.axmodel")
    ap.add_argument("--dacvae", default="build/axmodel_dacvae_b0/compiled.axmodel")
    ap.add_argument("--cond-dir", default="/tmp/sweep_cond")
    ap.add_argument("--out-dir", default="/tmp/sweep")
    ap.add_argument("--t-valid", type=int, default=119)
    ap.add_argument("--cfg-text", type=float, default=3.0)
    ap.add_argument("--cfg-spk", type=float, default=5.0)
    ap.add_argument("--cfg-min-t", type=float, default=0.5)
    ap.add_argument("--cfg-max-t", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sr", type=int, default=48000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dit = axe.InferenceSession(args.dit)
    dac = axe.InferenceSession(args.dacvae)
    ish = {i.name: tuple(i.shape) for i in dit.get_inputs()}
    T = ish["x_t"][1]; latent_dim = ish["x_t"][2]
    kv = [n for n in ish if n not in ("x_t", "t", "text_mask", "speaker_mask", "latent_mask")]
    print(f"[dit] T={T} latent_dim={latent_dim} #kv={len(kv)} t_valid={args.t_valid} steps={STEPS}", flush=True)

    cond_files = sorted(glob.glob(f"{args.cond_dir}/cond_*.npz"))
    print(f"[conds] {[os.path.basename(c) for c in cond_files]}", flush=True)

    def run_N(cond, N, seed=None):
        return sample(dit, dac, cond, kv, T, latent_dim, args.t_valid, N,
                      args.cfg_text, args.cfg_spk, args.cfg_min_t, args.cfg_max_t,
                      args.seed if seed is None else seed)

    # determinism check on first cond at N=32
    if cond_files:
        c0 = np.load(cond_files[0])
        z_a, _, _, _, _ = run_N(c0, 32)
        z_b, _, _, _, _ = run_N(c0, 32)
        eq = np.array_equal(z_a, z_b)
        print(f"[determinism] N=32 x2 latent bitwise-equal: {eq}"
              f"{'' if eq else f'  (max|Δ|={np.abs(z_a-z_b).max():.2e}, cos={np.dot(z_a.ravel(),z_b.ravel())/(np.linalg.norm(z_a)*np.linalg.norm(z_b)):.6f})'}",
              flush=True)

    for cf in cond_files:
        label = os.path.basename(cf)[5:-4]  # cond_<label>.npz
        cond = np.load(cf)
        print(f"\n===== text='{label}' =====", flush=True)
        results = {}
        for N in sorted(STEPS, reverse=True):  # 32 first = reference
            z, wav, dit_ms, calls, dac_ms = run_N(cond, N)
            write_wav(f"{args.out_dir}/{label}_n{N}.wav", wav, args.sr)
            results[N] = dict(z=z, wav=wav, dit_ms=dit_ms, calls=calls, dac_ms=dac_ms)
        ref = results[32]
        ref_logS = log_stft(ref["wav"])
        print(f"{'N':>3} {'calls':>5} {'dit_ms':>7} {'ms/call':>7} {'latcos':>8} {'latL2%':>7} "
              f"{'logSTFT_L1':>10} {'clip%':>6} {'std':>6}", flush=True)
        for N in sorted(STEPS, reverse=True):
            r = results[N]
            lc = float(np.dot(r["z"].ravel(), ref["z"].ravel()) / (np.linalg.norm(r["z"])*np.linalg.norm(ref["z"])))
            l2 = float(np.linalg.norm(r["z"]-ref["z"]) / np.linalg.norm(ref["z"]) * 100)
            ls = float(np.mean(np.abs(log_stft(r["wav"]) - ref_logS)))
            clip = float(np.mean(np.abs(r["wav"]) >= 0.999) * 100)
            print(f"{N:>3} {r['calls']:>5} {r['dit_ms']:>7.0f} {r['dit_ms']/r['calls']:>7.1f} "
                  f"{lc:>8.5f} {l2:>7.2f} {ls:>10.4f} {clip:>6.3f} {r['wav'].std():>6.4f}", flush=True)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
