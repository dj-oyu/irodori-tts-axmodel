#!/usr/bin/env python3
"""Stage B+C on NPU (axengine, root): DiT(53-input, latent_mask) sampling -> trim
-> DACVAE(T=119) decode -> wav. Composes the deploy DiT(T=201) with dacvae_b0(T=119)
by generating at T_max=201 with latent_mask valid=[:T_valid] and trimming to T_valid.

Mirrors b_sample_npu.py CFG sampling, adds latent_mask to every feed (53rd input).
"""
from __future__ import annotations
import argparse, math, time, wave
import numpy as np
import axengine as axe

def sway_schedule(num_steps, sway_coeff=-1.0, init_scale=0.999):
    u = np.linspace(0.0, 1.0, num_steps + 1)
    u = u + sway_coeff * (np.cos(0.5 * math.pi * u) + u - 1.0)
    u = np.clip(u, 0.0, 1.0)
    t = (1.0 - u) * init_scale
    assert np.all(t[:-1] > t[1:])
    return t.astype(np.float32)

def write_wav(path, wav, sr):
    x = np.clip(wav, -1.0, 1.0)
    pcm = (x * 32767.0).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dit", default="build/axmodel_kv_long_lm_cosu16/compiled.axmodel")
    ap.add_argument("--dacvae", default="build/axmodel_dacvae_b0/compiled.axmodel")
    ap.add_argument("--cond", default="/tmp/e2e_cond.npz")
    ap.add_argument("--t-valid", type=int, default=119, help="real latent frames (= dacvae T)")
    ap.add_argument("--num-steps", type=int, default=32)
    ap.add_argument("--cfg-text", type=float, default=3.0)
    ap.add_argument("--cfg-spk", type=float, default=5.0)
    ap.add_argument("--cfg-min-t", type=float, default=0.5)
    ap.add_argument("--cfg-max-t", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cfg", action="store_true", help="cond-only (no CFG); cond needs only cond_* keys")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--out-wav", default="/tmp/e2e_out.wav")
    ap.add_argument("--out-latent", default="/tmp/e2e_latent.npy")
    args = ap.parse_args()

    cond = np.load(args.cond)
    dit = axe.InferenceSession(args.dit)
    ishapes = {i.name: tuple(i.shape) for i in dit.get_inputs()}
    T = ishapes["x_t"][1]; latent_dim = ishapes["x_t"][2]
    reserved = ("x_t", "t", "text_mask", "speaker_mask", "latent_mask")
    kv_names = [n for n in ishapes if n not in reserved]
    print(f"[dit] T_max={T} latent_dim={latent_dim} #kv={len(kv_names)} t_valid={args.t_valid}")

    text_mask = cond["text_mask"]; speaker_mask = cond["speaker_mask"]
    zero_tm = np.zeros_like(text_mask); zero_sm = np.zeros_like(speaker_mask)
    latent_mask = np.zeros((1, T), np.uint8); latent_mask[:, :args.t_valid] = 1

    def branch_feed(branch, tm, sm):
        f = {"text_mask": tm, "speaker_mask": sm, "latent_mask": latent_mask}
        for n in kv_names:
            f[n] = cond[f"{branch}_{n}"].astype(np.float32)
        return f
    feeds = {"cond": branch_feed("cond", text_mask, speaker_mask)}
    if not args.no_cfg:
        feeds["text"] = branch_feed("text", zero_tm, speaker_mask)
        feeds["spk"] = branch_feed("spk", text_mask, zero_sm)

    t_sched = sway_schedule(args.num_steps)
    rng = np.random.default_rng(args.seed)
    x_t = rng.standard_normal((1, T, latent_dim)).astype(np.float32)

    def run(branch, x, t):
        f = dict(feeds[branch]); f["x_t"] = x; f["t"] = np.array([t], np.float32)
        return dit.run(None, f)[0]

    tot = 0.0
    for i in range(args.num_steps):
        t = float(t_sched[i]); tn = float(t_sched[i+1]); s = time.time()
        if (not args.no_cfg) and args.cfg_min_t <= t <= args.cfg_max_t:
            vc = run("cond", x_t, t); vt = run("text", x_t, t); vs = run("spk", x_t, t)
            v = vc + args.cfg_text*(vc - vt) + args.cfg_spk*(vc - vs); calls = 3
        else:
            v = run("cond", x_t, t); calls = 1
        ms = (time.time()-s)*1000; tot += ms
        x_t = x_t + v * (tn - t)
        print(f"  step {i}: t={t:.3f}->{tn:.3f} {calls}call {ms:.0f}ms")
    print(f"[dit] sampling done, total {tot:.0f}ms ({tot/1000:.1f}s)")

    z_full = np.transpose(x_t, (0, 2, 1)).astype(np.float32)   # (1,32,T_max)
    z = np.ascontiguousarray(z_full[:, :, :args.t_valid])      # trim -> (1,32,t_valid)
    np.save(args.out_latent, z)
    print(f"[trim] z={z.shape} std={z.std():.3f}")

    dac = axe.InferenceSession(args.dacvae)
    zin = dac.get_inputs()[0].name
    s = time.time(); audio = np.asarray(dac.run(None, {zin: z})[0]); dms = (time.time()-s)*1000
    wav = audio.reshape(-1)
    print(f"[dacvae] audio={audio.shape} finite={np.isfinite(audio).all()} "
          f"range=±{np.abs(wav).max():.3f} std={wav.std():.4f} {dms:.0f}ms")
    write_wav(args.out_wav, wav, args.sr)
    print(f"[saved] {args.out_wav} dur={len(wav)/args.sr:.2f}s samples={len(wav)}")

if __name__ == "__main__":
    main()
