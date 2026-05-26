#!/usr/bin/env python3
"""完全NPU化 text->wav（実機 AX8850, ROOT）。torch モデル / model.safetensors 不要。

条件付けを ① cond axmodel (text KV) + bake 済定数(speaker/text-branch KV) で構成し、
DiT(allfcu16_npu3) → dacvae(T201/b0) まで全段 NPU。tokenizer のみ torch 非依存で使用。

no_ref 専用（話者は --seed）。A3: ① が token_logits を出すと t_valid を自動予測（可変長）。
検証は実機で行い、結果は RESULT.md に書いて GitHub 共有。

  sudo -n PYTHONPATH=/home/exe/ai/Irodori-TTS /usr/bin/python3.10 e2e_demo/run_npu_full.py \
    --text "今日はとても良い天気ですね。" --num-steps 16 --seed 0 --out-wav /tmp/npu_full.wav
  # 既定: cond=axmodel_cond_textkv_dur(A3付), dacvae=T201, t-valid=自動。
  # 手動長さ: --t-valid 119。duration補正: --duration-scale 1.1 等。
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
    return t.astype(np.float32)


def write_wav(path, wav, sr):
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


def tokenize(text, seq=256):
    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from irodori_tts.text_normalization import normalize_text
    import json, dataclasses
    from irodori_tts.config import ModelConfig
    cfg_all = json.load(open("build/model_introspection.json"))["model_cfg"]
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_all.items() if k in fields})
    tok = PretrainedTextTokenizer.from_pretrained(cfg.text_tokenizer_repo, add_bos=cfg.text_add_bos)
    ids, mask = tok.batch_encode([normalize_text(text).strip()], max_length=seq)
    return ids.numpy().astype(np.int64), mask.numpy(), int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--cond", default="build/axmodel_cond_textkv_dur/compiled.axmodel",
                    help="① cond axmodel (text KV [+ token_logits=A3 duration])")
    ap.add_argument("--constants", default="build/cond_constants.npz")
    ap.add_argument("--dit", default="build/axmodel_kv_long_lm_allfcu16_npu3/compiled.axmodel")
    ap.add_argument("--dacvae", default="build/axmodel_dacvae_T201/compiled.axmodel")
    ap.add_argument("--t-valid", type=int, default=0,
                    help="0 = A3自動(token_logits から duration予測); >0 で手動上書き")
    ap.add_argument("--duration-scale", type=float, default=1.0)
    ap.add_argument("--min-sec", type=float, default=0.3)
    ap.add_argument("--max-sec", type=float, default=8.0)
    ap.add_argument("--hop", type=int, default=1920)
    ap.add_argument("--num-steps", type=int, default=16)
    ap.add_argument("--cfg-text", type=float, default=3.0)
    ap.add_argument("--cfg-spk", type=float, default=5.0)
    ap.add_argument("--cfg-min-t", type=float, default=0.5)
    ap.add_argument("--cfg-max-t", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--out-wav", default="/tmp/npu_full.wav")
    ap.add_argument("--dump-cond", default="", help="save assembled cond npz (for A/B vs torch slim cond)")
    args = ap.parse_args()

    # 1) tokenize (no torch model / no safetensors)
    t0 = time.time()
    ids, text_mask, ntok = tokenize(args.text)
    print(f"[tok] {ntok} tokens, {time.time()-t0:.2f}s", flush=True)

    # 2) ① cond axmodel: ids,mask -> 24 text KV (text-dependent part)
    t0 = time.time()
    cond = axe.InferenceSession(args.cond)
    cin = {i.name: i for i in cond.get_inputs()}
    feed_cond = {"input_ids": ids.astype(np.int32), "mask": text_mask.astype(np.uint8)}
    feed_cond = {n: feed_cond[n] for n in cin}
    out_names = [o.name for o in cond.get_outputs()]
    cout = {out_names[i]: np.asarray(cout_i, np.float32) for i, cout_i in enumerate(cond.run(None, feed_cond))}
    token_logits = cout.pop("token_logits", None)  # A3 duration head (per-token, pre-softplus)
    text_kv = cout
    print(f"[cond①] {len(text_kv)} text-KV{' + token_logits(A3)' if token_logits is not None else ''}, "
          f"load+run {time.time()-t0:.2f}s", flush=True)

    K = np.load(args.constants)  # baked constants

    # 3) DiT setup
    dit = axe.InferenceSession(args.dit)
    ishapes = {i.name: tuple(i.shape) for i in dit.get_inputs()}
    T = ishapes["x_t"][1]; latent_dim = ishapes["x_t"][2]
    nlayers = sum(1 for n in ishapes if n.startswith("k_text_"))

    # A3: t_valid from duration head (CPU softplus + masked-sum), or manual override
    if args.t_valid > 0:
        t_valid = min(args.t_valid, T)
        print(f"[A3] t_valid={t_valid} (manual override)", flush=True)
    elif token_logits is not None:
        token_frames = np.logaddexp(0.0, token_logits.astype(np.float64))  # softplus
        pred_frames = float((token_frames * text_mask.astype(np.float64)).sum())
        min_f = max(1, math.ceil(args.min_sec * args.sr / args.hop))
        max_f = min(T, max(1, math.floor(args.max_sec * args.sr / args.hop)))
        t_valid = int(round(pred_frames * args.duration_scale))
        t_valid = max(min_f, min(max_f, t_valid))
        print(f"[A3] predicted frames={pred_frames:.1f} scale={args.duration_scale} "
              f"-> t_valid={t_valid} ({t_valid*args.hop/args.sr:.2f}s)", flush=True)
    else:
        t_valid = min(119, T)
        print(f"[A3] no token_logits; fallback t_valid={t_valid}", flush=True)
    latent_mask = np.zeros((1, T), np.uint8); latent_mask[:, :t_valid] = 1
    zero_tm = np.zeros_like(text_mask.astype(np.uint8))
    spk_mask = K["speaker_mask"]; zero_sm = np.zeros_like(spk_mask)

    def assemble(branch):
        f = {"latent_mask": latent_mask}
        # masks per branch (CFG: text branch zeros text, spk branch zeros speaker)
        f["text_mask"] = (zero_tm if branch == "text" else text_mask.astype(np.uint8))
        f["speaker_mask"] = (zero_sm if branch == "spk" else spk_mask)
        for L in range(nlayers):
            # text KV: cond/spk use ① output; text branch uses baked zero-text
            if branch == "text":
                f[f"k_text_{L}"] = K[f"text_zero_k_text_{L}"]; f[f"v_text_{L}"] = K[f"text_zero_v_text_{L}"]
            else:
                f[f"k_text_{L}"] = text_kv[f"k_text_{L}"]; f[f"v_text_{L}"] = text_kv[f"v_text_{L}"]
            # speaker KV: cond/text use spk_real; spk branch uses spk_zero
            if branch == "spk":
                f[f"k_spk_{L}"] = K[f"spk_zero_k_spk_{L}"]; f[f"v_spk_{L}"] = K[f"spk_zero_v_spk_{L}"]
            else:
                f[f"k_spk_{L}"] = K[f"spk_real_k_spk_{L}"]; f[f"v_spk_{L}"] = K[f"spk_real_v_spk_{L}"]
        return {n: np.asarray(v, np.float32) if n not in ("text_mask", "speaker_mask", "latent_mask") else v
                for n, v in f.items()}

    feeds = {b: assemble(b) for b in ("cond", "text", "spk")}
    if args.dump_cond:  # for A/B: save in e2e_npu cond-npz format
        dump = {}
        for b, f in feeds.items():
            for n, v in f.items():
                if n in ("latent_mask",): continue
                if n in ("text_mask", "speaker_mask"):
                    if b == "cond": dump[n] = v
                else: dump[f"{b}_{n}"] = v
        np.savez(args.dump_cond, **dump); print(f"[dump] {args.dump_cond}", flush=True)

    # 4) RF sampling loop (Euler + independent CFG), same as e2e_npu
    t_sched = sway_schedule(args.num_steps)
    rng = np.random.default_rng(args.seed)
    x_t = rng.standard_normal((1, T, latent_dim)).astype(np.float32)

    def run(branch, x, t):
        f = dict(feeds[branch]); f["x_t"] = x; f["t"] = np.array([t], np.float32)
        return dit.run(None, f)[0]

    tot = 0.0
    for i in range(args.num_steps):
        t = float(t_sched[i]); tn = float(t_sched[i + 1]); s = time.time()
        if args.cfg_min_t <= t <= args.cfg_max_t:
            vc = run("cond", x_t, t); vt = run("text", x_t, t); vs = run("spk", x_t, t)
            v = vc + args.cfg_text * (vc - vt) + args.cfg_spk * (vc - vs); calls = 3
        else:
            v = run("cond", x_t, t); calls = 1
        ms = (time.time() - s) * 1000; tot += ms
        x_t = x_t + v * (tn - t)
        print(f"  step {i}: t={t:.3f} {calls}call {ms:.0f}ms", flush=True)
    print(f"[dit] sampling {tot/1000:.1f}s ({tot/ max(1,(args.num_steps)):.0f}ms/step avg)", flush=True)

    # 5) trim -> dacvae -> wav
    z = np.ascontiguousarray(np.transpose(x_t, (0, 2, 1))[:, :, :t_valid]).astype(np.float32)
    dac = axe.InferenceSession(args.dacvae)
    zin = dac.get_inputs()[0].name
    s = time.time(); audio = np.asarray(dac.run(None, {zin: z})[0]).reshape(-1)
    print(f"[dacvae] {audio.shape} ±{np.abs(audio).max():.3f} {(time.time()-s)*1000:.0f}ms", flush=True)
    write_wav(args.out_wav, audio, args.sr)
    print(f"[saved] {args.out_wav} dur={len(audio)/args.sr:.2f}s", flush=True)


if __name__ == "__main__":
    main()
