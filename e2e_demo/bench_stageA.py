#!/usr/bin/env python3
"""Bench Stage A (torch CPU): load model ONCE, encode N texts, time each phase.

Separates one-time model load (swap-bound on this device) from per-text encode+KV,
so per-utterance timing is meaningful. Dumps cond_<i>.npz for the NPU bench + timings.json.
"""
from __future__ import annotations
import argparse, dataclasses, json, os, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer
from irodori_tts.text_normalization import normalize_text

TEXTS = [
    "はい、わかりました。",
    "今日はいい天気です。",
    "今日はとても良い天気ですね。少し散歩に出かけませんか。",
    "音声合成のテストを行っています。品質を確認しましょう。",
    "人工知能技術の進歩により、様々な分野で自動化が進んでいます。",
    "むかしむかし、あるところに、おじいさんとおばあさんが住んでいました。",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.environ.get("IRODORI_WEIGHTS", "model.safetensors"),
                    help="fp32 checkpoint; set IRODORI_WEIGHTS or pass --weights")
    ap.add_argument("--model-cfg-json", default="build/model_introspection.json")
    ap.add_argument("--out-dir", default="/tmp/bench_cond")
    ap.add_argument("--max-text-len", type=int, default=256)
    args = ap.parse_args()
    torch.set_num_threads(8)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cfg_all = json.loads(Path(args.model_cfg_json).read_text())["model_cfg"]
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_all.items() if k in fields})

    t0 = time.time()
    model = TextToLatentRFDiT(cfg); sd = model.state_dict()
    with safe_open(args.weights, framework="pt", device="cpu") as f:
        for n in f.keys():
            if n in sd:
                with torch.no_grad(): sd[n].copy_(f.get_tensor(n))
    model.eval()
    t_load = time.time() - t0

    t0 = time.time()
    tok = PretrainedTextTokenizer.from_pretrained(cfg.text_tokenizer_repo, add_bos=cfg.text_add_bos)
    t_tok = time.time() - t0

    ref_len = max(1, int(cfg.speaker_patch_size))
    ref_latent = torch.zeros((1, ref_len, cfg.latent_dim * cfg.latent_patch_size), dtype=torch.float32)
    ref_mask = torch.zeros((1, ref_len), dtype=torch.bool)
    kinds = ["k_text", "v_text", "k_spk", "v_spk"]

    rows = []
    for i, text in enumerate(TEXTS):
        norm = normalize_text(text).strip()
        text_ids, text_mask = tok.batch_encode([norm], max_length=args.max_text_len)
        ntok = int(text_mask.sum().item())
        t0 = time.time()
        with torch.inference_mode():
            (text_state, tmc, speaker_state, smc, _a, _b) = model.encode_conditions(
                text_input_ids=text_ids, text_mask=text_mask, ref_latent=ref_latent,
                ref_mask=ref_mask, speaker_state_override=None, speaker_mask_override=None,
                speaker_uncond_mode="mask")
            tsu = torch.zeros_like(text_state); ssu = torch.zeros_like(speaker_state)
            caches = {
                "cond": model.build_context_kv_cache(text_state=text_state, speaker_state=speaker_state, caption_state=None),
                "text": model.build_context_kv_cache(text_state=tsu, speaker_state=speaker_state, caption_state=None),
                "spk":  model.build_context_kv_cache(text_state=text_state, speaker_state=ssu, caption_state=None),
            }
        dt = time.time() - t0
        save = {}
        for branch, cache in caches.items():
            for li, layer in enumerate(cache):
                for kind, ten in zip(kinds, layer):
                    save[f"{branch}_{kind}_{li}"] = ten.detach().cpu().float().numpy()
        save["text_mask"] = tmc.detach().cpu().numpy().astype(np.uint8)
        save["speaker_mask"] = smc.detach().cpu().numpy().astype(np.uint8)
        np.savez(out / f"cond_{i}.npz", **save)
        rows.append({"i": i, "text": text, "tokens": ntok, "encode_s": round(dt, 3)})
        print(f"  [{i}] tok={ntok:3d} encode={dt:.2f}s  {text}")

    timings = {"phase": "stageA_torch_cpu", "model_load_s": round(t_load, 2),
               "tokenizer_load_s": round(t_tok, 2), "per_text": rows,
               "note": "model_load is swap-bound on 1.9Gi RAM device; one-time cost"}
    (out / "stageA_timings.json").write_text(json.dumps(timings, ensure_ascii=False, indent=2))
    enc = np.array([r["encode_s"] for r in rows])
    print(f"[load] {t_load:.1f}s (tok {t_tok:.1f}s) | encode n={len(enc)} "
          f"mean={enc.mean():.2f} std={enc.std(ddof=1):.2f} min={enc.min():.2f} max={enc.max():.2f}s")

if __name__ == "__main__":
    main()
