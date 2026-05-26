#!/usr/bin/env python3
"""Stage A for emoji-control eval: load model ONCE, encode emoji/plain A-B pairs.
Outputs named cond_<label>.npz so the NPU stage can synth each and you can judge
the emoji effect by ear (plain vs emoji same text). normalize_text preserves emoji.
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

# (label, text) — plain/emoji A-B + one extra. cond-only (no CFG) to cut Stage A memory ~3x.
ITEMS = [
    ("plain_tanoshii", "今日はとても楽しいです。"),
    ("emoji_tanoshii", "今日はとても楽しいです😄"),
    ("emoji_laugh",    "あはは、おかしいね😄😄"),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.environ.get("IRODORI_WEIGHTS", "model.safetensors"),
                    help="fp32 checkpoint; set IRODORI_WEIGHTS or pass --weights")
    ap.add_argument("--model-cfg-json", default="build/model_introspection.json")
    ap.add_argument("--out-dir", default="/tmp/emoji_cond")
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
    print(f"[load] {time.time()-t0:.1f}s", flush=True)
    tok = PretrainedTextTokenizer.from_pretrained(cfg.text_tokenizer_repo, add_bos=cfg.text_add_bos)
    ref_len = max(1, int(cfg.speaker_patch_size))
    ref_latent = torch.zeros((1, ref_len, cfg.latent_dim * cfg.latent_patch_size), dtype=torch.float32)
    ref_mask = torch.zeros((1, ref_len), dtype=torch.bool)
    kinds = ["k_text", "v_text", "k_spk", "v_spk"]

    for label, text in ITEMS:
        norm = normalize_text(text).strip()
        ids, tmask = tok.batch_encode([norm], max_length=256)
        with torch.inference_mode():
            (ts, tmc, ss, smc, _a, _b) = model.encode_conditions(
                text_input_ids=ids, text_mask=tmask, ref_latent=ref_latent, ref_mask=ref_mask,
                speaker_state_override=None, speaker_mask_override=None, speaker_uncond_mode="mask")
            caches = {"cond": model.build_context_kv_cache(text_state=ts, speaker_state=ss, caption_state=None)}
        save = {}
        for br, cache in caches.items():
            for li, layer in enumerate(cache):
                for k, ten in zip(kinds, layer):
                    save[f"{br}_{k}_{li}"] = ten.detach().cpu().float().numpy()
        save["text_mask"] = tmc.detach().cpu().numpy().astype(np.uint8)
        save["speaker_mask"] = smc.detach().cpu().numpy().astype(np.uint8)
        np.savez(out / f"cond_{label}.npz", **save)
        print(f"  [{label}] tok={int(tmask.sum())} '{norm}'", flush=True)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
