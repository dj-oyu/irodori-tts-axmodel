#!/usr/bin/env python3
"""Slim Stage A: minimum-footprint conditioning runner producing 3 CFG branches.

Same output as a_build_cond/emoji_stageA (cond/text/spk KV caches + masks), but built
on the META device with only the conditioning params materialized (no-copy mmap), so the
swap-competing footprint is ~360MB instead of ~2GB. See slim_cond_probe.py for the
footprint measurement and conditioning-footprint memory.

  PYTHONPATH=/path/to/Irodori-TTS python3 e2e_demo/slim_stageA.py \
    --weights /path/to/model.safetensors --text "今日はとても楽しいです。" \
    --label plain --out-dir /tmp/slim_e2e [--ref /tmp/emoji_cond/cond_plain_tanoshii.npz]
    # or set IRODORI_WEIGHTS instead of --weights
"""
from __future__ import annotations
import argparse, dataclasses, json, re, os, gc
from pathlib import Path


def load_slim(weights, cfg):
    import torch
    from safetensors import safe_open
    from irodori_tts.model import TextToLatentRFDiT
    with torch.device("meta"):
        model = TextToLatentRFDiT(cfg)
    blk = re.compile(r"^blocks\.\d+\.attention\.(wk_text|wv_text|wk_speaker|wv_speaker|k_norm)\.")
    def needed(n):
        return (n.startswith("text_encoder.") or n == "text_norm.weight"
                or n.startswith("speaker_encoder.") or n == "speaker_norm.weight"
                or bool(blk.match(n)))
    def set_param(root, dotted, t):
        *path, leaf = dotted.split(".")
        m = root
        for p in path:
            m = getattr(m, p)
        m._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)
    nbytes = 0
    with safe_open(weights, framework="pt", device="cpu") as f:
        for name in f.keys():
            if needed(name):
                t = f.get_tensor(name)
                set_param(model, name, t)
                nbytes += t.numel() * t.element_size()
    model.eval()
    print(f"[slim-load] {nbytes/1024/1024:.0f}MB materialized (meta+mmap)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.environ.get("IRODORI_WEIGHTS", "model.safetensors"))
    ap.add_argument("--model-cfg-json", default="build/model_introspection.json")
    ap.add_argument("--text", default="今日はとても楽しいです。")
    ap.add_argument("--label", default="plain")
    ap.add_argument("--out-dir", default="/tmp/slim_e2e")
    ap.add_argument("--ref", default="")
    args = ap.parse_args()

    import numpy as np, torch
    from irodori_tts.config import ModelConfig
    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from irodori_tts.text_normalization import normalize_text
    torch.set_num_threads(8)

    cfg_all = json.loads(Path(args.model_cfg_json).read_text())["model_cfg"]
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_all.items() if k in fields})

    model = load_slim(args.weights, cfg)
    tok = PretrainedTextTokenizer.from_pretrained(cfg.text_tokenizer_repo, add_bos=cfg.text_add_bos)
    ref_len = max(1, int(cfg.speaker_patch_size))
    ref_latent = torch.zeros((1, ref_len, cfg.latent_dim * cfg.latent_patch_size), dtype=torch.float32)
    ref_mask = torch.zeros((1, ref_len), dtype=torch.bool)
    kinds = ["k_text", "v_text", "k_spk", "v_spk"]

    norm = normalize_text(args.text).strip()
    ids, tmask = tok.batch_encode([norm], max_length=256)
    with torch.inference_mode():
        ts, tmc, ss, smc, _a, _b = model.encode_conditions(
            text_input_ids=ids, text_mask=tmask, ref_latent=ref_latent, ref_mask=ref_mask,
            speaker_state_override=None, speaker_mask_override=None, speaker_uncond_mode="mask")
        ts_u = torch.zeros_like(ts); ss_u = torch.zeros_like(ss)
        caches = {
            "cond": model.build_context_kv_cache(text_state=ts, speaker_state=ss, caption_state=None),
            "text": model.build_context_kv_cache(text_state=ts_u, speaker_state=ss, caption_state=None),
            "spk":  model.build_context_kv_cache(text_state=ts, speaker_state=ss_u, caption_state=None),
        }
    save = {}
    for br, cache in caches.items():
        for li, layer in enumerate(cache):
            for k, ten in zip(kinds, layer):
                save[f"{br}_{k}_{li}"] = ten.detach().cpu().float().numpy()
    save["text_mask"] = tmc.detach().cpu().numpy().astype(np.uint8)
    save["speaker_mask"] = smc.detach().cpu().numpy().astype(np.uint8)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"cond_{args.label}.npz", **save)
    print(f"[saved] {out}/cond_{args.label}.npz  tok={int(tmask.sum())} '{norm}'  ({len(save)} arrays)", flush=True)

    if args.ref and Path(args.ref).exists():
        ref = np.load(args.ref)
        bad = [k for k in save if k in ref and not np.array_equal(save[k], ref[k])]
        common = [k for k in save if k in ref]
        print(f"[correctness] {len(common)} keys vs ref: "
              f"{'ALL BITWISE-EQUAL ✅' if not bad else f'MISMATCH ❌ {bad[:5]}'}", flush=True)


if __name__ == "__main__":
    main()
