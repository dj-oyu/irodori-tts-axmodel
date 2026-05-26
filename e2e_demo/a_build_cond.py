#!/usr/bin/env python3
"""
Stage A (torch, CPU, no root): text -> conditioning KV cache.

Full-load-then-discard: instantiate the 500M TextToLatentRFDiT, load the 2GB
checkpoint (assign=True to keep peak RAM ~2GB), run tokenizer + encode_conditions
(no_ref / unconditional speaker) + build_context_kv_cache once, dump the per-layer
KV (named to match the DiT axmodel inputs) + masks to disk, then exit so the 2GB
model is released before the NPU stage.

Run (PYTHONPATH must point at the cloned Irodori-TTS):
  PYTHONPATH=/path/to/Irodori-TTS \
    python3 e2e_demo/a_build_cond.py --weights /path/to/model.safetensors \
      --text "こんにちは。" --out /tmp/e2e_cond.npz
  (or set IRODORI_WEIGHTS instead of --weights)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer
from irodori_tts.text_normalization import normalize_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.environ.get("IRODORI_WEIGHTS", "model.safetensors"),
                    help="fp32 checkpoint; set IRODORI_WEIGHTS or pass --weights")
    ap.add_argument("--model-cfg-json", default="build/model_introspection.json")
    ap.add_argument("--text", default="こんにちは。")
    ap.add_argument("--max-text-len", type=int, default=256)
    ap.add_argument("--out", default="/tmp/e2e_cond.npz")
    args = ap.parse_args()

    torch.set_num_threads(8)
    cfg_all = json.loads(Path(args.model_cfg_json).read_text())["model_cfg"]
    field_names = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_all.items() if k in field_names})
    print(f"[cfg] layers={cfg.num_layers} text_dim={cfg.text_dim} speaker_dim={cfg.speaker_dim} "
          f"use_speaker={cfg.use_speaker_condition} tokenizer={cfg.text_tokenizer_repo}")

    # Low-RAM load: construct model (~2GB), then copy weights one tensor at a time
    # from the mmap'd safetensors. Avoids materializing a full 2GB state dict on top
    # of the model's params (which would peak ~4GB and OOM the 2.8GB system).
    t0 = time.time()
    model = TextToLatentRFDiT(cfg)
    sd = model.state_dict()
    copied = 0
    unexpected = 0
    with safe_open(args.weights, framework="pt", device="cpu") as f:
        file_keys = set(f.keys())
        for name in file_keys:
            if name in sd:
                with torch.no_grad():
                    sd[name].copy_(f.get_tensor(name))
                copied += 1
            else:
                unexpected += 1
    missing = [k for k in sd if k not in file_keys]
    model.eval()
    print(f"[load] copied={copied} missing={len(missing)} unexpected={unexpected} "
          f"in {time.time()-t0:.1f}s")
    if missing:
        print(f"[load] missing keys (first 10): {missing[:10]}")

    tok = PretrainedTextTokenizer.from_pretrained(cfg.text_tokenizer_repo, add_bos=cfg.text_add_bos)
    norm = normalize_text(args.text).strip()
    text_ids, text_mask = tok.batch_encode([norm], max_length=args.max_text_len)
    print(f"[tok] '{norm}' -> ids {tuple(text_ids.shape)} tokens={int(text_mask.sum().item())}")

    # no_ref / unconditional speaker: mirror InferenceRuntime._load_reference_latent's
    # no_ref branch — a length-1 zero reference latent with an all-False mask. The
    # speaker_encoder then yields a null speaker state (prepended masked-mean token =>
    # 2 speaker tokens, matching the axmodel's speaker_mask (1,2)).
    ref_len = max(1, int(cfg.speaker_patch_size))
    ref_latent = torch.zeros((1, ref_len, cfg.latent_dim * cfg.latent_patch_size),
                             dtype=torch.float32)
    ref_mask = torch.zeros((1, ref_len), dtype=torch.bool)
    t1 = time.time()
    with torch.inference_mode():
        (text_state, text_mask_c, speaker_state, speaker_mask_c,
         _cap_s, _cap_m) = model.encode_conditions(
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            speaker_state_override=None,
            speaker_mask_override=None,
            speaker_uncond_mode="mask",
        )
        # CFG (independent mode, branches=text+speaker) needs 3 KV caches. uncond =
        # zeros (speaker_uncond_mode="mask"). See irodori_tts.rf.sample_euler_rf_cfg.
        text_state_uncond = torch.zeros_like(text_state)
        speaker_state_uncond = torch.zeros_like(speaker_state)
        caches = {
            "cond": model.build_context_kv_cache(text_state=text_state, speaker_state=speaker_state, caption_state=None),
            "text": model.build_context_kv_cache(text_state=text_state_uncond, speaker_state=speaker_state, caption_state=None),
            "spk":  model.build_context_kv_cache(text_state=text_state, speaker_state=speaker_state_uncond, caption_state=None),
        }
    print(f"[encode+kv] text_state={tuple(text_state.shape)} "
          f"speaker_state={None if speaker_state is None else tuple(speaker_state.shape)} "
          f"layers={len(caches['cond'])} per_layer={len(caches['cond'][0])} (x3 caches) in {time.time()-t1:.1f}s")

    # Flatten each cache in the axmodel's input name order: k_text_i,v_text_i,k_spk_i,v_spk_i,
    # prefixed by branch (cond_/text_/spk_).
    kinds = ["k_text", "v_text", "k_spk", "v_spk"]
    save: dict[str, np.ndarray] = {}
    for branch, cache in caches.items():
        for i, layer in enumerate(cache):
            assert len(layer) == len(kinds), (len(layer), len(kinds))
            for kind, ten in zip(kinds, layer):
                save[f"{branch}_{kind}_{i}"] = ten.detach().cpu().float().numpy()
    save["text_mask"] = text_mask_c.detach().cpu().numpy().astype(np.uint8)
    save["speaker_mask"] = speaker_mask_c.detach().cpu().numpy().astype(np.uint8)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **save)
    print(f"[saved] {args.out}: {len(save)} arrays | "
          f"cond_k_text_0={save['cond_k_text_0'].shape} cond_k_spk_0={save['cond_k_spk_0'].shape} "
          f"text_mask={save['text_mask'].shape} speaker_mask={save['speaker_mask'].shape}")


if __name__ == "__main__":
    main()
