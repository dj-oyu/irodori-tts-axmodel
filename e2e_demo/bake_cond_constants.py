#!/usr/bin/env python3
"""Bake no_ref conditioning constants (build host, torch fp32 — run ONCE).

In no_ref mode the speaker path and the zero-text CFG branch are text-independent
constants (verified: speaker_state/KV bit-identical across texts). This precomputes
them so the device runtime only needs the ① cond axmodel (text KV) + these consts.

Outputs npz with, per layer L (0..11):
  text_zero_k_text_L, text_zero_v_text_L  = proj_text(0)      (CFG "text" branch text KV)
  spk_real_k_spk_L,   spk_real_v_spk_L    = proj_speaker(ss)  (cond & text branch speaker KV)
  spk_zero_k_spk_L,   spk_zero_v_spk_L    = proj_speaker(0)   (CFG "spk" branch speaker KV)
  speaker_mask                            = const speaker mask (no_ref)

Runtime branch assembly (device): k_text/v_text for cond&spk = ① output (text-dep);
text branch text KV = text_zero_*; speaker KV: cond/text = spk_real_*, spk = spk_zero_*.

  PYTHONPATH=/home/exe/ai/Irodori-TTS uv run --project irodori_ax650_preprocess python \
    e2e_demo/bake_cond_constants.py --weights <model.safetensors> \
    --model-cfg-json build/model_introspection.json --out build/cond_constants.npz
"""
from __future__ import annotations
import argparse, dataclasses, glob, json
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file
from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model-cfg-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=256)
    args = ap.parse_args()

    cfg_all = json.loads(Path(args.model_cfg_json).read_text())["model_cfg"]
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_all.items() if k in fields})
    model = TextToLatentRFDiT(cfg)
    w = sorted(glob.glob(args.weights))[0]
    model.load_state_dict(load_file(w, device="cpu"), strict=False)
    model.eval()

    S = args.seq
    ref_len = max(1, int(cfg.speaker_patch_size))
    ref_latent = torch.zeros((1, ref_len, cfg.latent_dim * cfg.latent_patch_size))
    ref_mask = torch.zeros((1, ref_len), dtype=torch.bool)
    ids = torch.zeros(1, S, dtype=torch.long)
    tmask = torch.ones(1, S, dtype=torch.bool)

    with torch.no_grad():
        ts, tmc, ss, smc, _, _ = model.encode_conditions(
            text_input_ids=ids, text_mask=tmask, ref_latent=ref_latent, ref_mask=ref_mask,
            speaker_state_override=None, speaker_mask_override=None, speaker_uncond_mode="mask")
        zero_ts = torch.zeros_like(ts)
        zero_ss = torch.zeros_like(ss)
        # call 1: (text=0, speaker=ss) -> text_zero text-KV + spk_real speaker-KV
        cache_textzero = model.build_context_kv_cache(text_state=zero_ts, speaker_state=ss, caption_state=None)
        # call 2: (text=ts, speaker=0) -> spk_zero speaker-KV
        cache_spkzero = model.build_context_kv_cache(text_state=ts, speaker_state=zero_ss, caption_state=None)

    save = {}
    for L, layer in enumerate(cache_textzero):
        k_text, v_text, k_spk, v_spk = layer
        save[f"text_zero_k_text_{L}"] = k_text.cpu().float().numpy()
        save[f"text_zero_v_text_{L}"] = v_text.cpu().float().numpy()
        save[f"spk_real_k_spk_{L}"] = k_spk.cpu().float().numpy()
        save[f"spk_real_v_spk_{L}"] = v_spk.cpu().float().numpy()
    for L, layer in enumerate(cache_spkzero):
        _, _, k_spk0, v_spk0 = layer
        save[f"spk_zero_k_spk_{L}"] = k_spk0.cpu().float().numpy()
        save[f"spk_zero_v_spk_{L}"] = v_spk0.cpu().float().numpy()
    save["speaker_mask"] = smc.cpu().numpy().astype(np.uint8)

    np.savez_compressed(args.out, **save)
    print(f"[baked] {args.out}: {len(save)} arrays, {len(cache_textzero)} layers")
    print(f"  speaker_mask={save['speaker_mask'].shape}, "
          f"k_spk={save['spk_real_k_spk_0'].shape}, k_text={save['text_zero_k_text_0'].shape}")


if __name__ == "__main__":
    main()
