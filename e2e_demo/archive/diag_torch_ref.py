#!/usr/bin/env python3
"""
Diagnostic: produce reference latents fully in torch to isolate "is CFG needed"
from "is stage B / quant wrong".

Runs sample_euler_rf_cfg twice on the loaded torch model with the SAME seed
(same initial x_t), 8-step sway, no_ref:
  1. CFG ON  (default scales: text 3.0, speaker 5.0) -> /tmp/diag_cfg_on.npy
  2. CFG OFF (cfg_scale=0, single forward/step)      -> /tmp/diag_cfg_off.npy
Both saved as (1, latent_dim, T) to match e2e_demo/c_decode.py's onnx DACVAE input.

Decode each with: python3 e2e_demo/c_decode.py --latent /tmp/diag_cfg_on.npy --out /tmp/diag_cfg_on.wav
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer
from irodori_tts.text_normalization import normalize_text
from irodori_tts.rf import sample_euler_rf_cfg


def load_model(weights: str, cfg_json: str):
    cfg_all = json.loads(Path(cfg_json).read_text())["model_cfg"]
    field_names = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_all.items() if k in field_names})
    model = TextToLatentRFDiT(cfg)
    sd = model.state_dict()
    t0 = time.time()
    with safe_open(weights, framework="pt", device="cpu") as f:
        for name in f.keys():
            if name in sd:
                with torch.no_grad():
                    sd[name].copy_(f.get_tensor(name))
    model.eval()
    print(f"[load] {time.time()-t0:.1f}s")
    return model, cfg


def run(model, cfg, text_ids, text_mask, seq_len, num_steps, seed, cfg_scale, use_kv=False):
    ref_len = max(1, int(cfg.speaker_patch_size))
    ref_latent = torch.zeros((1, ref_len, cfg.latent_dim * cfg.latent_patch_size), dtype=torch.float32)
    ref_mask = torch.zeros((1, ref_len), dtype=torch.bool)
    kw = dict(
        model=model, text_input_ids=text_ids, text_mask=text_mask,
        ref_latent=ref_latent, ref_mask=ref_mask, sequence_length=seq_len,
        speaker_uncond_mode="mask", num_steps=num_steps, seed=seed,
        use_context_kv_cache=use_kv, t_schedule_mode="sway", sway_coeff=-1.0,
    )
    if cfg_scale is not None:
        kw["cfg_scale"] = float(cfg_scale)  # 0.0 -> all guidance off -> single forward
    with torch.inference_mode():
        z = sample_euler_rf_cfg(**kw)  # (1, seq_len, latent_dim)
    return np.transpose(z.cpu().float().numpy(), (0, 2, 1))  # (1, latent_dim, T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/admin-user/github/Irodori-TTS/model.safetensors")
    ap.add_argument("--model-cfg-json", default="build/model_introspection.json")
    ap.add_argument("--text", default="こんにちは。")
    ap.add_argument("--seq-len", type=int, default=119)
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(8)
    model, cfg = load_model(args.weights, args.model_cfg_json)
    tok = PretrainedTextTokenizer.from_pretrained(cfg.text_tokenizer_repo, add_bos=cfg.text_add_bos)
    text_ids, text_mask = tok.batch_encode([normalize_text(args.text).strip()], max_length=256)

    # Probe: CFG-on using the precomputed-KV path (Option A, what stage A/B use).
    z_kv = run(model, cfg, text_ids, text_mask, args.seq_len, args.num_steps, args.seed,
               cfg_scale=None, use_kv=True)
    np.save("/tmp/diag_cfg_on_kv.npy", z_kv)
    print(f"[cfg_on_kv ] saved /tmp/diag_cfg_on_kv.npy std={z_kv.std():.3f} range=[{z_kv.min():.2f},{z_kv.max():.2f}]")


if __name__ == "__main__":
    main()
