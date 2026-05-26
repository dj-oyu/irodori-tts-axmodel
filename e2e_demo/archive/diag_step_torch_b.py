#!/usr/bin/env python3
"""
Option-B single-step torch reference: compute v_pred via forward_with_encoded_conditions
WITHOUT a precomputed KV cache (KV built internally from text_state/speaker_state) — this
is exactly what the Option-B axmodel (axmodel_b1, inputs x_t/t/text_state/text_mask/
speaker_state/speaker_mask) should reproduce. Saves the inputs + v_torch for the NPU compare.
no_ref / unconditional speaker (cond branch only).
"""
from __future__ import annotations
import argparse, dataclasses, json, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer
from irodori_tts.text_normalization import normalize_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/admin-user/github/Irodori-TTS/model.safetensors")
    ap.add_argument("--model-cfg-json", default="build/model_introspection.json")
    ap.add_argument("--text", default="こんにちは。")
    ap.add_argument("--out", default="/tmp/step_ref_b.npz")
    ap.add_argument("--t", type=float, default=0.8)
    ap.add_argument("--seq-len", type=int, default=119)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.set_num_threads(8)

    cfg_all = json.loads(Path(args.model_cfg_json).read_text())["model_cfg"]
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_all.items() if k in fields})
    model = TextToLatentRFDiT(cfg)
    sd = model.state_dict()
    t0 = time.time()
    with safe_open(args.weights, framework="pt", device="cpu") as f:
        for n in f.keys():
            if n in sd:
                with torch.no_grad():
                    sd[n].copy_(f.get_tensor(n))
    model.eval()
    print(f"[load] {time.time()-t0:.1f}s")

    tok = PretrainedTextTokenizer.from_pretrained(cfg.text_tokenizer_repo, add_bos=cfg.text_add_bos)
    text_ids, text_mask = tok.batch_encode([normalize_text(args.text).strip()], max_length=256)
    ref_len = max(1, int(cfg.speaker_patch_size))
    ref_latent = torch.zeros((1, ref_len, cfg.latent_dim * cfg.latent_patch_size), dtype=torch.float32)
    ref_mask = torch.zeros((1, ref_len), dtype=torch.bool)

    with torch.inference_mode():
        (text_state, text_mask_c, speaker_state, speaker_mask_c, _a, _b) = model.encode_conditions(
            text_input_ids=text_ids, text_mask=text_mask, ref_latent=ref_latent, ref_mask=ref_mask,
            speaker_state_override=None, speaker_mask_override=None, speaker_uncond_mode="mask")

        rng = np.random.default_rng(args.seed)
        x_t_np = rng.standard_normal((1, args.seq_len, cfg.latent_dim)).astype(np.float32)
        x_t = torch.from_numpy(x_t_np)
        t = torch.tensor([args.t], dtype=torch.float32)
        v = model.forward_with_encoded_conditions(
            x_t=x_t, t=t, text_state=text_state, text_mask=text_mask_c,
            speaker_state=speaker_state, speaker_mask=speaker_mask_c,
            caption_state=None, caption_mask=None, latent_mask=None, context_kv_cache=None)

    v_np = v.cpu().float().numpy()
    np.savez(args.out,
             x_t=x_t_np, t=np.array([args.t], np.float32),
             text_state=text_state.cpu().float().numpy(),
             text_mask=text_mask_c.cpu().numpy().astype(np.uint8),
             speaker_state=speaker_state.cpu().float().numpy(),
             speaker_mask=speaker_mask_c.cpu().numpy().astype(np.uint8),
             v_torch=v_np)
    print(f"[saved] {args.out} v_torch={v_np.shape} std={v_np.std():.4f} "
          f"text_state={text_state.shape} speaker_state={speaker_state.shape}")


if __name__ == "__main__":
    main()
