#!/usr/bin/env python3
"""
Single-step torch reference: feed the EXACT cond KV cache from stage A's cond.npz
into forward_with_encoded_conditions for a fixed (x_t, t), save v_pred. The NPU
side (diag_step_npu.py) runs the axmodel on the identical inputs and compares.
Isolates "is the W8A8 axmodel itself wrong" from any loop/CFG effects.
"""
from __future__ import annotations
import argparse, dataclasses, json, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/admin-user/github/Irodori-TTS/model.safetensors")
    ap.add_argument("--model-cfg-json", default="build/model_introspection.json")
    ap.add_argument("--cond", default="/tmp/e2e_cond.npz")
    ap.add_argument("--out", default="/tmp/step_ref.npz")
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

    cond = np.load(args.cond)
    # reconstruct cond KV cache: 12 layers x (k_text, v_text, k_spk, v_spk)
    kinds = ["k_text", "v_text", "k_spk", "v_spk"]
    cache = []
    for i in range(cfg.num_layers):
        cache.append(tuple(torch.from_numpy(cond[f"cond_{k}_{i}"]) for k in kinds))
    text_mask = torch.from_numpy(cond["text_mask"]).to(torch.bool)
    speaker_mask = torch.from_numpy(cond["speaker_mask"]).to(torch.bool)

    rng = np.random.default_rng(args.seed)
    x_t_np = rng.standard_normal((1, args.seq_len, cfg.latent_dim)).astype(np.float32)
    x_t = torch.from_numpy(x_t_np)
    t = torch.tensor([args.t], dtype=torch.float32)
    text_state = torch.zeros((1, text_mask.shape[1], cfg.text_dim))
    speaker_state = torch.zeros((1, speaker_mask.shape[1], cfg.speaker_dim))

    with torch.inference_mode():
        v = model.forward_with_encoded_conditions(
            x_t=x_t, t=t, text_state=text_state, text_mask=text_mask,
            speaker_state=speaker_state, speaker_mask=speaker_mask,
            caption_state=None, caption_mask=None, latent_mask=None,
            context_kv_cache=cache,
        )
    v_np = v.cpu().float().numpy()
    np.savez(args.out, x_t=x_t_np, t=np.array([args.t], np.float32), v_torch=v_np)
    print(f"[saved] {args.out} v_torch={v_np.shape} std={v_np.std():.4f} "
          f"mean={v_np.mean():.4f} range=[{v_np.min():.3f},{v_np.max():.3f}]")


if __name__ == "__main__":
    main()
