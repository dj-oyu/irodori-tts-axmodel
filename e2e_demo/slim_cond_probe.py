#!/usr/bin/env python3
"""Path-B probe: minimum-footprint conditioning runner.

Builds TextToLatentRFDiT on the META device (zero real storage), then materializes
ONLY the params encode_conditions + build_context_kv_cache touch:
  text_encoder.* , text_norm , speaker_encoder.* , speaker_norm ,
  blocks.N.attention.{wk_text,wv_text,wk_speaker,wv_speaker,k_norm}
Everything else (DiT self-attn/MLP/AdaLN, cond_module, in/out_proj, duration) stays
on meta = no memory. RoPE _freqs_cis_cache is persistent=False (recomputed in fwd).

Measures VmRSS/VmHWM at 3 points and checks the resulting cond is BITWISE-equal to a
full-model reference npz (same text/seed). Run with services stopped.

  PYTHONPATH=/path/to/Irodori-TTS python3 e2e_demo/slim_cond_probe.py \
    --weights /path/to/model.safetensors \
    --ref /tmp/emoji_cond/cond_plain_tanoshii.npz   # or set IRODORI_WEIGHTS
"""
from __future__ import annotations
import argparse, dataclasses, json, re, os, gc
from pathlib import Path


def meminfo(tag):
    d = {}
    with open("/proc/self/status") as f:
        for line in f:
            for key in ("VmRSS:", "VmHWM:", "RssAnon:", "RssFile:"):
                if line.startswith(key):
                    d[key.strip(":")] = int(line.split()[1]) / 1024
    print(f"[mem:{tag}] VmRSS={d.get('VmRSS',0):.0f}MB peak={d.get('VmHWM',0):.0f}MB "
          f"| anon={d.get('RssAnon',0):.0f}MB file={d.get('RssFile',0):.0f}MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.environ.get("IRODORI_WEIGHTS", "model.safetensors"))
    ap.add_argument("--model-cfg-json", default="build/model_introspection.json")
    ap.add_argument("--ref", default="/tmp/emoji_cond/cond_plain_tanoshii.npz")
    ap.add_argument("--text", default="今日はとても楽しいです。")
    ap.add_argument("--out", default="/tmp/slim_cond/cond_plain.npz")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    meminfo("start")
    import numpy as np, torch
    from safetensors import safe_open
    from irodori_tts.config import ModelConfig
    from irodori_tts.model import TextToLatentRFDiT
    from irodori_tts.tokenizer import PretrainedTextTokenizer
    from irodori_tts.text_normalization import normalize_text
    torch.set_num_threads(8)
    meminfo("after import torch")

    cfg_all = json.loads(Path(args.model_cfg_json).read_text())["model_cfg"]
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_all.items() if k in fields})

    # 1) construct on meta (no storage)
    with torch.device("meta"):
        model = TextToLatentRFDiT(cfg)

    # 2) decide which params to materialize
    blk = re.compile(r"^blocks\.\d+\.attention\.(wk_text|wv_text|wk_speaker|wv_speaker|k_norm)\.")
    def needed(name: str) -> bool:
        return (name.startswith("text_encoder.") or name == "text_norm.weight"
                or name.startswith("speaker_encoder.") or name == "speaker_norm.weight"
                or bool(blk.match(name)))

    def set_param(root, dotted, tensor):
        *path, leaf = dotted.split(".")
        m = root
        for p in path:
            m = getattr(m, p)
        m._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)

    loaded_bytes = loaded_n = 0
    with safe_open(args.weights, framework="pt", device="cpu") as f:
        for name in f.keys():
            if needed(name):
                t = f.get_tensor(name)
                set_param(model, name, t)
                loaded_bytes += t.numel() * t.element_size()
                loaded_n += 1
    model.eval()
    print(f"[load] materialized {loaded_n} params = {loaded_bytes/1024/1024:.1f}MB "
          f"(rest stay on meta)", flush=True)
    gc.collect()
    meminfo("after meta+materialize (STARTUP)")

    # 3) run conditioning (cond branch only, matches emoji_stageA 'cond')
    tok = PretrainedTextTokenizer.from_pretrained(cfg.text_tokenizer_repo, add_bos=cfg.text_add_bos)
    ref_len = max(1, int(cfg.speaker_patch_size))
    ref_latent = torch.zeros((1, ref_len, cfg.latent_dim * cfg.latent_patch_size), dtype=torch.float32)
    ref_mask = torch.zeros((1, ref_len), dtype=torch.bool)
    kinds = ["k_text", "v_text", "k_spk", "v_spk"]

    def run_once():
        norm = normalize_text(args.text).strip()
        ids, tmask = tok.batch_encode([norm], max_length=256)
        with torch.inference_mode():
            ts, tmc, ss, smc, _a, _b = model.encode_conditions(
                text_input_ids=ids, text_mask=tmask, ref_latent=ref_latent, ref_mask=ref_mask,
                speaker_state_override=None, speaker_mask_override=None, speaker_uncond_mode="mask")
            cache = model.build_context_kv_cache(text_state=ts, speaker_state=ss, caption_state=None)
        save = {}
        for li, layer in enumerate(cache):
            for k, ten in zip(kinds, layer):
                save[f"cond_{k}_{li}"] = ten.detach().cpu().float().numpy()
        save["text_mask"] = tmc.detach().cpu().numpy().astype(np.uint8)
        save["speaker_mask"] = smc.detach().cpu().numpy().astype(np.uint8)
        return save, int(tmask.sum())

    save, ntok = run_once()
    meminfo(f"after encode+kv (tok={ntok}, OPERATING)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **save)

    # 4) bitwise correctness vs full-model reference
    if Path(args.ref).exists():
        ref = np.load(args.ref)
        bad = []
        checked = 0
        for k in save:
            if k in ref:
                checked += 1
                if not np.array_equal(save[k], ref[k]):
                    bad.append(k)
        print(f"[correctness] checked {checked} keys vs ref; "
              f"{'ALL BITWISE-EQUAL ✅' if not bad else f'MISMATCH ❌ in {bad[:5]}'}", flush=True)
    else:
        print(f"[correctness] ref {args.ref} not found — skipped", flush=True)

    # 5) leak check: repeat encodes, watch RSS
    for i in range(args.repeats):
        run_once()
    meminfo(f"after {args.repeats} extra encodes (LEAK check)")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
