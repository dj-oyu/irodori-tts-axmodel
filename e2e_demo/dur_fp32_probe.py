#!/usr/bin/env python3
"""fp32 reference for the A3 duration head — splits quant-vs-model/formula.
meta-slim load (text_encoder/text_norm/speaker_encoder/speaker_norm/duration_predictor),
no_ref encode_conditions, capture per-token duration logits via forward hook on
token_out_proj, compare to the NPU ① axmodel's token_logits. Also prints BOS ids.
  PYTHONPATH=/path/to/Irodori-TTS python3 e2e_demo/dur_fp32_probe.py --weights <model.safetensors>
"""
from __future__ import annotations
import argparse, dataclasses, json
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer
from irodori_tts.text_normalization import normalize_text

# NPU ① token_logits->softplus per valid token (from on-device run, for side-by-side)
NPU = {
    "plain":    [41.1, 15.8, 28.0, 21.2, 27.5],
    "sibilant": [4.9, 13.0, 3.4, 7.9, 3.7, 10.4, 5.4, 8.7, 3.1, 10.6, 8.4],
}
TEXTS = [("plain", "今日はとても楽しいです。"),
         ("sibilant", "資料を探して整理する作業を続けましょう。"),
         ("long", "今日はとても良い天気ですね。少し散歩に出かけませんか。それから近くのカフェでコーヒーを飲みながら、最近読んだ本の話でもゆっくりしましょう。")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model-cfg-json", default="build/model_introspection.json")
    args = ap.parse_args()
    cfg_all = json.loads(Path(args.model_cfg_json).read_text())["model_cfg"]
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_all.items() if k in fields})

    with torch.device("meta"):
        model = TextToLatentRFDiT(cfg)
    def needed(n):
        return (n.startswith("text_encoder.") or n == "text_norm.weight"
                or n.startswith("speaker_encoder.") or n == "speaker_norm.weight"
                or n.startswith("duration_predictor."))
    def setp(root, dotted, t):
        *path, leaf = dotted.split(".")
        m = root
        for p in path: m = getattr(m, p)
        m._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)
    with safe_open(args.weights, framework="pt", device="cpu") as f:
        for n in f.keys():
            if needed(n): setp(model, n, f.get_tensor(n))
    model.eval()
    print(f"[load] duration_architecture={cfg.duration_architecture} aux_dim={cfg.duration_aux_dim}")

    tok = PretrainedTextTokenizer.from_pretrained(cfg.text_tokenizer_repo, add_bos=cfg.text_add_bos)
    rl = max(1, int(cfg.speaker_patch_size))
    ref_latent = torch.zeros((1, rl, cfg.latent_dim * cfg.latent_patch_size), dtype=torch.float32)
    ref_mask = torch.zeros((1, rl), dtype=torch.bool)

    for label, text in TEXTS:
        ids, tmask = tok.batch_encode([normalize_text(text).strip()], max_length=256)
        ntok = int(tmask.sum())
        cap = {}
        h = model.duration_predictor.token_out_proj.register_forward_hook(
            lambda m, i, o: cap.__setitem__("tl", o.detach()))
        with torch.inference_mode():
            ts, tmc, ss, smc, _, _ = model.encode_conditions(
                text_input_ids=ids, text_mask=tmask, ref_latent=ref_latent, ref_mask=ref_mask,
                speaker_state_override=None, speaker_mask_override=None, speaker_uncond_mode="mask")
            lf = model.predict_duration_log_frames(
                text_state=ts, text_mask=tmc, speaker_state=ss, speaker_mask=smc,
                duration_features=torch.zeros(1, cfg.duration_aux_dim), has_speaker=torch.tensor([False]))
        h.remove()
        tl = cap["tl"].squeeze(-1).float().numpy().ravel()
        sp = np.logaddexp(0.0, tl)[:ntok]
        total = float(np.expm1(lf.item()))
        print(f"\n[{label}] tok={ntok}  first_ids={ids[0,:6].tolist()}  bos={cfg.text_add_bos}")
        print(f"  fp32 softplus/token(valid): {np.round(sp,1)}")
        print(f"  fp32 sum={sp.sum():.1f}  total(expm1 of log_frames)={total:.1f}")
        if label in NPU:
            print(f"  NPU  softplus/token(valid): {np.round(NPU[label],1)}  sum={sum(NPU[label]):.1f}")


if __name__ == "__main__":
    main()
