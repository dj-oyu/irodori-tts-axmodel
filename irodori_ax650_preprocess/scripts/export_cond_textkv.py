#!/usr/bin/env python3
"""
export_cond_textkv.py  (A1 完全NPU化 ①: no_ref 最小 cond axmodel)

no_ref 運用では text 依存なのは text KV のみ（speaker KV / text-branch は定数）。
本 export は「ids,mask -> text_encoder -> text_norm -> 各block の text KV射影」を
1本の ONNX にする。出力 = k_text_0..11, v_text_0..11 (各 [1,256,heads,head_dim])。

build_context_kv_cache の text 部分 (project_context_kv 内 wk_text/wv_text/k_norm) と
ビット等価。speaker KV と text-branch(zero-text) KV は別途 torch で一度計算し定数 bake。

I/O: input_ids[1,256] int64, mask[1,256] bool -> 24 text KV テンソル

実行（preprocess env, irodori_tts import するので PYTHONPATH 付与）:
  PYTHONPATH=/home/exe/ai/Irodori-TTS uv run --project irodori_ax650_preprocess \
    --with onnxruntime python irodori_ax650_preprocess/scripts/export_cond_textkv.py \
    --weights ~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v3/snapshots/*/model.safetensors \
    --model-cfg-json build/model_introspection.json --out build/cond_textkv.onnx
"""
from __future__ import annotations
import argparse, dataclasses, glob, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from safetensors.torch import load_file as load_sft

from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT


class CondTextKVWrapper(nn.Module):
    """ids,mask -> text_state -> per-block (k_text, v_text) [+ duration token_logits].

    no_ref duration head: speaker_vec is constant (baked) so the duration sub-graph is a
    pure function of text_state. Outputs per-token token_logits[1,S] (pre-softplus); the
    masked-sum + softplus + clamp -> t_valid is done on CPU (quant-robust, padded positions
    masked out like the KV)."""
    def __init__(self, model: nn.Module, speaker_vec_const: torch.Tensor | None = None):
        super().__init__()
        self.text_encoder = model.text_encoder
        self.text_norm = model.text_norm
        self.attentions = nn.ModuleList(b.attention for b in model.blocks)
        self.with_duration = speaker_vec_const is not None
        if self.with_duration:
            dp = model.duration_predictor
            self.token_input_proj = dp.token_input_proj
            self.token_blocks = dp.token_blocks
            self.token_out_norm = dp.token_out_norm
            self.token_out_proj = dp.token_out_proj
            self.register_buffer("speaker_vec", speaker_vec_const)

    def forward(self, input_ids, mask):
        ts = self.text_norm(self.text_encoder(input_ids, mask))   # (1,S,dim)
        bsz, S = ts.shape[0], ts.shape[1]
        outs = []
        for att in self.attentions:
            k = att.wk_text(ts).reshape(bsz, S, att.heads, att.head_dim)
            k = att.k_norm(k)
            v = att.wv_text(ts).reshape(bsz, S, att.heads, att.head_dim)
            outs.append(k); outs.append(v)
        if self.with_duration:
            h = self.token_input_proj(ts)
            for block in self.token_blocks:
                h = block(h, cond=self.speaker_vec)
            token_logits = self.token_out_proj(self.token_out_norm(h)).squeeze(-1)  # (1,S)
            outs.append(token_logits)
        return tuple(outs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model-cfg-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-duration", action="store_true", help="text KV のみ(duration head 除外)")
    args = ap.parse_args()

    cfg_all = json.loads(Path(args.model_cfg_json).read_text())["model_cfg"]
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_all.items() if k in fields})
    model = TextToLatentRFDiT(cfg)
    w = sorted(glob.glob(args.weights))[0]
    miss, unexp = model.load_state_dict(load_sft(w, device="cpu"), strict=False)
    print(f"[load] {w}\n[load] missing={len(miss)} unexpected={len(unexp)}")

    from rope_export_patch import apply_export_patches
    apply_export_patches(model)
    model.eval()

    # no_ref constant speaker_vec for the duration head (text-independent)
    speaker_vec = None
    if not args.no_duration and getattr(model, "duration_predictor", None) is not None:
        ref_len = max(1, int(cfg.speaker_patch_size))
        rl = torch.zeros((1, ref_len, cfg.latent_dim * cfg.latent_patch_size))
        rm = torch.zeros((1, ref_len), dtype=torch.bool)
        d_ids = torch.zeros(1, args.seq, dtype=torch.long)
        d_mask = torch.ones(1, args.seq, dtype=torch.bool)
        with torch.no_grad():
            _ts, _tmc, _ss, _smc, _, _ = model.encode_conditions(
                text_input_ids=d_ids, text_mask=d_mask, ref_latent=rl, ref_mask=rm,
                speaker_state_override=None, speaker_mask_override=None, speaker_uncond_mode="mask")
            speaker_vec = model.duration_predictor._speaker_vec(
                batch_size=1, device=_ss.device, dtype=_ss.dtype,
                speaker_state=_ss, has_speaker=torch.zeros(1, dtype=torch.bool))
        print(f"[duration] baked speaker_vec {tuple(speaker_vec.shape)} (no_ref const)")

    wrapper = CondTextKVWrapper(model, speaker_vec_const=speaker_vec).eval()
    nblk = len(wrapper.attentions)
    out_names = [f"{kind}_text_{i}" for i in range(nblk) for kind in ("k", "v")]
    if wrapper.with_duration:
        out_names.append("token_logits")

    input_ids = torch.arange(args.seq, dtype=torch.int64).unsqueeze(0) % cfg.text_vocab_size
    mask = torch.ones(1, args.seq, dtype=torch.bool)
    with torch.no_grad():
        ref = wrapper(input_ids, mask)
    print(f"[ref] {len(ref)} outputs, e.g. {out_names[0]}={tuple(ref[0].shape)} last={out_names[-1]}={tuple(ref[-1].shape)}")

    # duration fp32 cross-check: CPU softplus+masked-sum of token_logits vs torch predict_duration
    if wrapper.with_duration:
        import torch.nn.functional as F
        tl = ref[-1]
        total_cpu = (F.softplus(tl.float()) * mask.float()).sum(dim=1)  # = pred_frames
        aux_dim = int(cfg.duration_aux_dim)
        with torch.no_grad():
            ts2 = model.text_norm(model.text_encoder(input_ids, mask))
            plf = model.predict_duration_log_frames(
                text_state=ts2, text_mask=mask, speaker_state=_ss, speaker_mask=_smc,
                duration_features=torch.zeros(1, aux_dim), has_speaker=torch.zeros(1, dtype=torch.bool))
            pred_frames_torch = torch.expm1(plf).float()
        print(f"[duration] frames CPU(via token_logits)={total_cpu.item():.3f} vs torch={pred_frames_torch.item():.3f} "
              f"diff={abs(total_cpu.item()-pred_frames_torch.item()):.3e}")

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (input_ids, mask), str(out_path),
            input_names=["input_ids", "mask"], output_names=out_names,
            opset_version=args.opset, dynamo=True,
        )
    print("[export] OK ->", out_path)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feed = {"input_ids": input_ids.numpy(), "mask": mask.numpy()}
    feed = {n: feed[n] for n in [i.name for i in sess.get_inputs()]}
    outs = sess.run(out_names, feed)
    err = max(float(np.max(np.abs(ref[i].numpy() - outs[i]))) for i in range(len(ref)))
    print(f"[validate] {len(outs)} outputs, max_abs_err={err:.3e}")
    # cross-check vs build_context_kv_cache (cond branch text KV)
    print(f"[done] wrote {out_path}; output names: {out_names[:2]} ... {out_names[-2:]}")


if __name__ == "__main__":
    main()
