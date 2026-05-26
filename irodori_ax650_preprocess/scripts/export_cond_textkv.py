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
    """ids,mask -> text_state -> per-block (k_text, v_text)."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.text_encoder = model.text_encoder
        self.text_norm = model.text_norm
        self.attentions = nn.ModuleList(b.attention for b in model.blocks)

    def forward(self, input_ids, mask):
        ts = self.text_norm(self.text_encoder(input_ids, mask))   # (1,S,dim)
        bsz, S = ts.shape[0], ts.shape[1]
        outs = []
        for att in self.attentions:
            k = att.wk_text(ts).reshape(bsz, S, att.heads, att.head_dim)
            k = att.k_norm(k)
            v = att.wv_text(ts).reshape(bsz, S, att.heads, att.head_dim)
            outs.append(k); outs.append(v)
        return tuple(outs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model-cfg-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--opset", type=int, default=17)
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
    wrapper = CondTextKVWrapper(model).eval()
    nblk = len(wrapper.attentions)
    out_names = [f"{kind}_text_{i}" for i in range(nblk) for kind in ("k", "v")]

    input_ids = torch.arange(args.seq, dtype=torch.int64).unsqueeze(0) % cfg.text_vocab_size
    mask = torch.ones(1, args.seq, dtype=torch.bool)
    with torch.no_grad():
        ref = wrapper(input_ids, mask)
    print(f"[ref] {len(ref)} outputs, e.g. {out_names[0]}={tuple(ref[0].shape)} {out_names[1]}={tuple(ref[1].shape)}")

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
