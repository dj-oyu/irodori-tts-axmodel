#!/usr/bin/env python3
"""
export_dit_step_kvcache.py

Option A: context_kv_cache を「入力」に取る DiT 1-step を ONNX 化する。
KV projection を host で 1 回計算（build_context_kv_cache）→ 全 step 再利用する形。
CMM(ユニファイドメモリ) 前提の AX8850 では帯域収支がこちらが有利な可能性が高い本命候補
（FINDINGS「DiT export: 2 つの形」参照）。入力は 6 + 12層x4=48 = 計 54。

KV cache テンソルは保存済み inputs(.ref.pt) から model.build_context_kv_cache で生成。
runtime は import しない（protobuf 衝突回避, irodori_tts.model のみ）。

実行（preprocess env）:
  cd /path/to/irodori-tts-axmodel/irodori_ax650_preprocess
  PYTHONPATH=/path/to/Irodori-TTS uv run python scripts/export_dit_step_kvcache.py \
    --weights ~/.cache/huggingface/hub/models--Aratako--Irodori-TTS-500M-v3/snapshots/*/model.safetensors \
    --model-cfg-json ../build/model_introspection.json \
    --inputs ../build/dit_step_b3.ref.pt \
    --out ../build/dit_step_kv_b3_fp32.onnx
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file

from irodori_tts.config import ModelConfig
from irodori_tts.model import TextToLatentRFDiT


class DiTStepWrapperKV(nn.Module):
    """
    Option A: 平坦化した context_kv_cache を入力に取り、毎 step の KV 再計算を省く。

    KV cache 提供時、text_state/speaker_state は forward_with_encoded_conditions 内で
    実際には使われない（K/V は cache 由来、mask は別入力）。これらを ONNX 入力にすると
    Pulsar2 の onnxsim が dead input を prune して入力数が食い違う（54 vs 52）。
    → 入力に取らず内部で zeros 定数として構築し、ONNX を本質的に 52 入力にする。
    """

    def __init__(self, model: nn.Module, num_layers: int, per_layer: int,
                 text_dim: int, speaker_dim: int):
        super().__init__()
        self.model = model
        self.num_layers = int(num_layers)
        self.per_layer = int(per_layer)
        self.text_dim = int(text_dim)
        self.speaker_dim = int(speaker_dim)

    def forward(self, x_t, t, text_mask, speaker_mask, *kv):
        b = x_t.shape[0]
        text_state = x_t.new_zeros((b, text_mask.shape[1], self.text_dim))
        speaker_state = x_t.new_zeros((b, speaker_mask.shape[1], self.speaker_dim))
        cache = []
        for layer in range(self.num_layers):
            base = layer * self.per_layer
            cache.append(tuple(kv[base : base + self.per_layer]))
        return self.model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            caption_state=None,
            caption_mask=None,
            latent_mask=None,
            context_kv_cache=cache,
        )


def build_model(weights: str, cfg_json: str) -> TextToLatentRFDiT:
    cfg_all = json.loads(Path(cfg_json).read_text())["model_cfg"]
    field_names = {f.name for f in dataclasses.fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in cfg_all.items() if k in field_names})
    model = TextToLatentRFDiT(cfg)
    state = load_safetensors_file(weights, device="cpu")
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unexp)} params={sum(p.numel() for p in model.parameters())}")
    return model.eval(), cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model-cfg-json", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--dynamic", action="store_true")
    args = ap.parse_args()

    weights = sorted(glob.glob(args.weights))
    if not weights:
        raise FileNotFoundError(args.weights)
    model, cfg = build_model(weights[0], args.model_cfg_json)

    from rope_export_patch import apply_export_patches

    apply_export_patches(model)

    has_speaker = bool(cfg.use_speaker_condition)
    has_caption = bool(cfg.use_caption_condition)
    per_layer = 2 + (2 if has_speaker else 0) + (2 if has_caption else 0)
    num_layers = int(cfg.num_layers)
    wrapper = DiTStepWrapperKV(
        model, num_layers, per_layer, int(cfg.text_dim), int(cfg.speaker_dim)
    ).eval()

    blob = torch.load(args.inputs, map_location="cpu", weights_only=False)
    ins = blob["inputs"]
    x_t, t = ins["x_t"], ins["t"]
    text_state, text_mask = ins["text_state"], ins["text_mask"]
    speaker_state, speaker_mask = ins["speaker_state"], ins["speaker_mask"]

    # KV cache を生成（k_norm 込み）。
    with torch.no_grad():
        cache = model.build_context_kv_cache(text_state, speaker_state)
    flat_kv = [tensor for layer in cache for tensor in layer]
    assert len(flat_kv) == num_layers * per_layer, (len(flat_kv), num_layers, per_layer)

    # text_state/speaker_state は KV cache 提供時 未使用。内部 zeros で等価なことを確認。
    with torch.no_grad():
        out_real = model.forward_with_encoded_conditions(
            x_t=x_t, t=t, text_state=text_state, text_mask=text_mask,
            speaker_state=speaker_state, speaker_mask=speaker_mask,
            caption_state=None, caption_mask=None, latent_mask=None, context_kv_cache=cache)
        out_zeros = wrapper(x_t, t, text_mask, speaker_mask, *flat_kv)
    eq = (out_real - out_zeros).abs().max().item()
    print(f"[check] zeros-vs-real text/speaker_state max_abs_err={eq:.3e}")
    assert eq < 1e-4, "text_state/speaker_state が実際に使われている（zeros 化不可）"

    # 入力(52): x_t, t, text_mask, speaker_mask + 12層x(k_text,v_text,k_spk,v_spk)
    base_names = ["x_t", "t", "text_mask", "speaker_mask"]
    kinds = ["k_text", "v_text"] + (["k_spk", "v_spk"] if has_speaker else [])
    kv_names = [f"{kind}_{i}" for i in range(num_layers) for kind in kinds]
    names = base_names + kv_names

    base = [x_t, t, text_mask, speaker_mask]
    if args.fp16:
        model.half()
        base = [b.half() if b.is_floating_point() else b for b in base]
        flat_kv = [k.half() for k in flat_kv]
    dummy = tuple(base) + tuple(flat_kv)

    print(f"[shapes] x_t={tuple(x_t.shape)} ; inputs={len(names)} (kv={len(flat_kv)}) "
          f"e.g. k_text_0={tuple(flat_kv[0].shape)} k_spk_0={tuple(flat_kv[2].shape)}")

    with torch.no_grad():
        ref = wrapper(*dummy)
    print(f"[ref] v_pred {tuple(ref.shape)} {ref.dtype}")

    dynamic_shapes = None
    if args.dynamic:
        from torch.export import Dim

        B = Dim("B", min=1, max=8)
        T = Dim("T", min=1, max=4096)
        # base 4 + vararg(KV) group。
        ds = [{0: B, 1: T}, {0: B}, {0: B}, {0: B}]
        ds.append(tuple({0: B} for _ in flat_kv))
        dynamic_shapes = tuple(ds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("[export] dynamo exporter ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(out_path),
            input_names=names, output_names=["v_pred"],
            opset_version=args.opset, dynamo=True, dynamic_shapes=dynamic_shapes,
        )
    print("[export] dynamo OK")

    # 数値検証（onnxruntime, グラフ実入力のみ feed）
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    graph_inputs = {i.name for i in sess.get_inputs()}
    all_feed = {n: d for n, d in zip(names, dummy)}
    feed = {}
    for n in graph_inputs:
        d = all_feed[n]
        arr = d.cpu().numpy()
        feed[n] = arr.astype(np.bool_) if d.dtype == torch.bool else arr.astype(np.float32)
    pruned = [n for n in names if n not in graph_inputs]
    if pruned:
        print(f"[note] exporter pruned unused inputs: {pruned}")
    ort_out = sess.run(["v_pred"], feed)[0]
    ref_np = ref.cpu().float().numpy()
    max_abs = float(np.max(np.abs(ref_np - ort_out.astype(np.float32))))
    denom = float(np.max(np.abs(ref_np))) + 1e-9
    print(f"[validate] graph_inputs={len(graph_inputs)}  max_abs_err={max_abs:.3e}  rel={max_abs/denom:.3e}")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
