#!/usr/bin/env python3
"""
validate_full_loop.py

DiT 1-step ONNX を「サンプリングループ全体」で検証する。
runtime.model.forward_with_encoded_conditions を onnxruntime 呼び出しに差し替えて
synthesize を丸ごと回し、純 PyTorch の synthesize（同 seed）と wav を比較する。

単発スナップショット検証では見えない「全 timestep / CFG batch B=3」を実地に通す。

Irodori env で onnxruntime を載せて実行:
  cd /path/to/Irodori-TTS
  PYTHONPATH=/path/to/Irodori-TTS uv run --with onnxruntime python \
    /path/to/irodori-tts-axmodel/irodori_ax650_preprocess/scripts/validate_full_loop.py \
    --onnx /path/to/irodori-tts-axmodel/build/dit_step_dyn_fp32.onnx
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-500M-v3")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mode", choices=["nokv", "kv"], default="nokv",
                    help="nokv=Option B(6入力) / kv=Option A(context_kv_cache 入力)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text", default="こんにちは、これはテスト音声です。")
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    ckpt = hf_hub_download(repo_id=args.hf_checkpoint, filename="model.safetensors")
    runtime = InferenceRuntime.from_key(
        RuntimeKey(checkpoint=ckpt, model_device=args.device, codec_device=args.device)
    )
    model = runtime.model

    use_kv = args.mode == "kv"

    def make_req():
        # nokv: recompute 経路に統一(B と apples-to-apples)。kv: cache 経路(A と一致)。
        return SamplingRequest(
            text=args.text, no_ref=True, num_steps=args.num_steps,
            seed=args.seed, context_kv_cache=use_kv,
        )

    # 1) 純 PyTorch
    res_pt = runtime.synthesize(make_req())
    wav_pt = res_pt.audio.detach().cpu().float().numpy()
    print(f"[pytorch] wav {wav_pt.shape} seed={res_pt.used_seed}")

    # 2) ONNX shim
    import onnxruntime as ort

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    graph_inputs = [i.name for i in sess.get_inputs()]
    print(f"[onnx] mode={args.mode} num_inputs={len(graph_inputs)}")

    orig = model.forward_with_encoded_conditions
    call_count = {"n": 0}

    def to_np(t):
        a = t.detach().cpu().numpy()
        return a.astype(np.bool_) if t.dtype == torch.bool else a.astype(np.float32)

    def shim(*a, **kw):
        x_t = kw["x_t"]
        all_feed = {n: to_np(kw[n]) for n in
                    ["x_t", "t", "text_state", "text_mask", "speaker_state", "speaker_mask"]}
        if use_kv:
            # context_kv_cache: list[tuple(k_text,v_text,k_spk,v_spk)] を平坦化して名前付け。
            cache = kw["context_kv_cache"]
            kinds = ["k_text", "v_text", "k_spk", "v_spk"]
            for i, layer in enumerate(cache):
                for kind, tns in zip(kinds, layer):
                    all_feed[f"{kind}_{i}"] = to_np(tns)
        feed = {n: all_feed[n] for n in graph_inputs}
        out = sess.run(["v_pred"], feed)[0]
        call_count["n"] += 1
        return torch.from_numpy(out).to(device=x_t.device, dtype=x_t.dtype)

    model.forward_with_encoded_conditions = shim
    try:
        res_onnx = runtime.synthesize(make_req())
    finally:
        model.forward_with_encoded_conditions = orig
    wav_onnx = res_onnx.audio.detach().cpu().float().numpy()
    print(f"[onnx] wav {wav_onnx.shape} fwec_calls={call_count['n']}")

    # 3) 比較
    n = min(wav_pt.shape[-1], wav_onnx.shape[-1])
    a = wav_pt[..., :n].ravel()
    b = wav_onnx[..., :n].ravel()
    max_abs = float(np.max(np.abs(a - b)))
    rms = float(np.sqrt(np.mean((a - b) ** 2)))
    sig = float(np.sqrt(np.mean(a**2))) + 1e-12
    snr = 20.0 * np.log10(sig / (rms + 1e-12))
    corr = float(np.corrcoef(a, b)[0, 1])
    print("[compare] len_pt=%d len_onnx=%d (cmp %d)" % (wav_pt.shape[-1], wav_onnx.shape[-1], n))
    print(f"[compare] max_abs={max_abs:.3e}  rms_err={rms:.3e}  SNR={snr:.1f} dB  corr={corr:.6f}")
    if wav_pt.shape[-1] != wav_onnx.shape[-1]:
        print("[compare] NOTE: 長さ差あり（trim_tail/duration の僅差）。波形相関で判断。")


if __name__ == "__main__":
    main()
