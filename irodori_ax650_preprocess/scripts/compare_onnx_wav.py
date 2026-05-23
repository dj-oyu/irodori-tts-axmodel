#!/usr/bin/env python3
"""
compare_onnx_wav.py

「最初の onnx(fp32)」と「軽量化版 onnx(fp16)」を runtime の DiT step に差し込んで
それぞれ end-to-end の wav を生成し、聴き比べ用に保存 + mel-L1 を出す。
（真の INT8 axmodel は NPU 実行が必要なため、x86 で回せる軽量化代理として fp16 を使う）

Irodori env で onnxruntime 込みで実行:
  cd /path/to/Irodori-TTS
  PYTHONPATH=/path/to/Irodori-TTS uv run --with onnxruntime python \
    /path/to/irodori-tts-axmodel/irodori_ax650_preprocess/scripts/compare_onnx_wav.py \
    --fp32-onnx /path/to/build/dit_step_dyn_fp32.onnx \
    --fp16-onnx /path/to/build/dit_step_dyn_fp16.onnx \
    --out-dir /path/to/build/compare
再生: paplay <wav>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest, save_wav

BASE = ["x_t", "t", "text_state", "text_mask", "speaker_state", "speaker_mask"]


def make_shim(sess, orig):
    in_types = {i.name: i.type for i in sess.get_inputs()}

    def to_np(name, t):
        a = t.detach().cpu().numpy()
        ty = in_types[name]
        if "float16" in ty:
            return a.astype(np.float16)
        if "uint8" in ty:
            return a.astype(np.uint8)
        if "bool" in ty:
            return a.astype(np.bool_)
        if "int64" in ty:
            return a.astype(np.int64)
        return a.astype(np.float32)

    def shim(*a, **kw):
        x_t = kw["x_t"]
        feed = {n: to_np(n, kw[n]) for n in BASE if n in in_types}
        out = sess.run(["v_pred"], feed)[0]
        return torch.from_numpy(out.astype(np.float32)).to(device=x_t.device, dtype=x_t.dtype)

    return shim


def log_mel(wav, sr):
    return torch.log(torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128)(wav) + 1e-5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-500M-v3")
    ap.add_argument("--fp32-onnx", required=True)
    ap.add_argument("--fp16-onnx", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text", default="こんにちは、これはテスト音声です。今日はいい天気ですね。")
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import onnxruntime as ort

    ck = hf_hub_download(args.hf_checkpoint, "model.safetensors")
    rt = InferenceRuntime.from_key(
        RuntimeKey(checkpoint=ck, model_device=args.device, codec_device=args.device))
    model = rt.model
    orig = model.forward_with_encoded_conditions
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def synth_with(onnx_path, tag):
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        model.forward_with_encoded_conditions = make_shim(sess, orig)
        try:
            res = rt.synthesize(SamplingRequest(
                text=args.text, no_ref=True, num_steps=args.num_steps,
                seed=args.seed, context_kv_cache=False, t_schedule_mode="sway"))
        finally:
            model.forward_with_encoded_conditions = orig
        p = out_dir / f"{tag}.wav"
        save_wav(p, res.audio, res.sample_rate)
        print(f"[{tag}] saved {p}  ({res.audio.shape[-1]} samples)")
        return res.audio.detach().cpu().float(), res.sample_rate

    w32, sr = synth_with(args.fp32_onnx, "dit_fp32")
    w16, _ = synth_with(args.fp16_onnx, "dit_fp16")

    n = min(w32.shape[-1], w16.shape[-1])
    l1 = (log_mel(w32[..., :n], sr) - log_mel(w16[..., :n], sr)).abs().mean().item()
    a, b = w32[..., :n].flatten(), w16[..., :n].flatten()
    snr = 20 * np.log10(float(torch.sqrt((a**2).mean())) / (float(torch.sqrt(((a-b)**2).mean())) + 1e-12) + 1e-12)
    print(f"\n[compare] fp32 vs fp16:  mel_L1={l1:.4f}  waveform_SNR={snr:.1f} dB")
    print(f"[play] paplay {out_dir/'dit_fp32.wav'}")
    print(f"[play] paplay {out_dir/'dit_fp16.wav'}")


if __name__ == "__main__":
    main()
