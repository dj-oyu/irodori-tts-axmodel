#!/usr/bin/env python3
"""
step_sweep.py

few-step sampling の品質/コスト スイープ。
高ステップ基準(既定 32, linear)に対し、num_steps x schedule を変えて
- log-mel L1 距離（基準への近さ = 品質劣化の代理指標）
- sample_rf 時間（NPU step×N に比例）
を出す。同 seed/text, trim_tail=False で長さを揃えて比較。wav も保存して試聴可能に。

Irodori env で実行:
  cd /path/to/Irodori-TTS
  PYTHONPATH=/path/to/Irodori-TTS uv run python \
    /path/to/irodori-tts-axmodel/irodori_ax650_preprocess/scripts/step_sweep.py \
    --out-dir /path/to/irodori-tts-axmodel/build/fewstep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest, save_wav


def log_mel(wav: torch.Tensor, sr: int) -> torch.Tensor:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128
    )(wav)
    return torch.log(mel + 1e-5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-500M-v3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text", default="こんにちは、これはテスト音声です。今日はいい天気ですね。")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ref-steps", type=int, default=32)
    args = ap.parse_args()

    ckpt = hf_hub_download(repo_id=args.hf_checkpoint, filename="model.safetensors")
    rt = InferenceRuntime.from_key(
        RuntimeKey(checkpoint=ckpt, model_device=args.device, codec_device=args.device)
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def synth(num_steps: int, schedule: str, sway: float):
        req = SamplingRequest(
            text=args.text, no_ref=True, num_steps=num_steps, seed=args.seed,
            t_schedule_mode=schedule, sway_coeff=sway, trim_tail=False,
        )
        res = rt.synthesize(req)
        t = dict(res.stage_timings).get("sample_rf", 0.0)
        return res.audio.detach().cpu().float(), res.sample_rate, t

    # 基準（高ステップ・linear）
    ref_wav, sr, ref_t = synth(args.ref_steps, "linear", -1.0)
    save_wav(out_dir / f"ref_{args.ref_steps}step.wav", ref_wav, sr)
    ref_lm = log_mel(ref_wav, sr)
    print(f"[ref] {args.ref_steps} step linear: sample_rf={ref_t*1000:.0f}ms")

    configs = []
    for n in [4, 6, 8, 12, 16]:
        for sched, sway in [("linear", -1.0), ("sway", -1.0)]:
            configs.append((n, sched, sway))

    print(f"\n{'steps':>5} {'sched':>6} {'sample_rf(ms)':>13} {'mel_L1_vs_ref':>14}")
    rows = []
    for n, sched, sway in configs:
        wav, sr, t = synth(n, sched, sway)
        save_wav(out_dir / f"{n:02d}step_{sched}.wav", wav, sr)
        m = min(ref_lm.shape[-1], log_mel(wav, sr).shape[-1])
        lm = log_mel(wav, sr)
        l1 = (ref_lm[..., :m] - lm[..., :m]).abs().mean().item()
        rows.append((n, sched, t * 1000, l1))
        print(f"{n:>5} {sched:>6} {t*1000:>13.0f} {l1:>14.4f}")

    print("\n[guide] mel_L1 が ref と近い(小さい)ほど劣化が少ない。"
          "step を下げて L1 が急増する手前が実用下限の目安。wav を試聴して最終判断。")
    print(f"[done] wavs under {out_dir}")


if __name__ == "__main__":
    main()
