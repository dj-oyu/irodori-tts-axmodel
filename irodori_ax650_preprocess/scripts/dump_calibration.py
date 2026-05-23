#!/usr/bin/env python3
"""
dump_calibration.py

Pulsar2 PTQ 用の実 activation calibration データを作る。
DiT 1-step(Option B, 6入力) の固定 shape (B=1, T) に合わせ、実 synth の各 step・各 CFG 要素を
1 サンプルとして .npy に書き出し、入力ごとに tar.gz を作る。

ランダム入力だと DiT の量子化品質が落ちるため、実 activation を使うのが重要（FINDINGS）。

Irodori env で実行:
  cd /path/to/Irodori-TTS
  PYTHONPATH=/path/to/Irodori-TTS uv run python \
    /path/to/irodori-tts-axmodel/irodori_ax650_preprocess/scripts/dump_calibration.py \
    --out-dir /path/to/irodori-tts-axmodel/build/calib_b1 --text "こんにちは、テストです。" --num-steps 20
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

NAMES = ["x_t", "t", "text_state", "text_mask", "speaker_state", "speaker_mask"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-500M-v3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text", default="こんにちは、テストです。")
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--max-samples", type=int, default=64)
    ap.add_argument("--kv", action="store_true",
                    help="Option A 用に context_kv_cache(48 tensor) も dump する")
    args = ap.parse_args()

    ckpt = hf_hub_download(repo_id=args.hf_checkpoint, filename="model.safetensors")
    rt = InferenceRuntime.from_key(
        RuntimeKey(checkpoint=ckpt, model_device=args.device, codec_device=args.device)
    )
    model = rt.model
    orig = model.forward_with_encoded_conditions
    samples: list[dict] = []

    def hook(*a, **kw):
        out = orig(*a, **kw)
        b = kw["x_t"].shape[0]
        for i in range(b):  # CFG 各要素を別サンプルに（多様性）
            samples.append({n: kw[n][i : i + 1].detach().cpu() for n in NAMES})
        return out

    model.forward_with_encoded_conditions = hook
    try:
        rt.synthesize(
            SamplingRequest(text=args.text, no_ref=True, num_steps=args.num_steps,
                            seed=0, context_kv_cache=False)
        )
    finally:
        model.forward_with_encoded_conditions = orig

    samples = samples[: args.max_samples]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[calib] {len(samples)} samples, shapes:")
    for n in NAMES:
        print(f"  {n}: {tuple(samples[0][n].shape)} {samples[0][n].dtype}")

    # Option A: 各サンプルの (text_state, speaker_state) から KV cache を計算して追加。
    kv_names: list[str] = []
    if args.kv:
        kinds = ["k_text", "v_text", "k_spk", "v_spk"]
        with torch.no_grad():
            for s in samples:
                cache = model.build_context_kv_cache(
                    s["text_state"].to(args.device), s["speaker_state"].to(args.device)
                )
                for i, layer in enumerate(cache):
                    for kind, tns in zip(kinds, layer):
                        s[f"{kind}_{i}"] = tns.detach().cpu()
        n_layers = len(cache)
        kv_names = [f"{kind}_{i}" for i in range(n_layers) for kind in kinds]
        print(f"[calib] +KV {len(kv_names)} tensors, e.g. k_text_0={tuple(samples[0]['k_text_0'].shape)}")

    # 入力ごとに tar.gz（中は flat な NNN.npy）
    for n in NAMES + kv_names:
        d = out_dir / n
        d.mkdir(exist_ok=True)
        for j, s in enumerate(samples):
            arr = s[n].numpy()
            # Pulsar2 は calib .npy の dtype を ONNX 入力 dtype と一致させる必要がある
            # （mask は bool のまま保存。uint8 にすると dtype mismatch で弾かれる）。
            np.save(d / f"{j:04d}.npy", arr)
        tar_path = out_dir / f"{n}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            for j in range(len(samples)):
                tf.add(d / f"{j:04d}.npy", arcname=f"{j:04d}.npy")
        print(f"  wrote {tar_path}")
    print(f"[done] calibration under {out_dir}")


if __name__ == "__main__":
    main()
