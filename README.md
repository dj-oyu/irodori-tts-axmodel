# irodori-tts-axmodel

**v0.2 (2026-05-23)**

[Irodori-TTS-500M-v3](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)（Flow Matching ベースの
日本語 TTS）を **Axera AX8850 / M5Stack AI Pyramid Pro** の NPU で動かすための変換・前処理ツール群。

支配的コストの **DiT denoiser** と **TextEncoder** を ONNX 経由で `.axmodel` に変換し AX650/AX8850 NPU で実行。
DACVAE vocoder は Pulsar2 制約で NPU 不可のため CPU 据え置き、tokenizer/watermark 等も CPU。

> ⚠️ 本リポジトリは**変換パイプラインとスクリプト**を管理する。
> 巨大な生成物（`.onnx` / `.onnx.data` / `.axmodel` / calibration / wav / `.venv`）は
> `.gitignore` 済みで**コミットされない**（スクリプトで再生成可能）。

## 現状（2026-05-23）

| 項目 | 状態 |
|---|---|
| v3 baseline 推論（4090）| ✅ |
| DiT 1-step → ONNX（Option A/B, 数値 ~1e-6, full-loop SNR 66–67dB）| ✅ |
| **DiT → `.axmodel`**（Option B 379MB / Option A 338MB, 単一 NPU subgraph）| ✅ |
| **TextEncoder → `.axmodel`**（単一 NPU subgraph）| ✅ |
| few-step sampling（**sway + 6–8 step** が実用域, DiT 呼び出し ~2×減）| ✅ |
| W8A8 量子化精度（v_pred **cosine 0.9997**）+ fp16 可聴比較（SNR 44dB）| ✅ |
| **DACVAE → NPU** | ❌ Pulsar2 6.0 の Snake op 制約で**不可**（全 chunk 検証済→CPU 確定）|
| W8A16 化 / 実機 end-to-end latency 実測 | ⏳ 未 |

### 変換済み artifact（`build/`, gitignore 済み・要再生成）
| モジュール | axmodel | size | 実行先 |
|---|---|---|---|
| DiT Option B（6入力）| `axmodel_b1/` | 379MB | NPU |
| DiT Option A（KV cache, 52入力）| `axmodel_kv_b1/` | 338MB | NPU（性能本命）|
| TextEncoder | `axmodel_textenc/` | — | NPU |
| DACVAE decoder | （ONNX のみ）| — | **CPU**（NPU 不可）|

詳細な技術メモ・設計判断・実地で潰した変換ブロッカは
[`irodori_ax650_preprocess/FINDINGS.md`](irodori_ax650_preprocess/FINDINGS.md) を参照（必読）。

## 構成

```
irodori_ax650_preprocess/
  scripts/
    inspect_model.py            # v3 の実構造・I/O shape をダンプ
    capture_dit_inputs.py       # 実 synth から DiT 実入力/出力を捕捉（検証・calib の土台）
    rope_export_patch.py        # export-safe patch（RoPE実数化/SDPA additive/RMSNorm・AdaLN rsqrt除去）
    export_dit_step.py          # DiT 1-step → ONNX（Option B: 6入力, fp16/dynamic 対応）
    export_dit_step_kvcache.py  # DiT 1-step → ONNX（Option A: KV cache 入力, 52入力）
    export_text_encoder.py      # TextEncoder → ONNX
    export_dacvae_decoder.py    # DACVAE decoder → ONNX（NPU 不可だが CPU/onnxruntime 用）
    dump_calibration.py         # 実 activation の PTQ calibration 生成（--kv で Option A 用）
    validate_full_loop.py       # sampling ループ全体を ONNX で回し wav 一致を検証（--mode nokv/kv）
    step_sweep.py               # few-step 品質/コスト スイープ（mel-L1 vs 高 step）
    compare_onnx_wav.py         # fp32 vs fp16 の end-to-end wav 比較（paplay 用）
  configs/                      # 入力 shape spec
  FINDINGS.md                   # 技術メモ（必読・全変換ブロッカと結論）
docs/
  deploy_ai_pyramid_pro.md      # AI Pyramid Pro 実機デプロイ手順
build/                          # 生成物（gitignore）。pulsar_configs と model_introspection.json のみ追跡
```

## 前提

- 変換ホスト: x86_64 + NVIDIA GPU 推奨, Docker, [uv](https://docs.astral.sh/uv/)
- 隣に [`Irodori-TTS`](https://github.com/Aratako/Irodori-TTS) を clone（`../Irodori-TTS`, `uv sync --extra cu128`）
- [Pulsar2](https://huggingface.co/AXERA-TECH/Pulsar2) 6.0 Docker イメージ（`docker load` → `pulsar2:6.0`）

## クイックスタート（DiT → .axmodel, Option B）

```bash
# 0) 実 DiT 入力を捕捉（Irodori-TTS env）
cd irodori_ax650_preprocess
PYTHONPATH=/path/to/Irodori-TTS uv run python scripts/capture_dit_inputs.py \
  --out ../build/dit_step_b1_fp32.ref.pt --text "こんにちは、テストです。"
# 1) ONNX 化（export patch 内包, 数値検証付き）
PYTHONPATH=/path/to/Irodori-TTS uv run python scripts/export_dit_step.py \
  --weights ~/.cache/huggingface/.../model.safetensors \
  --model-cfg-json ../build/model_introspection.json \
  --inputs ../build/dit_step_b1_fp32.ref.pt --out ../build/dit_step_b1_fp32.onnx
# 2) calibration 生成
PYTHONPATH=/path/to/Irodori-TTS uv run python scripts/dump_calibration.py \
  --out-dir ../build/calib_b1 --text "こんにちは、テストです。" --num-steps 20
# 3) Pulsar2 で .axmodel 化（AX650 = AX8850）
docker run --rm -v "$PWD/..:/data" pulsar2:6.0 -c \
  "cd /data && pulsar2 build --target_hardware AX650 \
     --input build/dit_step_b1_fp32.onnx --output_dir build/axmodel_b1 \
     --config build/pulsar_configs/b1_min.json"
```

実機での利用・CPU/NPU 分担・統合コードは [`docs/deploy_ai_pyramid_pro.md`](docs/deploy_ai_pyramid_pro.md)。

## ライセンス

[MIT](LICENSE)。上流 [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) も MIT で抵触しない
（本リポジトリは Irodori-TTS のコードを再配布せず、依存として import するのみ）。
