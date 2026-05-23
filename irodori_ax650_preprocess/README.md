# Irodori-TTS Lite/v3 -> AX650/Pulsar 前処理 scaffold（初期版・歴史的記録）

> ## ⚠️ この README は初期 scaffold の記録です（v0.1）。最新は以下を参照:
> - **リポジトリ概要・現状・クイックスタート**: [`../README.md`](../README.md)（root, v0.2）
> - **技術メモ・全変換ブロッカ・結論**: [`FINDINGS.md`](FINDINGS.md)（必読・TL;DR あり）
> - **実機デプロイ手順**: [`../docs/deploy_ai_pyramid_pro.md`](../docs/deploy_ai_pyramid_pro.md)
>
> 本ファイル以下の本文は当初の想定で書かれており、実モデル検証で判明した相違がある:
> - ターゲット SoC は **AX8850**（Pulsar2 では `--target_hardware AX650`）。
> - 実 export は `scripts/export_dit_step.py` 等（下記 `prepare_ax650_export.py` は **未使用の旧 scaffold**）。
> - `InferenceRuntime.from_key` は `RuntimeKey` dataclass を取る（HF 文字列ではない）。
> - v3 実構成: latent_dim=32, model_dim=1280, 12 層, num_heads=20, fp32/bf16 のみ。
> 以下は参考のため残置。

## 目的

Irodori-TTS を AX650/Pulsar に持っていく前に、NPU向きのサブモジュールだけを
固定shape ONNXとして切り出すための最小プロジェクトです。

Lite の INT4/Triton fused runtime を AX650に直接移植するのではなく、export時だけ
ONNXに落としやすい fp16/torch module 形へ戻す前提です。

## セットアップ

```bash
uv sync
```

Irodori-TTS / Irodori-TTS-Lite を同じ環境に入れてください。ローカルcloneを使うなら例:

```bash
uv pip install -e ../Irodori-TTS
uv pip install -e ../Irodori-TTS-Lite
```

## まずモジュール一覧を見る

loader と loader-arg は実際の Irodori-TTS runtime に合わせて調整してください。

```bash
uv run irodori-ax650-prep \
  --loader irodori_tts.inference_runtime:InferenceRuntime.from_key \
  --loader-arg '"Aratako/Irodori-TTS-500M-v3"' \
  --list-modules
```

## DiT 1-step を切り出す例

`--submodule` と `configs/dit_step_256.yaml` は実モデルのforwardに合わせて修正してください。

```bash
uv run irodori-ax650-prep \
  --loader irodori_tts.inference_runtime:InferenceRuntime.from_key \
  --loader-arg '"Aratako/Irodori-TTS-500M-v3"' \
  --submodule model.dit \
  --input-spec configs/dit_step_256.yaml \
  --out build/dit_step_256.onnx \
  --output-names pred
```

## AX650/Pulsarでの分割方針

推奨順:

1. duration predictor
2. text encoder
3. DiT block 1個
4. DiT denoiser 1-step全体
5. DACVAE decoder

CPUに残すもの:

- tokenizer / text normalization
- sampling schedule生成
- Euler / flow matching update
- wav保存 / watermark / 例外処理

NPUへ渡すもの:

- 固定shape tensor
- できれば fp16 または int8再量子化前提の ONNX
- Triton packed INT4ではなく、Pulsarが解釈可能な通常opグラフ

## 注意

Irodori-TTS-Lite は DiT INT4 packed + Triton fused kernel が主目的です。
AX650/Pulsar ではこのカーネルはそのまま使えないので、`use_fused=False` で patch し、
export時は通常の `nn.Linear` / ONNX op へ戻す設計にしています。

実キャリブレーションデータが重要です。DiTの量子化はランダム入力では品質が落ちやすいので、
最終的には実際のTTS推論中の activation を保存して calibration に使ってください。
