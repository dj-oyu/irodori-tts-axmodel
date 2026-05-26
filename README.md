# irodori-tts-axmodel

**v0.3 (2026-05-26)** — 全段NPU・safetensorsレス成立

[Irodori-TTS-500M-v3](https://huggingface.co/Aratako/Irodori-TTS-500M-v3)（Flow Matching ベースの
日本語 TTS）を **Axera AX8850 / M5Stack AI Pyramid Pro** の NPU で動かすための変換・前処理・デプロイツール群。

条件付け(cond)・DiT denoiser・DACVAE vocoder の3コンポーネントを全て `.axmodel` 化し、
**text→wav を全段 NPU で実行**（実行時に torch モデル / model.safetensors 不要、tokenizer のみ）。
AX8850 実機で検証済み。

> ⚠️ 本リポジトリは**変換パイプライン・スクリプト・デプロイ**を管理する。
> 巨大な生成物（`.onnx` / `.axmodel` / calibration / wav / `.venv` / `.npz`）は
> `.gitignore` 済みで**コミットされない**（スクリプトで再生成 or build host から転送）。
> 例外: `build/cond_constants.npz`(52KB), `build/model_introspection.json`, `build/pulsar_configs/` は追跡。

## 現状（2026-05-26）

| 項目 | 状態 |
|---|---|
| DiT 1-step → `.axmodel`（KV cache, allfcu16 true-A16, triple-core ~56ms/call）| ✅ |
| **条件付け(cond①) → `.axmodel`**（text-KV + duration head, W8A16）| ✅ |
| **DACVAE → `.axmodel`**（T=201, b0）| ✅ **NPU化成功**（v0.2 の「Snake op で NPU 不可」を解消）|
| **全段 NPU text→wav**（torch/safetensors レス）| ✅ 実機検証（`runs/*_npu_full`, `*_npu_a3`）|
| few-step sampling（sway + N=16 が内容横断で実用, 2.2s）| ✅ |
| emoji によるスタイル/感情制御（NPUパイプライン全体を通る）| ✅ n=1 |
| 量子化品質（cond 聴感等価, DiT true-A16 で帯域 ~3kHz）| ✅ |
| **duration head の自動長予測** | 🔴 過大予測（モデル欠陥, 要再学習）。手動 t-valid で回避 |
| 常駐サーバ化（per-call 数秒, kokoro 置換）| ⏳ 未（現状 cold ~23s 単発）|

詳細な技術メモ・設計判断・変換ブロッカは
[`irodori_ax650_preprocess/FINDINGS.md`](irodori_ax650_preprocess/FINDINGS.md)（必読）。

### axmodel artifact（`build/`, gitignore 済み・build host から転送）
| 役割 | axmodel | size | 実行先 |
|---|---|---|---|
| ① 条件付け + duration | `axmodel_cond_textkv_dur/` | 162MB | NPU |
| ② DiT 1-step（本命）| `axmodel_kv_long_lm_allfcu16_npu3/` | 329MB | NPU |
| ③ DACVAE decoder | `axmodel_dacvae_T201/` | 87MB | NPU |

## 実機デプロイ（AX8850）

全段 NPU のワンショット text→wav。詳細は [`docs/deploy_ai_pyramid_pro.md`](docs/deploy_ai_pyramid_pro.md)。

```bash
export IRODORI_TTS_HOME=$HOME/github/Irodori-TTS    # irodori_tts のパス（デバイス固有）
deploy/tts.sh "今日はとても良い天気ですね。" -o /tmp/out.wav --play
```

NPU 排他のためのサービス停止→合成→復帰を自動化。話者は `--seed`、step は `--steps`(既定16)。
**既知の制約**: duration head が過大予測する欠陥があり、短文の自動長(`--t-valid 0`)で末尾/冒頭ノイズが
出ることがある（`--t-valid <frames>` 手動指定で確実にクリーン）。本質解決はモデル所有者による
duration head 再学習（`docs/deploy_ai_pyramid_pro.md` §5, `runs/*_npu_a3/RESULT.md`）。

## 構成

```
deploy/
  tts.sh                        # 実機デプロイCLI（サービス停止→全段NPU合成→復帰）
e2e_demo/                       # 実機 e2e スクリプト（aarch64, AX8850 上で実行）
  run_npu_full.py               # ★正準ランナー: ① cond → ② DiT → ③ DACVAE（torch/safetensorsレス）
  bake_cond_constants.py        # cond_constants.npz（speaker/text-uncond KV 定数）を再生成
  slim_stageA.py                # torch fp32 条件付けリファレンス（A/B 比較用, meta+mmap で 360MB）
  dur_fp32_probe.py             # duration head の fp32 突合（量子化 vs モデル欠陥の切り分け）
  step_sweep.py                 # step×CFG×内容長 のsweep
  archive/                      # 旧・診断用ワンオフ（run_npu_full に統合済 or 歴史的）
irodori_ax650_preprocess/       # 変換ホスト側（x86）: ONNX export / calibration / 検証 / 設計メモ
  scripts/                      # export_*.py, dump_calibration.py, validate_full_loop.py 等
  FINDINGS.md                   # 技術メモ（必読）
  QUANTIZATION_NOTES.md         # 量子化方針
  ビルド依頼.md / TTS_NPU_workflow.md / 実機検証_*.md  # build host 依頼・実機手順
docs/
  deploy_ai_pyramid_pro.md      # ★実機デプロイ手順（全段NPU版, v0.3）
  dacvae_npu_research.md        # DACVAE NPU化の調査
build/                          # 生成物（gitignore）。cond_constants.npz / model_introspection.json / pulsar_configs のみ追跡
runs/                           # 実機検証ログ（RESULT.md を追跡, *.log/*.npz は除外）
```

## 前提（変換ホスト側）

- x86_64 + NVIDIA GPU 推奨, Docker, [uv](https://docs.astral.sh/uv/)
- 隣に [`Irodori-TTS`](https://github.com/Aratako/Irodori-TTS) を clone（`uv sync --extra cu128`）
- [Pulsar2](https://huggingface.co/AXERA-TECH/Pulsar2) 6.0 Docker イメージ（`docker load` → `pulsar2:6.0`）

再ビルド手順は [`docs/deploy_ai_pyramid_pro.md`](docs/deploy_ai_pyramid_pro.md) §6 +
[`irodori_ax650_preprocess/FINDINGS.md`](irodori_ax650_preprocess/FINDINGS.md)。

## ライセンス

[MIT](LICENSE)。上流 [Irodori-TTS](https://github.com/Aratako/Irodori-TTS) も MIT で抵触しない
（本リポジトリは Irodori-TTS のコードを再配布せず、依存として import するのみ）。
