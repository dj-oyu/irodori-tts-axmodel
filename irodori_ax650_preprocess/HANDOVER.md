# HANDOVER — Irodori-TTS → AX8850 量子化プロジェクト

> 引き継ぎ用エントリポイント（2026-05-26）。詳細は各 doc を参照。新規参加者はまず本書 →
> `テスト観点.md`（検証手順）→ `QUANTIZATION_NOTES.md`（知見）の順で読む。

## このプロジェクトは何か
日本語 TTS **Irodori-TTS-500M-v3**（rectified-flow DiT + DACVAE ボコーダ）を **AX8850 NPU** 向けに
**Pulsar2 6.0** で量子化・axmodel 化する。主戦場は「量子化で音質を保つ」こと。

## ドキュメント地図（全て本ディレクトリ）
| doc | 内容 |
|---|---|
| `HANDOVER.md`（本書） | 現状・成果物・次タスクの索引 |
| `テスト観点.md` | 検証手順・指標信頼度・落とし穴チェックリスト |
| `QUANTIZATION_NOTES.md` | 再利用可能な量子化知見（Pulsar2 挙動・レシピ） |
| `デプロイ設計.md` | 実機デプロイ設計（2バリアント×可変長） |
| `量子化トラブル解説.md` | 前提知識ゼロ向けの解説読み物 + Q&A |
| `FINDINGS.md` | 時系列の全作業ログ |

## 現状（PTQ 実用上限まで到達）
- **DiT**: `cosU16`（短文=fp32 透明） / `true-A16`=全FC入力U16（長文 mel_L1 3.68→2.73, くぐもり解消＋軽い bubbling）。
  残 bubbling = **S8 重み量子化の床 = PTQ 天井**（Pulsar2 は FC 重みを上げられない）。AdaRound は本環境で OOM/NaN で不可。
- **DACVAE**: W8A8 + 実 calib で実用レベル（少しホワイトノイズ・音声良好）。
- **可変長**: `latent_mask` を export 入力化（`export_dit_step_kvcache.py --latent-mask`）→ axmodel ビルド済（53入力）。
  fp32 で「max-T+mask+trim → 有効領域が直接生成と cosine 1.0」を検証済。

## ビルド済 NPU axmodel（実機に置く3点・各 compiled.axmodel 単一ファイル）
| 役割 | パス | サイズ |
|---|---|---|
| text encoder | `build/axmodel_textenc/` | 230MB |
| DiT（可変長） | `build/axmodel_kv_long_lm_cosu16/` | 325MB |
| DACVAE | `build/axmodel_dacvae_b0/` | 84MB |
（DiT は用途で選択: 長文品質=`axmodel_kv_long_allfcu16`(true-A16), 短文=`axmodel_kv_short_cosu16` 等）

## 次タスク（優先順）
1. ★**実機ランタイム移植（最大の穴・未着手）**: tokenize → textenc(axmodel) → `build_context_kv_cache`(KV射影, host)
   → duration predictor → RF サンプリングループ(DiT axmodel ×8 + latent_mask 構築) → DACVAE(axmodel) → 末尾 trim → wav。
   現状は x86 の PyTorch `inference_runtime` のみ。実機は **axengine 呼び出しの軽量オーケストレータ**が必要。
2. compiled-axmodel の latent_mask masking をフル loop で検証（~111分, harness に latent_mask 注入）。
3. **実機 latency 計測**（true-A16 の U16 活性化コスト）→ 量子化バリアントを true-A16/cosU16 のどちらにするか確定。
4. DACVAE のバケツ/可変長対応（T ごとに別 DACVAE or 可変長 export）。
5. （長文の完全品質が要件なら）QAT/蒸留 or 長文 fp32-CPU 据置。

## 環境 / 実行
- **ビルド**: `docker run --rm --gpus all -v "$PWD:/data" pulsar2:6.0 -c "pulsar2 build --target_hardware AX650 --input <onnx> --output_dir <dir> --config <json>"`（`device:cuda:0` で calib 高速化, ~15分。precision_analysis ON で +~9分）。
- **synth/calib**: Irodori env →`cd /home/exe/ai/Irodori-TTS && PYTHONPATH=. uv run python ...`。
- **export/検証**: preprocess env →`uv run --project irodori_ax650_preprocess python ...`（ただし irodori_tts を import する export は `PYTHONPATH=/home/exe/ai/Irodori-TTS` も付ける）。
- **wav arbiter**: `scripts/full_loop_axmodel.py`（env: `AXMODEL_DIR / SYNTH_TEXT / SYNTH_SEED / SIM_ROOT`）。1本 74〜111分。

## 鉄則（詳細は テスト観点.md）
- 指標信頼度: **per-step cosine < SNR/corr < mel_L1 < 実聴**。最終判定は必ず実聴。
- config は **silent に drop/部分適用**する → ビルド後に必ず `quant_axmodel.onnx` で dtype 検証（`scripts/verify_layer_dtypes.py`）。
- 活性化精度を上げるなら **U16**（S16 は heuristic で U8 に降格）。FC 入力を上げるには **FC 名を直接** layer_names 指定。
- Pulsar2 ビルドは **no-checkpoint**（途中 kill = 全ロス）。本ホスト 15GB で **長T + U16 + AdaRound は OOM**。
- 別モデルを測る事故防止に `full_loop` は env 経由 + `[config]` ガード（結果が別ビルドと bit 一致 → routing ミス）。
