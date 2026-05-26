# Irodori-TTS を AI Pyramid Pro (AX8850) で動かす

**v0.3 (2026-05-26)** — 全段NPU・safetensorsレス版

Irodori-TTS-500M-v3 を AX8850 NPU で **text→wav 全段 NPU** で動かすデプロイ手順。
条件付け(cond)・DiT・DACVAE を全て `.axmodel` 化済みで、実行時に **torch モデル / model.safetensors は不要**
（tokenizer のみ torch lib を import）。実機（AX8850）で検証済み。

> **v0.2 からの変更**: v0.2 は「DACVAE は NPU 不可・CPU 据え置き」「条件付けは torch CPU」という
> 別アーキだった。現在は **条件付け(cond①) も DACVAE も NPU 化済み**で全段 NPU が成立している
> （`runs/20260526T091616Z_npu_full/`, `runs/20260526T093950Z_npu_a3/` で実機検証）。
> 変換経緯・全ブロッカは [`../irodori_ax650_preprocess/FINDINGS.md`](../irodori_ax650_preprocess/FINDINGS.md)。

---

## 1. 全体像 — 全段 NPU の3コンポーネント

```
text ─[CPU] 正規化 + tokenizer
        │
        ├─[NPU] ① cond axmodel ──────────► 24 text-KV + token_logits(A3 duration)
        │        (text_encoder + 各block KV射影 + duration head)
        │        speaker-KV / text-uncond-KV は baked 定数(cond_constants.npz)
        ▼
   noise z_T ─[CPU] RF Euler ループ(N=16, sway, 独立CFG 3分岐)
        │  各 step:
        └─[NPU] ② DiT 1-step (allfcu16_npu3, triple-core)  ★支配的
        ▼
   latent z_0 ─[NPU] ③ DACVAE decoder(T201) ─[CPU] trim ─► wav(48kHz)
```

| 段 | 実行先 | axmodel | 備考 |
|---|---|---|---|
| 正規化 / tokenizer | CPU | — | torch lib import（2GBモデルは読まない）|
| **① 条件付け + duration** | **NPU** | `axmodel_cond_textkv_dur` | `input_ids,mask → 24 text-KV + token_logits` |
| **② DiT 1-step (×N)** | **NPU** | `axmodel_kv_long_lm_allfcu16_npu3` | 支配的。triple-core ~56ms/call |
| RF Euler 更新 / schedule / CFG / softplus duration | CPU | — | 制御ロジック（軽い）|
| **③ DACVAE decoder** | **NPU** | `axmodel_dacvae_T201` | T=201 固定→ audio を t_valid に trim |
| wav 保存 | CPU | — | |

- **no_ref 専用**（参照音声 voice clone ではなく、話者は `--seed` で振る）。
- CFG は **independent モード3分岐**（cond / text-uncond / speaker-uncond, cfg_text=3 / cfg_spk=5）。
  **guidance=1（cond-only）は使用不可＝ノイズに崩壊する**（[CFG required メモ](../e2e_demo/run_npu_full.py) 参照）。

---

## 2. 必要なもの（実機側）

- **AI Pyramid Pro (AX8850)** — octa Cortex-A55 + 24 TOPS NPU、8GB LPDDR4x（system ~2GB / CMM 6GB）。
- **PyAXEngine**（axengine 0.1.3）。root 実行のため、PYTHONPATH で user の site-packages を明示する必要あり。
- `irodori_tts` **パッケージ**（~384KB, tokenizer/text正規化/ModelConfig のため）+ aarch64 torch（CPU）。
  - **`model.safetensors`(~2GB) は実行時不要** — 全段NPUは重みを一切ロードしない（検証済）。
    ストレージ削減のため削除してよい（A/B検証 `dur_fp32_probe.py`/`slim_stageA.py` を回す時だけ要再DL）。
  - tokenizer は別repo `llm-jp/llm-jp-3-150m`（HFキャッシュ ~6MB、tokenizerファイルのみ）。`IRODORI_TTS_HOME` は
    この `irodori_tts` パッケージを指す（PYTHONPATH）。重みのパスではない。
- 依存: torch 2.10.0+cpu / onnxruntime 1.23.2 / onnx 1.21.0 / axengine 0.1.3。

参考: [AI Pyramid-Pro (m5-docs)](https://docs.m5stack.com/en/ai_hardware/AI_Pyramid-Pro) /
[PyAXEngine](https://github.com/AXERA-TECH/pyaxengine)

---

## 3. 持ち込むもの

**git pull で来る**（このリポジトリ）:
- `deploy/tts.sh`（デプロイCLI）、`e2e_demo/run_npu_full.py`（ランナー）、`e2e_demo/bake_cond_constants.py`
- `build/cond_constants.npz`（52KB, force-add）、`build/model_introspection.json`、`build/pulsar_configs/*.json`

**手動転送（gitignore済バイナリ, build host から）**:
| axmodel | size | 役割 |
|---|---|---|
| `build/axmodel_cond_textkv_dur/compiled.axmodel` | 162MB | ① 条件付け + duration head |
| `build/axmodel_kv_long_lm_allfcu16_npu3/compiled.axmodel` | 329MB | ② DiT（triple-core, true-A16）|
| `build/axmodel_dacvae_T201/compiled.axmodel` | 87MB | ③ DACVAE（T=201）|

転送例:
```bash
scp -r buildhost:.../build/axmodel_cond_textkv_dur          build/
scp -r buildhost:.../build/axmodel_kv_long_lm_allfcu16_npu3 build/
scp -r buildhost:.../build/axmodel_dacvae_T201              build/
```
（`cond_constants.npz` は git 管理。再生成は `e2e_demo/bake_cond_constants.py`。）

---

## 4. デプロイCLI — `deploy/tts.sh`

ワンショット text→wav。NPU 排他のためのサービス停止→合成→サービス復帰を自動化する
（cold ~23s。**常駐ではない**＝kokoro-tts 置き換え用途ではなく単発合成）。

```bash
export IRODORI_TTS_HOME=$HOME/github/Irodori-TTS    # irodori_tts のパス（必須・デバイス固有）
deploy/tts.sh "今日はとても良い天気ですね。" -o /tmp/out.wav --play
deploy/tts.sh "テキスト" --seed 3 --steps 16 --t-valid 70
deploy/tts.sh "テキスト" -o - | aplay               # wav を stdout に流して直接パイプ
```

| 引数 / 環境変数 | 既定 | 意味 |
|---|---|---|
| 位置引数 | — | 合成テキスト（必須）|
| `-o, --out` | `/tmp/tts.wav` | 出力 wav。**`-o -` で wav を stdout へ**（`\| aplay` 等にパイプ）|
| `--seed` | 0 | 話者（no_ref では seed が声色・性別を決める）|
| `--steps` | 16 | RF step 数（16=2.2s 推奨。32=最高品質）|
| `--t-valid` | 0 | 0=A3自動 / >0=手動フレーム数(25fps)。**下記5の制約参照** |
| `--duration-scale` | 1.0 | A3予測フレームの倍率 |
| `--play` | off | 合成後 `aplay` で再生 |
| `--keep-services` | off | サービスを停止しない（NPU が空いている前提）|
| `IRODORI_TTS_HOME` | （必須）| `irodori_tts` の場所 → PYTHONPATH |
| `IRODORI_PYSITE` | `$HOME/.local/lib/python3.10/site-packages` | axengine の場所（root から見えないため明示）|
| `IRODORI_AXLLM_UNIT` | 実行中インスタンスを動的解決 | axllm は templated unit `axllm-serve@<model>.service`。モデル名はデバイス固有 |
| `IRODORI_SERVICES` | `pet-album ax-yolo-daemon <axllm>` | 停止する NPU 排他サービス |
| `IRODORI_RESTART` | `<axllm> ax-yolo-daemon pet-album` | 復帰順（yolo の ExecStartPre が axllm:8000 待ち→axllm先）|

- root 実行（axengine が /dev/mem）には NOPASSWD sudo が要る（このデバイスは設定済）。
- サービス復帰は `trap EXIT` で**異常終了時も必ず実行**される。
- **タイムアウトは付けていない**: 初回の tokenizer init が cold/swap 競合で数十秒〜数分かかることがある。

---

## 5. 既知の制約（正直版）

### 🔴 duration head の予測ミスキャリブレーション（最重要）
`--t-valid 0`（A3自動）は **duration head が過大予測する既知欠陥**を持つ。モデルは余剰フレームを
**無音でなくゴミ音声で埋める**ため、予測−自然長の余剰量に応じて段階的に劣化する:

| 余剰フレーム | 症状 |
|---|---|
| ~5–10 | クリーン |
| ~15 | 末尾ノイズ（後方伸長）|
| ~30 | 冒頭に音声ゴミ「のおー」（前方伸長）|
| ~70 | 全文 2 回再生 |

- 短文は 1.2–1.9× 過大予測。モデルは ±10fr に過敏なので **単一 `--duration-scale` では直らない**
  （誤差が文により変動）。
- **on-device の robust な自動 duration は存在しない**。気になる場合は `--t-valid <frames>`（25fps、
  例: 2.4s→60）を手動指定すると確実にクリーン。
- **これは量子化ではなくモデル(duration head)の欠陥**（実機 fp32 突合で確定, `e2e_demo/dur_fp32_probe.py`）。
  **本質解決 = duration head の再学習/校正（モデル所有者作業）**。`token_sum_adarn_zero_no_aux` head、
  特に no-speaker(null_speaker) 経路。詳細: `runs/20260526T093950Z_npu_a3/RESULT.md`。

### 🟡 cold ~23s・常駐でない
毎回 tokenizer init + axmodel ロードが走る。kokoro-tts 同等の per-call 数秒運用には
**常駐サーバ化（axmodel を RAM 保持 + axllm 排他ハンドシェイク）が必要**（未実装）。

### 🟡 長さ上限 T_max=201（~8s）
それ以上はチャンク分割が必要（未実装）。

### 参考（健全な部分）
- 条件付けの量子化は**聴感等価**（full-tensor cosine 0.16 は masked-padded ±32 飽和由来の誤導値。
  valid 領域 cos 0.77 + 実聴で等価）。判定は valid 領域 or 実聴で行う。
- step は **N=16 が内容横断で実用**（長文崩壊は step でなく t-valid 不一致が原因だった）。

---

## 6. 別 shape / 別レシピでの再ビルド（ホスト側, x86 + Pulsar2）

実機ではなく**変換ホスト**で行う。配布した Pulsar2 build config は `build/pulsar_configs/`:
- `cond_textkv.json`（①）、`kv_long_lm_allfcu16_npu3.json`（②）、`dacvae_T201.json`（③）。

変換スクリプト一式・経緯は `irodori_ax650_preprocess/`（`scripts/`, `FINDINGS.md`,
`QUANTIZATION_NOTES.md`, `ビルド依頼.md`）を参照。

---

## 7. トラブルシュート

| 症状 | 原因 / 対処 |
|---|---|
| `No module named 'axengine'` | root から user site が見えない → `IRODORI_PYSITE` を確認 |
| `sudo: a password is required` | `sudo -n` が NOPASSWD にマッチせず。`env` を挟むと不可、`sudo -n PYTHONPATH=... python` 形 |
| `missing: build/axmodel_*` | gitignore バイナリ未転送（§3）|
| SEGV | NPU 排他違反（2プロセス同時）。`--keep-services` を外してサービス停止させる |
| 短文で末尾/冒頭ノイズ | duration head 欠陥（§5）。`--t-valid` 手動指定 |
| 音が出ない | `amixer sset 'DAC VOLUME' 55%`、`aplay -D plughw:0,0`（内蔵 ES8311 = card0）|
