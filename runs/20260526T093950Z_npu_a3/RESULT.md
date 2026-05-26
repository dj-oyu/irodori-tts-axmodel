# A3統合版（duration入り ① cond axmodel）実機検証 — 2026-05-26 (AX8850)

`axmodel_cond_textkv_dur`(169MB, A3 duration head 付) + `run_npu_full.py`（`--t-valid 0`=自動）。
手順: `実機検証_完全NPU化_条件付け.md`「A3統合版」。N=16, seed=0, dacvae_T201, 3文。

## 判定: ⚠️ 全段safetensorsレス成立、but A3 duration予測は要校正

### ✅ できたこと
- **完全 safetensors レス**で text→wav が通る（① cond+duration head → DiT → dacvae、torchモデル無し、tokenizerのみ）。duration_predictor も NPU化。
- ① cond axmodel: 24 text-KV **＋ token_logits(A3)** を出力、load+run ~1.7s。

### 🔴 問題1: run_npu_full の dacvae shape mismatch（修正済）
`run_npu_full` は z を t_valid にトリムして dacvae へ渡すが、dacvae は**固定shape**(T201)。
A3自動 t_valid(=134/80/200 等)が 201 でないと **AssertionError**（`z expect [1,32,201] got [1,32,134]`）。
**修正**: latent を dacvae の T(=201)に合わせて投入し、**audio側を t_valid にトリム**（max-T+mask+trim、デプロイ設計§2/§3準拠）。`run_npu_full.py` で対応済（要 build host 反映）。

### 🔴 問題2: A3 duration 予測が miscalibrated（実聴で確定）
formula: `pred_frames = Σ_valid softplus(token_logits)`、25 frame/s（hop1920@48k）。

| 文 | tok | 予測frames | frames/tok | 実聴 | 誤り |
|---|---|---|---|---|---|
| plain | 5 | 133.6 (5.36s) | 26.7 | **同一文が2回再生** | **過大**（理想~60）→繰返しで埋める |
| sibilant | 11 | 79.5 (3.20s) | 7.2 | かなり不明瞭(rushed) | **過小**（理想~110）→詰め込み崩れ |
| long | 29 | 362→clamp200 (8.0s) | 12.5 | **うまくいっている** | clamp偶然命中（生362は過大） |

- **誤差が両方向**（plain過大・sibilant過小）→ **単一 `--duration-scale` では直らない**＝per-token miscalibration。
- per-token frames が 26.7/7.2/12.5 と乱高下（日本語は~0.1-0.2s/token=2.5-5frame が自然）→ 全体に過大寄り、文で不安定。
- long が良かったのは raw362 が clamp(max 200)で偶然妥当長に落ちただけ（35tok等ならclampで過小化し崩れる想定）。

### 原因究明 = **量子化ではなく duration head のモデル挙動**（実機 fp32突合で確定, `dur_fp32_probe.py`）
on-device で fp32 duration_predictor(meta-slim) を回し、NPU① の token_logits と per-token 突合:

| 文 | fp32 sum | NPU sum | fp32 BOS(tok1) |
|---|---|---|---|
| plain | 123.8 | 133.6 | 48.7 |
| sibilant | 81.5 | 79.5 | 5.3 |
| long | 334.3 | 362 | 35.0 |

- **fp32 ≈ NPU**（量子化誤差は小: 123.8→133.6 等）→ **量子化は原因でない。duration head 自体が over/under予測**。**rebuild不要**。
- formula(softplus+masked-sum)も model の forward と一致＝正しい。
- token1=**BOS**(id=1)が大きな可変frame（plain48.7≈2s/long35、但しsibilant5.3）→「BOS=先頭pause」説は部分的（一貫せず）。**A1検証の冒頭アーティファクトもBOS由来の可能性**（両経路がBOSを通る）。
- 誤差が両方向（plain過大124 vs理想~60 / sibilant過小81 vs理想~110）＝**単一scaleで直らない**。**no_ref / no-speaker 推論configで duration predictor が不正確**（has_speaker=False のadarn_zero経路が弱い可能性）。

### 究明の決着 = D1（モデルは正常、duration head だけが欠陥）
plain「今日はとても楽しいです。」を手動 t_valid でスイープ（実聴）:

| t_valid | 秒 | 実聴 |
|---|---|---|
| 60 | 2.40 | **ちょうどいい（クリーン）** ✅ |
| 70 | 2.80 | **ちょうどいい（クリーン）** ✅ |
| 80 | 3.20 | 「今日は」後に不自然なタメ |
| 110 | 4.40 | 2回言う（繰返し） |

→ **モデルは t_valid≈60-70（自然な内容長）でクリーンにレンダリングできる**。defect は **duration head の予測値**（134≈理想65の2倍）だけ。
- BOS-drop（→92, 1.4×過大）も不可: 「破綻/phaserエフェクト」。モデルは**過剰割当に過敏**（+15fr→不自然なタメ、+45fr→繰返し）→ headは±数frで当てる必要があるのに2×外す。
- has_speaker=True でも plain 93（1.4×過大）→ 参照話者でも直らない。

### 短文パターン掃引 → 統一理論（余剰フレーム = 先頭/末尾アーティファクト）
8短文の予測frames(BOS除)を予測長で生成し実聴。症状が**余剰フレーム量(予測−自然長)で段階的**:

| 余剰fr | 症状 | 例（実聴） |
|---|---|---|
| ~5–10 | **クリーン** | おはよう / また明日 / 元気ですか / こんにちは / ありがとう |
| ~15 | 末尾ノイズ（後方伸長） | 今日はいい天気だ (51fr) |
| ~30 | 冒頭に音声ゴミ「のおー」（前方伸長） | お腹が空きました (64fr) |
| ~70 | 全文2回再生 | 今日はとても楽しいです (134fr) |

- モデルは**余剰フレームを無音でなくゴミ音声で埋める**（±10frで過敏）。
- **A1の「冒頭アーティファクト」= この余剰フレームの先頭描画と同一根本原因**（t-valid過大→先頭ゴミ）。→ **duration を ±10fr精度で正確化すれば、durationズレ も 冒頭アーティファクト も同時に消える（1修正で2問題）**。
- 短文over-predictionは方向一貫（常に過大）だが量1.2–1.9×でバラつき＋±10fr過敏 → scale/heuristicでは精度に届かず robust化しない。
- BOS frame=41.1 は量子化飽和（fp32 48.7）＝二次要因。

## 推奨 / 次（確定）
1. **robust な自動durationの本筋 = duration head の再学習/校正（モデル所有者）**。`token_sum_adarn_zero_no_aux` head、特に **no-speaker(null_speaker)経路**が短文2倍過大・中文過小（両方向誤差）。**量子化・config・frame-rate・BOS・has_speaker は全て切り分け済＝原因でない**。
2. **オンデバイスのrobust自動修正は存在しない**: 誤差両方向ゆえ単一scale/BOS-drop/ヒューリスティックで両立不可。手動t-valid(plain60-70)は効くが運用不安定。
3. **健全な部分**（モデル本体レンダリング・cond NPU(A1)・DiT・dacvae・safetensorsレス全段）はそのまま使える。defect は duration head に**完全に限局**。
4. shape fix（問題1, max-T+mask+trim audio）と BOS-exclude フラグ（`--keep-bos-frames`）は `run_npu_full` に反映済。
5. 冒頭アーティファクト（A1宿題）: BOS-drop でも残存（plain破綻に紛れ未確定）。duration修正後に再確認。

## 成果物（追加）
- `e2e_demo/dur_fp32_probe.py`（fp32 duration突合）、`run_npu_full.py`（shape fix + `--keep-bos-frames`）。
- 長さスイープ wav: `/tmp/a3_len{60,70,80,110}_plain.wav`（device一時）。

## 成果物
- wav: `/tmp/a3_{plain,sibilant,long}.wav`（device一時）。
- `e2e_demo/run_npu_full.py`（dacvae shape fix 済）。
- 関連: `runs/…_npu_full`(A1), `…_stepsweep`(内容長×duration), `実機検証_完全NPU化_条件付け.md`。
