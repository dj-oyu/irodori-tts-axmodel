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

## 推奨 / 次
1. **当面は手動 `--t-valid`**（step_sweep知見: 内容長に合わせる。plain~60/sibilant~110/long200）。A3自動は未だ本番不可。
2. ✅ **fp32突合 完了 → 量子化でなく duration head のモデル挙動**（`dur_fp32_probe.py`）。**duration head の再ビルドは不要**。次は **モデル側**: (a) 学習時の推論config確認（has_speaker=True/参照話者付きで予測が改善するか＝no-speaker経路が弱いか）、(b) BOSトークンのframe扱い（先頭pause設計か）、(c) ダメなら token数ベースのヒューリスティック or 手動t-valid運用。
3. shape fix（問題1）は汎用に有用＝反映推奨。
4. **冒頭アーティファクト**（A1の宿題）も BOS の大frame由来かを切り分け（plain BOS=48.7frame≈2s が先頭に何を生成しているか）。

## 成果物
- wav: `/tmp/a3_{plain,sibilant,long}.wav`（device一時）。
- `e2e_demo/run_npu_full.py`（dacvae shape fix 済）。
- 関連: `runs/…_npu_full`(A1), `…_stepsweep`(内容長×duration), `実機検証_完全NPU化_条件付け.md`。
