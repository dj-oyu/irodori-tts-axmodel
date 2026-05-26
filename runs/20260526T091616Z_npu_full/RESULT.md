# 完全NPU化 条件付け(A1) 実機検証結果 — 2026-05-26 (AX8850)

`run_npu_full.py`（① cond axmodel + baked constants → DiT → dacvae, torchモデル/safetensors不要）。
手順: `実機検証_完全NPU化_条件付け.md`。文「今日はとても良い天気ですね。」(6tok), N=16, seed=0, dacvae_b0(t119)。

## 判定: ✅ Path A 成立（① cond axmodel は品質的に使える）

### 1. 全段NPUで text→wav 通った（torchモデル不起動）
- ① cond axmodel: `input_ids[1,256],mask → 24 text-KV`。load+run **1.22s**、Model type **0 (single core)**。
- DiT(allfcu16_npu3) 2.2s(N=16) → dacvae_b0 0.17s → wav 4.76s。**torch model/safetensors ロード無し**。
- **cold 23s**（slim torch経路の37sから短縮。内訳: tokenizer初回 ~13s + ① load/run 1.2s + DiT/dacvae load）。tokenizerはtorch lib importするが2GBモデルは読まない。
- ⚠️ run_npu_full.py に dtype バグ（`input_ids` を int64 で feed → axmodel は **int32** 要求）。実機で int32 に修正して通した（要 build host 反映, 下記）。

### 2. A/B（量子化 cond が品質を壊すか）★ → 壊さない
| 比較 | 結果 |
|---|---|
| **実聴** wav_npucond vs wav_torchcond | **同じ品質**（差を感じない）= ① cond量子化は聴感劣化なし |
| wav SNR/cos | SNR 0.6dB / cos 0.55（同seed/DiT/dacvae） |
| cond text-KV cos (full tensor) | 0.16（誤導値） |
| cond text-KV cos (**valid 6トークン領域のみ**) | **0.77** |
| NPU cond値域 | std 1.36, **±32.37で飽和** vs fp32 std0.167/range -19〜+8 |

**数値と実聴の食い違いの正体**: full-tensor cos(0.16)は **masked-padded領域(位置6〜256)の±32飽和ゴミ**が支配。
DiTは `text_mask` でpadded領域を捨てるので**出力に効かない**。valid領域 cos 0.77 + masking で**聴感等価**。
→ **教訓: cond KVの品質は full-tensor cos でなく valid領域 or 実聴で判定**。padded±32飽和は無害だがレシピ的に不潔。

### 3. cold起動
torch 2GBモデルロード(~37s, slim)→**消滅**。完全NPUで **~23s**（主に tokenizer初回 + axmodel load群）。D2常駐化すれば tokenizer/axmodel も1回償却 → per-call は DiT 2.2s + dacvae 0.17s + ①1.2s ≈ 数秒。

### 4. 別件（cond量子化ではない・直交）: 冒頭アーティファクト
**A/B 両経路とも**、音声**冒頭に短い非日本語の無関係音**。共通パス(DiT/dacvae/speaker/duration)由来。
候補: onset/BOS / no_ref の masked-mean speaker トークン / 過充填(6tok→119frame の先頭余白)。要調査だが Path A 判定とは独立。

## build host への反映依頼
1. **`run_npu_full.py` の `input_ids` を int32 に**（axmodelが int32 要求。実機で修正済→PR化）。
2. ① cond axmodel: **padded(masked)領域の ±32 飽和**を調査（calib に現実的な mask 領域が入っていたか）。無害だが、cos-U16 未適用(W8A16 base)＝valid領域も 0.77 止まり。**true-A16化で valid領域の忠実度↑余地**（DiTのR3同様）。
3. 冒頭アーティファクトの切り分け（onset/speaker-token/duration）。
4. 次: **A3(duration) を ① に統合** → 完全 safetensors レス + 可変長自動。

## 成果物
- `e2e_demo/run_npu_full.py`（int32修正済）, `build/cond_constants.npz`, `build/axmodel_cond_textkv/`(158MB, 手動転送)。
- wav: `/tmp/ab/wav_{npucond,torchcond}.wav`（device一時）。
- 関連: `実機検証_完全NPU化_条件付け.md`, `slim_torch_conditioning.md`, `TTS_NPU_workflow.md`, `runs/…_stepsweep`(N=16)。
