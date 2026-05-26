# 実機検証ガイド — 完全NPU化 条件付け（A1, 2026-05-26）

> 条件付け(Stage A)を NPU 化した **① cond axmodel** を実機検証する。torch / model.safetensors 不要で
> text→wav が全段 NPU で回るかを確認。検証は **実機（sim でない）**、結果は本リポジトリに RESULT.md で push。
> 背景: `TTS_NPU_workflow.md` / `slim_torch_conditioning.md` / `NPU実用化_ギャップ分析.md`(A1)。

## 何ができたか
no_ref 運用では条件付けのうち **テキスト依存は text KV のみ**（speaker KV と CFG text-branch は定数、検証済 bit一致）。
- **① cond axmodel** (`axmodel_cond_textkv`, 158MB): `input_ids[1,256], mask[1,256] → 24 text KV(k_text/v_text 0..11)`。
  = `text_encoder + text_norm + 各block の text KV射影`。fp32 で実 `build_context_kv_cache` と **ビット一致**確認済。
- **定数** (`cond_constants.npz`, 49KB): speaker KV(cond/text/spk branch分) + text-branch zero-text KV + speaker_mask。
- これで条件付けの torch / safetensors が**実行時不要**に（tokenizer のみ使用）。
- ※ duration(A3) はこのビルドに**未含**（t-valid は手動）。完全 safetensos レスには duration の NPU化が次。

## 実機に必要なもの
**git pull で来る**: `e2e_demo/run_npu_full.py`, `e2e_demo/bake_cond_constants.py`, `build/cond_constants.npz`(49KB, force-add), `build/pulsar_configs/cond_textkv.json`, 本doc。
**手動転送（gitignore バイナリ）**: `build/axmodel_cond_textkv/compiled.axmodel`（**158MB**, build host から）。
**既存（実機にあり）**: DiT `axmodel_kv_long_lm_allfcu16_npu3`, dacvae `axmodel_dacvae_b0`/`_T201`, tokenizer。

## 実行（全段 NPU, torch/safetensos 不要）
```bash
sudo -n PYTHONPATH=/home/exe/ai/Irodori-TTS /usr/bin/python3.10 e2e_demo/run_npu_full.py \
  --text "今日はとても良い天気ですね。" \
  --cond build/axmodel_cond_textkv/compiled.axmodel \
  --constants build/cond_constants.npz \
  --dit build/axmodel_kv_long_lm_allfcu16_npu3/compiled.axmodel \
  --dacvae build/axmodel_dacvae_b0/compiled.axmodel --t-valid 119 \
  --num-steps 16 --seed 0 --out-wav /tmp/npu_full.wav
```
（長文は `--dacvae .../axmodel_dacvae_T201 --t-valid 201`。話者は `--seed`。）

## 検証1: 動作（最優先）
- text→wav が **torch を一切起動せず**最後まで通るか（① cond → DiT → dacvae）。
- 実聴: 既存 slim_stageA 経路の wav と**音として同等か**（量子化 cond でも自然か）。
- 起動時間: torch モデルロード(~37s)が消えて cold が短縮されるか（① axmodel load + tokenizer のみ）。

## 検証2: A/B（量子化 cond が品質を壊さないか）★重要
NPU cond と torch fp32 cond で同一 text/seed の wav を比較:
```bash
# A: torch slim cond 経路（リファレンス）
PYTHONPATH=/home/exe/ai/Irodori-TTS python3 e2e_demo/slim_stageA.py \
  --weights <model.safetensors> --text "…" --label ref --out-dir /tmp/ab
sudo -n PYTHONPATH=. /usr/bin/python3.10 e2e_demo/e2e_npu.py --cond /tmp/ab/cond_ref.npz \
  --dit .../allfcu16_npu3/compiled.axmodel --dacvae .../dacvae_b0/compiled.axmodel \
  --t-valid 119 --num-steps 16 --seed 0 --out-wav /tmp/ab/wav_torchcond.wav
# B: NPU cond 経路（本番）
sudo -n PYTHONPATH=/home/exe/ai/Irodori-TTS /usr/bin/python3.10 e2e_demo/run_npu_full.py \
  --text "…" --num-steps 16 --seed 0 --out-wav /tmp/ab/wav_npucond.wav
# 比較: 実聴 + wav SNR/cosine（同一seedなので近いはず）
```
判定: **wav_npucond が wav_torchcond と聴感上ほぼ同じ**なら、条件付けの NPU化(量子化)は成功。
大きく劣化するなら text KV の量子化誤差が DiT に伝播 → ① の量子化レシピ見直し（現状 W8A16 base、cos U16 は text_encoder に node_cos 無く未適用）。

## 報告（GitHub へ）
`runs/<ts>_npu_full/RESULT.md` を作って push してください。記載してほしい点:
1. 全段 NPU で text→wav が通ったか（torch 不起動の確認）。
2. A/B: wav_npucond vs wav_torchcond の実聴差 + SNR/cosine。
3. cold 起動時間（torch ロード消滅の効果）。① axmodel の load 時間 / Model type。
4. もし劣化があれば、どのテキスト/どの程度か（レシピ見直しの材料）。

## 次（このフィードバック次第）
- A/B OK → 実機 safetensos 削除可（duration 未対応なので t-valid 手動運用）。
- duration(A3) を ① に統合 → 完全 safetensos レス + 可変長自動。
- 量子化劣化あり → ① レシピ調整（true-A16 / text_encoder の cos ノード特定して U16）。

---

# A3統合版（duration入り ①）— 2026-05-26 追記

text-KV版 ① の A/B が実機で品質 OK だったので、**duration(A3) を ① に統合**＝完全 safetensos レス + 自動可変長を実現。

## 変わった点
- **① `axmodel_cond_textkv_dur`**(169MB, gitignore=手動転送): 出力が **25個**（24 text KV + **`token_logits[1,256]`**）。
  duration head(token_sum_adarn_zero_no_aux)を統合。no_ref で speaker_vec 定数を bake、softplus/masked-sum は CPU。
  fp32 検証: CPU(token_logits→softplus+masked-sum)=104.989 vs torch duration=104.989（**diff 7.6e-6 = 一致**）。
- **`run_npu_full.py`**: `--t-valid 0`(既定)で **token_logits から t_valid を自動予測**（A3）。dacvae 既定 `T201`、長さ自動。
- これで **duration_predictor も NPU 化 → model.safetensors は実行時完全不要**（tokenizer のみ）。

## 実行（自動可変長, 全段NPU, safetensos不要）
```bash
sudo -n PYTHONPATH=/home/exe/ai/Irodori-TTS /usr/bin/python3.10 e2e_demo/run_npu_full.py \
  --text "今日はとても良い天気ですね。" --num-steps 16 --seed 0 --out-wav /tmp/npu_a3.wav
# 既定で cond=axmodel_cond_textkv_dur, dacvae=T201, t-valid=自動予測。手動上書きは --t-valid N。
```

## 追加検証（実機, GitHub報告）
1. **自動 t_valid が妥当か**: 短文/中文/長文で `[A3] predicted frames=… -> t_valid=…` のログ長さが内容に整合するか。
   特に **長文**: 旧 t-valid=119固定で崩れた長文が、A3自動 t_valid + dacvae_T201 で**自然な長さ**で出るか（step_sweep の「長文=duration問題」が解消するか）。
2. duration の量子化影響: A3自動長 vs 手動最適長(`--t-valid`)で**実聴差**があるか（token_logits は W8A16・per-token なので padded はmask、valid領域の量子化が長さに効くか）。
3. wav が完全 NPU（torch model/safetensos 無し）で通るか・cold 時間。
4. もし長さがズレる → `--duration-scale` で補正可能か、または ① の duration head を true-A16 等で再量子化要か。

報告: `runs/<ts>_npu_a3/RESULT.md` に push。

## 手動転送ファイル更新
- 旧 `axmodel_cond_textkv`(158MB, text KVのみ) → **`axmodel_cond_textkv_dur`(169MB, A3付)** に差し替え推奨。
- `cond_constants.npz` は不変（同じ）。
