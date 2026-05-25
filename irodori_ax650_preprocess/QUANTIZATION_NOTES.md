# Irodori-TTS DiT 量子化 詳細知見（Pulsar2 6.0 / AX8850）

> FINDINGS.md は時系列の作業ログ。本書はそこから抽出した**再利用可能な知見**をトピック別に整理した
> リファレンス。RF-DiT を Pulsar2 で量子化する際の「ハマりどころ」と対処をまとめる。
> 最終更新: 2026-05-24。

---

## 0. TL;DR（現状到達点）

- **plain W8A16 は出荷不可**（full-loop wav **SNR −4.93 dB / corr 0.069**）。原因は2つ:
  ① timestep `cos` の角度が **U8(256段) に潰れる Pulsar2 グラフ artifact**、
  ② 「W8A16」のはずが **194 op が U8 出力**（AdaLN 経路に集中）＝活性化が実質 8bit。
- **判明した最良レシピ** = `kv_b1_w8a16_cosu16.json`（W8A16 + `layer_names:["node_cos"]→U16`）。
  config 1行で wav **−4.93 → −1.96 dB / corr 0.069 → 0.306 / per-step 0.93 → 0.997**。
  **★品質は文長依存（実聴）**: 短文(T=37)は fp32 と**透明**、長文(T=201)は**可聴劣化**（くぐもり/声質低下）。
  SNR/corr は知覚と乖離、**mel_L1 が良 proxy**（短1.33/長3.68）。plain はどの長さも破綻。
  → 短文は出荷可、長文は残差(AdaLN 経路 U8 活性化)の S16 化が要る見込み（§3.1b, §9）。
- 残差を詰めるには ~240 producer op を S16 化（真の A16）が必要 = **精度↔速度のトレードオフ（設計判断）**。
- **per-channel S8 重み量子化は問題ではない**（Lite の W4-group より精密）。真因は活性化精度と感度層の扱い。

---

## 1. ハードウェア / ツール前提

- **AX8850**（AX650 系, Pulsar2 `--target_hardware AX650`）。**18 TOPS@INT8 / 72 TOPS@INT4**。
- Pulsar2 の活性化 dtype: **U8 / S8 / U16 / S16 / FP32**（中間活性化として FP32 は実質非対応, §3）。
  **INT4 なし・group-wise weight quant なし・FP16 層計算なし**（proto で確認済）。
- `weight_data_type` は **Conv 専用**（FullyConnected には無効）。
- `pulsar2 run` = **x86 シミュレータ**（NPU 不要）。compiled.axmodel を Reference/NPUBackend mode で実行でき、
  **実機の数値破綻を x86 で再現できる**（実機 cosine 0.16 を x86 で 0.45 として再現・原因特定した実績）。

---

## 2. Pulsar2 PTQ の挙動 ★最重要（非自明・再利用価値大）

夜通しの実験で判明した、ドキュメントに無い実挙動。**これを知らないと config が黙って効かず時間を溶かす。**

### 2.1 `data_type` は op の「出力」を制御し、「入力」はしない
- layer_config の `data_type` はその op の**出力**量子化を決める。**入力の精度は上流 op の出力で決まる**。
- 例: FC の出力は S16 でも、入力は上流（SiLU/Add 等）が U8 出力なら U8。FC を S16 指定しても入力は U8 のまま。
- **したがって特定 op の入力精度を上げたいときは、その「上流 producer」を狙う**（FC 本体ではなく）。

### 2.2 override が効く指定・効かない指定（実測）
| 指定方法 | 効く? | 備考 |
|---|---|---|
| `layer_names:["node_cos"], data_type:"U16"` | ✅ **効く** | cos 入力 requant が U8→U16 に実際に変化（グラフ検証済） |
| `op_types:["Cos","Sin"], data_type:"FP32"` | ❌ **黙殺** | グラフ無変化＝完全 no-op |
| `op_types:[...], data_type:"S16"`（global） | △ 部分的 | 出力には効くが多数の活性化は U8 のまま |
- **FP32 は中間活性化として非対応**らしく silent に無視される。精度を上げたいときは **U16/S16** を使う。
- **op_types より layer_names が確実**。op_types は global default 程度に考える。

### 2.3 U8 vs S16 は per-tensor の calibration ヒューリスティック
- 同じ「S16 指定」でも Pulsar2 は tensor ごとに U8/S16 を選ぶ（非負・小レンジは U8 に落とす）。
- これは config の無視ではなく**最適化判断**。が、周期関数/大レンジ tensor を U8 に落とすと壊れる（§4.1）。

### 2.4 config は silent に no-op / 部分適用する → **必ずビルド後にグラフ検証**
- 名前マッピングは黙って外れる（例: 148 指定 → 実適用 146、cond/timestep が脱落して 12 分を無駄に）。
- **対策**: ビルド後 `quant_axmodel.onnx` を読み、intended と実 dtype を突き合わせ **fail-loud**。
  → `scripts/verify_layer_dtypes.py`（FullyConnected の重み dtype を shape 別に集計し intended と assert）。
- 「bit 単位で前回と同一の結果」は **config が no-op だった**サインと疑え（wav 結果が plain と一致 → 配線/設定ミス）。

### 2.5 precision_analysis の正しい読み方
- `precision_analysis_method`:
  - **EndToEnd**（NPUBackend mode）= **累積**。各 op は上流の量子化誤差を全て継承。
    → 深層 op が cosine≈0 でも「その op が壊れている」のではなく**累積の帰結**。
    「1186中971 op が <0.99」は累積アーティファクトで、80% が個別に壊れている意味ではない。
  - **PerLayer** = 各層を **float 入力で単独**評価。累積を見ないので楽観的（旧記録の「W8A8 v_pred 0.9997」は
    これで、実 end-to-end は 0.45。**誤解の元**）。
- **真の seed を見るには EndToEnd 表の「早期 op（累積が支配する前）」を読む**。
  本件で cos 0.63 / cat 0.79 / silu 0.85 / cond-linear 0.94 が早期 seed として浮上した。

### 2.6 主な knob
`calibration_method`(MinMax/Percentile/MSE/KL), `layer_configs`(layer_names | op_types + data_type + weight_data_type),
`precision_analysis` + `_method`(EndToEnd/PerLayer) + `_mode`(NPUBackend), `enable_smooth_quant`, `enable_adaround`,
`transformer_opt_level`(0-2), `device`("cuda:0" で calib 大幅高速化), `onnx_opt.enable_onnxsim`。

---

## 3. 評価の落とし穴（測り方）★ここを外すと結論を誤る

### 3.1 per-step cosine は品質 proxy にならない
RF（rectified flow）は **8 step 反復 × CFG**。各 step の v_pred 誤差が累積増幅される。実測:
| per-step cosine | full-loop wav SNR |
|---|---|
| 0.93（plain W8A16） | **−4.93 dB**（破綻） |
| 0.997（cosu16） | **−1.96 dB**（まだ不可） |
- per-step 0.997 でも 8 step × 12 block で増幅され wav は出荷不可。**per-step cosine の改善＝wav 改善とは限らない**。

### 3.1b ★ SNR/corr も知覚品質の proxy として弱い（実聴で判明, 2026-05-24）
cosU16 の短文 wav は **SNR −2.65 dB / corr 0.137** と数値は悪いが、**実聴では fp32 とそん色ない高品質**
（声質・聞き取りやすさとも良好, ユーザ確認）。一方 plain W8A16 は −4.93dB/corr0.069 で**実聴でも破綻**。
- = SNR/corr のわずかな差（−4.9→−2.6）が知覚では「破綻→透明」の天地の差。**サンプル単位 SNR は
  RF sampling の微小な時間/位相シフトで大きく落ちるが、声質・明瞭度は保たれる**。
- **mel_L1 の方が知覚に近い**（plain 4.41 → cosU16 1.33, 3.3倍改善）。
- 教訓: **最終判定は実聴**。SNR/corr は「破綻検出」には使えるが「出荷可否」の閾値には使えない。
  per-step cosine < SNR/corr < mel_L1 < 実聴 の順で知覚相関が上がる。
- → **cos 修正版(cosU16)の品質は文長依存**（実聴, 2026-05-24）:
  実聴3点（文長 vs 品質, 全て cosU16 8step）:
  | 長さ | T | mel_L1 | 実聴 |
  |---|---|---|---|
  | 短 (ありがとう。) | 37 | 1.33 | fp32 と**透明** |
  | 中 (seed11, B) | 110 | 2.16 | **くぐもり=電話品質**・明瞭・内容OK |
  | 長 (8秒) | 201 | 3.68 | くぐもり＋**声質低下**＋一部句崩れ |
  - 劣化モードは一貫して**「くぐもり＝高域(HF)欠落」**で、文長とともに単調に悪化（mel_L1 と一致）。
    機序: 活性化量子化(U8/S16)が HF を平滑化し、長い潜在系列ほど attention/RF 反復で累積。
  - **mel_L1 が知覚をよく予測（1.33→2.16→3.68）。SNR/corr は不可**（短文の −2.65dB が透明だった通り）。
  - 透明なのは概ね T≲40 の短文のみ。T≳110 で電話品質、T≳200 で声質も低下。
  - 推定原因: 長い T ほど activation 量子化(U8/S16)誤差が attention/RF 反復で累積 → HF が平滑化されくぐもる。
  - → 短文は出荷可・長文は要改善。改善レバー = overnight 特定の **AdaLN 経路 ~240 U8 op を S16 化（真の A16）**
    で長文の累積を抑える（精度↔速度トレードオフ）。中文(T=110)の実聴で閾値を見る。

### 3.2 full-loop wav は必要だが十分でない arbiter
- 指標: **SNR(dB) / 相関 corr / mel_L1**（fp32 torch 出力との比較）。
- 実装: `$CLAUDE_JOB_DIR/full_loop_axmodel.py`。fp32 torch を ref に、各 step の denoiser を
  compiled.axmodel（`pulsar2 run` x86 sim）で差し替えてフル sampling。**~74 分/本**（律速）。
- **教訓**: per-step cosine や PerLayer 表で「いける」と判断せず、必ず wav で決着させる。
- **ハーネス検証**: AXMODEL_DIR は env で渡す＋起動時に `[config]` print と `assert compiled.axmodel` ガード
  （過去にパスをハードコードして別モデルを 75 分測る事故 → 結果が plain と bit 一致して発覚）。

### 3.3 高価な wav の前に precision_analysis seed で gate する
- ビルド後に早期 seed の cosine を読み、**改善していなければ wav(74分) を回さない**。
  例: MSE calib は seed が悪化 → wav skip で 74 分節約。cosu16 は cos 0.63→1.0 → wav 価値が立った。

---

## 4. DiT 固有の失敗モードと真因

### 4.1 ★決定的: timestep `cos` の角度が U8 に潰れる（Pulsar2 グラフ artifact）
- timestep 正弦波埋め込みで `sin` と `cos` は**同一の角度 tensor `mul_2`(S16, = timestep × 周波数バンド)** を読む。
- `sin` は S16 のまま読む → cosine **0.9999**（無傷）。
- `cos` は手前に **`AxRequantizeLinear` が挿入され `mul_2` を S16→U8(256段) に落として**から読む
  → 大レンジの角度が 256 段に潰れ、周期関数 cos 通過後に巨大誤差 → cosine **0.6315**。
- **= 同一 tensor を cos 側だけ U8 に requant する Pulsar2 の非対称アーティファクト**。
  周期関数の原理的問題ではなく per-tensor 量子化割当のバグ。**sin が無傷なのが動かぬ証拠**。
- **下流の cat / silu / cond-linear が壊れていたのは、入力の cos が壊れていたせい**（それ自体は無傷）。
- **修正**: `layer_names:["node_cos"], data_type:"U16"` → 入力 requant が U16 になり cos 0.63→**1.0**、
  cat/silu/cond-linear も連動して ~1.0 に回復。

### 4.2 「W8A16」のはずが活性化はほぼ U8（AdaLN 経路に集中）
- cosu16 グラフの全 op 出力: **S16=1183 / U8=194 / FP32=4 / U16=1**（大半 S16, 弱点は U8 の 194 op）。
- FC 入力 dtype: **U8=242 / S16=3**（§2.1 のとおり入力は上流由来）。
- U8 を FC に供給する producer: **SiLU 72 + Add 72 + FC 72 + Mul 25 ≈ 242 op、AdaLN 変調経路に集中**。
- これが cos 修正後の残差 −1.96 dB の主因。真の A16 にするには ~240 producer を S16 化 = 設計判断（§9）。

### 4.3 RF 反復 × depth の誤差増幅
- 12 DiffusionBlock × 8 step。per-step の僅かな摂動が壊滅増幅。
- だから **block drop / block merge / 任意の小摂動はすべて失敗**（fine-tune 無しでは −0.5〜−2 dB）。
  学習無しで効くのは **per-step ほぼ無損失の近似のみ**（= 真因のピンポイント修正 or mixed-precision or QAT）。

### 4.4 感度層 vs 頑健層
- **頑健（量子化しても 0.999）**: AdaLN(LowRankAdaLN rank192) の重み matmul、bulk attention(1280,1280)、SwiGLU MLP(3680)。
- **感度（壊れやすい）**: timestep 埋め込み経路（cos/cat/silu/cond-linear）、活性化精度全般。
- ※ 注意: 「AdaLN 重みが感度」という拡散 PTQ の通説に引きずられ AdaLN 重みを FP32 化したが、本件では**逆効果**
  （AdaLN 重みは元々無傷、FP32 境界の requant 雑音で悪化、しかも FP32 は黙殺）。**通説でなく precision_analysis を見る**。

---

## 5. 試した手法の結果一覧

| 手法 | 結果 | 教訓 |
|---|---|---|
| W8A8（既定） | end-to-end cosine **0.45**（出力 std が半減） | transformer に A8 は不可 |
| W8A16 plain | per-step 0.93 / **wav −4.93 dB** | cos U8 バグ + U8 活性化で破綻 |
| calib MinMax→**MSE** | seed −0.04〜−0.07 悪化 | range 問題でない。calib では直らない |
| Cos/Sin op_types→FP32 | **silent 無視**（no-op） | FP32/op_types は効かない |
| mixed-precision: AdaLN 重み→FP32 | per-step 0.93→**0.66 悪化** | 直す層を間違え（cond 脱落）+FP32 境界雑音+FP32黙殺。inverted test |
| block drop / merge | −0.5〜−2 dB | RF 増幅。fine-tune 必須 |
| **`node_cos`→U16** | **wav −4.93→−1.96 dB / per-step 0.997** | ★真因ピンポイント修正。採用 |
| 真の A16（~240 op→S16） | **未実施**（設計判断待ち） | 精度↔速度トレードオフ |

---

## 6. Lite(OneCompression W4) 比較と重み再構成誤差

コミュニティ版 `irodori-tts-lite` は W4 で「音質ほぼ劣化なし」。そのレシピ解析と重み誤差の実測から真因を確定:

### Lite のレシピ（`kizuna-intelligence/Irodori-TTS-500M-v3-int4` メタデータ）
- **AutoGPTQ-v1, 4-bit, group-wise(gs32), GPTQ**。回転も fine-tune も無し（base 重みを PTQ するだけ）。
- **4bit 量子化**: 全 block の attention(wq/wk/wv/wo/gate/wk_text/…) と mlp(w1/w2/w3)。
- **fp16 で残す（感度高）**: LowRankAdaLN 投影 / cond_module(timestep) / in_proj・out_proj / norms / duration_predictor。

### 重み再構成の相対 Frobenius 誤差（実測）
| bulk Linear | per-channel **S8**(Pulsar2) | group-wise **W4**(Lite naive) | Lite **GPTQ**(実) |
|---|---|---|---|
| 代表 relerr | **~1%** | ~8% | ~13% |
- **Pulsar2 の per-channel S8 は Lite の W4-group より ~8倍 精密**（256 vs 16 段）。GPTQ は weight 誤差を上げてでも
  output 誤差を下げるので 13% でも音質維持。
- **結論: group-wise の欠如は無関係。bulk 重み量子化（per-channel S8）は問題でない。**
- **真因は活性化精度（A8/U8）と感度層の扱い**。Lite が効くのは fp16 感度層 + A16 + 良 calib（GPTQ）だから。
  我々の plain は感度経路（cos）を U8 に落とし、活性化も実質 A8 だった。

---

## 7. 診断プレイブック（再利用手順）

1. **precision_analysis を EndToEnd で出す**（`precision_analysis_method=EndToEnd, _mode=NPUBackend`）。
2. **表の早期 op（累積前）を cosine 昇順で読む** → 真の seed を特定（深層 linear の 0.00x は無視＝累積）。
3. seed op を `quant_axmodel.onnx` で**上流トレース**（各 tensor の dtype を見る）→ **不要な dtype 降格 / requant** を探す。
4. **`layer_names:[seed_op], data_type:"U16"/"S16"`** で修正（FP32 は使わない）。
5. **ビルド後グラフ検証**: 当該 tensor dtype が実際に上がったか、intended と一致するか assert（silent no-op を弾く）。
6. seed cosine が改善したら **full-loop wav** で決着（per-step では判断しない）。改善しなければ wav を回さない。
7. config は 1 変数ずつ。結果が前回と bit 一致 → no-op を疑う。

---

## 8. 効くレシピ / config スニペット

```jsonc
// kv_b1_w8a16_cosu16.json の要点（採用済・wav −1.96dB）
"quant": {
  "calibration_method": "MinMax",        // MSE/Percentile は本件では効果なし
  "device": "cuda:0",                    // calib を大幅高速化（build ~15分）
  "layer_configs": [
    { "op_types": ["MatMul","Gemm","Add","Mul","Softmax","Cos","Sin", /*…*/],
      "data_type": "S16", "weight_data_type": "S8" },   // global W8A16
    { "layer_names": ["node_cos"], "data_type": "U16" } // ★cos 角度の U8 潰れを修正
  ],
  "precision_analysis": true,
  "precision_analysis_method": "EndToEnd",
  "precision_analysis_mode": "NPUBackend"
}
```
- ビルド: `pulsar2 build --target_hardware AX650 --input dit_step_kv_mask1e4_fp32.onnx --config <上記>`（GPU calib で ~15分）。
- **真の A16 を狙う場合の次案**（未検証・設計判断）: §4.2 の producer（SiLU/Add/Mul/FC ~240本）を
  `layer_names` で S16 化。サイズ/latency 増。`scripts/verify_layer_dtypes.py` で適用検証必須。

---

## 9. 残課題と判断待ち事項

- **DiT を出荷品質で NPU 化するには残差 −1.96dB を詰める必要**。選択肢:
  1. **真の A16**（AdaLN 経路 ~240 op を S16 化）→ wav と **速度コスト**を実測。速度重視なら割に合わない可能性。
  2. **DiT は fp32/CPU 据え置き**、cos 修正版は中間成果として保持、NPU は **DACVAE / TextEncoder** に集中。
  3. **QAT / 蒸留**（A8 速度で品質回復・最終手段・重い）。
- いずれも **wav が唯一の arbiter**。precision_analysis で gate してから 74 分 wav を回すこと。

## 関連ファイル
- `build/pulsar_configs/kv_b1_w8a16_cosu16.json`（採用レシピ）, `kv_b1_w8a16.json`(plain), `kv_b1_w8a16_mse.json`(棄却)
- `build/axmodel_kv_b1_cosu16/`（採用 axmodel + EndToEnd 表）, `build/fullloop_sim_cosu16/`（wav 結果）
- `scripts/verify_layer_dtypes.py`（ビルド後 dtype 検証ゲート）
- `$CLAUDE_JOB_DIR/full_loop_axmodel.py`（wav arbiter）
- FINDINGS.md（時系列ログ・全経緯）

## DACVAE の NPU 量子化検証（2026-05-24, 実 calib + cos→U16）
DACVAE デコーダ（Snake cos 書換版, z(1,32,119)→audio(1,1,228480)）を実 latent calib で量子化し x86 sim:
| variant | 29 cos 入力 | SNR | mel_L1 | corr |
|---|---|---|---|---|
| B0 (fix無) | U8 | 5.96dB | 1.95 | 0.877 |
| B1 (cos→U16) | U16 | 5.98dB | 1.94 | 0.878 |

- **★ cos→U16 は DACVAE では無効（B0≈B1）**。DiT で劇的に効いた同じ修正が効かない。
  理由: DiT の cos 角度は timestep×高周波=**大レンジ**で U8 潰れが致命的だったが、DACVAE の Snake `cos(2αx)` は
  活性化 x の**小レンジ**なので U8 で十分。**構造類似(cos×29)でも角度スケールが違えば同じ罠にならない**。
  教訓: ある層の修正が別モデルに転用できるとは限らない。**値の大きさを実測してから**。
- **DACVAE は DiT より素直に量子化できる**: SNR **正(+5.96dB)** / corr 0.88（DiT は負）。単一パス=RF増幅が無い恩恵。
- **実 calib は有効**: 過去のランダム calib(SNR4.8) → 実 calib(5.96)。
- 残差(mel_L1 1.94≈電話品質)は cos でなく Conv/アップサンプリング由来と推定（precision_analysis は Snake-path の
  AxQuantizedSub が uint16 overflow で停止＝別途の課題）。要・実聴で出荷可否判定。
- ビルド注意: `precision_analysis:true` だと `node_sub_15` で `integer 114239 does not fit uint16_t` エラーでビルド停止
  → DACVAE は analysis off で建てる。sim は重い（max_cycle 164M, x86 で1本>10分。実機NPUなら高速）。

### DACVAE 実聴判定（2026-05-25, ユーザ確認）
- `dacvae_b0_nofix_quant.wav`(W8A8+実calib): **「少しホワイトノイズが載るが音声は良好」**＝**NPU 化実用レベル**。
- DiT の長文（くぐもり/声質低下）より明確に良い。単一パス=RF増幅なしの恩恵が実音でも確認できた。
- 残るアーティファクト = 広帯域のホワイトノイズ（W8A8 の量子化雑音）。気になるなら W8A16/選択 S16 で低減余地ありだが、
  現状で「良好」判定なので**そのまま出荷も可**。cos→U16 は不要（§DACVAE検証）。

## true-A16（task10）暫定: partial では効果なし・bulk S16 適用が不安定（2026-05-25）
medB cosu16 に「U8出力 ~188 op → S16」を layer_names で指定しビルド:
- **適用が不完全**: all-op U8 出力 194→186（8減のみ）, FC入力 S16 3→88（85変化）。~188 指定の大半が黙って未適用。
- wav: SNR −2.38→−2.15, **mel_L1 2.16→2.14（≒変化なし）**。partial S16 では muffling 不変。
- → true-A16 の是非は**未決**（partial すぎて結論不可）。bulk activation を確実に S16 化する手段が要るが、
  layer_names 一括 S16 は mixedp と同様 silent drop する。Pulsar2 に「U8 を禁止し全 activation S16」global 設定が
  あるか要調査。なければ clean な true-A16 検証は困難。

## true-A16（task10）結論: activation 精度が muffling のレバー（2026-05-25）
long(T=201) で partial U16（FC入力 87/242 を U16 化）を sim:
| long | SNR | mel_L1 | corr |
|---|---|---|---|
| cosU16 | −2.36 | 3.68 | 0.083 |
| true-A16(U16) partial | −1.85 | **3.32** | 0.187 |
- mel_L1 3.68→3.32（~10%改善, partial coverage で）→ **activation 精度＝muffling のレバー**で確定。
- **★ U16 は heuristic を生き残る / S16 は U8 に降格される**（U16指定で 240 op が U16 化, S16指定では ~17 のみ）。
- **per-op layer_names は U8 op に当たらない**: 抽出名の多くが別ビルドでは既S16 op にマッチ（240→U16）し、
  肝心の U8 op は素通り（U8 194→187）。名前ベースの bulk 指定は不安定で full coverage 不可。
- **full coverage の本命 = global op_types の活性化 dtype を S16→U16 に変更**（heuristic 降格を回避し全活性化16bit化）。要検証。

### true-A16 FULL coverage 結論（2026-05-25, task10 完了）
真因: **Pulsar2 は各 FC の手前に U16→U8 requant を挿入**（producer 出力が U16 でも FC 入力は U8 に落とす）。
→ **全 FC を layer_names で U16 指定**すると各 FC 入力 requant が U16 になり **245/245 full coverage**。
long(T=201) dose-response（FC入力 U16 化率 vs 品質）:
| FC U16 | SNR | mel_L1 | corr |
|---|---|---|---|
| 0/245 (cosU16) | −2.36 | 3.68 | 0.083 |
| 87/245 (partial) | −1.85 | 3.32 | 0.187 |
| **245/245 (full)** | **−0.61** | **2.73** | **0.379** |
- mel_L1 単調改善 3.68→2.73（−26%）= **activation 精度が muffling のレバーで確定**。
- 長文は「くぐもり＋声質低下」→「電話品質(≈中文)」に改善。ただし**完全透明(短文1.33)には届かず**。
- レシピ: `cosU16 + layer_names=[全FC]→U16`（= 真の W8A16, 全 activation 16bit）。size 340→344MB（+4MB と僅少だが
  U16 活性化は実機で帯域/レイテンシ増。速度コストは実機計測要）。
- **config 教訓**: FC 入力精度を上げるには **FC 名を直接指定**（producer を指定しても FC 前 requant が U8 に戻す）。

### true-A16 long の実聴（2026-05-25, ユーザ確認）
- `long_truea16_FULL_quant.wav`: **こもりは改善**（true-A16 が muffling を実際に消すと実音で確認）。
  ただし**「水中でブクブク話すような」軽いノイズ**が残る。
- = muffling(HF欠落) → 解消、だが別の軽いアーティファクト（bubbling）が顕在化。
- 推定: 残差は **S8 重み量子化**（Pulsar2 では FC 重みは S8 固定・FP32 は Conv 専用＝**重み精度を上げられないハード上限**）
  と非FC U8 活性化(Mul/Add)由来。**activation を U16 にしても重みが S8 なのが PTQ の天井**。
- → 長文の完全透明は PTQ では頭打ち。さらに上は **QAT/蒸留**（重み込みで学習）か CPU 据置。
  進捗: 長文は plain「破綻」→ cosU16「muffled+声質低下」→ true-A16「muffled解消＋軽いbubbling」と単調改善。

## AdaRound（task12）: 本ホスト/モデルでは不調 → PTQ 天井確定（2026-05-25）
S8 重み量子化の床（long の bubbling）を AdaRound で押そうとしたが2連敗:
- long(T=201)+true-A16+AdaRound, calib32: **~4h で OOM**（rss 8.9GB > 15GBホスト枠）。
- medB(T=110)+true-A16+AdaRound, calib16: **~2h で NaN scale**（`v_pred_DequantizeLinear x_scale=nan`）。
  calib16 が少なすぎて scale 統計が退化したのが主因と推定。
- 教訓: AdaRound はメモリ大・遅い・calib に敏感。本15GBホストでは long は OOM、calib削ると NaN。
  calib32 で medB 再試行の余地はあるが（OOM 危険）、得られる bubbling 改善は小さい見込み。
- **結論**: PTQ の天井（S8 FC 重み）は本環境では AdaRound で越えられない。
  長文の完全品質が要るなら **QAT/蒸留**（重み込み学習）か **長文 fp32-CPU** 据置。
  現 true-A16（mel 2.73, muffling解消＋軽bubbling）が PTQ 実用上限。
