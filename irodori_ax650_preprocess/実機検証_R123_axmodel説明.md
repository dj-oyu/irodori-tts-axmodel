# 実機検証ガイド — R1/R2/R3 新規 axmodel（2026-05-26 転送分）

> build host で生成し AX8850（AI Pyramid Pro）へ転送した3 axmodel の説明と検証手順。
> 起点: `NPU実用化_ギャップ分析.md` + `runs/20260525T170818Z_e2e/RESULT.md`（電話品質=cosu16 の HF 欠落）。
> 詳細な納品報告: `ビルド依頼.md` 末尾「充足ステータス」。`*.axmodel` は .gitignore のためリポジトリには無い（手動転送済）。

## 転送した3ファイル

| ファイル（実機の置き先） | サイズ | 役割 | 由来 |
|---|---|---|---|
| `build/axmodel_kv_long_lm_allfcu16/compiled.axmodel` | ~328M | **#1 品質(R3)**: true-A16 DiT | lm onnx(53入力,latent_mask) × `kv_long_lm_allfcu16.json`(245 FC→U16). **NPU1** |
| `build/axmodel_kv_long_lm_allfcu16_npu3/compiled.axmodel` | ~329M | **#3 RT候補(R1)**: triple-core DiT | #1 と同一 quant + `npu_mode:NPU3` |
| `build/axmodel_dacvae_T201/compiled.axmodel` | ~87M | **#2 連結(R2)**: T=201 dacvae | `dacvae_decoder_201.onnx`(z=[1,32,201]) × `dacvae_T201_gpu.json` |

### 各ファイルの中身（事実）
- **#1 `axmodel_kv_long_lm_allfcu16`**: 現 deploy 中の `kv_long_lm_cosu16`（電話品質）の**品質置換版**。違いは量子化処方のみ — cosu16(FC入力 U16=0/245) → **true-A16(FC入力 U16=245/245)**。build host で `verify_activation_dtypes.py` により FC入力 U16 **245/245 (100%)** を確認済。入出力形状・latent_mask(53入力)は cosu16 と同一なので、`e2e_npu.py` で `--dit` を差し替えるだけで動く。
- **#3 `axmodel_kv_long_lm_allfcu16_npu3`**: #1 と**量子化は完全同一**（FC U16 245/245）。違いは `npu_mode: NPU3`（triple-core 指定）のみ。⚠️ **build が NPU3 config を受理しただけで、実機で本当に 3コア（Model type:2）として走るかは未確認**。ロード時のログで `Model type` を要確認。
- **#2 `axmodel_dacvae_T201`**: 現 `dacvae_b0`（T=119, 4.76s）の T=201（8.0s）版。DiT の T_max=201 と一致するので、DiT 出力 z をトリムせず直結できる。⚠️ calib は T=119 実データを時間軸タイルで T=201 化した**合成 calib**（MinMax レンジ用途）。末尾領域の品質は実聴で要確認。

## 検証手順（`e2e_demo/e2e_npu.py`）

cond（torch Stage A 出力 `/tmp/e2e_cond.npz`）は既存のものを流用。`sudo -n PYTHONPATH=... /usr/bin/python3.10` で root 実行（axengine が /dev/mem を要求）。

### #1 品質テスト（最優先＝最終判定は実聴）
DiT だけ差し替え、dacvae は既存 b0 + trim(`--t-valid 119`) のまま。電話品質(HF欠落)が true-A16 で消えるか実聴。
```bash
sudo -n PYTHONPATH=. /usr/bin/python3.10 e2e_demo/e2e_npu.py \
  --dit build/axmodel_kv_long_lm_allfcu16/compiled.axmodel \
  --dacvae build/axmodel_dacvae_b0/compiled.axmodel --t-valid 119 \
  --out-wav /tmp/q_allfcu16.wav
```
**見るべき点**: 帯域制限・くぐもりが取れるか。比較対象は既存 `kv_long_lm_cosu16`(電話品質)の wav。
過去 sim では mel_L1 3.68→2.73（くぐもり解消＋軽い bubbling）。bubbling（水中音）は S8 重み量子化の床＝PTQ 天井で残る想定。

### #3 RT テスト（triple-core 化の可否＝RT の本丸）
```bash
sudo -n PYTHONPATH=. /usr/bin/python3.10 e2e_demo/e2e_npu.py \
  --dit build/axmodel_kv_long_lm_allfcu16_npu3/compiled.axmodel \
  --dacvae build/axmodel_dacvae_b0/compiled.axmodel --t-valid 119 \
  --out-wav /tmp/q_npu3.wav
```
**見るべき点**: ① ロードログの `Model type` が **2 (triple core)** か（dacvae と同じ）。1 のままなら NPU3 は効いていない。② per-call ms が現 **103.9ms(single)** から下がるか（R1 目標 **<40ms/call**）。音質は #1 と同一のはず。

### #2 連結テスト（T=201 直結＝trim 不要化）
```bash
sudo -n PYTHONPATH=. /usr/bin/python3.10 e2e_demo/e2e_npu.py \
  --dit build/axmodel_kv_long_lm_allfcu16_npu3/compiled.axmodel \
  --dacvae build/axmodel_dacvae_T201/compiled.axmodel --t-valid 201 \
  --out-wav /tmp/q_t201.wav
```
**見るべき点**: z=(1,32,201) が dacvae T=201 にそのまま入り、audio=(1,1,385920)=8.0s が出るか。trim せず末尾まで自然か（合成 calib の末尾品質確認）。

## 注意・限界（過剰評価しないため）
- #1/#2 は **build 受入基準充足**まで。最終品質は**実聴**で判定（指標信頼度: per-step cosine < SNR < mel_L1 < 実聴）。
- #3 は **config 受理のみ**。triple-core 化は実機ロードで初めて判明。
- **fp32 リファレンスは未送付**: 実機 `build/dacvae_decoder.onnx` は重み `.onnx.data`(261MB) 欠落で動かない。C2(sim≡NPU 等価, ギャップ分析「致命」)に着手する段で別途 `.data` を送る。
- NPU 排他: TTS 実行中は VLM(axllm)/yolo 停止必須。復帰順 axllm→yolo→pet-album。

## 報告してほしいこと（build host 側の次アクション判断材料）
1. #1 の実聴: 電話品質は改善したか / bubbling の程度。
2. #3: `Model type` の値と ms/call。
3. #2: T=201 直結が成立したか / 末尾の不自然さの有無。
