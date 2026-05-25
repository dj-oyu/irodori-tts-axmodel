# 実機検証 結果レポート — 2026-05-26 (overnight autonomous run)

> 実行: AX8850 実機上で autonomous に Phase −1 → Ph0/Ph1 + stage smoke を実行。
> ログ全件は本ディレクトリ (`runs/20260525T163624Z/`)。**サービスは復帰済み（3つとも active）**。
> 正直な達成度: **「verified」ではなく「配線+構造+dtypeゲート通過。実聴とsim≡NPUは未了」**。

## TL;DR（朝イチで読む順）
1. ✅ **構造(Ph0)・DiT/dacvae の NPU 実行・latent_mask liveness・dtype 構成** はすべて取得できた。
2. 🔴 **最重要の発見（要・判断）**: デプロイ DiT `kv_long_lm_cosu16` は **cosu16 処方であって true-A16(allfcu16) ではない**（実測: FC入力 U8、U16化率 0/245）。
   長文くぐもりを直した allfcu16(245FC→U16) の処方が**この版には入っていない**。これが**意図的な選択**か**silent drop**かは実機側だけでは確定不能 → 朝の判断事項。
3. ⚠️ **dacvae_b0 は T=119**、DiT は **T=201**。デプロイ設計の「DACVAE も T_max で1本・T一致必須」を満たす **T=201 dacvae axmodel が未ビルド**。
4. ⛔ **今夜できなかった**: sim≡NPU 等価(Ph2, build host の sim .npy 無し)、fp32 e2e/実聴(2GB torch が**メモリゲート未達**で OOM 回避のため見送り)、dacvae fp32等価(`dacvae_decoder.onnx.data` 外部重み欠落)。

---

## Phase −1: サービス停止/復帰
- 停止: pet-album / axllm / ax-yolo-daemon → CMM **3304MB→276MB**、RAM available 528→774Mi、swap 0→1.9Gi free。NPU プロセス0。
- 復帰: **3つとも active**（`restart.log`）。
- 🐛 **発見した運用バグ**: yolo の `ExecStartPre`(drop-in `vlm-order.conf`) が axllm:8000 を待つため、**復帰は axllm→yolo→pet-album 順**でないと yolo が timeout fail する。`実機テスト計画.md` を修正済み。

## Ph0 構造/IO（`ph0_inspect.log`）— PASS
| model | 入力 | 出力 | 備考 |
|---|---|---|---|
| textenc | input_ids(1,256,i32), mask(1,256,u8) | text_state(1,256,512,f32) | single core |
| DiT kv_long_lm_cosu16 | **53入力**: x_t(1,**201**,32), t, text_mask(1,256), speaker_mask(1,2), **latent_mask(1,201)**, k/v_text/spk ×12層 | v_pred(1,201,32) | latent_mask 生存✅ T=201 |
| dacvae_b0 | z(1,32,**119**) | audio(1,1,228480) | **triple core**, T=119 |

## Ph1 dtype ゲート — 🔴 重要な不一致を検出
**FC 重み**（`ph1_fc_weight_dtypes.log`）: DiT 245 FC すべて量子化（FP32 isolation 0件）。dacvae FC=0（conv ボコーダ、想定通り）。
**活性化 dtype**（`ph1_activation_dtypes.log`, 新規 `verify_activation_dtypes.py`）:
- DiT: 活性化 1209×S16 / 197×U8 / **1×U16(=node_cos入力)**。**FC入力[0]: 242×U8 + 3×S16 → U16化率 0/245 (0%)**。cos→U16 のみ適用。
- dacvae: 全 U8、Snake cos 29 個すべて U8 in/out（§6「DACVAE に cos U16 不要」に合致 ✅）。

**配置元 config 比較**（git HEAD `build/pulsar_configs/`）で確定:
- `kv_long_allfcu16`: S16ops + cos U16 + **245 layer_names→U16** ← 長文くぐもり修正(mel3.68→2.73)の処方
- `kv_long_truea16u16`: 同上だが 190 FC→U16
- `kv_long_cosu16`: S16ops + cos U16 **のみ**（FC→U16 無し）← **デプロイ実物 kv_long_lm_cosu16 が一致**

> 確定事実: デプロイ実物 = **cosu16 処方 + latent_mask**（allfcu16 ではない）。S16-on-FC は heuristic で U8 に降格する（自前 pitfall #7/#8 通り）。
> 推論（要検証）: true-A16 比で長文品質が劣る**可能性**。意図的か silent drop かは実機側だけでは不明。
> 対応案（判断後）: 必要なら latent_mask 入り export で **allfcu16(245FC layer_names→U16)** 再ビルド。

## Stage smoke（メモリ安全・torch不要）
- **DiT NPU smoke**（`dit_smoke.log`, 新規 `dit_smoke.py`）: v_pred finite、**100ms/call**(min, single core)。CFG3×40step≈**~13s/utterance** の概算予算。
  ※ KV キャッシュは zeros で計測（compute は同一なので latency の上限 sanity としては妥当だが、**検証済みデプロイ性能値として引用しないこと**）。
- **latent_mask liveness**: full vs partial(valid=119/201) で max|Δ|=3.12 → **mask は runtime で生きている**（§7 axmodel側 pending を liveness レベルで解消。数値等価は torch 必要で未了）。
- **dacvae NPU smoke**（`dacvae_equiv.log`, 新規 `dacvae_equiv.py`）: z(1,32,119)→audio finite, range±0.81, **189ms**(triple core)。fp32 onnx 等価比較は **`dacvae_decoder.onnx.data` 外部重み欠落で実行不可**。

---

## 残ブロッカー / 次アクション（朝の判断事項）
1. **【判断】DiT を allfcu16 で再ビルドするか**（最重要）。現デプロイは true-A16 ではない。
2. **【判断/ビルド】T=201 dacvae axmodel** を build host で作る（現 b0 は T=119、DiT と長さ不一致）。
3. build host から回収: ① DiT/dacvae の過去 **sim 出力 .npy**（Ph2 等価判定に必須）② `dacvae_decoder.onnx.data`（fp32 等価/実聴の ref）。
4. **メモリ**: 2GB fp32 ref(a_build_cond)は実機 RAM 1.9Gi では OOM 危険 → 短文 or swap 増 or build host で生成。
5. 新規スクリプト（本run で作成、再利用可）: `e2e_demo/ph0_inspect.py`, `e2e_demo/dit_smoke.py`, `e2e_demo/dacvae_equiv.py`, `scripts/verify_activation_dtypes.py`。
