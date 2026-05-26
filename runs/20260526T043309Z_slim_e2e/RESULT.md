# Slim-torch conditioning e2e 結果 — 2026-05-26 (AX8850)

slim Stage A（`e2e_demo/slim_stageA.py`, meta-device + no-copy mmap）の e2e 検証。
build host 共有用。手法・背景: `irodori_ax650_preprocess/slim_torch_conditioning.md`。

## 構成
- 文: 「今日はとても楽しいです。」(tok=5), seed=0, CFG 既定(text=3/spk=5), no_ref speaker。
- DiT: `axmodel_kv_long_lm_allfcu16_npu3`(true-A16, triple-core) / dacvae: `axmodel_dacvae_b0`(T=119, trim t-valid=119)。
- 比較対象: フルモデル cond（`emoji_stageA`相当, `/tmp/emoji_cond/cond_plain_tanoshii.npz`）。

## 1. 正当性 — フルモデルと bitwise 一致 ✅
| 段 | 結果 |
|---|---|
| slim Stage A cond（3 CFGブランチ） | **146/146 key が np.array_equal でフルモデルと一致** |
| slim cond → DiT → dacvae → wav | latent `cmp` **IDENTICAL**、wav md5 **`d8ad527b…` 完全一致**（フルcond経路と同一） |

→ slim版は数値的に**完全等価**な drop-in。fp32品質を一切落とさない（同一ビット）。

## 2. フットプリント（全て main RAM, torch-CPU, CMM不使用）
| 計測点 | VmRSS | anon(swap競合) | file(再利用可) |
|---|---|---|---|
| import torch | 216MB | 140MB | 76MB |
| meta構築+材質化（**起動時**） | 286MB | 192MB | 95MB |
| encode+KV（**稼働時**） | **942MB** | **~333–360MB** | **~609MB** |

- 材質化 = **281 param / 703MB**（text_encoder 323 + speaker_encoder 231 + block-ctxKV 150 + norm）。残り1.2GB(DiT本体)は meta=メモリ0。
- file 609MB は safetensors の no-copy mmap＝再利用可ページキャッシュ（swapを食わない）。
- **テキスト長非依存**: tok=5 と tok=29 で稼働RSS同一（重み支配, activations無視可）。
- **リークなし**: 5回追加encodeで anon ~333→361MB（微増, fileは安定）。
- 対比: 旧 `a_build_cond`（フル丸読み+`.copy_`）= ~2GB anon。→ **swap競合 ~2GB → ~360MB（~5–6倍削減）**。

## 3. レイテンシ（実測, cold 1発）
| 段階 | 時間 |
|---|---|
| Stage A（slim, cold: import+mmap-fault+tokenizer+3branch encode） | **37.7s** |
| NPU e2e（axmodelロード~5.4s + DiT 4.3s + dacvae 0.18s + write） | **9.9s** |
| **cold 合計** | **47.5s** |
| 内 DiT sampling（32step CFG, npu3 triple-core, 76call） | 4319ms |
| 内 dacvae(b0, T119) | 166–175ms |

- cold 47.5s のうち **~43s は import+モデルロードの1回限り費用**。**計算実体は ~4.5s**(DiT+dacvae)。
- **常駐サーバ(D2)化で warm per-call ~6s 見込み**（DiT 4.3 + dacvae 0.2 + warm encode ~1–2s）→ 音声4.76sに対し RTF ~1.2–1.3。warm encode は未実測（要・常駐PoC）。

## 4. 結論
- slim Stage A は **bitwise完全等価・fp32品質維持**で、swap競合フットプリントを **~360MB** に圧縮。
- mem=2048M・排他なら swap無し常駐可。axllm 同時稼働も射程内（anon360+axllm~1G+OS~0.5≈1.86G ≈ 1.9Gi）。
- 残課題（直交, gap分析参照）: 常駐サーバ化(D2)で warm latency 実測 / hybrid(textenc→NPU)で更に~380MBへ / 長文T=201品質 / 声紋契約 / A3 duration。
- スクリプト: `e2e_demo/slim_stageA.py`, `e2e_demo/slim_cond_probe.py`（footprint+bitwise probe）。
