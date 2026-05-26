# 実機検証結果 — R1/R2/R3 新規 axmodel（2026-05-26, AX8850）

build host 転送の3 axmodel を実機検証。手順は `irodori_ax650_preprocess/実機検証_R123_axmodel説明.md`。
cond は本セッションの3ブランチCFG cond `/tmp/emoji_cond/cond_plain_tanoshii.npz`（文「今日はとても楽しいです。」, seed=0, CFG既定 text=3/spk=5）を流用（torch再ロード不要）。全テスト同一cond+同一seed → 差分は量子化/コア/dacvaeのみ。

## サマリ（build host 報告3項目）

| 項目 | 結果 | 判定 |
|---|---|---|
| **#1 品質 (R3 allfcu16)** | 95%ロールオフ **1007Hz→3000Hz**（3倍帯域拡張）、centroid 474→588Hz。**実聴: bubbling無し（元から無し）、AMラジオ風(帯域制限)ノイズが少し低減** | ✅ 改善（定量＋実聴で一致。~3kHz頭打ち=PTQ天井のため「少し」） |
| **#2 連結 (R2 dacvae T=201)** | z=(1,32,201) 直結 → audio=(1,1,385920)=**8.04s**、trim不要、末尾=実聴クリーン | ✅ 成立 |
| **#3 RT (R1 triple-core)** | DiTロードログ **Model type: 2 (triple core)**、**56ms/call**（single 104-143ms比） | ✅ triple-core稼働（ただし目標<40msは未達） |

## 詳細

### #1 品質（最重要・実聴＋スペクトル）
電話くぐもり = HFロールオフ。実聴の逐次A/Bは記憶保持が難しく判定困難だったため、wavのスペクトルで定量化（同一文/seed、量子化のみ差）:

| file | centroid | 95%roll | 0-1k | 1-4k | 4-8k | 8k+ |
|---|---|---|---|---|---|---|
| cosu16 (旧deploy) | 474Hz | **1007Hz** | 95.0% | 2.0% | 2.4% | 0.6% |
| allfcu16 (新R3) | 588Hz | **3000Hz** | 92.3% | 3.5% | 3.2% | 1.0% |
| npu3 (同quant) | 602Hz | **3704Hz** | 92.3% | 3.0% | 4.1% | 0.7% |

- cosu16 は ~1kHz 以下にエネルギー集中（極端な電話帯域）。allfcu16 で 95%ロールオフが3kHzへ＝**中高域を回復**。sim の mel_L1 3.68→2.73（改善・完全ではない）と整合。
- bubbling（水中音, S8重み量子化の床）= **ユーザー実聴で無し（「元からそれらしいノイズはなかった」, allfcu16を3ループ@62%で集中試聴）**。build host想定の残留bubblingは可聴閾値下＝非問題。
- 加えて「AMラジオ風ノイズが少し低減」= 帯域制限の実聴改善がロールオフ1k→3kHzの定量と一致（「少し」＝3kHz頭打ちのため）。
- 限界: 3kHz頭打ち。full帯域には fp32 リファレンス比較（C2, .onnx.data 未送付）が必要。

### #2 連結（T=201 直結）
- `--dacvae axmodel_dacvae_T201 --t-valid 201`: z=(1,32,201) をトリムせず投入 → audio=(1,1,**385920**)=8.04s。finite=True。dacvae 323ms（T119の166ms比, ~1.7x長で妥当）。
- 末尾（短文を201フレーム生成した余剰領域）= 実聴クリーン（合成calib末尾品質OK）。**trim hack 解消**。

### #3 RT（triple-core）
DiTロード時 `[INFO] Model type: 2 (triple core)`（cosu16/allfcu16 single は `0 (single core)`）。per-call = total/76calls（22 CFGstep×3 + 10×1）:

| DiT | total | ms/call | core |
|---|---|---|---|
| cosu16 | 7989ms | 105.1 | single (type0) |
| allfcu16 | 10859ms | 142.9 | single (type0) ← true-A16 U16活性化は重い |
| **allfcu16_npu3** | **4276ms** | **56.3** | **triple (type2)** |

- triple-core は実機で稼働確認。allfcu16-single 比 2.5x高速、cosu16-single 比 1.85x。
- ただし R1目標 <40ms/call は未達（56ms）。発話あたり DiT 4.3s（32step CFG）。
- npu3 と allfcu16(single) は同一quantのはずだがスペクトル/std微差（triple-coreの並列リダクション数値差と推定）。両者とも cosu16 >> で帯域拡張は一致。

## 次アクション候補
- emoji A/B を allfcu16 で再実施（本日 cosu16 で成立済の emoji制御が true-A16 でどう聞こえるか）。
- C2（sim≡NPU等価）: build host から `dacvae_decoder.onnx.data`(261MB) 受領で fp32リファレンス比較。
- R1 <40ms 追求 or 56msで実用判断。
