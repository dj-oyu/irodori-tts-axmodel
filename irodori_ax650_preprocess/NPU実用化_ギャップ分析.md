# NPU実用TTS ギャップ分析 — AX8850 (AI Pyramid Pro)

> 実機検証 2026-05-26 で確定した事実・制約・**実用的なNPU TTSに足りないピース**の記録。
> 詳細結果: `runs/20260525T163624Z/REPORT.md` / 再ビルド依頼: `ビルド依頼.md` / 実行計画: `実機テスト計画.md`。
> ゴール定義: **AX8850 上で実用レベルの TTS**（実テキスト→自然なwav、レイテンシ許容範囲、プロダクトに組込可能）。

---

## 1. 現状サマリ（実機で確定した事実）
- 各段は **NPU 上で SEGV せず動く**: textenc / DiT(kv_long_lm_cosu16) / dacvae_b0 をロード・実行確認。
- 形状: textenc(input_ids→text_state) / DiT 53入力(latent_mask生存)→v_pred(1,201,32) / dacvae z(1,32,119)→audio(1,1,228480, 48kHz/4.76s)。
- 速度: DiT **100ms/call single-core** / dacvae **189ms triple-core**。
- latent_mask は **runtime で生きている**（full vs partial で出力変化）。
- 量子化: DiT 245FC 全量子化だが **FC入力 U16化率 0/245**（= cosu16 処方, true-A16 ではない）。dacvae 全U8(§6通り)。

## 2. 環境制約（恒久的）
| 制約 | 内容 | 影響 |
|---|---|---|
| **NPU 排他** | 2プロセス同時で SEGV。VLM(axllm)+yolo で既に2つ | TTS実行中は VLM/yolo 停止必須 |
| **System RAM ~1.9Gi** | CMM(5GB)とは別枠。2GB torch ロードがギリギリ | fp32条件付けが OOM 危険 |
| **pulsar2 sim 無し** | x86 docker は build host 側のみ | sim≡NPU 等価を実機で確認不可（要 build host 成果物） |
| **root 必要** | axengine は /dev/mem アクセス | `sudo -n PYTHONPATH=... /usr/bin/python3.10`（NOPASSWD設定済） |
| **サービス復帰順** | yolo は axllm:8000 待ち | 復帰は axllm→yolo→pet-album |

---

## 3. 実用NPU TTSに足りないピース（カテゴリ別）

凡例: 🟥動作不能の壁 / 🟧速度の壁 / 🟨品質の壁 / 🟦運用の壁。〔→Rx〕は `ビルド依頼.md` 対応番号。

### A. パイプラインの完全性（そもそも実テキスト→wavが繋がるか）
| # | 足りないピース | 状態 | 区分 |
|---|---|---|---|
| A1 | **条件付けの実装が NPU化されていない** | text→KVキャッシュ は **全部 torch CPU**。textenc axmodel は存在するが**未使用**。speaker_encoder と build_context_kv_cache(3 CFG分岐×12層のK/V射影)に **axmodel が無い** | 🟥/🟧 |
| A2 | **DiT↔dacvae の T整合ペア** | DiT T=201 ↔ dacvae T=119。連結用に揃ったモデルが無い〔→R2〕 | 🟥 |
| A3 | **可変長運用ロジック** | テキスト長→生成フレーム数(T_real)の duration 決定が実機パイプライン(b_sample)に無い（seq_len ハードコード）。inference_runtime の duration ロジックを移植要 | 🟥 |
| A4 | **b_sample の latent_mask 53入力対応** | 既定 b_sample は 52入力(mask無)前提。deploy DiT は 53入力 | 🟥(小, コードのみ) |

### B. リアルタイム性能（速いか）
| # | 足りないピース | 状態 | 区分 |
|---|---|---|---|
| B1 | **DiT の multi-core 化** | single-core 100ms/call が律速。40step+CFG で RTF≈1.1〔→R1〕 | 🟧(最重要) |
| B2 | **few-step / CFG削減での品質維持** | 40step+CFG が本番品質。step蒸留や少stepでの品質確保手段が無い | 🟧 |
| B3 | **ストリーミング生成** | whole-utterance diffusion = 全step完了まで音が出ない。チャンク/逐次生成の仕組みが無い | 🟧(設計) |
| B4 | **バケツ or 真の動的長** | T_max固定で短文も最大長コスト〔→R5〕 | 🟧 |

### C. 品質保証（実用に耐えるか）
| # | 足りないピース | 状態 | 区分 |
|---|---|---|---|
| C1 | **最適な量子化処方** | deploy DiT が cosu16(true-A16でない)。長文くぐもり懸念〔→R3〕 | 🟨 |
| C2 | **sim≡NPU 等価の検証** | 全品質結論が sim 前提。実機で未検証。崩れると結論やり直し〔→R4-1〕 | 🟨(致命) |
| C3 | **fp32 ref / 実聴の基準** | dacvae_decoder.onnx.data 欠落・sim出力 .npy 無し → 実機で品質判定不能〔→R4〕 | 🟨 |

### D. 運用・プロダクト統合
| # | 足りないピース | 状態 | 区分 |
|---|---|---|---|
| D1 | **NPU 排他オーケストレーション** | TTSがNPU握る間 VLM/yolo を自動停止/再開する仕組み（kokoro前例あり）。TTSサービス化が無い | 🟦 |
| D2 | **常駐サーバ化** | 今は毎回 2GB torch ロード。load-once の resident server が無い | 🟦/🟧 |
| D3 | **条件付けのメモリ削減** | 2GB torch 常駐 vs RAM 1.9Gi。量子化/別プロセス/軽量化のいずれか | 🟦 |

---

## 4. クリティカルパス（実用最小セット）

**「実テキストで自然な wav が NPU で出る」までの最短**:
1. **A2**(dacvae T=201) + **A3/A4**(可変長ロジック + 53入力対応) → end-to-end が繋がる
2. **C2/C3**(sim≡NPU等価 + fp32 ref) → 品質が信用できる ← build host 成果物待ち
3. **C1**(allfcu16 再ビルド判断) → 長文品質

**「リアルタイムで実用」まで**:
4. **B1**(DiT triple-core) ← RT可否をほぼ決める単一レバー
5. **D2/D1**(常駐サービス化 + NPU排他制御) ← プロダクト化

**今すぐ実機で試せる唯一の前進**: 中程度(~4.76s)の文で `torch Stage A → DiT(T=201,trim[:119]) → dacvae(119) → wav` の擬似フルチェーン（A2回避の暫定連結）。ただし Stage A の 2GB が RAM 2.7GB に対し OOM 境界。

## 5. 一言まとめ
- **動作の壁**: 条件付けがtorch依存(A1) + T整合(A2) + 可変長ロジック(A3)。
- **速度の壁**: DiT single-core(B1) と multi-step/非ストリーミング(B2/B3)。
- **品質の壁**: sim≡NPU 未検証(C2) が最も怖い（全前提が崩れうる）。
- **運用の壁**: 常駐化(D2) と NPU排他制御(D1)。
- **唯一の単一最大レバー**: DiT を triple-core で建てる(B1/R1)。RT可否がこれでほぼ決まる。
