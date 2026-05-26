# Step削減 × CFG × 内容長 検証結果 — 2026-05-26 (AX8850)

B2(few-step/CFG削減)の実機検証。**step数はCPUループ引数＝axmodel再ビルド不要**。
harness: `e2e_demo/step_sweep.py`（DiT/dacvaeセッション1回ロード→texts×N単一プロセス）。
構成: DiT `allfcu16_npu3`(true-A16, triple-core) / dacvae `b0`(T=119,trim) / seed=0 / slim cond(3branch CFG)。
3文: plain(5tok) / sibilant(11tok) / long(29tok)。

## 確定事項（客観）
- **N=32×2 latent bitwise一致** → npu3 は run間決定論的。発散値にノイズ床なし。
- **per-call ~56ms が全N/全文で一定** → **レイテンシ = calls × 56ms**（クリーンな線形）。

| N | calls | DiT | 対32 |
|---|---|---|---|
| 32 | 76 | 4.3s | 1.0× |
| 20 | 48 | 2.7s | 0.63× |
| 16 | 38 | 2.2s | 0.50× |
| 12 | 28 | 1.6s | 0.37× |
| 8 | 20 | 1.1s | 0.26× |

客観品質(N=32からの発散, latcos / logSTFT-L1)は **非単調**で、N=8が全文で明確に最悪(logSTFT≈1.9–2.1)。
ただし「別サンプル化」と「劣化」を分離できず、**knee決定は実聴が決定的**（spectral centroid/rolloffは
under-integrationのザラつきを捉えない＝量子化帯域制限とは別アラ）。

## 実聴結果（決定的）— step floor は内容依存
| 文 | N=32 | 削減の挙動 |
|---|---|---|
| **plain (5tok)** | OK | cfg3/5 floor≈N=20。**N=8+低CFG(1.5/2.5)で実用**(1.1s, ~3.9倍速)。軽微な音素繰返し=短文の過充填 |
| **sibilant (11tok)** | OK | **低step+低CFG = 識別不能ノイズ**。低CFG-rescueは効かず（under-guide）。より多step/標準CFG要 |
| **long (29tok)** | **@119=ノイズ** / **@201=完璧** | step問題でなく **duration不一致(A3)** |

### 2種類のノイズの切り分け（実聴語彙→要因）
- 「ラジオみたいな」= **量子化の帯域制限**（R3の床。step/cfg非依存、常在）。
- 「水中ぶくぶく」= **CFG過剰 × 粗step のアーティファクト**（cfg下げると消える。S8重み量子化のbubblingとは別物）。

### CFG-rescue は内容依存
- 短文(plain): 低CFGが低stepを**救う**（n12: cfg3/5=ノイズ → cfg1.5/2.5=OK）。
- 中文(sibilant): 低CFGは低stepを**救わない/悪化**（under-guideでノイズ）。
→ **step削減とCFGは結合、かつ結合の符号が内容で変わる**。

### long のノイズ = duration(A3)、確定
long(29tok)@t-valid=119(4.76s)=ノイズ → @t-valid=201(8.04s)+dacvae_T201=**完璧に再生**。
29tokは119フレームに入らない（過去の動いたe2eは~20tok）。**R2(dacvae_T201)が実・長文で有効**と確認。
deploy設計「duration→T_real→latent_mask 必須」を実音で裏付け。

## 結論・推奨
1. **全文一律のstep削減は不可**。step floor は内容長 × CFG × duration(A3) と結合。
   - 短文: 攻め可（N=8 + cfg1.5/2.5, 1.1s）。
   - 中長文: 保守（標準CFG + 多step）。低CFG攻めは崩れる。
2. **本番step数は A3(text→T_real→latent_mask) 実装後に再最適化すべき**。今は t-valid=119固定が短文専用で、長文は崩壊（=長文のstep評価が交絡）。
3. 確定して使える成果: **レイテンシモデル(56ms×calls)** と **「低CFGは短文専用の攻め手」** という設計知見、**R2が実長文で有効**。
4. step削減を本番採用するなら **CFG再調整を必須随伴**（B2はCFG込みで1つの最適化）。

## 成果物
- harness: `e2e_demo/step_sweep.py`、wav群 `/tmp/sweep/`（device一時, 非追跡）。
- 関連: `irodori_ax650_preprocess/NPU実用化_ギャップ分析.md`(B2/A3), デプロイ設計.md(可変長), `runs/…_r123/RESULT.md`(R2/R3)。
