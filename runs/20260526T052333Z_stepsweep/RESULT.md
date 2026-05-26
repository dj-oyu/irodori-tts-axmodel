# Step削減 × CFG × 内容長 検証結果 — 2026-05-26 (AX8850)

B2(few-step/CFG削減)の実機検証。**step数はCPUループ引数＝axmodel再ビルド不要**。
harness: `e2e_demo/step_sweep.py`（DiT/dacvaeセッション1回ロード→texts×N単一プロセス）。
構成: DiT `allfcu16_npu3`(true-A16, triple-core) / dacvae `b0`(T=119) ＋ long は `dacvae_T201`(T=201) / seed=0 / slim cond(3branch CFG)。
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

客観品質(N=32からの発散)は非単調で knee 決定に使えず（spectral指標は under-integration の
ザラつきを捉えない＝量子化帯域制限とは別アラ）。**実聴が決定的**。

## 実聴結果（正しい設定＝標準CFG ＋ 内容長に合った t-valid）
| 文 | 設定 | 結果 |
|---|---|---|
| plain (5tok) | cfg3/5, t119 | floor≈N=20（N=16はやや）。**低CFG(1.5/2.5)なら N=8可**(過充填の音素繰返し軽微) |
| sibilant (11tok) | cfg3/5, t119 | **N=16違和感なし / N=12金属ノイズ若干だが許容** |
| long (29tok) | cfg3/5, **t201** | N=32でも元々ノイズ多め(量子化床)。**N=16/12で悪化なし** |

### 当初結論の訂正（交絡だった）
最初 "step floor は内容依存・中長文は保守" と出したが、これは**2つの交絡**:
- 中文を**低CFG**で試して崩した（低CFGは短文専用の攻め手で、濃い内容は under-guide で崩れる）。
- 長文を**誤った t-valid=119**で試して崩した（29tok は119フレームに入らない＝duration不一致 A3）。
→ **標準CFG ＋ 正しい t-valid に揃えると、step削減は内容横断でほぼ同等に効く**。step floorは内容固有ではなかった。

### 2種類のノイズの切り分け（実聴語彙→要因）
- 「ラジオみたいな」= **量子化の帯域制限**（R3の床。step/cfg/内容 非依存、常在。longのN=32ノイズ多めもこれ）。
- 「水中ぶくぶく」「金属っぽい」= **step粗 × CFG過剰 のアーティファクト**（標準CFG+十分step で消える。S8重み量子化のbubblingとは別物）。

### long のノイズ = duration(A3)、確定
long(29tok)@t-valid=119(4.76s)=ノイズ → @t-valid=201(8.04s)+dacvae_T201=**完璧/悪化なし**。
過去の動いたe2eは~20tok。**R2(dacvae_T201)が実・長文で有効**と確認。
deploy設計「duration→T_real→latent_mask 必須」を実音で裏付け。

## 結論・推奨
1. **(CFG・duration)を正せば step削減は内容横断で効く。安全な global = N=16〜20 + 標準CFG**。
   - N=20: 全文安全（短文込み, 2.7s, 0.63×）。
   - **N=16: 中長文クリーン・短文ぎりぎり（2.2s, 0.50×）= 実用推奨**。
   - N=12: 中文で金属ノイズ若干（許容）、短文は低CFG要。per-call一定ゆえ 16→12 の短縮は0.6sのみ → **N=16が費用対効果のスイートスポット**。
   - 低CFG(N=8)は**短文専用の攻め手**。global設定としては不要。
2. **architecture（ユーザー設計を裏付け）**: 短文専用モデルを持たず、**1本の T_max=201 axmodel + latent_mask + A3(per-utterance t-valid)** で可変長出力。長さルーティング不要・ビルド/検証1本で安全。本検証で「モデルを分けるのでなく t-valid を内容長に合わせるだけ」を実証。上限 T_max=201(~8s) 超は要チャンク分割。
3. **前提**: A3(duration予測) 必須（long で実証）。これ無しの t-valid固定は短文専用。
4. long残存の「ノイズ多め」は量子化床（step非依存・直交）。改善は量子化側(R3の先 = AdaRound/QAT)、step削減では戻らない。

## 成果物
- harness: `e2e_demo/step_sweep.py`、wav群 `/tmp/sweep/`（device一時, 非追跡）。
- 関連: `irodori_ax650_preprocess/NPU実用化_ギャップ分析.md`(B2/A3), デプロイ設計.md(§2可変長), `runs/…_r123/RESULT.md`(R2/R3), `runs/…_c2_dacvae/`(C2)。
