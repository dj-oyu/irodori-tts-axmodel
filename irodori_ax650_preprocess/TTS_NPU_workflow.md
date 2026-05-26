# TTS NPU フル活用 ワークフロー / Usage（実機 AX8850 向け, 2026-05-26）

> 現時点の**最良コンポーネントでの text→wav 実行手順**を1枚に集約。
> 背景・検証値は: `実機検証_R123_axmodel説明.md` / `runs/*_r123` `*_slim_e2e` `*_stepsweep` の RESULT.md /
> `slim_torch_conditioning.md` / `NPU実用化_ギャップ分析.md`。

## 全体像（3 段）
```
text
 └─[Stage A] slim_stageA.py   torch CPU(no root) ~360MB anon   ← 条件付け(まだCPU=A1未)
      → cond_<label>.npz (3 CFGブランチ: cond/text/spk の KV射影 + masks)
 └─[Stage B] e2e_npu.py       NPU axengine(ROOT)               ← DiT RFサンプリング
      DiT = allfcu16_npu3 (true-A16=帯域拡張, triple-core=56ms/call)
 └─[Stage C] e2e_npu.py 内    NPU axengine(ROOT)               ← DACVAE 復号
      dacvae = b0(T119) or T201(長文)  → wav(48kHz)
```
- **NPU で動くのは Stage B/C**（DiT・DACVAE）。**Stage A の条件付けは torch CPU のまま**（A1 未着手, slim 化で 360MB に圧縮済）。
- 「全段 NPU」ではない。条件付け NPU化(A1/hybrid)は将来レバー。

## 推奨コンポーネント（最新ビルド, 実機転送済）
| 役割 | 推奨 axmodel | 備考 |
|---|---|---|
| DiT（速度重視・本命） | `build/axmodel_kv_long_lm_allfcu16_npu3` | true-A16(帯域拡張)+triple-core(56ms/call)。**通常はこれ** |
| DiT（数値等価検証用） | `build/axmodel_kv_long_lm_allfcu16` | NPU1版。C2突合や品質基準はこちら（triple-coreの並列リダクション微差を避ける） |
| DACVAE（短〜中文 ~4.76s） | `build/axmodel_dacvae_b0` | T=119。`--t-valid 119` |
| DACVAE（長文 ~8.04s） | `build/axmodel_dacvae_T201` | T=201。`--t-valid 201`。**長文(>~20tok)はこちら必須** |
| text encoder | `build/axmodel_textenc` | A1/hybrid 用に存在（現行 slim では未使用） |

## End-to-end 手順
```bash
# Stage A: slim 条件付け（torch CPU, root不要, ~360MB anon）
PYTHONPATH=/home/exe/ai/Irodori-TTS python3 e2e_demo/slim_stageA.py \
  --weights <model.safetensors> --text "今日はとても楽しいです。" \
  --label plain --out-dir /tmp/cond
# → /tmp/cond/cond_plain.npz

# Stage B+C: NPU で DiT サンプリング → DACVAE 復号 → wav（root必須）
sudo -n PYTHONPATH=. /usr/bin/python3.10 e2e_demo/e2e_npu.py \
  --cond /tmp/cond/cond_plain.npz \
  --dit   build/axmodel_kv_long_lm_allfcu16_npu3/compiled.axmodel \
  --dacvae build/axmodel_dacvae_b0/compiled.axmodel --t-valid 119 \
  --num-steps 16 --cfg-text 3 --cfg-spk 5 --seed 0 \   # 推奨 global N=16(2.2s); 最高品質は 32
  --out-wav /tmp/out.wav
```
- `slim_stageA` の cond は `e2e_npu` の `--cond` に**直結互換**（キー命名一致確認済）。
- 話者バリエーション: `--seed` だけで性別含め振れる（参照音声/torch再実行 不要）。
- cond-only（CFG省略で高速・低品質）: `e2e_npu --no-cfg`。

## step / CFG 設定（step_sweep 実機知見, 2026-05-26 訂正版）
**レイテンシ = calls × 56ms**（線形, npu3）。calls は step数×CFG分岐（N=32→76call/4.3s, N=16→38call/2.2s, N=8→20call/1.1s）。

**推奨 global = `--num-steps 16` + 標準CFG(3/5)**（duration(A3)/t-valid さえ正しければ **step削減は内容横断で効く**）。
当初の「一律削減は不可」は **duration 交絡**だった（長文崩壊は step でなく t-valid 不一致が原因, 下表）。

| num-steps | 速度(対32) | 適性 |
|---|---|---|
| 32 | 4.3s (1.0×) | 最高品質・保守。長文の量子化床ノイズは step では戻らない |
| **16** | **2.2s (0.50×)** | **推奨スイートスポット**: 中長文クリーン・短文ぎりぎり。費用対効果最良（16→12 は0.6s短縮のみ） |
| 20 | 2.7s (0.63×) | 全文安全（短文込み）。最も無難な global |
| 8 + 低CFG(1.5/2.5) | 1.1s (3.9×) | **短文専用の攻め手**。global には不要（中文は under-guide で崩れる） |

**t-valid / DACVAE は内容長で必ず合わせる**（step とは独立・A3 の役割）:
| 文の長さ | t-valid / DACVAE | 理由 |
|---|---|---|
| 短〜中文(~4.76s, ≲20tok) | 119 / `dacvae_b0` | |
| 長文(~8.04s, ~29tok) | **201 / `dacvae_T201`** | t-valid=119 だとノイズ（duration不一致A3）→ T=201 で完璧再生 |
| ~8s 超 | 要チャンク分割 | T_max=201 が上限 |

→ アーキ: **短文専用モデルを持たず、1本の T_max=201 axmodel + latent_mask + A3(per-utterance t-valid)** で可変長。長さルーティング不要・ビルド/検証1本。**A3(duration予測)が前提**（無いと t-valid固定=短文専用）。

## ノイズの切り分け（実聴語彙→要因, 混同しない）
- 「AM/電話ラジオ風（帯域制限）」= **量子化の床(R3)**。step/CFG非依存・常在。~3kHz頭打ち=PTQ天井。
- 「水中ぶくぶく」= **CFG過剰×粗step のアーティファクト**。CFG下げると消える（S8重みbubblingとは別物）。
- 「長文で識別不能」= **duration不一致(A3)**。t-valid を実長に合わせる(=T201)と解消。

## NPU 排他（D1, 運用必須）
- NPU は**2プロセス同時で SEGV**。TTS実行中は **VLM(axllm)/yolo を停止**。
- 復帰順: **axllm → yolo → pet-album**。
- root 必須（axengine が /dev/mem）: `sudo -n PYTHONPATH=. /usr/bin/python3.10 ...`（NOPASSWD設定済）。

## 現状の限界（=次レバー）
| 限界 | 回避/状態 | 本質解決 |
|---|---|---|
| 条件付けが torch CPU | slim で 360MB（常駐可） | A1(NPU化) / D2(常駐サーバ warm~6s) |
| cold 47.5s（うち~43s 1回限りロード） | slim で短縮済 | **D2 常駐サーバ**（両doc最重要） |
| 長文 duration | t-valid=201 + dacvae_T201 で回避 | **A3**(text→T_real→latent_mask) |
| RT 56ms/call（<40ms未達） | 短文は実用 | R1追求 or 56ms実用判断 |
| 品質 ~3kHz頭打ち | true-A16 で電話→改善済 | QAT/蒸留 or 長文fp32-CPU |

## スクリプト早見表（`e2e_demo/`）
| script | 役割 |
|---|---|
| `slim_stageA.py` | **Stage A 推奨**: slim 条件付け(meta+mmap, 360MB, bitwise等価) → cond npz |
| `e2e_npu.py` | **Stage B+C 推奨**: cond → DiT(NPU) → trim → DACVAE(NPU) → wav |
| `a_build_cond.py` | Stage A フル版(2GB丸読み, slim前の旧版) |
| `b_sample_npu.py` / `c_decode.py` | Stage B / C を個別実行したい場合 |
| `step_sweep.py` | step×CFG×内容 のsweep（1ロードで texts×N） |
| `bench_npu.py` / `bench_stageA.py` | NPU/StageA のタイミング計測 |
| `dit_smoke.py` / `dacvae_equiv.py` / `ph0_inspect.py` | I/O・liveness・dtype 確認 |
| `dit_c2_singlestep.py`(scripts/) / `dacvae_sim.py`(scripts/) | C2 sim≡NPU 突合（build host）|
