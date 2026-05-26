# ベンチ＆音声バリエーション結果 — 2026-05-26

## 1. NPU フェーズ・タイミング統計（統計的に信頼できる値）
2テキスト×6反復、固定 num_steps=32 / T=201 / dacvae T=119。

| フェーズ | n | mean | std | median | min–max | p95 | CV |
|---|---|---|---|---|---|---|---|
| DiT sampling / 発話 | 12 | **7.91s** | 0.03 | 7.91 | 7.87–7.96 | 7.95 | **0.3%** |
| DiT per-call | 912 | **103.9ms** | 1.78 | 103.6 | 101.5–135.5 | 106.3 | 1.7% |
| dacvae decode | 12 | **165.9ms** | 0.11 | 165.9 | 165.8–166.1 | 166.1 | 0.1% |
| (axmodel load) DiT / dacvae | — | 2.5s / 0.45s | — | — | — | — | one-time |

**所見**: 短文・中文で sampling が同値（7.91s）→ **タイミングはテキスト非依存**（固定T・固定step・CFG閾値のため）。CV<2%で再現性高。
Stage A(torch) は別表（下記、swap律速で不安定）。

## 2. seed 話者バリエーション（torch不要・no_ref）
同一テキスト（中文）で seed のみ変更 → **声色が変わる**。実聴:
- seed 0 = 男性1 / seed 1 = 女性 / seed 2 = 男性2、**目立った破綻なし**（seed1 のみ波形 ±1.0 でクリップ気味）。
- **結論**: 話者バリエーションは参照音声(voice cloning)もtorch再実行も不要、**NPU段の --seed だけ**で性別含め振れる。

## 3. emoji コントロール — 現状ブロック
- normalize_text は emoji を**保持**（😊😄🤫😲 そのまま）→ 仕組み上は効くはず。
- だが emoji はテキスト→**torch Stage A** を通す必要があり、本セッションで torch が swap劣化:
  - load は通る（~144–152s）が、**encode が thrash で ~300s+/件に悪化**、OOM/timeout 非決定的。
  - cond-only（CFG省略でencode~1/3）でも不足。plain版1件のみ取得、emoji版は書込中timeoutで破損。
- **A/B（plain vs 😄 同一テキスト）は未達**。

## 4. 環境ブロッカーの更新（重要）
- torch Stage A（2GB fp32）は **RAM 1.9Gi に対し swap律速→反復で悪化**（load 139s→850s+、encode 17s→300s+）。
- swap committed 850MB+/2GB、断片化。**根本回復にはリブート**（swapクリア）か、build host 生成、常駐サーバ(D2)。
- → emoji 検証は **リブート後に一発ロード**すれば成立する見込み（load→A/Bペア encode→cond-only合成→A/B再生）。

## 成果物
- 統計: `npu_timings.json`、ログ `bench_npu.log`
- 音声: `seed_0/1/2.wav`(話者), `emoji_plain_tanoshii.wav`(plain単独)
- スクリプト: `bench_stageA.py` `bench_npu.py` `emoji_stageA.py`、`e2e_npu.py`(--no-cfg追加)
