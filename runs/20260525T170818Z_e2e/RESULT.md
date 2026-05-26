# E2E 実機結果 — 2026-05-26 (text→wav, NPU)

テキスト: 「今日はとても良い天気ですね。少し散歩に出かけませんか。」
パイプライン: torch Stage A → DiT(kv_long_lm_cosu16, T=201, mask valid=119, 32step) → trim(1,32,119) → dacvae_b0(NPU) → wav(4.76s/48kHz)

## 実聴結果（本体スピーカー card0/ES8311, DAC 45%→55%）
- **声になっている＝配線完全OK**。日本語として聴き取れる。
- 音質 = **「アナログ電話みたいな」帯域制限・くぐもり**。
- → テスト観点 品質マップ「**中 mel_L1≈2.16 = 電話**」と一致。**崩れ(バグ)ではなく量子化のHF欠落シグネチャ**。

## 解釈
1. **R3(allfcu16/true-A16) の必要性を実音で確定**: deploy DiT は cosu16 処方（FC入力U8, U16化率0/245。Ph1で検出）。電話品質はその帯域欠落。allfcu16(245FC→U16)が直接の改善レバー（過去 mel 3.68→2.73 でくぐもり改善）。
2. **sim≈NPU の弱い傍証**: 実機品質が sim予測の「中=電話」帯に着地（厳密なC2等価検証の代替ではない）。
3. **step数は無関係**: 電話品質は帯域制限でありundercookではない。40stepでも帯域は戻らない。

## 性能（実測, この構成）
- Stage A(torch条件付け): 156s（うちload 139s, swap律速）← 実用化には常駐化(D2)必須
- DiT(NPU): 7.9s/32step（CFG22×3call + 10×1call）← RT化はtriple-core(B1/R1)
- dacvae(NPU): 166ms

## 結論
端末上で **NPUベースの text→wav が成立**。品質は「電話」レベルで、**改善の本命は R3(allfcu16 再ビルド, build host)**。次点で T=201 正規連結(R2)・常駐化(D2)・triple-core(B1)。
