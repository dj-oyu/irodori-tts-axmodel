# C2 sim≡NPU 等価検証 — DiT allfcu16（品質本丸, build host, 2026-05-26）

> ギャップ分析 C2「sim≡NPU」(致命) の DiT 版。品質結論(mel_L1 等)は全て sim で出しているため、
> DiT が「sim≡実NPU」かが最重要。build host で **fp32 onnx vs pulsar2 x86-sim を single-step** 比較し、
> device が **同一53入力を実NPU実行 → 本 sim 出力(v_pred_sim)と突合**して確定する。
> スクリプト: `irodori_ax650_preprocess/scripts/dit_c2_singlestep.py`。

## build host 側で確定（fp32 vs pulsar2-sim, 1 step）
- モデル: `build/axmodel_kv_long_lm_allfcu16`(true-A16, FC U16 245/245)。fp32: `dit_step_kv_long_lm_mask1e4_fp32.onnx`。
- 入力: `build/calib_kv_long/<name>/0000.npy` の一貫した実 DiT ステップ 53入力（latent_mask 全有効 → valid=201）。
- **fp32 vs sim: cosine 0.99752 / SNR 22.95dB**（T=201 全域）。
  → per-step cosine 0.998 は QUANTIZATION_NOTES の cosu16 ~0.997 と整合。**DiT 量子化は sim 上で fp32 にほぼ忠実**。

## 同梱ファイル（device 突合用, 計~2.4MB）
| file | 内容 |
|---|---|
| `dit_inputs.npz` | 53入力一式（x_t/t/text_mask/speaker_mask/latent_mask + 48 kv）。圧縮 2.37MB |
| `v_pred_sim.npy` | (1,201,32) pulsar2-sim の v_pred（突合基準） |
| `v_pred_fp32.npy` | (1,201,32) fp32 onnx の v_pred（真値リファレンス） |

## device 側でやること（C2 完結手順）
```python
import numpy as np, axengine as axe
d = np.load("runs/20260526T052020Z_c2_dit/dit_inputs.npz")
dit = axe.InferenceSession("build/axmodel_kv_long_lm_allfcu16/compiled.axmodel")  # NPU1版で
ishapes = {i.name: i.shape for i in dit.get_inputs()}
MASKS = ("text_mask","speaker_mask","latent_mask")
feed = {n: (d[n].astype(np.uint8) if n in MASKS else d[n].astype(np.float32)) for n in ishapes}
v_npu = np.asarray(dit.run(None, feed)[0]).reshape(1,201,32)
sim = np.load("runs/20260526T052020Z_c2_dit/v_pred_sim.npy")
fp32 = np.load("runs/20260526T052020Z_c2_dit/v_pred_fp32.npy")
def m(a,b):
    a=a.ravel().astype(np.float64); b=b.ravel().astype(np.float64)
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)), \
           10*np.log10((a**2).sum()/(((a-b)**2).sum()+1e-12))
print("sim  vs NPU :", m(sim, v_npu))   # ★C2 の核心
print("fp32 vs NPU :", m(fp32, v_npu))
```
※ allfcu16(NPU1) で測ること。npu3版は triple-core 並列リダクションで微差が出る（実機検証で既報）ので等価判定は NPU1 が clean。

## 判定基準
- **sim vs NPU の cosine が ~1.0（fp32-vs-sim の 0.998 と同等以上）** なら → **sim≡NPU 成立**。
  「fp32 vs NPU ≈ fp32 vs sim(0.998/22.95dB)」となり、**sim で出した DiT 品質結論(帯域/mel_L1/処方比較)が実機に転送可能**と確定（致命リスク解消）。
- 乖離が大きいと → 量子化の sim 近似が実機 NPU で崩れる＝sim 前提の結論を実機で取り直す必要。

## 状況
- dacvae C2: `runs/20260526T051746Z_c2_dacvae/`（fp32 vs sim SNR 5.96dB, W8A8）。
- DiT C2（本書）: fp32 vs sim cosine 0.998。**両モデルとも sim は fp32 に忠実**。残るは device の sim-vs-実NPU 突合のみ。
