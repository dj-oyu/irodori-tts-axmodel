# C2 sim≡NPU 等価検証 — DACVAE b0（build host, 2026-05-26）

> ギャップ分析 C2「sim≡NPU 等価」(致命) の dacvae 版。build host で **fp32 onnx vs pulsar2 x86-sim** を測定し、
> device が **同一入力を実NPU実行 → 本 sim 出力と突合**することで「sim の品質結論が実機に転送可能か」を確定する。
> 元実装: `irodori_ax650_preprocess/scripts/dacvae_sim.py`。

## build host 側で確定（fp32 vs pulsar2-sim）
- モデル: `build/axmodel_dacvae_b0`(W8A8, T=119)。入力: `build/calib_dacvae/test_z.npy`(実 DiT 出力 z, (1,32,119))。
- fp32 ref = `dacvae_decoder_cos.onnx`(+.data) の onnxruntime 出力（= `test_z_fp32_wave.npy`）。
- **fp32 vs sim: SNR 5.96dB / corr 0.877 / mel_L1 1.95**（W8A8 の既知のホワイトノイズと整合）。

## 同梱ファイル（device 突合用）
| file | 形状 | 用途 |
|---|---|---|
| `input_z.npy` | (1,32,119) f32 | **device がこの z を実NPU dacvae_b0 に投入**（=sim と同一入力） |
| `sim_audio_b0.npy` | (228480,) f32 | build host の pulsar2-sim 出力（突合の基準） |
| `fp32_audio.npy` | (228480,) f32 | fp32 onnx 出力（真値リファレンス） |

## device 側でやること（C2 完結手順）
```python
# 実機 axengine で input_z を dacvae_b0 に投入し npu_audio を得る
import numpy as np, axengine as axe
z = np.load("runs/20260526T051746Z_c2_dacvae/input_z.npy")
dac = axe.InferenceSession("build/axmodel_dacvae_b0/compiled.axmodel")
zin = dac.get_inputs()[0].name
npu = np.asarray(dac.run(None, {zin: z})[0]).reshape(-1)
sim = np.load("runs/20260526T051746Z_c2_dacvae/sim_audio_b0.npy")
fp32 = np.load("runs/20260526T051746Z_c2_dacvae/fp32_audio.npy")
def snr(a,b):
    n=min(len(a),len(b)); a,b=a[:n],b[:n]
    return 10*np.log10((a**2).sum()/(((a-b)**2).sum()+1e-12)), float(np.corrcoef(a,b)[0,1])
print("sim vs NPU :", snr(sim, npu))   # ★これが C2 の核心
print("fp32 vs NPU:", snr(fp32, npu))
```

## 判定基準
- **sim vs NPU が高SNR（理想は bit 近い／SNR≫fp32-vs-sim の 5.96dB）** なら → **sim≡NPU 成立**。
  「fp32 vs NPU ≈ fp32 vs sim(5.96dB)」となり、**sim で出した全品質結論(mel_L1 等)が実機に転送可能**と確定。
- sim vs NPU が低いと → sim と実機がズレる＝sim 前提の品質結論を実機で取り直す必要（致命シナリオ）。

## 次（build host 側）
- DiT 版 C2（品質の本丸）: `full_loop_axmodel.py` の `run_axmodel_step` で allfcu16 を **single-step** sim → fp32 onnx の同一1ステップと per-step cosine/SNR 比較 + v_pred_sim を device へ。
