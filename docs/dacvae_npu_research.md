# DACVAE Decoder on AX8850 NPU — Untried-Methods Recon & Cheap Evaluation

Date: 2026-05-23. Toolchain: docker `pulsar2:6.0` (x86 build + `pulsar2 run` simulator, no NPU hardware).
Scope: research/recon only — no production code modified. All experiments live in
`/home/exe/.claude/jobs/f7b5721e/`.

## TL;DR — the prior "NPU impossible" conclusion is WRONG

**A single export-time rewrite of the Snake activation makes the whole DACVAE decoder build on
the NPU as one subgraph at production length (T=119), no chunking, no NoTilerException.**

Root cause of the prior failure (now identified): Pulsar2 6.0 **fuses** the ONNX op pattern
`Mul(α,x) → Sin → Pow(·,2) → Mul(1/α,·) → Add(x,·)` into a single quantized op `AxQuantizedSnake`.
That fused op's tiler/quantizer is the broken component:
- small T (<64): `AxQuantizedSnake` build → `OpBuildException "broadcast dim 2: T vs 1536"`
- large T (≥64): `AxQuantizedSnake` tiler → `NoTilerException` on `(1,192,T×960)`.

The Snake math is per-element in time, so the *primitives* tile fine — only the *fused op* doesn't.
**Rewriting Snake with the identity `sin²(αx) = (1 − cos(2αx))/2` prevents the fuser from matching**
(no `Sin`, no `Pow(·,2)`), leaving plain `Cos/Mul/Sub/Add` that the tiler splits correctly.
Verified: build succeeds, produces `compiled.axmodel` (87.5 MB), single NPU subgraph, max_cycle 164M,
**zero Snake ops** in the quant graph (29× `AxQuantizedCos` instead). Numerical equivalence of the
rewrite at fp32: max_abs_err 7.3e-6 vs original Snake.

---

## 1. Tried-vs-untried table

| # | Method | Status | Exact result / error |
|---|--------|--------|----------------------|
| P1 | Whole decoder → single NPU (NPU1/2/3), **original Snake** | TRIED (prior), FAIL | `AxQuantizedSnake` recognized; tiler `NoTilerException` on `(1,384,14280)`/`(1,192,114240)` |
| P2 | Chunked decode, original Snake, T=16/32/48 | TRIED (prior), FAIL | first Snake `(1,1536,T)` `OpBuildException "broadcast dim 2: T vs 1536"` |
| P3 | Chunked decode, original Snake, T=64/96/119 | TRIED (prior), FAIL | terminal Snake `(1,192,T×960)` `NoTilerException` |
| **N1** | **Snake cos-rewrite `(1-cos(2αx))/2`, T=119, NPU3** | **TRIED (this work), SUCCESS** | **Single NPU subgraph, compiled.axmodel 87.5MB, max_cycle 164.3M, 0 Snake ops, 29 AxQuantizedCos** |
| N2 | Snake alpha pre-expand to `(1,C,T)`, T=32, NPU3 | TRIED (this work), FAIL | `AssertionError: Quant don't support the shape(49152) of snake alpha now` — still fuses to Snake; worse |
| N3 | `compiler.enable_tile_mode`, T=119, NPU1 | TRIED (this work), INCONCLUSIVE | host OOM-killed in quant phase (3 parallel builds on 15GB host). Not a Pulsar2 signal. Superseded by N1. |
| N4 | `compiler.enable_slice_mode`, T=119, NPU1 | TRIED (this work), INCONCLUSIVE | host OOM-killed in quant phase. Superseded by N1. |
| N5 | `enable_tile_mode + enable_slice_mode + transformer_opt_level=2`, T=119, NPU3, original Snake | TRIED (this work), FAIL (but new) | Passed quant, reached NPU backend, then `merge_sliceop.chunk_support_op: ValueError: not enough values to unpack (expected 4, got 2)` — got *past* NoTilerException, hit a different slice-scheduler bug on the 3D Snake tensor |
| U1 | `layer_configs:[{op_type:"Snake",data_type:"FP32"}]` (keep Snake FP32) | UNTRIED | Feasible but unnecessary given N1. `data_type=FP32` is a quant-precision flag; would still route through the fused Snake op's backend path (tiler). Low expected value. |
| U2 | Hybrid NPU/CPU partition (`compiler.sub_configs` + start/end_tensor_names; or split ONNX, chain in host) | UNTRIED | Feasible as a fallback (proto confirms `SubgraphType{NPUDotNeu,ONNX}` = CPU/ONNX subgraph exists). Unnecessary given N1 builds the whole graph on NPU. |
| U3 | Layout transpose (B,T,C) so big dim is channel | UNTRIED, REJECTED | Would require rewriting every Conv1d; tiler limit is on total tensor size not axis order. High effort, low value. |
| U4 | Newer Pulsar2 (>6.0) with Snake fix | UNTRIED (can't, local image is 6.0) | Docs are at "V5.1"; HF binary tarball labeled 6.0. Versioning inconsistent. Not testable locally. The `"...alpha now"` wording (N2) hints the Snake limit is a known toolchain gap. Note as future work. |

---

## 2. Untried-method evaluations (with concrete results)

### N1 — Snake cos-identity rewrite  ★ THE FIX (verified on x86 sim build)
- **What**: export-time monkeypatch of `Snake1d.forward` to compute
  `x + (1/(α+1e-9)) · 0.5 · (1 − cos(2αx))` instead of `x + (1/α)·sin²(αx)`. Mathematically identical.
- **Why it works**: defeats Pulsar2's Snake fuser. The graph keeps primitive `Cos/Mul/Sub/Add`,
  which the tiler splits along time (confirmed in log: the fatal `(1,192,114240)` tensor was tiled
  into 17 slices of 6720 — `view_30[:1,:192,:6720] ... [:1,:192,107520:114240]`).
- **fp32 equivalence**: max_abs_err 7.302e-6 (T=119) / 1.3e-6 (T=32) vs original Snake (ONNXRuntime).
- **Build**: `pulsar2 build ... NPU3`, MinMax W8A8, 16 random calib samples →
  `out_T119cos_npu3/compiled.axmodel` = **87.5 MB**, single NPU subgraph, max_cycle 164,353,838,
  QuantAxModel MACs 260.9 G. Quant graph op-types: `AxQuantizedMul 58, AxQuantizedAdd 41,
  AxQuantizedCos 29, AxQuantizedSub 29, AxQuantizedConv 27, AxQuantizedConvTranspose 4, Tanh 1`.
  **Snake ops: 0.**
- **x86 sim numerical validation** (`pulsar2 run` on compiled.axmodel, no NPU HW): runs end-to-end,
  writes `audio.bin` of correct shape (228480 f32). vs fp32 cos-ONNX reference (random z, seed 7):
  **cosine 0.843, SNR 4.8 dB, ref_std 0.097 / sim_std 0.102, max_abs 0.91**.
  → structurally correct (right shape, matched std, positive correlation) but **W8A8 quantization
  quality is poor** — exactly the A8-saturation pattern seen on DiT (W8A8 cos 0.45 → W8A16 0.925).
  This number used **random calib + default W8A8 + random/OOD input**, so it is a lower bound, not a
  quality verdict. The structural point — *the decoder runs on NPU at all* — is what's proven.

### N2 — alpha pre-expansion  (negative result, informative)
- **What**: pre-expand α from `(1,C,1)` to full `(1,C,T)` with `.expand(...)` so there is no runtime
  broadcast (hypothesis: kills the "broadcast dim 2: T vs 1536" bug at T<64).
- **Result**: build reaches quant export then dies:
  `RuntimeError: Operator(...onnx.Snake) convert error: Quant don't support the shape(49152) of snake alpha now`.
  Pulsar2 STILL fuses to `Snake` and its quantizer hard-asserts α must be a small per-channel vector.
  Expanding α makes it strictly worse. **Conclusion: the trap is the fused op, not the ONNX broadcast** —
  this is what redirected the approach to N1.

### N5 — tile+slice scheduler on original Snake  (negative, but reframes prior conclusion)
- With `enable_tile_mode + enable_slice_mode + transformer_opt_level=2` on NPU3, the original-Snake
  graph passed quantization and entered the NPU backend compiler (further than any prior config),
  failing in the slice-merge pass: `chunk_support_op: ValueError: not enough values to unpack
  (expected 4, got 2)` — the slice scheduler expects 4D (NCHW) tensors and chokes on Snake's 3D
  `(B,C,T)`. So even the original Snake is closer to buildable than "NoTilerException" suggested;
  the prior swept matrix never tried these compiler flags. (Mooted by N1, which builds cleanly without them.)

### U1 — Snake-as-FP32 via layer_configs  (untried; low value)
- `build_config.proto` `LayerConfig.data_type` accepts `FP32` per `op_type`. But this is a *quantization
  precision* setting — the op still compiles through the same fused-Snake NPU backend path that throws
  the tiler exception. It does not change tiling. Not worth a build given N1 succeeds.

### U2 — hybrid NPU+CPU subgraph  (untried; valid fallback only)
- Confirmed mechanism exists: `axmodel_extra.proto` `SubgraphType{NPUDotNeu=0, ONNX=1}` — Pulsar2 can
  emit CPU/ONNX subgraphs within one axmodel. `build_config.proto` `CompilerConfig.sub_configs` +
  `start_tensor_names/end_tensor_names` allow manual subgraph cut points. This is the production-grade
  fallback (run conv/upsample on NPU, Snake-heavy tail on CPU) — but **unnecessary**: N1 puts the
  entire decoder on NPU. Recorded for completeness.

### U3 / U4 — rejected / not locally testable
- U3 layout transpose: every Conv1d would need rewriting; tiler limit is on tensor size not axis. Reject.
- U4 newer Pulsar2: docs version "V5.1" vs binary "6.0" inconsistent; can't test on the local image.
  The N2 error string "...alpha now" suggests Axera knows the Snake-op limit. Future work: check whether
  a Pulsar2 >6.0 fuses Snake more robustly (would make even the unmodified ONNX buildable).

---

## 3. Ranked recommendation for a full attempt

1. **N1 (Snake cos-rewrite) — do this.** Strongest evidence: it actually built the full decoder on NPU.
   - Production path: add a Snake monkeypatch to the existing export-patch module (sibling to
     `rope_export_patch.apply_export_patches`), re-export `dacvae_decoder.onnx`, build with a real
     activation calibration set (not the random calib used here), `precision_analysis_method=EndToEnd`,
     and W8A16 (`layer_configs` S16 activation) given DiT's experience that A8 saturates. Validate wav
     SNR vs the CPU/ONNX decoder before shipping.
   - Caveat: this work used **random calib + W8A8**, so the *numerical* result below is a structural/
     codegen check, not a final-quality number. Quality tuning (W8A16, real calib, smooth_quant) is the
     remaining work. The DiT W8A8→W8A16 uplift (0.45→0.925) is **expected** to carry over but is
     **unverified for vocoder activations** — DiT's saturation was attention/finfo.min outliers; the
     vocoder is conv+Cos+Mul with a different activation distribution, so the magnitude of the W8A16
     uplift here must be measured, not assumed. **W8A16 build attempted but host-OOM-killed** (U16
     activations ~2× memory; this 250MB model with a 228K-sample output exceeds the 15GB host even
     running alone — A8 succeeded because it needs less). The W8A16 quality number is therefore the
     one piece I could not obtain locally; needs a bigger-RAM build host.
   - **Perf is not a given.** The cos rewrite *adds* ops (Cos+Sub+Mul replacing Sin+Pow per Snake) and
     the decoder compiles to max_cycle 164M. Naive scaling from DiT (52M cyc ≈ 29 ms/step) puts NPU
     decode near ~90 ms vs the ~53 ms CPU baseline. So the real win is **single-axmodel / coherent NPU
     pipeline**, not necessarily faster decode. The go/no-go must hinge on a real-HW latency measurement.
   - **Reproducibility**: the calib generator here did NOT seed numpy, and W8A8 PTQ calib stats can vary
     run-to-run. A production build should use a deterministic / real-activation calib path.

2. **U2 (hybrid partition)** only as a fallback if W8A16 Snake-cos quality proves unacceptable on real audio.

3. Everything else (U1/U3/N2/N5) is ruled out or mooted.

---

## 4. Honest limits — what I could NOT verify

- **Quality (quantization) is not validated.** N1 used 16 *random* latent calib samples and default W8A8.
  Structural build + tiling is proven; audible quality is not. Vocoders are sensitive to activation
  quantization — expect to need W8A16 and real-activation calibration, mirroring the DiT findings.
- **No real NPU hardware.** Only x86 `pulsar2 run` simulator. Prior FINDINGS showed the on-device engine
  (2.12.0s) can add ~10% degradation vs sim; that is untested here.
- **Single shape / single sample.** Only T=119, B=1, one random z. Variable T (utterance-dependent) and
  the host-side fixed-shape strategy are unaddressed.
- **OOM artifact.** The N3/N4 parallel builds were host-OOM-killed (15GB RAM, 3 concurrent ~250MB-model
  builds). Run Pulsar2 builds **one at a time** on this host. Not a Pulsar2 limitation. Separately, the
  **W8A16 cos build was OOM-killed even running alone** (U16 doubles activation memory) — the W8A16
  quality measurement requires a larger-RAM build host and is the main untested follow-up.
- The cos-rewrite adds 29 `Cos` + extra `Mul/Sub` vs the fused Snake; max_cycle 164M for the decoder.
  Whether that on-NPU cost beats the A55 CPU decode (~53 ms baseline) is a perf question for real HW
  (naive cycle-scaling suggests ~90 ms — possibly *slower* than CPU; main value is pipeline coherence).
- **Watermark path**: the decoder includes a Watermarker with `nn.LSTM` (lstm_layers=2). The dynamo
  exporter decomposes the LSTM into elementwise ops (no `LSTM` op appears in the graph), so it is not a
  separate blocker and the cos rewrite is orthogonal to it. `alpha = wm_channels/d_wm_out = 0.25`, so the
  watermark contribution is present in the exported graph. Not separately validated for quantization.

## Artifacts (in /home/exe/.claude/jobs/f7b5721e/)
- `export_dacvae_variants.py` — export with `--variant {cos,expand,mulmul,plain}` (the cos monkeypatch is the fix)
- `dacvae_T119_cos.onnx(+.data)` — cos-rewrite decoder, fp32-validated 7.3e-6
- `out_T119cos_npu3/compiled.axmodel` — 87.5 MB, single NPU subgraph (THE proof)
- `out_T119cos_npu3/quant/quant_axmodel.onnx` — quant graph, 0 Snake / 29 AxQuantizedCos
- configs: `dacvae_T119cos_npu3.json`, `dacvae_tileslice_npu3.json` (N5), `dacvae_T32expand_npu3.json` (N2)
- `build_config.proto` (extracted from image) — for layer_configs / sub_configs reference
