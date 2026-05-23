# Irodori-TTS-500M-v3 実構造メモ（AX8850/Pulsar 前処理の土台）

baseline 推論を 4090 で通し、`scripts/inspect_model.py` で実体をダンプした結果。
配置: `build/model_introspection.json` に全文、`build/baseline_noref.wav` に出力例。

---
## TL;DR（v0.2, 2026-05-23 時点の到達点）

- ターゲット: M5Stack AI Pyramid Pro = **AX8850**（Pulsar2 では `--target_hardware AX650`）。
- **NPU 化済み（axmodel）**: DiT（Option B 379MB / Option A 338MB）, TextEncoder。全て単一 NPU subgraph。
- **NPU 不可**: DACVAE decoder（Snake op が T<64 broadcast バグ / T≥64 tiler 限界、chunk でも全滅）→ **CPU 確定**。
- **数値検証**: ONNX vs PyTorch ~1e-6、full-loop SNR 66–67dB、W8A8 v_pred cosine **0.9997**、fp16 wav SNR 44dB。
- **few-step**: `sway` + **6〜8 step** が実用域（DiT 呼び出し ~2×減）。最大の latency レバー。
- **export-safe patch**（`rope_export_patch.apply_export_patches`）: complex RoPE→実数 / SDPA bool mask→additive /
  RMSNorm・LowRankAdaLN の rsqrt→sqrt+div。いずれも数値等価。Pulsar2 の IsNaN/Reciprocal 非対応を回避。
- **残**: 実機 end-to-end latency 実測（焦点は A55 の DACVAE-CPU）、固定 T の汎用化（T_max + self-mask）。

以下は時系列の詳細ログ。
---

## ターゲット HW の訂正

- M5Stack **AI Pyramid Pro = Axera AX8850**（24 TOPS INT8, 8GB LPDDR4x）。
- scaffold の "AX650" 表記は family 名。SoC 実体は AX8850。Pulsar2 ツールチェーンは同系統。
- AX650N（72 TOPS INT4）とは別チップ。INT4 前提（Lite 相当）より **INT8 量子化**が AX8850 の素直な狙い。

## ソース

- 本体: `github.com/Aratako/Irodori-TTS` → `/path/to/Irodori-TTS`（uv env 構築済, cu128, torch 2.10）
- Lite: `github.com/kizuna-intelligence/Irodori-TTS-Lite` → `/path/to/Irodori-TTS-Lite`
  - `irodori_tts_lite.configure(use_fused=, force_fp16=, disable_eager=, codec_int4=, codec_int4_groupsize=, pack_rtn_extras=, duration_donor=)` と `patch()` は実在（`checkpoint_loader.py`）。
  - Lite は INT4 packed + Triton fused が主目的（DiT 279MB / peak 552MB）。AX8850 へは直接持ち込めない。

## ランタイム API（scaffold の想定と相違）

- `InferenceRuntime.from_key(key: RuntimeKey)` で、引数は **HF 文字列ではなく `RuntimeKey` dataclass**。
  `RuntimeKey.checkpoint` はローカル .safetensors パス（`infer.py` が `hf_hub_download(repo, "model.safetensors")` で取得）。
- precision は **fp32 / bf16 のみ**（runtime に fp16 経路なし）。ONNX export 時は module を `.half()` する形で別途対応。
- → scaffold の `prepare_ax650_export.py` の `--loader-arg '"...repo..."'` 前提は誤り。export 用 loader は RuntimeKey を組む形に直す必要あり。

## モデル実構成（v3 実値, ckpt metadata より）

| 項目 | 値 |
|---|---|
| latent_dim | 32（= patched, patch_size=1）|
| model_dim | 1280 |
| num_layers (DiT blocks) | 12 |
| num_heads / head_dim | 20 / 64 |
| text_dim / layers / heads / vocab | 512 / 10 / 8 / 99574 |
| speaker_dim / layers / heads | 768 / 8 / 12 |
| timestep_embed_dim / adaln_rank | 512 / 192 |
| use_speaker_condition | True |
| use_duration_predictor | True |
| use_caption_condition | False（v3 は caption 無し）|

named_children: `text_encoder, speaker_encoder(ReferenceLatentEncoder), text_norm, speaker_norm,
duration_predictor, cond_module(Sequential=timestep embed), in_proj(Linear 32→1280),
blocks(ModuleList[12] DiffusionBlock), out_norm, out_proj(Linear 1280→32)`

DiffusionBlock = JointAttention + LowRankAdaLN + SwiGLU。

## 実 I/O shape（hook 捕捉, no_ref / num_steps=4 時）

- **TextEncoder**: `(input_ids[1,256] i64, mask[1,256] bool) → [1,256,512]`
- **ReferenceLatentEncoder(speaker_encoder)**: `(latent[1,L,32] f32, mask[1,L] bool) → [1,L,768]`
  （no_ref で L=1。実 ref では patchify 後の長さ）
- **DurationPredictor**: text_state[1,256,512] ほか → log_frames[1]
- **DiT 1-step = `forward_with_encoded_conditions`**（kwargs 渡し）:
  - `x_t [B,T,32]`, `t [B]`
  - `text_state [B,256,512]`, `text_mask [B,256]`
  - `speaker_state [B,2,768]`, `speaker_mask [B,2]`
  - `caption_state/mask = None`
  - `context_kv_cache`: **12 層分の [k_text, v_text, k_spk, v_spk]**
    = `[B,256,20,64],[B,256,20,64],[B,2,20,64],[B,2,20,64]` × 12
  - B=3 は CFG independent（cond / text-uncond / speaker-uncond を batch 化）。T はパッチ後 latent 長（utterance 依存, 例 119）。

## 段階別コスト（4090, 16 step, no_ref, baseline）

| stage | 時間 |
|---|---|
| predict_duration | 85 ms |
| **sample_rf（DiT 12 層 ×16 step）** | **463 ms（支配的）** |
| decode_latent（DACVAE decoder）| 53 ms |
| watermark（CPU 側, NPU 対象外）| 272 ms |

→ NPU へ出すべき本命は **DiT denoiser**。次点 DACVAE decoder。

## export 設計上の含意（重要）

1. **DiT step は KV cache を入力に取る。** `build_context_kv_cache` で 1 回だけ作り、全 RF step で再利用する設計。
   AX8850 でも「KV cache を host/NPU で 1 回計算 → denoiser を step ごと再利用」が最高効率。
   ただし KV cache を入力にした ONNX は入力数が多い（12×4 tensor）。代替: KV を毎 step 再計算する版（I/O 単純, 計算増）。
2. `forward_with_encoded_conditions` は **kwargs 渡し**かつ KV cache が list-of-list。
   現行 `prepare_ax650_export.py`（positional + flat spec）では直接 export 不可。
   → DiT step 用に「固定 batch・固定 latent 長・KV cache を平坦化引数で受ける」**ラッパ nn.Module** が必要。
3. CFG batch=3（independent）。NPU 固定 shape は batch=3 を 1 グラフにするか、batch=1 で 3 回流すかの設計判断。
4. fixed latent 長 T の決定が必要（duration 予測で可変）。代表上限を 1 つ選び固定 shape にする。

## DiT export: 2 つの形（A/B 設計判断・CMM 反映で訂正）

`forward_with_encoded_conditions(context_kv_cache=...)` の扱いで 2 案:

- **Option A（KV cache を入力にする）**: KV projection を host で 1 回計算→全 step 再利用。入力 6+48=54。
- **Option B（毎 step 再計算, `context_kv_cache=None`）**: 入力 6 個。

### 性能: AX8850 は CMM（メイン LPDDR4x から切り出した連続領域）= ユニファイドメモリ
→ CPU↔NPU の PCIe 的コピーは無い。当初の「bus 転送が高い」前提は誤りで撤回。
ステップあたり DRAM 読み出し量（B=1, fp16 概算）で比較すると:

| | Option A | Option B |
|---|---|---|
| KV cache 読み | ~16MB | 0 |
| 射影重み (wk/wv_text, wk/wv_spk ×12層) | 0(グラフ外) | **~79MB** |
| 射影 MAC | 0 | あり |

precompute した KV(16MB) < 毎 step 読み直す射影重み(79MB)。**帯域・MAC とも A が有利**で、
CMM 前提なら **A が AX8850 の本命候補**（B が「帯域最適化の後回し」は逆だった）。
※ 数値は「毎 step グラフ再実行＝重み再ストリーム」前提の概算。Pulsar2 の fusion/on-chip 保持次第で
  差は縮みうるので最終判断は実測。

### 方針（訂正後）
- **B = 最初に Pulsar2 を確実に通すための de-risk 用**（性能根拠ではない）。
  54 入力(A)が Pulsar2/PTQ calib を詰まらせないかが唯一の懸念。
- **A = 本命候補。B が通った後に必ず A をベンチして比較**。
- A の host 側 `build_context_kv_cache` は per-step でなく **per-発話 1 回**（A55 で軽い）。

### export 上の既知ブロッカ
- RoPE が `torch.view_as_complex/view_as_real`（model.py:31-36）。TorchScript exporter では不可。
  → まず torch 2.10 の **dynamo exporter（`torch.onnx.export(dynamo=True)`）**を試す。
    複素 op が通れば patch 不要。ダメなら実数値 RoPE（cos/sin）に export 時だけ monkeypatch。
- `apply_rotary_emb` は `x.float()→複素→.type_as(x)`。fp16 export 時は内部 fp32 を再現しないと数値ズレ。
- 検証は **fp32-ONNX vs fp32-PyTorch を先に**（export 正当性）→ その後 fp16。

## DiT 1-step ONNX 化: 完了（Option B, fp32, 2026-05-23）

パイプライン（capture は Irodori env、export は preprocess env で分離）:
1. `scripts/capture_dit_inputs.py`（Irodori env）: 実 synth を hook し DiT 実入力+実出力を
   `build/dit_step_b1_fp32.ref.pt` に保存。
2. `scripts/rope_export_patch.py`: complex RoPE を実数(cos/sin)版へ monkeypatch。
   **数値等価を検証済（fp32/fp16 とも max_abs_err=0）**。export 時のみ適用、本体無改変。
3. `scripts/export_dit_step.py`（preprocess env, `PYTHONPATH=.../Irodori-TTS`）:
   `irodori_tts.model` だけを使い（runtime 非 import → protobuf 衝突回避）、
   `TextToLatentRFDiT` を weights から構築し Option B ラッパを dynamo exporter で ONNX 化。

### 結果（検証済）
- 出力: `build/dit_step_b1_fp32.onnx`（graph 2.4MB）+ `.onnx.data`（fp32 weights 1.4GB, external data）
- weight load: missing 0 / unexpected 0、param 512,049,441（500M）✓
- **standalone PyTorch vs 実 runtime 出力: max_abs_err 3.6e-6**
- **ONNX vs PyTorch: max_abs_err 2.4e-6（rel 6e-7, out_range ±4.09）**
  → ONNX は実 runtime の DiT step を ~1e-5 で再現。export 正当性 OK。
- 入力(6): x_t[1,119,32], t[1], text_state[1,256,512], text_mask[1,256](bool),
  speaker_state[1,2,768], speaker_mask[1,2](bool) / 出力 v_pred[1,119,32]

### Pulsar2 へ向けた観察
- **opset 18**（onnxscript の version_converter→17 は assertion で失敗。Pulsar2 の対応 opset 要確認）。
- op 種別は概ね標準: MatMul/Gemm/Softmax/Add/Mul/Reshape/Transpose/Concat、
  RMSNorm=ReduceMean+Sqrt+Reciprocal、Sigmoid/Tanh、RoPE=Cos/Sin。
- **要注意 op: `IsNaN`(12) と `Where`(13)** — SDPA + bool mask 分解由来（all-masked 行の
  softmax=NaN ガード、層ごと 1 個）。Pulsar2 が IsNaN を未サポートなら、bool mask をやめて
  **additive float mask（`(~mask)*-1e4` を score に加算）**で SDPA を呼ぶラッパ変更で `Add` のみに落ちる。
  （本体無改変・wrapper 側の小修正。Pulsar2 が実際に弾いた時だけ実施）。
- T(=latent長)=119 は捕捉 utterance 依存。固定 shape 化は要 latent 長確定（configs は 256 想定）。
- 重み external data 1.4GB は fp32。量子化前に fp16 化 or Pulsar2 側 PTQ。

### 検証スコープ（重要・正直な限界）
今回の 1e-5 一致は **単一スナップショット**の検証（1 timestep / CFG batch を B=1 に slice /
T=119 / no_ref）。以下は**未検証**で、Pulsar2 前に詰める:
- 複数 timestep・full CFG batch(B=3)・ref 条件付き(speaker seq≠2)・異なる latent 長。
- **推奨次テスト**: runtime の `forward_with_encoded_conditions` を onnxruntime 呼び出しに差し替え、
  `synthesize`(no_ref, 同 seed) を丸ごと回して **wav を `baseline_noref.wav` と比較**。
  サンプリングループ全体 vs export の end-to-end 検証になる。

## サンプリングループ全体の検証: 完了（2026-05-23）

`scripts/validate_full_loop.py`（Irodori env + `uv run --with onnxruntime`）:
runtime の `forward_with_encoded_conditions` を onnxruntime 呼び出しに差し替え、
`synthesize` を丸ごと回して純 PyTorch（同 seed）と wav 比較。

### 結果（text="こんにちは、これはテスト音声です。", num_steps=8, seed=1234, no_ref）
- 全 8 step が ONNX 経由（fwec_calls=8）。wav 長一致(182400)。
- **max_abs=7.9e-4, rms=6.5e-5, SNR=67.4 dB, corr=1.000000**
  → fp32 CPU(ONNX) vs CUDA(PyTorch) の 8 step 累積差のみ。可聴閾(>40dB)を大きく上回り実質一致。
- **CFG batch を実地に通過**: loop は cfg_min_t/max_t により B=3(誘導あり 4 step) と
  B=1(誘導なし 4 step) を交互に呼ぶ。動的 batch ONNX が両方を処理。

### dynamic ONNX export の教訓（はまりどころ）
1. RoPE の `reshape(*x.shape[:3], -1, 2)` の **`-1` は dynamic 時に次元誤推論**
   （H が混入）。`b,s,h,d=x.shape; reshape(b,s,h,d//2,2)` と**明示次元**にする。
2. legacy `dynamic_axes` は torch.export が内部 reshape の B を example 値に specialize
   → **`dynamic_shapes` + `torch.export.Dim`** を使う。
3. **size-1 次元は Dim 指定でも固定される**（torch.export の仕様）。batch を動的にするには
   **example 入力を B>1（=B=3 capture）**で export する。
4. onnxruntime は .onnx を C++ で読むため、Irodori env(protobuf 3.19.6) でも
   `uv run --with onnxruntime` で動く（python `onnx` を import しなければ衝突しない）。

### 成果物（build/）— 2 パターン揃え済み
**Option B（context_kv_cache なし, 6 入力, `export_dit_step.py`）**
- `dit_step_dyn_fp32.onnx`（B,T 動的）/ `dit_step_b3_fp32.onnx` / `dit_step_b1_fp32.onnx`（固定）

**Option A（context_kv_cache 入力, 54 入力, `export_dit_step_kvcache.py`）**
- `dit_step_kv_dyn_fp32.onnx`（B,T 動的）/ `dit_step_kv_b3_fp32.onnx`（固定 B=3）

- いずれも `.onnx.data`（fp32 weights）外部参照。dynamic は example B=3 で export（size-1 回避）。

### full-loop 検証（両パターン, num_steps=8, seed=1234, no_ref）
| pattern | onnx | fwec calls | SNR | corr |
|---|---|---|---|---|
| B (nokv) | dit_step_dyn_fp32.onnx | 8/8 | 67.4 dB | 1.000000 |
| A (kv)   | dit_step_kv_dyn_fp32.onnx | 8/8 | 66.1 dB | 1.000000 |

`validate_full_loop.py --mode {nokv,kv}` で再現。両者とも実 runtime の sampling ループに
差し込んで波形一致（CFG の B=3/B=1 両方を通過）。

## Pulsar2 導入（2026-05-23, 完了）

- 取得: HF `AXERA-TECH/Pulsar2` の `6.0/ax_pulsar2_6.0.tar.gz`(5.48GB) を DL →
  `docker load` → image `pulsar2:6.0`(10.6GB)。ENTRYPOINT=`/bin/bash`, WORKDIR=`/data`。
  実行例: `docker run --rm -v "$PWD:/data" pulsar2:6.0 -c "pulsar2 version"`（commit 48520c11）。
- **target_hardware enum(6.0): AX650, AX620E, M76H, M57, AX615, AX637 — AX8850 は無い**。
  → **AX8850 は `--target_hardware AX650` で変換**（NPU IP は AX650 系で共通。ax-llm 等も AX650 指定）。
  npu_mode は AX650 で NPU1/NPU2/NPU3（既定 NPU1）。
- 変換: `pulsar2 build --target_hardware AX650 --input m.onnx --output_dir out --config c.json`。
  **量子化(PTQ)必須**、float-only build は無い。事前に onnxsim 必須（`--onnx_opt.enable_onnxsim true` でも可）。

### config（multi-input / raw tensor / W8A16）の要点
- raw float 入力: `input_processors[].src_format="RAW"`, `src_dtype="FP32"`（画像前処理を skip）。
- multi-input calib: `quant.input_configs[]` に tensor_name ごと `calibration_format="Numpy"`,
  `calibration_dataset`(tar.gz), `calibration_size`。
- W8A16: `layer_configs[]` で `data_type="S16"`(activation) + `weight_data_type="S8"`、
  または op_type 単位指定。`precision_analysis=true` で層別影響を事前評価可。
- DiT 固有の懸念（要実地確認）: bool mask 入力(text_mask/speaker_mask)、`IsNaN` op、opset18、
  Option A の 54 入力、固定 shape 必須（dynamic NPU 不可 → B=1 や B=3 固定で build）。

## Pulsar2 build: 実地で潰した op 非対応チェーン（2026-05-23）

`pulsar2 build --target_hardware AX650`（W8A8 既定, 実 activation 40 サンプル calib, b1 固定 shape）を
反復し、変換でしか出ない blocker を順に解消。すべて **export 時 monkeypatch** で対応し本体は無改変、
数値等価（単発検証 ~2.4e-6 を維持）。`scripts/rope_export_patch.apply_export_patches(model)` に集約。

| # | blocker（Pulsar2 エラー） | 原因 | 対処（export patch） | 結果 |
|---|---|---|---|---|
| 1 | calib `dtype mismatch`（uint8≠bool）| mask を uint8 保存 | calib は **bool のまま保存** | 解消 |
| 2 | `dont support IsNaN opr` | SDPA bool mask 分解の NaN ガード | bool mask→**additive float mask**（finfo.min 加算, 全マスク行なしで等価）| IsNaN 12→0, Where 13→0 |
| 3 | `Quant doesn't support Reciprocal`（RMSNorm）| `torch.rsqrt`→Sqrt+Reciprocal | RMSNorm を **sqrt/div** に（rsqrt(v)=1/sqrt(v)）| Reciprocal 73→24 |
| 4 | 同上（残 24 = 12block×2）| LowRankAdaLN 内 inline rsqrt | LowRankAdaLN も sqrt/div に | Reciprocal 24→0 |

→ 現状の export-safe op: RoPE は Cos/Sin（実数）, norm は Sqrt+Div, mask は Not+Cast+Mul+Add。
  IsNaN/Reciprocal/complex は全て除去済み。これらは Option A/B 両 ONNX 共通で必要
  （他の ONNX も apply_export_patches 適用で再 export すること）。

## .axmodel 変換成功（Option B b1, 2026-05-23）

op 非対応チェーン解消後、build #4 が成功:
- 出力: `build/axmodel_b1/compiled.axmodel` = **379MB**（fp32 1.4GB → INT8 ~3.7x 圧縮）
- **全グラフが単一 NPU subgraph に**コンパイル（CPU fallback op 無し）。
- per-step MACs ≈ 37.4 GFLOPs、max_cycle ≈ 52.4M。
- target AX650（=AX8850）, 既定量子化（U8 系=実質 W8A8）, 実 activation 40 サンプル calib。

### 重要な未検証事項（次にやる）
1. **量子化精度は未検証**。`.axmodel` は構造的に有効・コンパイル成功だが、INT8 量子化が
   音質を保つかは別問題。`build/axmodel_b1/quant/quant_axmodel.onnx`（量子化済 ONNX）を
   onnxruntime で fp32 と比較 or full-loop で wav 比較して quant error を測ること。
2. **既定は W8A8 相当**。transformer は **W8A16** が本命（FINDINGS 量子化方針）。
   `layer_configs` で S16 activation 指定 + `precision_analysis=true` で層別評価して作り直す。
3. **Option A（KV cache 54 入力）未変換**。同じ patch 済 ONNX で build を試し、A の 54 入力が
   Pulsar2 を通るか・性能差を確認。
4. b1 固定（B=1,T=119）。実機運用は CFG の B=3/B=1・可変 T の固定 shape 戦略を確定要。

## few-step sampling スイープ（2026-05-23, `scripts/step_sweep.py`）

32-step linear 基準への log-mel L1（同 seed/text, trim_tail off, 4090）。sample_rf は step に線形。

| steps | linear L1 | sway L1 | sample_rf |
|---|---|---|---|
| 4 | 2.48 | 1.12 | ~75ms |
| 6 | 1.54 | **0.87** | ~105ms |
| 8 | 0.70 | **0.56** | ~135ms |
| 12 | 0.35 | 0.35 | ~190ms |
| 16 | 0.26 | 0.21 | ~250ms |

結論:
- **sway schedule が全 step で linear より良い**（`t_schedule_mode="sway"`。パラメタだけの無料の改善）。
- **実用下限は sway で 6〜8 step**（8 以下で L1 が増え始め、6→4 で急増）。
  16→8 で **NPU 呼び出し ~2x 削減**、32 比なら 3〜4x。最終は wav 試聴で確定（`build/fewstep/`）。
- DiT step×N が支配的コストなので、step 半減はそのまま latency 半減。**最優先で効くレバー**。

## Option A axmodel build: 未完（dead-input 問題, 2026-05-23）

Option A の ONNX(54入力, B=1) は patch 込みで再 export 済・full-loop 検証済だが、Pulsar2 build が
**dead input** で詰まる: KV cache 提供時 `text_state`/`speaker_state` は未使用 →
- config 54入力: optimize_onnx の onnxsim が 2 個 prune → 「inputs num 52 and 54 not match」
- config 52入力: quant fetch_data は prune 前(54)を見て「config of input(text_state) doesn't exist」
2 ステージで入力数が食い違う。**修正方針: Option A ラッパから text_state/speaker_state 入力を外し、
内部で zeros 定数として構築**（k_text/k_spk の seq 長から shape 導出）→ ONNX を 52 入力に固定して
両ステージを一致させる。Option B(6入力) は影響なし＝既に変換成功。

## Option A axmodel 変換成功（2026-05-23）

dead-input を根治して変換成功:
- 修正: Option A ラッパから `text_state`/`speaker_state` 入力を外し内部 zeros 化
  （**zeros-vs-real 誤差 0 = 真に未使用と確認**）。ONNX を 52 入力に固定し Pulsar2 の 2 ステージ食い違いを解消。
- 出力: `build/axmodel_kv_b1/compiled.axmodel` = **338MB**（Option B 379MB より小 = KV 射影重みがグラフ外）。
- 単一 NPU subgraph、**max_cycle 49.7M（Option B 52.4M より少）** = A は per-step の KV 射影計算を省く分軽い。
  → CMM 前提の帯域メリットに加え、コンパイル後サイクルでも A 有利を確認。

## 量子化精度の検証メモ

- `quant/quant_axmodel.onnx` は **Axera 独自 op（AxConcat 等）**を含み、**vanilla onnxruntime では読めない**。
  → onnxruntime での fp32 比較は不可。正しくは Pulsar2 の **`precision_analysis: true`**（層別 float vs quant
    類似度レポート）or 実機 / pulsar2 シミュレータで測る。
- 副産物: quant グラフで mask 入力は **uint8**（`ones/text_mask/speaker_mask: tensor(uint8)`）と確定
  → deploy doc の「mask uint8」想定が裏取りされた。
- W8A8 既定の音質可否は **precision_analysis で要計測**（次アクション）。transformer 本命は W8A16。

## 最適化バッチ結果（2026-05-23）

### 量子化精度（W8A8）— 検証済・十分
`precision_analysis: true` で再ビルド（`build/axmodel_b1_prec/quant/debug/precision_analysis_table.txt`）:
- **最終出力 `v_pred`: cosine 0.99972 / MSE 0.00062**（fp32 vs W8A8）→ 量子化は DiT 出力をほぼ保つ。
- 例外: RoPE の `cos` 中間 tensor が U8 で cosine 0.62 と劣化するが、最終 v_pred には影響なし
  （気になるなら freqs を高精度に残す手はある）。
- 注: `quant/quant_axmodel.onnx` は Axera 独自 op(AxConcat 等)で **vanilla onnxruntime 不可**。

### fp16 vs fp32 の可聴比較（軽量化代理）
`scripts/compare_onnx_wav.py` で end-to-end wav 生成（DiT のみ精度差, T=95 text）:
- **mel-L1 0.0576 / 波形 SNR 44.3 dB** → 可聴閾超えで実質聞き分け不可。
- `build/compare/dit_{fp32,fp16}.wav`。再生は `paplay <wav>`。

### CPU オフロード削減（#1）
| モジュール | ONNX | axmodel | 備考 |
|---|---|---|---|
| **TextEncoder** | ✅(4.2e-5) | ✅ build 成功（max_cycle 17.4M, 単一 NPU subgraph）| DiT 同型, 既存 patch で通過 |
| **DACVAE decoder** | ✅(6.3e-6) | ❌ **NPU 不可** | Snake tiler 制約（下記）→ **CPU 据え置き** |
| speaker_encoder / duration | 未 | 未 | 小さいので優先度低（残 CPU は軽い）|

→ 重量級 3 段（TextEncoder / DiT / DACVAE）を NPU 化すれば、残 CPU は tokenizer・小 encoder・
  RF グルー・watermark のみ＝軽量。

### DACVAE が NPU 不可な理由（Snake tiler 制約）
- Pulsar2 は Snake を `AxQuantizedSnake` として認識・量子化までは成功（ONNX parse / PTQ OK）。
- だが NPU backend の **tiler が大きな時間長で Snake を分割できず `NoTilerException`**。
  失敗 shape: NPU1 で `(1,384,14280)`、NPU3 で `(1,192,114240)`（vocoder の upsample 後半の巨大 tensor）。
  npu_mode を上げても解消せず（より進んだ段で同種の壁）。
- Snake は時間方向に pointwise（`x + (1/α)sin²(αx)`）なので原理的には T 分割可能 → **Pulsar2 のツール側ギャップ**。
- **対処（将来）**: ①DACVAE は CPU(A55) 据え置き（現実解, deploy doc の既定）。
  ②host で latent を T-chunk 分割し overlap-add で decode（conv 受容野の境界処理が要る）。
  ③Snake を tile 可能な活性に置換（要再学習, 非現実的）。④Axera に Snake tiler 対応を要望。
- 結論: 重量級では **DiT と TextEncoder を NPU 化**、**DACVAE は CPU**。watermark/tokenizer/小 encoder も CPU。

### chunk 分割で tile できる T は無い（x86 先行検証, 2026-05-23）
overlap-add の前段として「tile が通る chunk T」を探索（`dacvae_T{16,32,48,64,96}.onnx` を NPU3 build）:

| chunk T | 失敗 op / shape | 失敗種別 |
|---|---|---|
| 16 / 32 / 48 | 先頭 Snake `(1,1536,T)` | `OpBuildException`「broadcast dim 2: T vs 1536」|
| 64 / 96 / 119 | 終段 Snake `(1,192, T×960)`（≥61440）| `NoTilerException` |

→ **T<64 は Snake の broadcast バグ、T≥64 は Snake tiler 限界**で、2 つの失敗域が全 T を覆う＝
  **どの chunk でも build 不可**。64 は NPU tile 幅の閾値らしく、未満だと 1536ch の α broadcast が壊れ、
  以上だと upsample 後の時間長が tile 不能。
→ **chunked-DACVAE-on-NPU は Pulsar2 6.0 では不可**。DACVAE は CPU 据え置きが確定。
  打開には Axera 側の Snake op 修正（小 T broadcast / 大 T tiler 両方）が必要。
  代替: vocoder を tile 可能な op 構成のものに差し替え（要再学習）or CPU 最適化（A55 マルチスレッド）。

### axmodel 一覧（build/）
| pattern | file | size | max_cycle |
|---|---|---|---|
| DiT Option B | axmodel_b1/compiled.axmodel | 379MB | 52.4M |
| DiT Option A (KV) | axmodel_kv_b1/compiled.axmodel | 338MB | 49.7M |
| TextEncoder | axmodel_textenc/compiled.axmodel | — | 17.4M |
| DACVAE decoder | axmodel_dacvae/ | ⏳ | — |

## 次アクション候補

- [ ] DiT step ラッパ Module を書き、fp16 ONNX を実 KV cache 入力で export
- [ ] text_encoder / speaker_encoder / duration_predictor / DACVAE decoder を個別 ONNX 化
- [ ] onnxruntime で PyTorch 出力と数値一致を検証（export 正当性）
- [ ] Pulsar2（AX8850）導入、**W8A16** で量子化（DiT は実 activation で calibration）

## 量子化方針（調査済, 2026-05-23）

### 結論
- **AX8850 の transformer 量子化パスは W8A16 / W4A16（activation は 16bit 固定）。**
  Axera model zoo（Qwen2.5, MixFormerV2, LivePortrait 等）は全て w8a16 / w4a16。
  NPU に Transformer 加速ユニットがあり、効くのは W8A16。
- AX8850 NPU 実スペックは **72 TOPS@INT4 / 18 TOPS@INT8**（"24 TOPS" は実用値表記）。
- ベースラインは **W8A16**（CNN 向け W8A8 ではない。RF-DiT は transformer なので W8A16 パス）。
- サイズ/帯域を削るなら **W4A16**（重み -75%）。エッジは帯域律速になりやすく latency も縮みやすい。

### 各方式の評価

| 方式 | 重み | act | Axera サポート | DiT 品質 | サイズ |
|---|---|---|---|---|---|
| **W8A16** | 8bit | 16bit | ◎ zoo 標準 | ◎ 安全 | 中 |
| **W4A16** | 4bit | 16bit | ◯ zoo 実績 | ◯ 要 calib | 小(-75% 重み) |
| W4A8 | 4bit | 8bit | ✕ Pulsar2 標準外 | △ 研究レベル(劣化大) | 小 |
| W4A4 | 4bit | 4bit | △ 72TOPS 道だが | ✕ ハイリスク | 小 |

- **act 16bit は DiT に好都合**: 敏感な AdaLN 変調・timestep embed のダイナミクスを保てる。
  W4A8 / W4A4 はまさにここを潰すので不利。
- **Lite の INT4 は移植不可**: Triton fused W4A16（GPU 用）。AX8850 では Pulsar2 で再量子化が必要。
- DiT 量子化は **実 activation での calibration 必須**（ランダム入力だと品質劣化）。
  必要なら norm/embed/AdaLN を高精度に残す mixed precision。

### 推奨順序
1. **W8A16 で end-to-end を通して実測**（Transformer 加速ユニット活用、act16bit で品質安全）。
2. サイズ/帯域が足りなければ **W4A16**（実 activation calibration、音質確認）。
3. W4A8 / W4A4 は Pulsar2 標準外＋品質リスクで当面見送り。

> 訂正: 旧版で「INT8 ベースライン（W8A8）」と書いたが、transformer は W8A16 が正。
