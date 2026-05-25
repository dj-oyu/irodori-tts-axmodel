# Irodori-TTS-500M-v3 実構造メモ（AX8850/Pulsar 前処理の土台）

baseline 推論を 4090 で通し、`scripts/inspect_model.py` で実体をダンプした結果。
配置: `build/model_introspection.json` に全文、`build/baseline_noref.wav` に出力例。

---
## TL;DR（v0.2, 2026-05-23 時点の到達点）

- ターゲット: M5Stack AI Pyramid Pro = **AX8850**（Pulsar2 では `--target_hardware AX650`）。
- **NPU 化済み（axmodel）**: DiT（Option B 379MB / Option A 338MB）, TextEncoder。全て単一 NPU subgraph。
- **DACVAE decoder**: 旧「NPU 不可・CPU 確定」は **訂正**。Snake を `sin²(αx)=(1−cos(2αx))/2` に
  export 時書き換えると Pulsar2 の Snake 融合を回避でき、**T=119 全体が単一 NPU subgraph でビルド成功**
  （x86 sim で構造確認済, 87.5MB）。ただし**量子化品質・実機 latency は未検証**（末尾 DACVAE 節）。
- **数値検証**: ONNX vs PyTorch ~1e-6、full-loop SNR 66–67dB、fp16 wav SNR 44dB。
  ⚠️ 旧記載「W8A8 v_pred cosine 0.9997」は **PerLayer 値で誤り**。実 end-to-end は **W8A8≈0.45 / W8A16≈0.925**
  （x86 sim で実機 0.16 破綻を再現・原因特定済。末尾「x86 シミュレータで…」節）。
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

### 量子化精度（W8A8）— ⚠️ この節の結論は誤り（2026-05-23 夜の x86 sim 検証で訂正）
`precision_analysis: true` で再ビルド（`build/axmodel_b1_prec/quant/debug/precision_analysis_table.txt`）:
- **最終出力 `v_pred`: cosine 0.99972 / MSE 0.00062**（fp32 vs W8A8）→ 量子化は DiT 出力をほぼ保つ。
- 例外: RoPE の `cos` 中間 tensor が U8 で cosine 0.62 と劣化するが、最終 v_pred には影響なし
  （気になるなら freqs を高精度に残す手はある）。
- 注: `quant/quant_axmodel.onnx` は Axera 独自 op(AxConcat 等)で **vanilla onnxruntime 不可**。

> **訂正（重要）**: 上の 0.99972 は `precision_analysis_method=**PerLayer**`＝各層を **float 入力で単独**に
> 量子化評価した値で、**累積（end-to-end）誤差ではない**。実際に量子化グラフ／コンパイル axmodel を
> `pulsar2 run` で通すと **end-to-end cosine ≈ 0.45（W8A8）** しか出ない（下記 x86 sim 節）。
> 「W8A8 で十分」は誤り。transformer は本 FINDINGS の量子化方針どおり **A16 が必要**だった。

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
> **⚠️ この節の「NPU 不可」結論は訂正済**（2026-05-23 夜）。原因を Snake **融合**op と特定し回避策を
> 検証＝**ビルドは可能**になった。下の分析は失敗の機序として正しいが「不可」は誤り。末尾「DACVAE を
> NPU 化できた」節を参照。原因の核心: tiler 限界そのものより、Pulsar2 が Snake パターンを
> **単一 `AxQuantizedSnake` op に融合**し、その融合 op の tiler/quantizer だけが壊れている点。
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

---

## 実機検証で判明: 出荷 axmodel(kv, W8A8) は数値的に壊れている → x86 で要再検証（2026-05-23）

> **解決済（同日夜）**: x86 sim で原因特定 → **W8A8 の A8 累積飽和**が主因（engine 版不一致は副次）。
> 詳細と修正レバーは末尾「x86 シミュレータで実機破綻を再現＆原因特定」節。以下は当時の切り分けメモ。

実機 AX8850 で初めて DiT axmodel の**数値正しさ**を確認したところ不合格だった。**ロード・shape・latency は正常だが出力が誤り。**

### 症状（実機, 単一ステップ）
- 同一入力 `(x_t, t=0.8, cond KV cache, masks)` を **torch fp32 `forward_with_encoded_conditions(context_kv_cache=...)`** と
  **`axmodel_kv_b1`** に与えて直接比較 → **cosine = 0.16**（torch v_pred std 0.93 / npu 0.62）。
- レイアウト仮説（転置・reshape `(1,32,119)`・seq/latent 逆順 計6種）すべて ≒0 → **並び替えではない**。
- W8A8 の許容（通常 cosine ~0.99）に程遠い → 量子化の劣化ではなく **ビルドが誤っている**。
- torch fp32 サンプリング → 同じ onnx DACVAE で復号すると**明瞭な音声**が出る（CFG有り）。
  → 条件付け・CFG・`build_context_kv_cache`・DACVAE 経路はすべて正しく、**壊れているのは NPU DiT のみ**。

### 原因の所在（未確定。x86 で切り分け）
- 出荷した Option A (kv) axmodel は `pulsar_configs/kv_b1_min.json`（**`precision_analysis: false`**）でビルドされ、
  **数値検証が一度も行われていなかった**（`b1_precision.json` は Option B 版用で別物）。
- 校正データ自体は `dump_calibration.py` の実 activation キャプチャで適正。
- 疑い: ① W8A8 が DiT の AdaLN/timestep ダイナミクスを潰した（本 FINDINGS の「DiT は W8A16 が正」と整合）、
  ② `transformer_opt_level=0`、③ **Pulsar2 6.0 でコンパイル vs 実機 engine 2.12.0s の版不一致が数値実行に影響**（ロードは red herring と判定済だが数値は未検証）。

### x86 単独でできる検証（NPU 実機不要）
1. **`pulsar2 run`（x86 シミュレータ）** で `compiled.axmodel` を回し、`onnxruntime` の fp32 ONNX 出力 / torch と cosine 比較。
   - sim も cosine 低い → **Pulsar2 ビルド/量子化のバグ**（→ 下記再ビルド）。
   - sim は cosine 高い のに実機が低い → **engine 版不一致が真因**（実機 engine 更新 or 実機版に合わせて再build）。
2. **`precision_analysis: true` で Option A(kv) を再ビルド**（`kv_b1_min.json` に追加）→ 層ごとの float vs 量子化 cosine を取得し崩れる層を特定。
3. 直し方の候補（本 FINDINGS の推奨順に従う）: **W8A16**（act16bit で AdaLN/embed を保つ）、必要なら norm/embed/AdaLN を
   `layer_configs` で高精度に残す mixed precision、`transformer_opt_level≥1`。**per-step cosine ≥0.99 を確認してから配布**。
   `axmodel_textenc` も未検証なので同様に確認。

### 検証ベクトル（実機側 `e2e_demo/` で生成済。x86 へ持ち出すと sim==実機 / sim==torch が一発で分かる）
- `diag_step_torch.py` が出す `step_ref.npz`（`x_t`, `t`, `v_torch`=fp32正解）
- 実機 axmodel の実出力 `v_npu.npy`（cosine 0.16 のもの）
- `a_build_cond.py` が出す cond KV + masks（`e2e_cond.npz`）
- 比較用 ONNX `dit_step_kv_b1_fp32.onnx`(+`.onnx.data`) と calib は x86 ビルド環境に既にあるはず。

> 補足: 実機側の最小 end-to-end 検証コードは `e2e_demo/`（A=torch 条件付け / B=NPU Euler+CFG / C=onnx DACVAE）。
> torch DiT 経路なら実機でも音声は出る（A55 CPU で低速）。NPU DiT を直すのが本筋。

---

## x86 シミュレータで実機破綻を再現＆原因特定（2026-05-23 夜, NPU 不要）

`pulsar2 run`（x86 シミュレータ。NPU 実機なしで axmodel/quant ONNX を計算実行できる）で実機の
cosine 0.16 破綻を **x86 上で再現し、原因を W8A8 の累積量子化劣化に特定**した。engine 版不一致は
**主因ではない**（コンパイル済 axmodel が x86 sim でも壊れる）。

### 再生成パイプライン（すべて再現可能）
1. `capture_dit_inputs.py`（Irodori env, CUDA）→ `build/dit_step_b1_fp32.ref.pt`（実 DiT 入力＋torch fp32 v_pred, T=119）
2. `dump_calibration.py --kv`（Irodori env）→ `build/calib_kv_b1/`（54 tar.gz × 40 実 activation サンプル）
3. `export_dit_step_kvcache.py`（preprocess env）→ `build/dit_step_kv_b1_fp32.onnx`（52 入力, ONNX vs torch **2.1e-6**）
4. Pulsar2 build（docker `pulsar2:6.0`）: configs は `build/pulsar_configs/{kv_b1_precision,kv_b1_w8a16}.json`
5. 入力 .bin 生成は `$JOB/prep_sim_inputs.py`（masks=uint8, 他=fp32）。`pulsar2 run --model <axmodel|quant.onnx> --input_dir <dir> --output_dir <dir> [--mode Reference]`

### 計測結果（cosine vs torch fp32, 同一 T=119, no_ref）
| 計測 | cosine | 出力 std | 意味 |
|---|---|---|---|
| **precision_analysis PerLayer** v_pred (W8A8) | 0.9997 | — | **各層 float 入力で単独評価。累積でない＝誤解の元** |
| onnxruntime fp32 ONNX(自前 .bin) vs torch | **1.00000** | — | .bin 値は正しい（入力misformat を否定）|
| `pulsar2 run` quant ONNX, 自前入力 | 0.46 | 0.57 | 真の end-to-end（W8A8）|
| `pulsar2 run` **compiled axmodel**, 自前入力 | **0.45** | 0.57 | 実機で動くもの。実機 0.16 と同クラス |
| `pulsar2 run` quant ONNX, calib サンプル | 0.61 | 0.57 | calib データでも壊れる |
| `pulsar2 run` compiled axmodel, calib サンプル | 0.62 | 0.56 | 〃（量子化したデータ自体で破綻）|
| 実機 compiled (既出) | 0.16 | — | 破綻 |

### 切り分けで潰した仮説
- **入力 misformat ではない**: onnxruntime が自前 .bin から torch を cosine 1.00000 で再現。
- **コンパイル/codegen バグではない**: quant ONNX 単体（コンパイル前）が既に 0.46。compiled とほぼ一致。
- **engine 版不一致が主因ではない**: コンパイル済 axmodel が **x86 sim でも 0.45** で壊れる（実機 0.16 はこれに
  engine 差が上乗せされた程度。engine 仮説は **副次要因に格下げ**, 実機再検証で確認すること）。
- **PerLayer ≠ end-to-end**: precision_analysis PerLayer(0.9997) と実 end-to-end(0.45) の乖離が核心。
  Pulsar2 には `precision_analysis_method=EndToEnd`, `precision_analysis_mode=NPUBackend` がある（こちらを使え）。

### 原因: A8（activation 8bit）の累積飽和
W8A8 は出力 std が **半分**（0.57 vs fp32 1.06）に潰れ cosine 0.45。**W8A16 で 0.45→0.925 に回復し
std も 0.98 に戻る**（`build/axmodel_kv_b1_w8a16/`, max_cycle 53.4M, EndToEnd v_pred 0.889）。
→ 破綻の主因は **activation 量子化（A8）**。FINDINGS 量子化方針「transformer は W8A16 が正」が裏取りされた。

### 有力な根本原因候補（次パスで最優先に検証）: additive mask の `finfo.min`
`rope_export_patch._sdpa_additive_mask` は masked 位置に **`torch.finfo(dtype).min`（≈ -3.4e38）** を加算する。
このスコア tensor（0 と -3.4e38 が混在）を量子化すると scale が天文学的になり、本来 O(1〜10) の実スコアが
1 bin に潰れて softmax が壊れる。EndToEnd 表で **AxSoftmax cosine 0.496** や多数の linear 劣化が出るのと整合。
本 FINDINGS が当初挙げた **`(~mask)*-1e4`** 程度の穏当な負値に変えれば A8 でも量子化可能になる可能性が高い。

### 推奨レバー（ランク順, ≥0.99 を狙う別パスで）
1. **additive mask を `finfo.min`→`-1e4`〜`-3e4` に変更**して re-export（飽和の発生源を断つ。最安・最有望）。
2. **W8A16 をベースライン化**（A16 で 0.925。1 と併用で底上げ）。
3. **calibration_method を MinMax→MSE/Percentile** または **smooth_quant 有効化**（残る linear の outlier 対策）。
4. 最悪層だけ `layer_configs` で **FP32/高精度 mixed precision**。

### まだ未検証（正直な限界）
- ≥0.99 達成後に **実機で音質が戻るか**（sim cosine と可聴品質の対応）。
- 実機 engine **2.12.0s** が x86 sim に対し追加で ~10% 劣化を載せないか（実機再計測要）。
- 計測は単一 step / B=1 / T=119 / no_ref。複数 step・CFG B=3・ref 条件は別途。

### 成果物（build/, すべて .gitignore 対象＝再生成前提）
- `dit_step_kv_b1_fp32.onnx`(+.data), `calib_kv_b1/`, `sim_kv_inputs/`(自前 .bin + `v_torch_ref.npy`)
- `axmodel_kv_b1_prec/`（W8A8 + PerLayer 表）, `axmodel_kv_b1_w8a16/`（W8A16 + EndToEnd 表）
- `sim_kv_out/`,`sim_kv_quant_out/`,`sim_calib_*`,`sim_kv_w8a16_out/`（各 `pulsar2 run` の v_pred.bin）
- config: `build/pulsar_configs/kv_b1_precision.json`, `kv_b1_w8a16.json`（git 追跡）

---

## DACVAE を NPU 化できた（Snake cos 書き換え, 2026-05-23 夜, x86 sim 検証）

旧結論「Snake tiler 制約で DACVAE は NPU 不可・CPU 確定」は**誤り**。原因を特定し回避して
**T=119 全体を単一 NPU subgraph でビルド成功**（x86 `pulsar2 run` で構造確認）。詳細レポート:
`docs/dacvae_npu_research.md`、export スクリプト: `scripts/export_dacvae_variants.py`。

### 真因: Snake の「融合 op」
Pulsar2 6.0 は ONNX の `Mul(α,x)→Sin→Pow(·,2)→Mul(1/α,·)→Add(x,·)` を**単一 `AxQuantizedSnake` に融合**し、
その融合 op の tiler/quantizer **だけ**が壊れている（T<64 broadcast バグ / T≥64 NoTilerException は両方とも
融合 op 起因）。Snake は時間方向 pointwise なので**素の primitive なら tile 可能** = ツール側の融合バグ。
- 傍証: α を `(1,C,T)` に事前 expand しても**まだ融合**し `Quant don't support the shape(49152) of snake alpha now`
  で悪化（N2）。→ 罠は ONNX broadcast でなく**融合 op そのもの**。

### 回避策（検証済の fix）= export 時に Snake を三角恒等式で書き換え
`sin²(αx) = (1 − cos(2αx))/2` を使い `Snake1d.forward` を
**`x + (1/(α+1e-9)) · 0.5 · (1 − cos(2αx))`** に monkeypatch（既存 `rope_export_patch` の兄弟）。
`Sin`/`Pow` が消え融合がマッチしない → 素の `Cos/Mul/Sub/Add` が残り tiler が時間分割できる
（致命の `(1,192,114240)` が 6720 幅 ×17 slice に分割されるのをログで確認）。
- **fp32 等価**: 元 Snake と max_abs_err **7.3e-6**（T=119）。
- **build**: `dacvae_T119cos_npu3.json`（NPU3, W8A8, ランダム calib 16）→ `compiled.axmodel` **87.5MB**,
  単一 NPU subgraph, max_cycle **164.3M**, quant graph は **Snake 0 / AxQuantizedCos 29**。

### 未検証（productionize 前に必須）
- **量子化品質**: x86 sim で end-to-end は通るが W8A8 + ランダム calib + OOD 入力で **cosine 0.843 / SNR 4.8dB**
  ＝構造は正しいが品質は低い（DiT と同じ A8 飽和パターン）。**W8A16 + 実 activation calib** で測り直し要。
  ※ DiT の W8A8→W8A16 改善(0.45→0.925)が vocoder(conv+Cos+Mul, 分布が別)でも同等に効く保証はない＝要実測。
- **W8A16 ビルドが本ホスト(15GB)で OOM**（U16 で activation メモリ ~2倍 / 出力 228K サンプル）。
  → **大容量 RAM のビルドホストが必要**。Pulsar2 ビルドは本ホストでは**1 本ずつ**（並列で OOM 実績）。
- **perf: 実機で DACVAE-CPU(A55) は「相当遅い」と判明（ユーザ実機計測, 2026-05-23）**。
  → 旧保留「NPU ~90ms は CPU ~53ms より遅いかも」は撤回（その 53ms は **4090(GPU)** の数字だった）。
  実機 A55 CPU が遅いなら **NPU offload(~90ms 概算) は速度でも勝つ公算大** ＝ NPU 化は速度面でも価値あり。
  最終 go/no-go は NPU 実機 latency 実測だが、**productionize 優先度は上がった**。
- 単一 shape(T=119)/単発 z のみ。可変 T・watermark(LSTM は dynamo で素 op 分解済=別ブロッカではない) は別途。

### 次アクション
1. `scripts/export_dacvae_variants.py` の cos monkeypatch を export-patch 本体に統合し、実 calib + W8A16 で
   再ビルド（大 RAM ホスト）→ wav SNR を CPU/ONNX decoder と比較。
2. ダメなら hybrid 分割（`CompilerConfig.sub_configs` + `start/end_tensor_names`、`SubgraphType.ONNX`=CPU
   subgraph は proto で存在確認済）で Snake 周辺だけ CPU。

---

## DiT W8A16 量子化最適化: 0.93 で頭打ち＋SmoothQuant の罠（2026-05-23 深夜, x86 sim）

W8A16(0.925) を ≥0.99 に詰める試行。**安いレバーは出尽くし、plain W8A16 ≈ 0.93 が天井**と判明。
全数値は compiled.axmodel を `pulsar2 run`(x86 sim) で実行し torch fp32 と比較した cosine（接地真値）。

| config | compiled-sim cosine | 備考 |
|---|---|---|
| W8A8 | 0.45 | A8 飽和で破綻 |
| **W8A16 plain (MinMax, opt0)** | **0.925** | 現状ベスト |
| W8A16 + SmoothQuant + opt2 | 0.233 | 大幅悪化 |
| W8A16 + SmoothQuant (opt0, 旧mask) | 0.210 | opt2 は無関係＝SmoothQuant が犯人 |
| W8A16 + SmoothQuant (mask -1e4) | 0.827 | mask 修正で救済も plain 未満 |
| W8A16 plain (mask -1e4) | 0.932 | mask 修正は plain にほぼ無効(+0.007) |

### 重要な教訓
1. **SmoothQuant は additive mask の `finfo.min`(-3.4e38) に毒される**。SmoothQuant は raw activation
   統計から per-channel scale を計算するため、-3.4e38 が混入すると scale が壊れ 0.925→0.21 に崩壊。
   `IRODORI_MASK_NEG=-1e4`（`rope_export_patch` を env 化）で再 export すると 0.21→0.83 に回復。
   → **PTQ で SmoothQuant/percentile/MSE 等 activation 統計系を使うなら mask を穏当な負値にすること**。
   （-1e4 は fp32 で exp(score-1e4)→0＝完全マスクで出力数値等価、ONNX vs torch 2.1e-6 維持）。
2. **ただし SmoothQuant 自体がこのグラフでは net-negative**（mask 修正後 0.83 < plain 0.925）。不採用。
3. **mask 修正は plain W8A16 をほぼ改善しない**（0.925→0.932）。mask の害は SmoothQuant 限定だった。
4. **transformer_opt_level=2 は単独要因ではない**（smooth と分離して確認済）。
5. activation 精度は Pulsar2 最大の S16 で頭打ち。残差 0.93→0.99 は単純な activation outlier 問題ではない。
   `weight_data_type=FP32` は **Conv 専用**で FullyConnected には無効（重み分離診断は機構的に不可、要検証で確認済）。

### 方針転換（重要）: proxy(per-step cosine) でなく実 wav を測る
**「≥0.99」は W8A8 破綻時に置いた経験則で、本モデルの実測閾値ではない。** per-step cosine 0.93 が
8 step + CFG を通して最終 wav にどう効くかは未測。fp32 ONNX は 1e-6→SNR 67dB だったが、0.93/step の
axmodel が何 dB になるかが**唯一の判断材料**。→ **次アクション: full-loop に `pulsar2 run`(compiled axmodel sim)
を差し込み、wav SNR/mel-L1 を fp32 torch wav と比較**（>40dB なら 0.93 で配布可、低ければ要追加）。

### wav が不足なら（ランク順, 全レバー試さず順に）
1. **AdaRound（GPU, `finetune_epochs=100`, 時間限定）**: A8→A16 で活性は搾り切ったので残差は重み側の可能性。
   +0.02 超なら採用。※既定 epochs=500 は CPU で 82分・GPU34%＝過剰だった。GPU+短epochs で ~15分に。
2. **最悪 FullyConnected 層だけ FP32 mixed precision**（EndToEnd 表の下位5-10層を `layer_name`+`data_type:FP32`）。確実だが FP32 op 増でサイズ/latency 増。
3. **Hadamard 回転前処理**（最終手段, 実装重）。SmoothQuant 失敗は回転失敗を含意しない（別軸）。

### 成果物（build/, gitignore）
- `dit_step_kv_mask1e4_fp32.onnx`（mask -1e4 版, torch 2.1e-6）
- `axmodel_kv_b1_{mask1e4_plain, mask1e4_smooth, smoothonly, gpu_lean}/`（各 compiled.axmodel + EndToEnd 表）
- config: `kv_b1_{w8a16, gpu_lean, smoothonly, mask1e4_smooth}.json`（git 追跡）
- `rope_export_patch.py`: mask 負値を `IRODORI_MASK_NEG` で可変化（既定 finfo.min＝従来挙動）

---

## 実 wav 測定で決着: W8A16 plain は配布不可（per-step 0.93 は品質 proxy として無効, 2026-05-24, x86 sim）

上記「次アクション」を実施。full-loop（runtime の `synthesize`）の DiT step を
`build/axmodel_kv_b1_mask1e4_plain/compiled.axmodel`（W8A16 plain, per-step cosine 0.932）に差し替え、
各 step を docker `pulsar2 run`(x86 sim, NPU 不要) で実行して最終 wav を生成、fp32 torch wav と比較した。

### 条件（torch / axmodel で完全一致）
- text="こんにちは、テストです。"（固定 axmodel の T=119 を満たす唯一の発話）, no_ref, num_steps=8,
  t_schedule_mode="sway", seed=1234, context_kv_cache=True。sr=48000, wav 長 228480 で torch/axmodel 一致。
- fwec calls=8。schedule は call 0–6 が **B=3 (CFG)**, call 7 が **B=1**（t<cfg_min_t=0.5）= 計 **22 回**の
  `pulsar2 run`（各 ~3.7 分, 合計 ~80 分）。B=3 step は KV cache を batch slice して b ごとに 1 回ずつ流す。
- harness: `$JOB/full_loop_axmodel.py`（production 無改変, 新規）。wav 保存は torchcodec(ffmpeg欠)回避で soundfile。

### 配線健全性（48 分の本走前に確認, OK）
**初回 fwec(B=3) の axmodel 出力 vs その live 入力に対する torch fwec 出力**を per-element 比較:
cosine **0.911 / 0.931 / 0.911** = FINDINGS の per-step 0.932 と一致。→ 悪い wav は **harness バグではなく
量子化誤差の累積**。（注: 旧版で `v_torch_ref.npy`(別 x_t の gold) と比較し cosine 0.02 を出したのは
apples-to-oranges な誤チェック。live 入力の torch fwec を都度計算する形に修正済。）

### 結果（W8A16 axmodel wav vs fp32 torch wav）
| 指標 | 値 | 解釈 |
|---|---|---|
| **SNR** | **-4.93 dB** | 誤差が信号より大 = 別物 |
| **波形 corr** | **0.069** | ほぼ無相関 |
| mel-L1（log-mel, n_fft1024/hop256/80mel） | **4.41** | fp32 ONNX の 8 step は SNR 67dB / corr 1.0 だった対比で壊滅 |
| max_abs | 1.33 | — |

**時間ズレ説の否定**: 最適 lag での正規化相互相関 peak **0.10**, エネルギー包絡 corr **0.25**
→ 位相/トリム差ではなく**波形そのものが別物**。包絡 0.25 は「発話/無音の粗い時間構造」だけ僅かに残る程度。
**スペクトル**: fp32 は centroid 7925Hz・<4kHz エネルギー 45%（明瞭な広帯域音声）に対し、axmodel は
centroid **1092Hz**・<4kHz **94%**（低域に潰れた濁音）。人間には「何か喋っぽいが内容の壊れた / 別発話の」音。

### 結論（重要・決着）
- **per-step cosine 0.93 は最終 wav 品質の proxy として無効**。RF の反復精製では各 step の残差が
  次 step の入力条件付けに乗って軌道がドリフトし、0.93/step × 8 step + CFG で wav は完全に崩れる。
  fp32 ONNX(誤差 1e-6) が SNR 67dB だったのと対照的に、0.93/step は SNR **-4.93dB**。
- タスクの判定基準（>40dB ≈ 可聴 OK / 低 SNR ≈ 可聴劣化）に照らし **W8A16 plain axmodel は配布不可**。
- **「≥0.99 を狙う」方針は誤りではなく必須だった**: 0.93 では足りないことが実 wav で確定。上記「wav が
  不足なら」のレバー（AdaRound / 最悪 FC 層 FP32 mixed / Hadamard）が**必要**。次は 1 つ適用→**再度この
  full-loop で wav 実測**（per-step cosine の改善幅でなく wav SNR で判定すること）。

### 成果物
- `build/wav_fp32_ref.wav`, `build/wav_w8a16_axmodel.wav`（試聴用, `paplay <wav>`）
- harness: `/home/exe/.claude/jobs/f7b5721e/full_loop_axmodel.py`、metrics: `build/fullloop_sim/metrics.txt`、
  各 step の `pulsar2 run` 入出力: `build/fullloop_sim/step{NN}_b{B}/{in,out}/`

---

## 突破口: Lite(OneCompression) の量子化レシピを解析（2026-05-24）

W8A16 plain が wav SNR -4.93dB で破綻（per-step cosine 0.93 は proxy にならず、RF が誤差増幅）。
「なぜ失敗」の答えと突破口を **Lite の公開 int4 ckpt メタデータ**から得た。

### Lite の実レシピ（`kizuna-intelligence/Irodori-TTS-500M-v3-int4` の safetensors __metadata__）
- 量子化 = **AutoGPTQ-v1, 4-bit, group-wise(groupsize 32), GPTQ**（`irodori_tts_lite/quant_utils.py`=OneComp vendored）。
  **回転(Hadamard)も fine-tune も無し**＝同じ base 重みを PTQ しているだけ。「音質ほぼ劣化なし」。
- **4bit 量子化する層(235)**: `blocks.N.attention.{wq,wk,wv,wo,gate,wk_text,wv_text,wk_speaker,wv_speaker}`
  と `blocks.N.mlp.{w1,w2,w3}`（DiT + text/speaker encoder の全 block の attention/MLP）。
- **fp16 で残す層**（量子化しない＝感度高）: **LowRankAdaLN 投影 / cond_module(timestep embed) /
  in_proj・out_proj / norms / duration_predictor**。text_embedding は group-wise embedding 量子化。

### 含意（重要）
1. **モデルは綺麗に量子化可能**（Lite が証明）。"intrinsically 無理" は誤り。
2. **感度の高い層 = AdaLN・timestep cond・proj/norm**（拡散 PTQ 文献と一致）。これらを高精度に残すのが鍵。
3. **Lite の効く要素のうち Pulsar2 に移植できるもの/できないもの**:
   - ✅ 感度層を高精度に残す → Pulsar2 `layer_configs data_type=FP32`（mixed-precision）で移植可。
   - ❌ bulk の **group-wise(gs32)** → Pulsar2 に group-wise weight quant が**無い**（proto 確認済）。
     bulk は per-channel S8 になる（group より粗いが levels は多い）。

### 次の実験（Lite 由来の principled mixed-precision）
Pulsar2 で **AdaLN + cond_module + in/out_proj + norms → FP32**, attention/MLP → S8A16, +AdaRound,
**→ wav テスト**（per-step cosine でなく wav SNR で判定）。
- 成功 → 出荷可能レシピ。group-wise 無しでも per-channel S8(bulk)+FP32(感度層) で足りると判明。
- 失敗 → group-wise の欠如が真のブロッカ＝Axera 待ち or DiT は NPU 外（fp32/CPU）。TextEncoder/DACVAE に注力。
- 実装: Lite の torch 層名 → 我々の ONNX node 名へのマッピングが要（dynamo export で改名済）。
  ※ Option A(kv) axmodel では wk/wv_text・wk/wv_speaker は host 側(build_context_kv_cache)で graph 外。

### 重み再構成の実測 → 失敗の真因が判明（2026-05-24, 突破）
base fp32 重みを各方式で量子化し相対 Frobenius 誤差を測定（`hf` で int4 ckpt DL、Lite GPTQ も実測）:

| bulk Linear | per-channel **S8**(Pulsar2) | group-wise **W4**(Lite naive) | Lite **GPTQ**(実) |
|---|---|---|---|
| 代表 relerr | **~0.010 (1%)** | ~0.081 (8%) | ~0.13 (13%) |

- **Pulsar2 の per-channel S8 は bulk 重みを Lite の W4 group-wise より ~8倍 正確**（256 vs 16 levels）。
  GPTQ は weight 誤差を上げてでも output 誤差を下げる手法なので 13% でも「音質劣化なし」。
- → **group-wise の欠如は無関係**。bulk attention/MLP は per-channel S8 で十分（Lite の W4 より良い）。
- → **W8A16 が壊れた真因 = Lite が fp16 で残す感度層（AdaLN/cond_module/timestep/proj/norm）まで量子化したこと**。
  これらは拡散の変調ダイナミクスを担い、量子化すると RF 反復で誤差増幅 → wav -4.9dB。
- **修正は安い**: 感度層は小さい（LowRankAdaLN rank192, timestep embed）。FP32 で残しても size/compute 微増、
  重い attention/MLP は S8A16 のまま。
- **予測レシピ（要 wav 検証）**: AdaLN + cond_module + in/out_proj + norms → FP32, attention/MLP → S8A16。
  Lite が W4+A16+fp16感度層 で動く以上、bulk がより穏当な本レシピは動く公算大。

### 深さ削減（block drop / merge）は不可（2026-05-24, 思い付き検証・短報）
Block Influence(入出力 cos)で低寄与ブロックを推定 → ablation。**fine-tune 無しでは全滅**:
- block drop（identity 化）: 最低BIの block1 単体でも wav **SNR −2dB**（全 drop 組合せ −1〜−2dB）。
- block merge（隣接2ブロックを重み平均→深さ−1）: 同水準 **−0.5〜−2.2dB**（合成≠平均・neuron 非整合）。
- BI は proxy として外れた（最良は block6 で最低BIの block1 でない）。
→ 教訓: **RF 反復が per-step の任意摂動を壊滅増幅**（量子化 −4.9dB / drop / merge すべて同根）。
  学習無しで効くのは per-step ほぼ無損失の近似のみ＝**mixed-precision(感度層FP32)** か **QAT/蒸留**。

## ビルド所要時間の記録（2026-05-24）
mixedp ビルドの実測（`build/axmodel_kv_b1_mixedp`, kv_b1_mixedp2.json: 感度層148層→FP32 / bulk→S8A16,
enable_onnxsim=true, device=cuda:0, host RTX-class GPU + 32core CPU）。docker ログは UTC, ファイル mtime は JST(+1h)。
※ ビルド冒頭の `rm -rf` が前回 build の root 所有物で permission denied → stale dir 混在のため、本ビルドの
単調増加成果物のみ採用。

| フェーズ | 完了時刻(JST) | 区間 |
|---|---|---|
| タスク起動 | 02:30:48 | — |
| frontend (onnx load + onnxsim) | 02:31:39 | ~1分 |
| PTQ 量子化 (GPU calibration) | 02:33:10 (quant_axmodel.onnx) | ~1.5分 |
| precision_analysis / debug dump | 02:42:43 | **~9.5分（最重）** |
| NPU compile + assemble | 02:46:58 (compiled.axmodel) | ~4分 |
| タスク終了 | 02:47:04 | — |
| **総 wall-clock** | | **≈ 16分** |

- compiled.axmodel = **441MB**（純 W8A16 の 338MB より +103MB＝感度148層を FP32 化した分）, **単一 NPU subgraph に fuse**（CPU fallback 無し）, QuantAxModel MACs 38.1G。
- 支配項は **precision_analysis（EndToEnd, NPUBackend mode）の ~9.5分**。本走でなく品質判定のための解析なので、出荷ビルドでは off にすれば総時間は ~6〜7分に短縮可。
- 参考: 以前の AdaRound 入りビルドは 82分（GPU 化前・single-core CPU）で kill した（→ memory: process-intervention-criteria）。GPU calibration + AdaRound 無しの本構成は桁違いに速い。

### wav テスト所要時間
- **無効な初回 run（routing bug）**: 02:47:47→04:02:32 = **約75分**（8 step × CFG 2要素 × pulsar2 run sim ≈ 各3.7分）。
  → ただし `full_loop_axmodel.py` の `run_axmodel_one` が `cd .../axmodel_kv_b1_mask1e4_plain` を**ハードコード**しており
  `AXMODEL_DIR` を無視 → **plain W8A16 を測っていた**（SNR −4.93dB/corr 0.069 が plain とビット一致したのが発覚の端緒）。
  修正: line61 を `dock_path(AXMODEL_DIR)` 化 + `main()` 冒頭に `[config]` print と `assert compiled.axmodel` ガード追加。
- **修正後 run（mixedp 正routing）**: 04:09:22→04:21:26 = **12分で step0 中断**。理由は次節。

### 夜通しビルドの所要時間まとめ（2026-05-24, config-only / GPU calib / precision_analysis on）
| build | 内容 | ビルド時間 | サイズ | 結果 |
|---|---|---|---|---|
| w8a16_mse | calib MinMax→MSE | 15.5分 | 339MB | seed 悪化・棄却 |
| cosfp32 | Cos/Sin op_types→FP32 | 14.6分 | 339MB | silent 無視（no-op） |
| cosu16 | node_cos→U16 | 14.4分 | 339MB | **cos seed 0.63→1.0・採用** |
| (wav) cosu16 full-loop | 8step×CFG sim | 74.0分 | — | **SNR −1.96dB / corr 0.31**（plain −4.93/0.069 から大幅改善も未出荷） |
- config-only ビルドは GPU calib + precision_analysis 込みで **~15分**で安定。wav 本走が律速（74分）。
- cosu16 のサイズは baseline とほぼ同じ（1 op の精度上げは無視できる増分）。

## mixed-precision(感度層FP32) は per-step を *悪化* させた（2026-05-24, 仮説に反する結果）
修正済 full_loop を mixedp（routing を `[config]` ログで確認済）で再走 → **step0 の wiring チェックで abort**。
- wiring: axmodel-vs-torch 各CFG要素 cosine = **0.657 / 0.669 / 0.657**（plain W8A16 は ~0.93）。
- つまり **感度層を FP32 にしたら per-step が 0.93→0.66 に悪化**（FP32化は精度を上げるはずなので直感に反する）。
- routing は確定的に mixedp（`[config] AXMODEL_DIR=...axmodel_kv_b1_mixedp`）。1-step 単独比較でも cos(plain,mixedp)=0.756, cos(mixedp,torch)=0.657 < cos(plain,torch)~0.93。→ harness バグではなく**ビルドの実回帰**。
- **FP32層の割当は正しい**ことを確認: 245 FullyConnected のうち FP32=148 = AdaLN-up72 + AdaLN-down144内72 + in_proj + out_proj + cond/timestep + (3840,1280)1本。bulk attention(1280,1280)×61 と SwiGLU-MLP(実体は **3680**,1280)は S8A16 のまま（=狙い通り）。
- **示唆**: Pulsar2 の mixed-precision は FP32↔A16 境界の dequant/requant か、AdaLN 変調の fusion 破壊で、むしろ誤差を増やす疑い。
  「感度層の量子化が真因」という仮説と矛盾 or Pulsar2 の混合精度実装自体が lossy。
- **注意**: per-step cosine は品質 proxy として無効（plain 0.93 でも wav −4.9dB）。0.66 から wav を断定はできないが、
  既に失敗している plain より悪い per-step で 75分を投じる価値は低い → wav 本走は保留し方針を再検討（advisor 相談）。

### 検証2点（advisor 助言）でビルドの実態と真因が判明
**Check1: mixedp グラフの実 FP32 層を直接確認 → 指定が一部落ちていた。**
- mixedp の `quant_axmodel.onnx` で FP32 重みの FullyConnected は **146本**（指定は148）。実体: 72×(192,1280)+72×(1280,192)=AdaLN144 + in_proj + out_proj。
- **落ちた2本 = cond/timestep `(1280,512)` と stray `(3840,1280)`** → S8A16 のまま。W8A16 グラフから採った node 名が mixedp グラフで再生成され不一致（`node_linear` 等が carry-over せず）。**最重要の timestep 条件層が量子化されたまま**だった。
- → mixedp は意図したレシピではない。混合精度の是非を語るには不適格な build。

**Check2: precision_analysis_table（EndToEnd NPUBackend）で真因判明。これが決定打。**
グラフ順（早い op）の量子化ダメージ:
| op | 出力 | quant後 cosine | MSE |
|---|---|---|---|
| `cos` | (1,256) | **0.632** | 0.440 |
| `cat` | (1,512) | **0.789** | 0.220 |
| `silu` | (1,1280) | **0.850** | 0.010 |
| `linear`(cond 512→1280) | (1,1280) | **0.939** | 0.073 |
| AdaLN/attention 大多数 | — | 0.999+ | ~0 |

- **真因の所在は timestep 正弦波埋め込み → cond MLP の経路に局在**（`cos` 0.63 / `cat` 0.79 / `silu` 0.85 / cond-linear 0.94）。
  私が FP32 化した AdaLN 重みは量子化しても 0.999＝**元から無傷**。狙う層を完全に間違えていた。
- ※ EndToEnd は**累積**指標（各 op が上流の量子化誤差を全て継承）なので「1186中971 op が cosine<0.99／深層 linear が 0.0002」は
  **非ゼロの per-op 誤差が深さ方向に累積した数学的帰結であり、80% の op が個別に壊れている証拠ではない**（←初稿の誤読、advisor 指摘）。
  独立な信号は**累積が支配する前の早期 op = timestep 経路**のみ。真因は局在しており分散していない。

### 訂正: mixedp は「混合精度の是非」の検証として不適格（inverted test）
- Check1 の通り、FP32 にすべき cond/timestep (1280,512) が**指定から脱落して量子化のまま**、一方で無傷の AdaLN144 を FP32 化していた。
  = **直すべき層を量子化のまま残し、直す必要のない層を FP32 にした**逆向きの実験。0.66 への悪化は
  「FP32 境界 requant のコストだけ増え、真犯人は壊れたまま」と整合 → **「Pulsar2 の混合精度が本質的に lossy」と結論するのは早計**（wrong-layer 仮説で 1/1 当たり）。

### 次の候補レバー（要ユーザ判断・無人で重ビルドはしない）
DiT-PTQ は「終わった」のではなく、**早期 seed（timestep 経路）を潰す**のが次手。安価な順:
1. **calibration を MinMax → MSE/Percentile**。`cos`(範囲[-1,1], MSE0.44) は外れ値で MinMax が破綻している公算大。早期 seed を底上げ。最安・まず試す。
2. **timestep 埋め込みを host 側で precompute して入力化**（KV cache=Option A と同じ手）。`cos/cat/silu/cond-linear` を NPU 外へ。アーキ的にクリーン。
3. **混合精度を正しく**: cond/timestep(1280,512, **shape 一意**で確実に当たる) を FP32。ただし name-mapping を堅牢化してから（次節）。
4. QAT/蒸留（重い、最終手段） / DiT は fp32-CPU 据え置き＋NPU は DACVAE・TextEncoder に集中。
→ 朝、ユーザと方針決定。1 が最安・最有望。

### レバー1（MSE calibration）は不発・むしろ悪化（2026-05-24 夜, 実測）
`kv_b1_w8a16_mse.json`（calibration_method=MSE, 他は W8A16 同一）をビルド（338MB, 単一subgraph, ~16分）。
precision_analysis で timestep 経路の seed を MinMax と比較 → **全項目わずかに悪化**:
| tensor | MinMax | MSE | Δ |
|---|---|---|---|
| cos | 0.632 | 0.572 | −0.059 |
| sin | 1.000 | 0.952 | −0.048 |
| cat | 0.789 | 0.732 | −0.057 |
| silu | 0.850 | 0.777 | −0.073 |
| cond-linear | 0.939 | 0.899 | −0.040 |

- **解釈**: 変化幅は小さい（−0.04〜−0.07）が方向は一貫して悪化。**MSE は seed を改善しなかった**＝
  range-clipping/外れ値が主因という説と整合しない（MSE は外れ値にロバストなので効くはず）。
  ※ Percentile/KL は未試行なので「calibration では原理的に直せない」とまでは断定しない。ただし
  理論上の主因は**周期関数の前で角度を量子化していること**（timestep × 高周波バンド = 大きな角度。
  角度のわずかな量子化誤差が cos 後に巨大化）と考えられ、calibration 変更で救える見込みは薄い。
- gate 判定により wav 本走（75分）はスキップ（seed が悪化＝wav も悪化が確実）。
- **→ 残る正攻法はレバー2: timestep 埋め込みを host 側で fp32 計算し入力化**（角度の量子化を NPU 外へ）。
  cut 点候補: ① cond_embed(1,1280) 丸ごと入力化（cos/sin/cat/silu/cond-linear を全部 NPU 外）＝最もクリーン、
  ② sin/cos 出力だけ入力化。①は Option A(KVキャッシュ) と同じ手で再 export が要る。

### 真因を正確に特定: cos の角度入力だけ U8 に落ちる Pulsar2 グラフ artifact（2026-05-24 夜, 突破）
診断ビルド `cosfp32`（Cos/Sin op_types→FP32）は **silent に無視**された（baseline とグラフ完全一致。
op_types=FP32 override は効かない＝mixedp の layer_names 脱落と同じ silent-drop。**FP32 はグラフ中間 activation
として非対応で黙殺された可能性大**）。ただしグラフ調査で**非対称の真因が判明**:
- `node_sin` は角度 `mul_2`(S16) を**直接** S16 で読む → cosine **0.9999**（無傷）。
- `node_cos` は手前に **`AxRequantizeLinear` が挿入され、同じ `mul_2` を S16→U8 に落として**から読む
  → 角度（timestep×高周波 = 大レンジ）が **256 levels** に潰れ cos 後に巨大誤差 → cosine **0.6315**。
- = **同一テンソルを cos 側だけ U8 に requant する Pulsar2 のグラフ生成 artifact**。周期関数の原理的問題ではなく
  **per-tensor 量子化割当のバグ/ヒューリスティック**。sin が S16 で無傷なのが動かぬ証拠。
- **含意（重要・希望）**: cos 入力 requant を U8→U16/S16 に上げられれば cos seed は sin 同様 ~0.99 に戻る公算。
  これが timestep 経路の支配的 seed なので、効けば DiT-PTQ が再び有望になる。calibration や host-precompute より
  軽い config 修正で済む可能性。**ただし op-level override が効くか要検証**（FP32 は黙殺された前科あり→U16 で試す）。

### 突破: `node_cos`→U16 で timestep 経路が完全回復（2026-05-24 夜, 実測）
`kv_b1_w8a16_cosu16.json` = W8A16(MinMax) + `layer_configs:[{layer_names:["node_cos"], data_type:"U16"}]`。
**グラフ検証**: node_cos の入力 dtype が **U8 → U16 に実際に変化**（config が効いた。**layer_names は効く / op_types=FP32 は黙殺**）。
precision_analysis seed:
| tensor | baseline | cosU16 | Δ |
|---|---|---|---|
| **cos** | 0.632 | **1.0000** | +0.368 |
| sin | 1.000 | 1.000 | 0 |
| cat | 0.789 | 1.0000 | +0.211 |
| silu | 0.850 | 0.9999 | +0.150 |
| cond-linear | 0.939 | 1.0000 | +0.061 |

- **timestep 経路が丸ごと ~1.0 に回復**。cat/silu/cond-linear が一緒に直ったのは、それらが壊れていた原因が
  「入力の cos が U8 で壊れていた」せいだったから（下流は元々無傷）。**支配的 seed を config 1行で除去**。
- 教訓: Pulsar2 の op-level `data_type` override は **layer_names では効く**（FP32 を除く。FP32 は中間 activation 非対応で黙殺）。
  U16 等の正規 activation dtype を layer_names で指定すれば該当 op の入出力精度を上げられる。
- ※ これは early seed の cosine 改善であり **wav 品質はまだ未確認**（per-step cosine は proxy 無効の教訓）。
  ただし seed が劇的改善＝wav 本走の価値が初めて立った。他に同種の U8-requant seed が無いか確認後、wav 本走へ。

### cosU16 の wav 本走結果: 大幅改善だが未だ出荷不可（2026-05-24 朝, 74分）
| metric | plain W8A16 | **cosU16** | fp32目標 |
|---|---|---|---|
| per-step cosine | ~0.93 | **0.9966〜0.9985** | 1.0 |
| wav SNR | −4.93 dB | **−1.96 dB** | 高+ |
| corr | 0.069 | **0.306** | →1.0 |
| mel_L1 | 4.41 | **1.30** | →0 |

- cos 修正で per-step が 0.93→**0.997** に跳ね、wav 全指標が 2〜3倍改善（mel_L1 は 4.41→1.30＝3.4倍）。**修正は正しい軸**。
- だが **wav はまだ出荷不可**（SNR 負, corr 0.31）。per-step 0.3% の残差でも 8 step RF × 12 block で増幅 → −1.96dB。
  = 「PTQ 絶望」から「あと一息」に前進。cos 修正は**必須だが単独では不十分**。
- 残る誤差源 = 通常の linear 量子化累積（早期 linear で op_57=0.74, op_127=0.85 等 4本が <0.9）。次手はこれらを詰める。
- wav 本走 = 74分（05:28→06:42）。wiring per-step 0.997 で gate 通過。

### 残差の所在: U8 activation は AdaLN 経路の producer 群に集中（2026-05-24 朝, 精査）
※ 初稿の「242/245 が U8 ＝実質 W8A8」は**誤読**（夜通しビルドで2度目の早合点。advisor 指摘で訂正）。正確には:
- **FC 出力**: S16=158 / U8=87 → `data_type:S16` は**出力には効いている**。
- **FC 入力**: U8=242 / S16=3 → 入力が U8 なのは **`data_type` が op の「出力」を制御し「入力」はしないから**
  （cos 修正が効いたのは node_cos の入力 requant が別 node で、cos の quant に合わせて上流が引き上がったため）。
- **全 op 出力**: S16=**1183** / U8=194 / FP32=4 / U16=1 → モデルは**大半 S16**。残る弱点は **U8 出力の 194 op**。
- U8 を FC 入力に供給する producer = AxQuantizedSilu 72 + AxQuantizedAdd 72 + AxQuantizedFullyConnected 72 + AxQuantizedMul 25（≈242, **AdaLN 変調経路に集中**）。

**含意**: 残差 −1.96dB を詰めるには **AdaLN 経路の ~240 producer op を S16 に引き上げる**必要があり、
これは「小さな clean な bug fix」ではなく **精度↔速度のトレードオフを伴う設計判断**（モデルが大きく・遅くなる。
ユーザは生成速度を重視）。layer_names なら効く実証済だが ~240 名の列挙＝設計選択。
→ **無人ビルドはここで停止。朝、ユーザと方針判断**（下記）。

### 教訓: layer_configs の name-mapping は silent に外れる（要・堅牢化）
mixedp で 148指定→146適用と **2本が黙って脱落**し 12分を無駄にした。Pulsar2 の op 名は不安定 or config の
当たり判定が node 名と一致しないことがある。**今後の混合精度は必ず post-build 検証**:
ビルド後 `quant_axmodel.onnx` の実 FP32 集合を数え、intended と len/集合一致を assert（不一致なら fail-loud）。
shape が一意な層（cond 1280,512 / in_proj 32,1280 / out_proj 1280,32）は shape で確実に狙えるが、AdaLN 等は名前依存。
→ ヘルパ `scripts/verify_layer_dtypes.py` を用意（次セッションで本走前に必ず通す）。

