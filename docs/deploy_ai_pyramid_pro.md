# Irodori-TTS を AI Pyramid Pro (AX8850) で動かす

**v0.2 (2026-05-23)**

Irodori-TTS-500M-v3 の **DiT denoiser** と **TextEncoder** を AX8850 NPU で、DACVAE vocoder ほかを
Cortex-A55(CPU) で動かすためのデプロイ手順。DiT(Option A/B) と TextEncoder は `.axmodel` 変換済み。

> 状態（2026-05-23）: DiT(Option A/B) + TextEncoder の `.axmodel` 変換、full-loop 数値検証、
> W8A8 量子化精度（v_pred cosine 0.9997）まで x86 で確認済み。**実機 end-to-end latency は未測定**
> （AX8850 実機が手元に無いため）。**DACVAE は NPU 不可で CPU 据え置きが確定**（§10）。
> 詳細な変換経緯・全ブロッカは [`../irodori_ax650_preprocess/FINDINGS.md`](../irodori_ax650_preprocess/FINDINGS.md)。

---

## 1. 全体像 — CPU/NPU の分担

支配的コスト（4090 実測で sample_rf=DiT が ~80%）の **DiT** と、次点の **TextEncoder** を NPU に出す。
**DACVAE は Pulsar2 の Snake op 制約で NPU 不可**（§10）なので CPU 据え置き。

```
text ──► [CPU] 正規化 + tokenizer
            │
            ├─► [NPU] TextEncoder ──────────┐   ★ NPU
            ├─► [CPU] ReferenceLatentEncoder ├─► conditioning (text_state / speaker_state)
            ├─► [CPU] DurationPredictor ────►│   → latent 長 T を決定
            ▼                                ▼
        noise z_T ──► RF Euler ループ(N steps, sway + 6〜8 step 推奨, [CPU] グルー)
                          │   各 step:
                          └─► [NPU] DiT 1-step (compiled.axmodel)  ★ NPU（支配的）
                          ▼
                       latent z_0 ──► [CPU] DACVAE decoder ──► wav ──► [CPU] watermark
                                       ↑ NPU 不可（Snake）= A55 で最重量の残 CPU
```

| 段 | 実行先 | 備考 |
|---|---|---|
| text 正規化 / tokenizer | CPU | 軽い |
| **TextEncoder** | **NPU (.axmodel)** | `axmodel_textenc/`。変換済み |
| ReferenceLatentEncoder / DurationPredictor | CPU(PyTorch) | 小さい。NPU 化未（優先度低）|
| **DiT 1-step (×N)** | **NPU (.axmodel)** | **支配的。Option A/B 変換済み** |
| RF Euler 更新 / schedule | CPU | 制御ロジック。**sway + 6〜8 step** 推奨（§7.5）|
| **DACVAE decoder** | **CPU(PyTorch)** | **NPU 不可（Snake op 制約, §10）**。A55 で最重量の残 CPU |
| wav 保存 / watermark | CPU | |

---

## 2. 必要なもの（実機側）

- **AI Pyramid Pro (AX8850)** — octa Cortex-A55 + 24 TOPS@INT8 NPU、8GB LPDDR4x
  （8GB 版は system 4GB / NPU・video 用 4GB に分割）。
- 電源: **PD 9V@3A (27W) 以上**。5V では起動しない。
- OS: 出荷時の Debian/Ubuntu（AXCL/AXEngine runtime 同梱）。
- **PyAXEngine**（axengine python wheel）。Python ≥3.8, numpy ≥1.22, cffi, ml-dtypes。
- ホスト CPU 側で Irodori-TTS を動かすための aarch64 版 PyTorch + `irodori_tts` 一式
  （tokenizer/encoder/duration/DACVAE/watermark のため）。

参考: [AI Pyramid-Pro (m5-docs)](https://docs.m5stack.com/en/ai_hardware/AI_Pyramid-Pro) /
[PyAXEngine](https://github.com/AXERA-TECH/pyaxengine)

---

## 3. 成果物（このリポジトリから持ち込むもの）

| axmodel | 内容 | size | 備考 |
|---|---|---|---|
| `build/axmodel_b1/compiled.axmodel` | DiT Option B（6入力, B=1, T=119, W8A8）| 379MB | 堅実・Pulsar2 親和 |
| `build/axmodel_kv_b1/compiled.axmodel` | DiT Option A（KV cache 52入力）| 338MB | **性能本命**（§9）|
| `build/axmodel_textenc/compiled.axmodel` | TextEncoder | — | NPU 化済み |

- DACVAE は `build/dacvae_decoder.onnx` のみ（NPU 不可 → CPU/onnxruntime か PyTorch で実行）。
- 生成・検証スクリプト一式（`irodori_ax650_preprocess/scripts/`）。再ビルド/別 shape は §8 + FINDINGS。

`.axmodel` を実機へ転送（例）:
```bash
scp build/axmodel_b1/compiled.axmodel       pyramid:/opt/irodori/dit_b1.axmodel
scp build/axmodel_textenc/compiled.axmodel  pyramid:/opt/irodori/text_encoder.axmodel
```

---

## 4. PyAXEngine 導入（実機）

wheel は [PyAXEngine (GitHub releases)](https://github.com/AXERA-TECH/pyaxengine/releases) または
HF [`AXERA-TECH/PyAXEngine`](https://huggingface.co/AXERA-TECH/PyAXEngine) から入手。

```bash
# wheel を入れる（バージョンは配布物に合わせる）
pip3 install axengine-*.whl
python3 -c "import axengine; print('axengine OK')"
```

スモークテスト（速度のみ。正しさは見ない）:
```bash
# NPU 上で 10 回回して latency を見る
ax_run_model -m /opt/irodori/dit_b1.axmodel -r 10
# あるいは
axcl_run_model -m /opt/irodori/dit_b1.axmodel -r 10
```

---

## 5. DiT 1-step .axmodel の I/O 契約（重要）

`compiled.axmodel`（Option B）の固定 shape 入出力。**dtype を厳密に合わせること**
（特に mask は実機で **uint8**。学習/PyTorch 側の bool ではない）。

| name | shape | dtype | 意味 |
|---|---|---|---|
| `x_t` | (1, 119, 32) | float32 | ノイズ付き latent（patched, latent_dim=32）|
| `t` | (1,) | float32 | timestep |
| `text_state` | (1, 256, 512) | float32 | TextEncoder 出力 |
| `text_mask` | (1, 256) | uint8 ※ | 1=有効 0=pad |
| `speaker_state` | (1, 2, 768) | float32 | ReferenceLatentEncoder 出力(+masked-mean token) |
| `speaker_mask` | (1, 2) | uint8 ※ | |
| **出力** `v_pred` | (1, 119, 32) | float32 | 速度場予測 |

> ※ mask の dtype は **uint8**（量子化グラフで `text_mask/speaker_mask: tensor(uint8)` を確認済。
>   PyTorch 側の bool ではない）。実機 `get_inputs()` で最終確認推奨。
> T=119 はこの .axmodel をビルドした発話の latent 長。**固定 shape なので別長は別ビルドが必要**（§8）。
> CFG は実行時 B=3/B=1 を交互に呼ぶが、本 .axmodel は B=1 固定 → **CFG 各要素を B=1 で個別実行**（§7）。
> **この .axmodel は `--no-ref` 用**。`speaker_state` の seq=2 は no_ref 時の値で、実 ref 音声を使うと
> seq 長が ref 長に依存して変わるため、ref 条件付き運用は別 calibration + 別ビルドが必要（§10）。

実機で確認:
```python
import axengine, numpy as np
s = axengine.InferenceSession("/opt/irodori/dit_b1.axmodel",
                              providers=["AxEngineExecutionProvider"])
print([(i.name, i.shape, i.dtype) for i in s.get_inputs()])
print([(o.name, o.shape, o.dtype) for o in s.get_outputs()])
```

---

## 6. 統合の要 — runtime の DiT step を NPU 呼び出しに差し替える

ホストで Irodori-TTS の `synthesize` をそのまま使い、`forward_with_encoded_conditions` だけを
axengine 呼び出しに差し替える。**ホスト側検証 `scripts/validate_full_loop.py` の onnxruntime 版を
axengine に置換しただけ**（API はほぼ同一なので移植は容易）。

```python
# device_dit_shim.py  （AI Pyramid Pro 上で実行）
import axengine, numpy as np, torch
from huggingface_hub import hf_hub_download
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest, save_wav

DIT = "/opt/irodori/dit_b1.axmodel"
sess = axengine.InferenceSession(DIT, providers=["AxEngineExecutionProvider"])
in_meta = {i.name: i for i in sess.get_inputs()}

ckpt = hf_hub_download("Aratako/Irodori-TTS-500M-v3", "model.safetensors")
rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=ckpt, model_device="cpu", codec_device="cpu"))
model = rt.model
orig = model.forward_with_encoded_conditions

def _np(name, t):
    a = t.detach().cpu().numpy()
    # axmodel の入力 dtype に合わせる（mask は uint8, それ以外 float32）
    want = np.uint8 if "mask" in name else np.float32
    return a.astype(want)

def shim(*a, **kw):
    x_t = kw["x_t"]; B = x_t.shape[0]
    outs = []
    for b in range(B):                       # B=3(CFG)/B=1 を B=1 ずつ NPU 実行
        feed = {n: _np(n, kw[n][b:b+1]) for n in
                ["x_t","t","text_state","text_mask","speaker_state","speaker_mask"]}
        outs.append(sess.run(["v_pred"], feed)[0])
    out = np.concatenate(outs, axis=0)
    return torch.from_numpy(out).to(device=x_t.device, dtype=x_t.dtype)

model.forward_with_encoded_conditions = shim

# Option B 経路に合わせて context_kv_cache=False。T は .axmodel と一致する発話を使う。
res = rt.synthesize(SamplingRequest(text="こんにちは、これはテスト音声です。",
                                    no_ref=True, num_steps=16, context_kv_cache=False))
save_wav("out.wav", res.audio, res.sample_rate)
```

ポイント:
- `context_kv_cache=False`（Option B の .axmodel と一致）。Option A(KV cache 入力)を使う場合は §9。
- `t` の dtype は float32。`text_mask/speaker_mask` は **uint8** に変換。
- **TextEncoder も同様に axmodel 化済み**（`text_encoder.axmodel`）→ 同じ要領で encode_conditions の
  text 経路を NPU 呼び出しに差し替え可能。
- **DACVAE は NPU 不可**なので PyTorch/onnxruntime CPU で動かす（§10）。A55 で最重量の残 CPU。
- step 数は **sway + 6〜8** を推奨（§7.5）。

---

## 7. 固定 shape 戦略（CFG batch と latent 長 T）

NPU の .axmodel は固定 shape。Irodori の実ループは 2 つの可変軸を持つ:

1. **CFG batch**: `cfg_min_t/max_t` の窓内 step は B=3（cond + text-uncond + speaker-uncond）、
   窓外は B=1。→ **B=1 .axmodel を CFG 要素ごとに実行**（上記 shim の `for b in range(B)`）。
   B=3 専用 .axmodel を作って 1 回で回す手もある（要再ビルド）。
2. **latent 長 T**: DurationPredictor の出力で発話ごとに変わる。固定 shape の解は 2 択:
   - (a) 代表 **T_max** で 1 個ビルドし、短い発話は **pad + latent self-mask** で無効化。
     ※ 現状の Option B ラッパは latent self-mask を入力に取っていない。T_max 運用には
       ラッパに self-mask 入力を追加して再 export が必要（TODO）。
   - (b) よく使う数種の T で複数 .axmodel を用意し、最も近い長さに丸める。
   - 暫定: 本 PoC は T=119 固定（特定発話）。汎用運用は (a) を推奨。

---

## 7.5 few-step sampling（最優先の latency 削減）

DiT step×N が支配的なので、step 数削減がそのまま latency に効く。x86 スイープ（`step_sweep.py`）で
32-step 基準への log-mel L1 を計測した結果:

- **`t_schedule_mode="sway"` が全 step で `linear` より良い**（パラメタだけの無料改善）。
- **実用下限は sway で 6〜8 step**（8 以下で劣化が増え、6→4 で急増）。16→8 で NPU 呼び出し ~2×減。

```python
SamplingRequest(..., num_steps=8, t_schedule_mode="sway")
```

### 量子化精度（参考, x86 で検証済み）
- W8A8 の DiT 出力 `v_pred`: **cosine 0.9997 / MSE 0.0006**（対 fp32, precision_analysis）。
- fp16 vs fp32 の end-to-end wav: **SNR 44dB / mel-L1 0.058**（聞き分け不可, `compare_onnx_wav.py`）。
- → 既定 W8A8 で音質は十分。W8A16 化は必須ではない（必要なら §8 の layer_configs で S16）。

---

## 8. 別 shape / 量子化方式で作り直す（ホスト側, x86 + Pulsar2）

実機ではなく**変換ホスト**（このリポジトリの環境）で行う。

```bash
# 0) 実 DiT 入力+基準出力を捕捉（Irodori env）。export と calibration の土台になる .ref.pt を作る
cd irodori_ax650_preprocess
PYTHONPATH=/path/to/Irodori-TTS uv run python scripts/capture_dit_inputs.py \
  --out ../build/dit_step_b1_fp32.ref.pt --text "こんにちは、テストです。" --num-steps 8
# 1) export-safe ONNX を作る（RoPE実数化 / SDPA additive mask / RMSNorm・AdaLN の rsqrt除去 を内包）
PYTHONPATH=/path/to/Irodori-TTS uv run python scripts/export_dit_step.py \
  --weights .../model.safetensors --model-cfg-json ../build/model_introspection.json \
  --inputs ../build/dit_step_b1_fp32.ref.pt --out ../build/dit_step_b1_fp32.onnx
# 2) 実 activation で calibration データ生成（Irodori env）
PYTHONPATH=/path/to/Irodori-TTS uv run python scripts/dump_calibration.py \
  --out-dir ../build/calib_b1 --text "..." --num-steps 20
# 3) Pulsar2 で .axmodel 化（target は AX650 = AX8850）
docker run --rm -v "$PWD/..:/data" pulsar2:6.0 -c \
  "cd /data && pulsar2 build --target_hardware AX650 \
     --input build/dit_step_b1_fp32.onnx --output_dir build/axmodel_b1 \
     --config build/pulsar_configs/b1_min.json"
```

**W8A16（transformer 本命, 推奨）にするには** `build/pulsar_configs/*.json` の `quant` に
`layer_configs` を足し、activation を `S16`（weight `S8`）に。`precision_analysis: true` で
層別の量子化誤差を出してから効かせると安全（FINDINGS「量子化方針」）。

---

## 9. Option A（KV cache を入力に取る版）— 変換済み・性能本命

CMM（ユニファイドメモリ）の AX8850 では、KV 射影を host で 1 回計算して全 step 再利用する
**Option A が帯域・MAC 的に有利**（per-step 重み再ストリーム ~79MB を KV 再利用 16MB に置換）。
- 変換済み: `build/axmodel_kv_b1/compiled.axmodel`（**338MB**, B 比で小, max_cycle 49.7M < B 52.4M）。
- 入力は **52**（x_t, t, text_mask, speaker_mask + KV 48）。text_state/speaker_state は KV 提供時
  未使用なのでラッパ内部 zeros 化して入力から外した（dead-input 回避）。
- 実機 shim: `context_kv_cache=True` で `build_context_kv_cache(text_state, speaker_state)` を
  **発話ごと 1 回**計算（A55, 軽い）し、その 48 tensor を毎 step axengine に渡す
  （`validate_full_loop.py --mode kv` が雛形, full-loop SNR 66dB で検証済み）。

---

## 10. 現状の制約と TODO（正直版）

- [x] **量子化精度 検証済み**。W8A8 の v_pred cosine **0.9997**/MSE 0.0006（precision_analysis）。
      fp16 vs fp32 wav は SNR 44dB で聞き分け不可。→ 既定 W8A8 で音質十分。
- [x] **W8A16 は必須でない**（W8A8 で十分）。必要なら §8 の `layer_configs` で S16。
- [x] **TextEncoder は NPU 化済**（`build/axmodel_textenc/`）。
- [x] **Option A 変換済み**（§9, `build/axmodel_kv_b1/`）。
- [x] **few-step 確定**（sway + 6〜8 step, §7.5）。
- [ ] **DACVAE decoder は NPU 不可で CPU 確定**。Pulsar2 6.0 の Snake op が
      **T<64 で broadcast バグ / T≥64 で tiler 限界**となり、**chunk 分割でも全 T で build 不可**（検証済）。
      打開は Axera の Snake op 修正待ち or vocoder 差し替え。当面 A55(CPU) で最重量の残 stage。
- [ ] **固定 T**（現状 119）。汎用化は T_max + latent self-mask 入力の追加が必要（§7-2a）。
      動的 ONNX も T が export 例値に固定される箇所があり、任意長は未対応。
- [ ] 実機実行・end-to-end latency 未測定（AX8850 実機待ち）。**残る最適化焦点は A55 の DACVAE-CPU latency**。

---

## 11. このディレクトリ/スクリプトの対応

| ファイル | 役割 |
|---|---|
| `irodori_ax650_preprocess/scripts/export_dit_step.py` | Option B DiT→ONNX（patch 内包, fp16/dynamic 可）|
| `irodori_ax650_preprocess/scripts/export_dit_step_kvcache.py` | Option A DiT→ONNX（KV cache 入力）|
| `irodori_ax650_preprocess/scripts/export_text_encoder.py` | TextEncoder→ONNX |
| `irodori_ax650_preprocess/scripts/export_dacvae_decoder.py` | DACVAE decoder→ONNX（NPU 不可, CPU 用）|
| `irodori_ax650_preprocess/scripts/rope_export_patch.py` | RoPE実数化/SDPA additive/RMSNorm・AdaLN rsqrt除去 + 等価検証 |
| `irodori_ax650_preprocess/scripts/dump_calibration.py` | 実 activation calibration 生成（`--kv` で Option A）|
| `irodori_ax650_preprocess/scripts/validate_full_loop.py` | sampling ループ全体検証（`--mode nokv/kv`, 実機 shim 雛形）|
| `irodori_ax650_preprocess/scripts/step_sweep.py` | few-step 品質/コスト スイープ |
| `irodori_ax650_preprocess/scripts/compare_onnx_wav.py` | fp32 vs fp16 の wav 比較（`paplay` 用）|
| `build/pulsar_configs/*.json` | Pulsar2 build config（b1_min / kv_b1_min / textenc 等）|
| `build/axmodel_{b1,kv_b1,textenc}/compiled.axmodel` | 変換済み（実機へ転送するファイル）|
