# e2e_demo/archive

`e2e_demo/run_npu_full.py`（全段NPU・safetensorsレスの正準ランナー）に統合された、
あるいは歴史的役割を終えたワンオフ・診断スクリプト。**現在のデプロイには不要**だが、
経緯の参照・再検証のために残置。

| script | 旧役割 | 後継 |
|---|---|---|
| `a_build_cond.py` | Stage A 条件付け（2GB フル読み）| `slim_stageA.py` → `run_npu_full.py`(① NPU化) |
| `b_sample_npu.py` | Stage B DiT サンプリング単体 | `run_npu_full.py` |
| `c_decode.py` | Stage C DACVAE 復号単体 | `run_npu_full.py` |
| `e2e_npu.py` | cond npz → DiT → DACVAE（torch cond 前提）| `run_npu_full.py`（cond も NPU）|
| `slim_cond_probe.py` | meta+mmap の条件付けメモリ計測 probe | `slim_stageA.py`（本番 slim）|
| `ph0_inspect.py` | axmodel I/O shape ダンプ | （単発確認用）|
| `dit_smoke.py` | DiT liveness/dtype smoke | （単発確認用）|
| `dacvae_equiv.py` | DACVAE 数値等価確認 | （単発確認用）|
| `emoji_stageA.py` | emoji A/B（3-branch CFG cond 生成）| `run_npu_full.py` で text に emoji を含めるだけ |
| `bench_npu.py` / `bench_stageA.py` | NPU / StageA タイミング計測 | `runs/*_bench/RESULT.md` に結果 |
| `diag_step_*.py` / `diag_torch_ref.py` | 完全NPU化以前の step 単位デバッグ（未コミットだった）| — |
