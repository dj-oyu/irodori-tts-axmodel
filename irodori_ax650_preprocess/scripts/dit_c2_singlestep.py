#!/usr/bin/env python3
"""DiT single-step C2: fp32 onnx vs pulsar2 x86-sim on ONE real DiT step input.

Builds one consistent 53-input set from calib_kv_long/<name>/<idx>.npy, runs the
fp32 onnx (onnxruntime) and the compiled axmodel (pulsar2 run sim), compares
v_pred (per-step cosine / SNR over full T and over the latent_mask-valid region),
and saves inputs + v_pred_sim + v_pred_fp32 for the device to run the SAME input
on real NPU and confirm sim≡NPU (the C2/「致命」 question for the quality model).

  PYTHONPATH=/home/exe/ai/Irodori-TTS uv run --project irodori_ax650_preprocess \
    --with onnxruntime python irodori_ax650_preprocess/scripts/dit_c2_singlestep.py \
    --axmodel build/axmodel_kv_long_lm_allfcu16 --fp32 build/dit_step_kv_long_lm_mask1e4_fp32.onnx \
    --calib build/calib_kv_long --idx 0 --out-dir runs/<ts>_c2_dit
"""
from __future__ import annotations
import argparse, subprocess, shutil
from pathlib import Path
import numpy as np
import onnxruntime as ort

REPO = Path("/home/exe/ai/irodori-tts-axmodel")
MASKS = ("text_mask", "speaker_mask", "latent_mask")


def dock(p: Path) -> str:
    return f"/data/{p.resolve().relative_to(REPO.resolve())}"


def metrics(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    snr = 10 * np.log10((a ** 2).sum() / (((a - b) ** 2).sum() + 1e-12))
    return cos, float(snr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axmodel", required=True)
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--calib", default="build/calib_kv_long")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"])
    in_meta = {i.name: i.type for i in sess.get_inputs()}
    calib = Path(args.calib)

    feed = {}
    for name in in_meta:
        a = np.load(calib / name / f"{args.idx:04d}.npy")
        feed[name] = a
    # fp32 onnx run (cast masks to the onnx-declared dtype)
    ofeed = {}
    for n, a in feed.items():
        if "bool" in in_meta[n]:
            ofeed[n] = a.astype(np.bool_)
        elif "float" in in_meta[n]:
            ofeed[n] = a.astype(np.float32)
        else:
            ofeed[n] = a
    v_fp32 = sess.run(["v_pred"], ofeed)[0].astype(np.float32)   # (1,201,32)

    # pulsar2 sim: write .bin (masks uint8, rest float32), run, read v_pred.bin
    in_dir = REPO / args.out_dir / "sim_in"; out_dir = REPO / args.out_dir / "sim_out"
    if in_dir.exists(): shutil.rmtree(in_dir)
    in_dir.mkdir(parents=True, exist_ok=True)
    for n, a in feed.items():
        (a.astype(np.uint8) if n in MASKS else a.astype(np.float32)).tofile(in_dir / f"{n}.bin")
    cmd = ["docker", "run", "--rm", "-v", f"{REPO}:/data", "pulsar2:6.0", "-c",
           f"cd {dock(REPO / args.axmodel)} && pulsar2 run --model compiled.axmodel "
           f"--input_dir {dock(in_dir)} --output_dir {dock(out_dir)}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:]); raise SystemExit(f"pulsar2 run rc={r.returncode}")
    vp = out_dir / "v_pred.bin"
    if not vp.exists(): vp = next(out_dir.glob("*.bin"))
    v_sim = np.fromfile(vp, dtype=np.float32).reshape(v_fp32.shape)

    # compare full + valid region (latent_mask)
    lm = feed["latent_mask"].astype(bool).ravel()
    valid = int(lm.sum())
    cos_f, snr_f = metrics(v_fp32, v_sim)
    cos_v, snr_v = metrics(v_fp32[:, :valid, :], v_sim[:, :valid, :])
    print(f"[DiT C2] T={v_fp32.shape[1]} valid={valid}")
    print(f"  fp32 vs sim  FULL : cosine={cos_f:.5f}  SNR={snr_f:.2f}dB")
    print(f"  fp32 vs sim  VALID: cosine={cos_v:.5f}  SNR={snr_v:.2f}dB")

    # save handoff
    out = REPO / args.out_dir
    np.savez(out / "dit_inputs.npz", **feed)
    np.save(out / "v_pred_sim.npy", v_sim)
    np.save(out / "v_pred_fp32.npy", v_fp32)
    print(f"[saved] {out}/dit_inputs.npz + v_pred_sim.npy + v_pred_fp32.npy")


if __name__ == "__main__":
    main()
