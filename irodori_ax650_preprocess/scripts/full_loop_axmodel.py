#!/usr/bin/env python3
"""
full_loop_axmodel.py

Run the full TTS sampling loop with the W8A16 compiled DiT axmodel as the
denoiser step (via `pulsar2 run` x86 simulator), and compare the resulting wav
to the fp32 torch reference wav.

Adapted from validate_full_loop.py: monkeypatch
model.forward_with_encoded_conditions to route each CFG-batch element through
the compiled axmodel via a subprocess `docker run ... pulsar2 run`.

Run in Irodori env:
  cd /home/exe/ai/Irodori-TTS
  PYTHONPATH=/home/exe/ai/Irodori-TTS uv run python \
    /home/exe/.claude/jobs/f7b5721e/full_loop_axmodel.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
from huggingface_hub import hf_hub_download


def save_wav(path, wav_1d_np, sr):
    sf.write(str(path), np.asarray(wav_1d_np, dtype=np.float32).reshape(-1), sr)

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

REPO = Path("/home/exe/ai/irodori-tts-axmodel")
JOB = Path("/home/exe/.claude/jobs/f7b5721e")
AXMODEL_DIR = Path(os.environ.get("AXMODEL_DIR", str(REPO / "build/axmodel_kv_b1_mask1e4_plain")))
SIM_ROOT = Path(os.environ.get("SIM_ROOT", str(REPO / "build/fullloop_sim")))

TEXT = os.environ.get("SYNTH_TEXT", "こんにちは、テストです。")  # T must match the built axmodel's fixed shape
NUM_STEPS = 8
SEED = int(os.environ.get("SYNTH_SEED", "1234"))
T_SCHEDULE = "sway"

# KV input names (Option A, kv): per layer [k_text, v_text, k_spk, v_spk] x 12 layers
NL = 12
KINDS = ["k_text", "v_text", "k_spk", "v_spk"]
KV_NAMES = [f"{kind}_{i}" for i in range(NL) for kind in KINDS]
INPUT_NAMES = ["x_t", "t", "text_mask", "speaker_mask"] + KV_NAMES


def run_axmodel_one(in_dir: Path, out_dir: Path) -> np.ndarray:
    """Invoke pulsar2 run on the compiled axmodel for one B=1 input set. Returns v_pred [1,119,32]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "-v", f"{REPO}:/data", "pulsar2:6.0", "-c",
        f"cd {dock_path(AXMODEL_DIR)} && "
        f"pulsar2 run --model compiled.axmodel "
        f"--input_dir {dock_path(in_dir)} --output_dir {dock_path(out_dir)}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        raise RuntimeError(f"pulsar2 run failed rc={res.returncode}")
    vp = out_dir / "v_pred.bin"
    if not vp.exists():
        raise RuntimeError(f"no v_pred.bin in {out_dir}; outputs={list(out_dir.iterdir())}")
    a = np.fromfile(vp, dtype=np.float32)
    return a.reshape(1, -1, 32)  # T inferred from output size (axmodel's fixed latent length)


def dock_path(p: Path) -> str:
    """Map a host path under REPO to its in-container /data path."""
    rel = p.resolve().relative_to(REPO.resolve())
    return f"/data/{rel}"


def write_bins(in_dir: Path, feed_b1: dict[str, np.ndarray]) -> None:
    """Write the 52 .bin for one B=1 element. masks uint8, rest float32."""
    if in_dir.exists():
        shutil.rmtree(in_dir, ignore_errors=True)
    in_dir.mkdir(parents=True, exist_ok=True)
    for n in INPUT_NAMES:
        a = feed_b1[n]
        if n in ("text_mask", "speaker_mask"):
            a = a.astype(np.uint8)
        else:
            a = a.astype(np.float32)
        a.tofile(in_dir / f"{n}.bin")


def to_np(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().numpy()
    if t.dtype == torch.bool:
        return a.astype(np.uint8)
    return a.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    print(f"[config] AXMODEL_DIR={AXMODEL_DIR}  SIM_ROOT={SIM_ROOT}", flush=True)
    assert (AXMODEL_DIR / "compiled.axmodel").exists(), f"no compiled.axmodel at {AXMODEL_DIR}"
    SIM_ROOT.mkdir(parents=True, exist_ok=True)
    ckpt = hf_hub_download(repo_id="Aratako/Irodori-TTS-500M-v3", filename="model.safetensors")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime = InferenceRuntime.from_key(
        RuntimeKey(checkpoint=ckpt, model_device=device, codec_device=device)
    )
    model = runtime.model

    def make_req():
        return SamplingRequest(
            text=TEXT, no_ref=True, num_steps=NUM_STEPS, seed=SEED,
            context_kv_cache=True, t_schedule_mode=T_SCHEDULE,
        )

    # ---- 1) fp32 torch reference wav (save FIRST, durably) ----
    print("[ref] running fp32 torch synthesize ...", flush=True)
    res_pt = runtime.synthesize(make_req())
    wav_pt = res_pt.audio.detach().cpu().float()
    sr = int(getattr(res_pt, "sample_rate", 44100) or 44100)
    ref_path = REPO / "build/wav_fp32_ref.wav"
    save_wav(ref_path, wav_pt.numpy(), sr)
    np.save(SIM_ROOT / "wav_fp32_ref.npy", wav_pt.numpy())
    print(f"[ref] saved {ref_path} shape={tuple(wav_pt.shape)} sr={sr} seed={res_pt.used_seed}", flush=True)

    # ---- 2) axmodel shim ----
    orig = model.forward_with_encoded_conditions
    state = {"call": 0, "checked": False}

    def shim(*a, **kw):
        x_t = kw["x_t"]
        B = x_t.shape[0]
        # On the very first call, compute the torch fwec output for THESE live inputs
        # (apples-to-apples per-step gold for the wiring sanity check).
        v_torch_live = None
        if not state["checked"]:
            with torch.no_grad():
                v_torch_live = orig(*a, **kw).detach().cpu().float().numpy()  # [B,119,32]
        base = {n: to_np(kw[n]) for n in ["x_t", "t", "text_mask", "speaker_mask"]}
        cache = kw["context_kv_cache"]  # list[ (k_text,v_text,k_spk,v_spk) ] per layer, batched B
        # flatten per layer into named np arrays, full batch
        kv_full = {}
        for i, layer in enumerate(cache):
            for kind, tns in zip(KINDS, layer):
                kv_full[f"{kind}_{i}"] = to_np(tns)
        # t may be shape [B] or scalar broadcast; ensure per-element
        t_np = base["t"]
        outs = []
        call = state["call"]
        for b in range(B):
            feed = {
                "x_t": base["x_t"][b:b + 1],
                "t": t_np[b:b + 1] if t_np.ndim >= 1 and t_np.shape[0] == B else t_np.reshape(-1)[:1],
                "text_mask": base["text_mask"][b:b + 1],
                "speaker_mask": base["speaker_mask"][b:b + 1],
            }
            for n in KV_NAMES:
                feed[n] = kv_full[n][b:b + 1]
            in_dir = SIM_ROOT / f"step{call:02d}_b{b}/in"
            out_dir = SIM_ROOT / f"step{call:02d}_b{b}/out"
            write_bins(in_dir, feed)
            t0 = time.time()
            vp = run_axmodel_one(in_dir, out_dir)
            dt = time.time() - t0
            outs.append(vp)
            print(f"[axmodel] call={call} b={b}/{B} {dt:.0f}s", flush=True)
        out = np.concatenate(outs, axis=0)  # [B,119,32]
        # wiring sanity on the first call: per-element cosine vs torch fwec (same live input)
        if not state["checked"]:
            cmin = 1.0
            for bi in range(B):
                c = cosine(out[bi:bi + 1], v_torch_live[bi:bi + 1])
                print(f"[WIRING] b={bi} axmodel-vs-torch cosine={c:.4f} (expect ~0.93)", flush=True)
                cmin = min(cmin, c)
            state["checked"] = True
            if cmin < 0.85:
                raise RuntimeError(f"Wiring check failed: min cosine {cmin:.4f} << 0.93. Aborting.")
        state["call"] += 1
        return torch.from_numpy(out).to(device=x_t.device, dtype=x_t.dtype)

    model.forward_with_encoded_conditions = shim
    try:
        print("[axmodel] running synthesize with axmodel DiT shim ...", flush=True)
        res_ax = runtime.synthesize(make_req())
    finally:
        model.forward_with_encoded_conditions = orig
    wav_ax = res_ax.audio.detach().cpu().float()
    ax_path = REPO / "build/wav_w8a16_axmodel.wav"
    save_wav(ax_path, wav_ax.numpy(), sr)
    np.save(SIM_ROOT / "wav_w8a16_axmodel.npy", wav_ax.numpy())
    print(f"[axmodel] saved {ax_path} shape={tuple(wav_ax.shape)} fwec_calls={state['call']}", flush=True)

    # ---- 3) compare ----
    a = wav_pt.reshape(-1).numpy()
    b = wav_ax.reshape(-1).numpy()
    n = min(a.shape[0], b.shape[0])
    a = a[:n]; b = b[:n]
    rms_err = float(np.sqrt(np.mean((a - b) ** 2)))
    sig = float(np.sqrt(np.mean(a ** 2))) + 1e-12
    snr = 20.0 * np.log10(sig / (rms_err + 1e-12))
    corr = float(np.corrcoef(a, b)[0, 1])
    max_abs = float(np.max(np.abs(a - b)))

    # mel-L1
    melT = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80
    )
    ma = torch.log(melT(torch.from_numpy(a).float()) + 1e-5)
    mb = torch.log(melT(torch.from_numpy(b).float()) + 1e-5)
    mel_l1 = float((ma - mb).abs().mean())

    print("=" * 60)
    print(f"len_fp32={wav_pt.shape[-1]} len_ax={wav_ax.shape[-1]} (cmp {n})")
    print(f"RESULT  SNR={snr:.2f} dB  mel_L1={mel_l1:.4f}  corr={corr:.6f}  max_abs={max_abs:.3e}")
    print("=" * 60)

    # durable metrics
    (SIM_ROOT / "metrics.txt").write_text(
        f"SNR_dB={snr:.4f}\nmel_L1={mel_l1:.6f}\ncorr={corr:.6f}\n"
        f"max_abs={max_abs:.6e}\nlen_fp32={wav_pt.shape[-1]}\nlen_ax={wav_ax.shape[-1]}\n"
        f"fwec_calls={state['call']}\nsr={sr}\n"
    )
    (SIM_ROOT / "done.flag").write_text("done\n")


if __name__ == "__main__":
    main()
