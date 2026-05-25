#!/usr/bin/env python3
"""Sim a DACVAE decoder axmodel on the test latent z (x86 pulsar2 run), compare wave to fp32 ref.
Usage: python dacvae_sim.py <axmodel_dir> <tag>   (run in Irodori env for soundfile/torch)
Writes build/test_audio_cosu16/dacvae_<tag>_quant.wav + prints SNR/mel_L1/corr."""
import sys, subprocess, numpy as np, torch, torchaudio, soundfile as sf
from pathlib import Path

REPO = Path("/home/exe/ai/irodori-tts-axmodel")
axdir = Path(sys.argv[1]); tag = sys.argv[2]
SR = 48000
calib = REPO / "build/calib_dacvae"
test_z = np.load(calib / "test_z.npy").astype(np.float32)          # (1,32,119)
fp32_wave = np.load(calib / "test_z_fp32_wave.npy").reshape(-1).astype(np.float32)

# write z.bin, run pulsar2 sim
indir = REPO / "build/dacvae_sim" / tag / "in"; outdir = REPO / "build/dacvae_sim" / tag / "out"
indir.mkdir(parents=True, exist_ok=True); outdir.mkdir(parents=True, exist_ok=True)
test_z.tofile(indir / "z.bin")
cmd = ["docker","run","--rm","-v",f"{REPO}:/data","pulsar2:6.0","-c",
       f"cd /data/{axdir.relative_to(REPO)} && pulsar2 run --model compiled.axmodel "
       f"--input_dir /data/{indir.relative_to(REPO)} --output_dir /data/{outdir.relative_to(REPO)}"]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
if r.returncode != 0:
    print(r.stdout[-1500:]); print(r.stderr[-1500:]); sys.exit(f"pulsar2 run failed rc={r.returncode}")

ab = outdir / "audio.bin"
if not ab.exists():
    ab = next(outdir.glob("*.bin"))
q = np.fromfile(ab, dtype=np.float32).reshape(-1)
n = min(len(q), len(fp32_wave)); a = fp32_wave[:n]; b = q[:n]
snr = 10*np.log10((a**2).sum() / (((a-b)**2).sum()+1e-12))
corr = float(np.corrcoef(a, b)[0,1])
# mel_L1 (perceptual proxy that tracked DiT quality)
mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=256, n_mels=80)
ma = torch.log(mel(torch.from_numpy(a))+1e-5); mb = torch.log(mel(torch.from_numpy(b))+1e-5)
mel_l1 = (ma-mb).abs().mean().item()
sf.write(REPO/f"build/test_audio_cosu16/dacvae_{tag}_quant.wav", b, SR)
print(f"[{tag}] SNR={snr:.2f}dB mel_L1={mel_l1:.4f} corr={corr:.4f} len={len(q)} peak={np.abs(b).max():.3f}")
