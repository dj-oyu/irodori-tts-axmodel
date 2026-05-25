#!/usr/bin/env python3
"""Capture real DACVAE decoder I/O: calib latents (z.tar.gz) + one test z + its fp32 wave.
Run in Irodori env. T=119 (matches dacvae_decoder_cos.onnx fixed shape)."""
import tarfile, numpy as np, torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

OUT = Path("/home/exe/ai/irodori-tts-axmodel/build/calib_dacvae")
T = 119
TEXT = "こんにちは、テストです。"   # yields T=119 latent
N = 24

ckpt = hf_hub_download(repo_id="Aratako/Irodori-TTS-500M-v3", filename="model.safetensors")
rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=ckpt, model_device="cuda", codec_device="cuda"))

# find the codec whose .model.decode(z) we export
codec = None
for name in dir(rt):
    c = getattr(rt, name, None)
    if hasattr(c, "model") and hasattr(getattr(c, "model"), "decode"):
        codec = c; break
assert codec is not None, "codec not found"
orig = codec.model.decode
zs = []
def hook(z, *a, **kw):
    zs.append(z[:1].detach().float().cpu().numpy())  # (1,32,T)
    return orig(z, *a, **kw)
codec.model.decode = hook
try:
    for s in range(N):
        rt.synthesize(SamplingRequest(text=TEXT, no_ref=True, num_steps=8, seed=s))
finally:
    codec.model.decode = orig

zs = [z for z in zs if z.shape[-1] == T]
print(f"captured {len(zs)} z @ T={T}, shape {zs[0].shape}")

zdir = OUT / "z"; zdir.mkdir(parents=True, exist_ok=True)
for j, z in enumerate(zs):
    np.save(zdir / f"{j:04d}.npy", z.astype(np.float32))
with tarfile.open(OUT / "z.tar.gz", "w:gz") as tf:
    for j in range(len(zs)):
        tf.add(zdir / f"{j:04d}.npy", arcname=f"{j:04d}.npy")
print(f"[calib] wrote {OUT/'z.tar.gz'} ({len(zs)} samples)")

# test sample: z[0] -> fp32 wave reference via the real decoder
test_z = zs[0]
with torch.no_grad():
    wav = codec.model.decode(torch.from_numpy(test_z).cuda()).float().cpu().numpy()
np.save(OUT / "test_z.npy", test_z.astype(np.float32))
np.save(OUT / "test_z_fp32_wave.npy", wav.reshape(-1).astype(np.float32))
print(f"[test] test_z {test_z.shape} -> fp32 wave {wav.reshape(-1).shape} "
      f"({wav.reshape(-1).shape[0]/48000:.2f}s) peak={np.abs(wav).max():.3f}")
