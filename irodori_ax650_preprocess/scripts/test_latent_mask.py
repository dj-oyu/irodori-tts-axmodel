#!/usr/bin/env python3
"""
Test the max-T + latent_mask hypothesis for variable-length output.
Q: if we sample at sequence_length=T_max but inject latent_mask marking the first
T_real frames valid, do those T_real frames match a DIRECT T_real generation?
If yes -> one max-T axmodel can output any length<=T_max (mask + trim). If no
(model stretches / valid region differs) -> latent_mask doesn't decouple duration.

Run in Irodori env. fp32, no axmodel.
"""
import numpy as np, torch
from huggingface_hub import hf_hub_download
from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest

ckpt = hf_hub_download(repo_id="Aratako/Irodori-TTS-500M-v3", filename="model.safetensors")
rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=ckpt, model_device="cuda", codec_device="cuda"))
model = rt.model
TEXT, SEED = "ありがとう。", 1234

orig_fwd = model.forward_with_encoded_conditions
state = {"capture_T": None, "inject_valid": None, "seen_T": None}

def patched(*a, **kw):
    x_t = kw.get("x_t", a[0] if a else None)
    T = x_t.shape[1]
    state["seen_T"] = T
    if state["inject_valid"] is not None and T == state["capture_T"]:
        v = state["inject_valid"]
        m = torch.zeros((x_t.shape[0], T), dtype=torch.bool, device=x_t.device)
        m[:, :v] = True
        kw["latent_mask"] = m
    return orig_fwd(*a, **kw)
model.forward_with_encoded_conditions = patched

# capture the final decoder-input latent z
latents = {}
def cap(tag):
    codec = next(getattr(rt,n) for n in dir(rt) if hasattr(getattr(rt,n,None),"model") and hasattr(getattr(getattr(rt,n),"model"),"decode"))
    od = codec.model.decode
    def h(z,*a,**k):
        latents[tag]=z[:1].detach().float().cpu().numpy(); return od(z,*a,**k)
    codec.model.decode = h
    return codec, od

# --- A: natural T_real generation ---
c,od = cap("real")
rt.synthesize(SamplingRequest(text=TEXT, no_ref=True, num_steps=40, seed=SEED))
c.model.decode = od
T_real = latents["real"].shape[-1]
print(f"[A] natural T_real = {T_real}")

# --- B: force longer length (T_real + pad) WITH latent_mask valid=[:T_real] ---
T_max = T_real + 24
state["capture_T"] = T_max; state["inject_valid"] = T_real
c,od = cap("masked")
# force sequence_length ~ T_max via seconds (framerate ~25/s); adjust if seen_T mismatches
secs = T_max / 25.0
rt.synthesize(SamplingRequest(text=TEXT, no_ref=True, num_steps=40, seed=SEED, seconds=secs))
c.model.decode = od
print(f"[B] forced seconds={secs:.2f} -> seen latent T = {state['seen_T']}, captured z T = {latents['masked'].shape[-1]}")

zr = latents["real"][0]                 # (32, T_real)
zm = latents["masked"][0]               # (32, T_b)
Tb = zm.shape[-1]
k = min(T_real, Tb)
vr, vm = zr[:, :k].ravel(), zm[:, :k].ravel()
cos = float(vr@vm/(np.linalg.norm(vr)*np.linalg.norm(vm)+1e-9))
print(f"\n=== valid-region (first {k} frames) match: direct-T_real vs masked-T_max ===")
print(f"cosine = {cos:.4f}   (close to 1.0 => latent_mask isolates valid frames => variable-length OK)")
# padding region energy (is it garbage or quiet?)
if Tb > T_real:
    pad = zm[:, T_real:]
    print(f"padding region ({Tb-T_real} frames) RMS = {np.sqrt((pad**2).mean()):.4f}  vs valid RMS {np.sqrt((zm[:,:T_real]**2).mean()):.4f}")
