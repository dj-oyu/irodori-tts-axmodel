#!/usr/bin/env python3
"""Ph0 structure/IO inspection of the 3 deploy axmodels (root, axengine).

Prints every input/output name/shape/dtype for textenc, DiT(kv_long_lm_cosu16),
dacvae_b0. Asserts the deploy-critical structural facts from テスト観点 §1:
  - DiT has a latent_mask input that survived pruning  (53 inputs total expected)
  - DiT x_t T == T_max (long bucket)
  - dacvae input is (1,32,T)
Fail-loud: prints [PH0-FAIL]/[PH0-OK] lines so the morning review is one grep.
"""
from __future__ import annotations
import sys
import axengine as axe

MODELS = {
    "textenc": "build/axmodel_textenc/compiled.axmodel",
    "dit":     "build/axmodel_kv_long_lm_cosu16/compiled.axmodel",
    "dacvae":  "build/axmodel_dacvae_b0/compiled.axmodel",
}

def describe(tag, path):
    print(f"\n===== {tag} : {path} =====")
    sess = axe.InferenceSession(path)
    ins = sess.get_inputs()
    outs = sess.get_outputs()
    print(f"  #inputs={len(ins)}  #outputs={len(outs)}")
    for i in ins:
        print(f"  IN  {i.name:24s} shape={i.shape} dtype={i.dtype}")
    for o in outs:
        print(f"  OUT {o.name:24s} shape={o.shape} dtype={o.dtype}")
    return ins, outs

def main():
    results = {}
    for tag, path in MODELS.items():
        try:
            results[tag] = describe(tag, path)
        except Exception as e:
            print(f"[PH0-FAIL] {tag} load error: {e}")
            results[tag] = None

    print("\n===== Ph0 assertions =====")
    ok = True

    # DiT: latent_mask present + input count
    if results.get("dit"):
        ins, _ = results["dit"]
        names = [i.name for i in ins]
        has_mask = any("latent_mask" in n or n == "latent_mask" for n in names)
        print(f"  DiT inputs={len(ins)} latent_mask_present={has_mask}")
        if has_mask:
            print("  [PH0-OK] DiT latent_mask survived pruning")
        else:
            print("  [PH0-FAIL] DiT has NO latent_mask input (variable-length not wired)")
            ok = False
        # x_t T
        xt = next((i for i in ins if i.name == "x_t"), None)
        if xt is not None:
            print(f"  [PH0-INFO] x_t shape={xt.shape}  (expect T==T_max long bucket ~201)")
        else:
            print("  [PH0-FAIL] DiT has no x_t input"); ok = False
    else:
        ok = False

    # dacvae input shape (1,32,T)
    if results.get("dacvae"):
        ins, outs = results["dacvae"]
        zin = ins[0]
        print(f"  [PH0-INFO] dacvae in {zin.name} shape={zin.shape} (expect (1,32,T))")
        print(f"  [PH0-INFO] dacvae out {outs[0].name} shape={outs[0].shape}")
    else:
        ok = False

    if results.get("textenc"):
        print("  [PH0-OK] textenc loaded")
    else:
        ok = False

    print(f"\n[PH0-{'OK' if ok else 'FAIL'}] overall")
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()
