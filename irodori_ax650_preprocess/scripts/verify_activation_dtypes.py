#!/usr/bin/env python3
"""Ph1 §1.2 activation-dtype coverage on a pulsar2 quant_axmodel.onnx.

verify_layer_dtypes.py checks FC *weight* dtype (FP32 isolation vs quantized).
This complements it by checking *activation* dtype — the real "true-A16 / cos→U16"
intent: FC inputs and the timestep-cos input should be U16 (not U8-downgraded).

Reads tensor dtypes from value_info/inputs/outputs (no external weights loaded).
Prints histograms + the FC-input and cos-input U16 coverage. Pure report (the
intended counts differ per build); fail-loud is left to the human reading coverage.
"""
from __future__ import annotations
import argparse, collections, sys
import onnx
from onnx import TensorProto

DT = {v: k for k, v in TensorProto.DataType.items()}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    args = ap.parse_args()
    m = onnx.load(args.onnx, load_external_data=False)
    g = m.graph

    # tensor name -> elem dtype, from every typed source
    tt: dict[str, int] = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        if vi.type.tensor_type.elem_type:
            tt[vi.name] = vi.type.tensor_type.elem_type
    for init in g.initializer:
        tt[init.name] = init.data_type

    op_hist = collections.Counter(n.op_type for n in g.node)
    print(f"total nodes={len(g.node)}  op_types:")
    for op, c in op_hist.most_common():
        print(f"  {c:4d}  {op}")

    # activation dtype histogram (exclude initializers = weights/consts)
    init_names = {i.name for i in g.initializer}
    act_hist = collections.Counter(
        DT.get(dt, dt) for n, dt in tt.items() if n not in init_names)
    print(f"\nactivation/value dtype histogram (non-initializer):")
    for d, c in act_hist.most_common():
        print(f"  {c:4d}  {d}")

    def in_dt(node, idx=0):
        if idx >= len(node.input):
            return "?"
        return DT.get(tt.get(node.input[idx]), "untyped")

    # FC input[0] (activation) dtype coverage
    fcs = [n for n in g.node if "FullyConnected" in n.op_type]
    if fcs:
        fc_in = collections.Counter(in_dt(n, 0) for n in fcs)
        print(f"\nFullyConnected ({len(fcs)}) activation-input[0] dtype:")
        for d, c in fc_in.most_common():
            print(f"  {c:4d}  {d}")
        u16 = fc_in.get("UINT16", 0)
        print(f"  -> FC-input U16 coverage: {u16}/{len(fcs)} "
              f"({100*u16/len(fcs):.0f}%)")

    # cos nodes (DiT timestep embedding) input dtype
    cos_nodes = [n for n in g.node
                 if "cos" in (n.name or "").lower() or n.op_type.lower().endswith("cos")]
    print(f"\ncos-related nodes: {len(cos_nodes)}")
    for n in cos_nodes[:20]:
        ins = [f"{i}:{DT.get(tt.get(i),'?')}" for i in n.input]
        outs = [f"{o}:{DT.get(tt.get(o),'?')}" for o in n.output]
        print(f"  {n.op_type} {n.name}  in={ins} out={outs}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
