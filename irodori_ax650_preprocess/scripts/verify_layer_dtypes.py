#!/usr/bin/env python3
"""
verify_layer_dtypes.py

Pulsar2 mixed-precision の **post-build 検証**。layer_configs の name-mapping は
silent に外れる（2026-05-24: 148指定→146適用で cond/timestep が脱落し 12分無駄に）。
Pulsar2 が割り当てる op 名は不安定で、config の当たり判定が node 名と一致しないことがあるため、
**ビルド後に実 FP32 集合を数え、intended と一致しなければ fail-loud** にする。

判定はビルド済 `quant_axmodel.onnx` の FullyConnected 重み initializer の dtype を直接見る
（FP32=isolation 成功 / INT8・INT16=量子化されたまま）。名前ではなく **重み shape** で集計するので
op 名の不安定性に依存しない。

使い方:
  # 実 FP32 集合を表示するだけ
  python verify_layer_dtypes.py --onnx build/axmodel_kv_b1_mixedp/quant/quant_axmodel.onnx

  # intended な FP32 shape 数を与えて assert（CI/本走前ゲート）
  #   --expect "SHAPE=COUNT" を複数。例: cond+in/out_proj だけ FP32 にしたい場合
  python verify_layer_dtypes.py --onnx .../quant_axmodel.onnx \
      --expect 1280,512=1 --expect 32,1280=1 --expect 1280,32=1
  # 不一致なら exit 1（fail-loud）。
"""
from __future__ import annotations

import argparse
import collections
import sys

import onnx
from onnx import TensorProto

_DTN = {
    TensorProto.FLOAT: "FP32",
    TensorProto.FLOAT16: "FP16",
    TensorProto.INT8: "S8",
    TensorProto.UINT8: "U8",
    TensorProto.INT16: "S16",
    TensorProto.UINT16: "U16",
}


def fc_weight_dtype_by_shape(onnx_path: str):
    """FullyConnected の 2D 重み initializer を dtype 別・shape 別に集計して返す。"""
    m = onnx.load(onnx_path, load_external_data=False)
    init = {i.name: (i.data_type, tuple(i.dims)) for i in m.graph.initializer}
    fp32 = collections.Counter()
    quant = collections.Counter()
    n_fc = 0
    for n in m.graph.node:
        if "FullyConnected" not in n.op_type:
            continue
        n_fc += 1
        for inp in n.input:
            if inp in init and len(init[inp][1]) == 2:
                dt, sh = init[inp]
                if dt == TensorProto.FLOAT:
                    fp32[sh] += 1
                else:
                    quant[sh] += 1
                break
    return n_fc, fp32, quant


def parse_shape(s: str):
    dims, _, cnt = s.partition("=")
    shape = tuple(int(x) for x in dims.split(","))
    return shape, int(cnt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="built quant_axmodel.onnx")
    ap.add_argument(
        "--expect",
        action="append",
        default=[],
        help='intended FP32 shape=count, e.g. "1280,512=1" (repeatable). '
        "省略時は表示のみ。",
    )
    args = ap.parse_args()

    n_fc, fp32, quant = fc_weight_dtype_by_shape(args.onnx)
    print(f"total FullyConnected = {n_fc}")
    print(f"\nFP32-weight FC by shape ({sum(fp32.values())} total):")
    for sh, c in fp32.most_common():
        print(f"  {c:4d}  {sh}")
    print(f"\nquantized-weight FC by shape ({sum(quant.values())} total):")
    for sh, c in quant.most_common():
        print(f"  {c:4d}  {sh}")

    if not args.expect:
        return 0

    expected = dict(parse_shape(s) for s in args.expect)
    ok = True
    print("\n=== assert intended FP32 set ===")
    for sh, want in expected.items():
        got = fp32.get(sh, 0)
        mark = "OK" if got == want else "MISMATCH"
        if got != want:
            ok = False
        print(f"  {sh}: want FP32={want}, got={got}  [{mark}]")
    # intended に無い shape が FP32 になっていないかも警告
    extra = {sh: c for sh, c in fp32.items() if sh not in expected}
    if extra:
        print(f"  WARNING: 未指定の shape が FP32 化: {dict(extra)}")
    if not ok:
        print("\nFP32 layer set does NOT match intent — name-mapping silently dropped layers. "
              "ABORT before any wav run.", file=sys.stderr)
        return 1
    print("\nFP32 layer set matches intent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
