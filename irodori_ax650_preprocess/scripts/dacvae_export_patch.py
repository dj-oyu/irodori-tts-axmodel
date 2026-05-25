#!/usr/bin/env python3
"""
dacvae_export_patch.py

DACVAE decoder を Pulsar2(AX8850) で NPU 化するための export 時 monkeypatch。
DiT 側の `rope_export_patch` の DACVAE 版（本体無改変・export 時のみ適用）。

## なぜ必要か（FINDINGS「DACVAE を NPU 化できた」節）
Pulsar2 6.0 は Snake の ONNX パターン `Mul(α,x)→Sin→Pow(·,2)→Mul(1/α,·)→Add(x,·)` を
**単一 `AxQuantizedSnake` op に融合**し、その融合 op の tiler/quantizer **だけ**が壊れている
（小 T で broadcast バグ、大 T で NoTilerException）。Snake は時間方向 pointwise なので
**素の primitive なら tile 可能** = ツール側の融合バグ。

## 回避策（検証済, fp32 等価 max_abs_err 7.3e-6 @ T=119）
三角恒等式 `sin²(αx) = (1 − cos(2αx))/2` で書き換えると `Sin`/`Pow` が消え、融合がマッチしない。
素の `Cos/Mul/Sub/Add` が残り tiler が時間分割できる → 全 decoder が単一 NPU subgraph でビルド成功。
数値はもとの Snake と等価（恒等式）。
"""

from __future__ import annotations

import torch


def _snake_forward_cos(self, x):
    """Snake1d.forward を `x + (1/α)·sin²(αx)` → `x + (1/α)·0.5·(1−cos(2αx))` に。

    元実装と数値等価（sin²(t)=(1−cos2t)/2）。α は (1,C,1) のまま broadcast。
    α の reciprocal は 1e-9 を足してゼロ割回避（元実装踏襲）。
    """
    shape = x.shape
    x2 = x.reshape(shape[0], shape[1], -1)
    recip = (self.alpha + 1e-9).reciprocal()
    x2 = x2 + recip * 0.5 * (1.0 - torch.cos(2.0 * self.alpha * x2))
    return x2.reshape(shape)


def apply_snake_cos_patch() -> None:
    """`dacvae.nn.layers.Snake1d.forward` を cos 恒等式版に差し替える（export 直前に 1 回）。"""
    import dacvae.nn.layers as Lmod

    Lmod.Snake1d.forward = _snake_forward_cos


def fold_weight_norm_all(m: torch.nn.Module) -> int:
    """weight_norm を fold（export 前）。dacvae は **hook ベース**（weight_g/weight_v）を使うため
    parametrize ベースの remove だけでは取り切れない。両方式を fold する。"""
    import torch.nn.utils.parametrize as parametrize

    n = 0
    # parametrize ベース（torch>=1.12 の新 API）
    for _, mod in m.named_modules():
        ph = getattr(mod, "parametrizations", None)
        if ph is not None and "weight" in ph:
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)
            n += 1
    # hook ベース（torch.nn.utils.weight_norm。dacvae はこちら）
    for mod in m.modules():
        if hasattr(mod, "weight_g") and hasattr(mod, "weight_v"):
            try:
                torch.nn.utils.remove_weight_norm(mod)
                n += 1
            except Exception:
                pass
    return n
