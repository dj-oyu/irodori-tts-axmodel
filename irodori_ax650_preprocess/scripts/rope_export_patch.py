#!/usr/bin/env python3
"""
rope_export_patch.py

Irodori-TTS の RoPE は `torch.view_as_complex` / `view_as_real`（model.py:24-36）を使う。
これは TorchScript / dynamo どちらの ONNX exporter でも未対応（"No decompositions registered
for the complex-valued input"）。

export 時だけ、数値的に等価な実数値（cos/sin）実装へ monkeypatch する。
複素数 (xr+ i*xi) * (cos + i*sin) = (xr*cos - xi*sin) + i*(xr*sin + xi*cos) を素直に展開。

`apply_rope_patch(model)` を export 直前に 1 回呼ぶ。元コードには手を入れない。
スクリプト単体実行で original との数値等価性を検証できる。
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

import irodori_tts.model as M

_ORIG_SDPA = F.scaled_dot_product_attention


def _sdpa_additive_mask(query, key, value, attn_mask=None, dropout_p=0.0,
                        is_causal=False, scale=None, enable_gqa=False):
    """
    bool の attn_mask を additive float mask に変換してから SDPA を呼ぶ。
    bool mask → SDPA 分解は Where/IsNaN（all-masked 行の softmax=NaN ガード）を生み、
    Pulsar2 が IsNaN 非対応で落ちる。本モデルは self(latent) token が常に有効で
    全マスク行が無いため、masked 位置に大きな負値(finfo.min)を足す additive 形と等価。
    → Add のみに分解され IsNaN/Where が消える。
    """
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        # 既定は finfo.min（数値的に完全マスク）。ただし PTQ では finfo.min(-3.4e38) が
        # SmoothQuant/MSE 等の activation 統計を汚染し量子化が壊れる（実測 0.89→0.35）。
        # IRODORI_MASK_NEG で穏当な負値(-1e4 等)に差し替え可能。-1e4 でも fp32 では
        # exp(score-1e4)→0 で完全マスクのまま＝出力は数値等価、量子化レンジだけ健全化。
        _env = os.environ.get("IRODORI_MASK_NEG")
        neg = float(_env) if _env else torch.finfo(query.dtype).min
        attn_mask = (~attn_mask).to(query.dtype) * neg  # 0 / neg の additive bias
    kw = {}
    if scale is not None:
        kw["scale"] = scale
    return _ORIG_SDPA(query, key, value, attn_mask=attn_mask,
                      dropout_p=dropout_p, is_causal=is_causal, **kw)


def precompute_freqs_cis_real(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """元の precompute_freqs_cis と同じ角度で、複素の代わりに [end, dim/2, 2]=(cos,sin) を返す。"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, dtype=torch.float32)
    f = torch.outer(t, freqs)  # [end, dim/2]
    return torch.stack([torch.cos(f), torch.sin(f)], dim=-1)  # [end, dim/2, 2]


def apply_rotary_emb_real(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    x: (B, S, H, Dh), freqs: (S, Dh/2, 2)=(cos,sin)。元 apply_rotary_emb と同じ規約
    （reshape(...,-1,2) の [...,0]=実部, [...,1]=虚部）で実数展開する。
    元実装と同様に内部 fp32 で計算し最後に type_as。
    """
    # 注: `-1` を使うと dynamic-axes export 時に onnxruntime が次元推論を誤る
    #     (Reshape target に H が混入する不具合)。x.shape から明示次元で reshape する。
    b, s, h, d = x.shape  # B,S は dynamic(SymInt) 可、H,D は固定
    xf = x.float().reshape(b, s, h, d // 2, 2)  # [B,S,H,Dh/2,2]
    xr = xf[..., 0]
    xi = xf[..., 1]
    cos = freqs[..., 0].float()[None, :, None, :]  # [1,S,1,Dh/2]
    sin = freqs[..., 1].float()[None, :, None, :]
    out_r = xr * cos - xi * sin
    out_i = xr * sin + xi * cos
    out = torch.stack([out_r, out_i], dim=-1).reshape(b, s, h, d)
    return out.type_as(x)


def apply_rope_patch(model: torch.nn.Module | None = None) -> None:
    """module-global の RoPE 関数を実数版に差し替え、必要なら freqs cache を無効化する。"""
    M.precompute_freqs_cis = precompute_freqs_cis_real
    M.apply_rotary_emb = apply_rotary_emb_real
    # 既存の複素 cache を破棄し、patched precompute で再計算させる。
    if model is not None:
        for name, mod in model.named_modules():
            if hasattr(mod, "_freqs_cis_cache"):
                mod._freqs_cis_cache = torch.zeros(0)


def _rmsnorm_forward_div(self, x):
    """
    RMSNorm の rsqrt を sqrt+div に書き換える（数値等価, rsqrt(v)==1/sqrt(v)）。
    rsqrt は ONNX で Sqrt+Reciprocal に分解され、Pulsar2 quant が Reciprocal 非対応で落ちる。
    Sqrt+Div は Reciprocal を出さない。
    """
    x_dtype = x.dtype
    xf = x.float()
    var = (xf * xf).mean(dim=-1, keepdim=True)
    xf = xf / torch.sqrt(var + self.eps)
    return (xf * self.weight).to(x_dtype)


def apply_rmsnorm_patch() -> None:
    """RMSNorm.forward を rsqrt→sqrt/div 版に差し替え（Reciprocal 除去）。"""
    M.RMSNorm.forward = _rmsnorm_forward_div


def _lowrank_adaln_forward_div(self, x, cond_embed):
    """LowRankAdaLN 内の inline rsqrt も sqrt/div へ（24 個 = 12 block x 2）。"""
    shift, scale, gate = cond_embed.chunk(3, dim=-1)
    shift = self.shift_up(self.shift_down(F.silu(shift))) + shift
    scale = self.scale_up(self.scale_down(F.silu(scale))) + scale
    gate = self.gate_up(self.gate_down(F.silu(gate))) + gate
    x_dtype = x.dtype
    xf = x.float()
    xf = xf / torch.sqrt((xf * xf).mean(dim=-1, keepdim=True) + self.eps)
    xf = xf * (1.0 + scale) + shift
    return xf.to(x_dtype), torch.tanh(gate)


def apply_adaln_patch() -> None:
    """LowRankAdaLN.forward の rsqrt を除去。"""
    M.LowRankAdaLN.forward = _lowrank_adaln_forward_div


def apply_sdpa_patch() -> None:
    """F.scaled_dot_product_attention を bool→additive mask 版に差し替え（IsNaN 除去）。"""
    F.scaled_dot_product_attention = _sdpa_additive_mask
    M.F.scaled_dot_product_attention = _sdpa_additive_mask


def apply_export_patches(model: torch.nn.Module | None = None) -> None:
    """export 前にまとめて適用（RoPE 実数化 + SDPA additive mask + RMSNorm rsqrt 除去）。"""
    apply_rope_patch(model)
    apply_sdpa_patch()
    apply_rmsnorm_patch()
    apply_adaln_patch()


def _verify(seq: int = 17, heads: int = 20, head_dim: int = 64) -> None:
    """original(複素) と patched(実数) の数値等価性を確認する。"""
    import importlib

    importlib.reload(M)  # original を確実にロード
    torch.manual_seed(0)
    x = torch.randn(2, seq, heads, head_dim)

    freqs_c = M.precompute_freqs_cis(head_dim, seq)[:seq]
    ref = M.apply_rotary_emb(x, freqs_c)

    freqs_r = precompute_freqs_cis_real(head_dim, seq)[:seq]
    got = apply_rotary_emb_real(x, freqs_r)

    max_abs = (ref - got).abs().max().item()
    print(f"[verify fp32] max_abs_err = {max_abs:.3e}")
    assert max_abs < 1e-5, "real RoPE diverges from complex RoPE"

    # fp16 経路
    xh = x.half()
    ref_h = M.apply_rotary_emb(xh, freqs_c)
    got_h = apply_rotary_emb_real(xh, freqs_r)
    max_abs_h = (ref_h.float() - got_h.float()).abs().max().item()
    print(f"[verify fp16] max_abs_err = {max_abs_h:.3e}")
    print("OK: real RoPE is numerically equivalent")


if __name__ == "__main__":
    _verify()
