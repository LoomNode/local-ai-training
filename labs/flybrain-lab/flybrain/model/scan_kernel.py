"""The diagonal recurrence h_t = a_t h_{t-1} + b_t as one fused kernel per direction (2026-11-05).

Each program owns BLOCK_D channels of one batch row and walks t sequentially in registers: one read of a and b, one
write of h, nothing in between. The backward is the same walk in reverse: g_t = dh_t + a_{t+1} g_{t+1}, db_t = g_t,
da_t = g_t h_{t-1}, dh0 = a_0 g_0. Exact fp32 arithmetic of the recurrence itself (no closed form, no division).
`scan_available()` is False without CUDA or Triton; callers then use MinGRULayer._scan_pairs.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - CPU-only boxes
    _HAVE_TRITON = False


def scan_available(t: Tensor) -> bool:
    return _HAVE_TRITON and t.is_cuda


if _HAVE_TRITON:

    @triton.jit
    def _fwd(a_ptr, b_ptr, h0_ptr, h_ptr, T, D, HAS_H0: tl.constexpr, BLOCK_D: tl.constexpr):
        pb = tl.program_id(0); pd = tl.program_id(1)
        offs = pd * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D
        base = pb * T * D
        if HAS_H0:
            h = tl.load(h0_ptr + pb * D + offs, mask=mask, other=0.0)
        else:
            h = tl.zeros([BLOCK_D], dtype=tl.float32)
        for t in range(T):
            p = base + t * D + offs
            a = tl.load(a_ptr + p, mask=mask, other=1.0)
            b = tl.load(b_ptr + p, mask=mask, other=0.0)
            h = a * h + b
            tl.store(h_ptr + p, h, mask=mask)

    @triton.jit
    def _bwd(a_ptr, h_ptr, h0_ptr, dh_ptr, da_ptr, db_ptr, dh0_ptr, T, D, HAS_H0: tl.constexpr, BLOCK_D: tl.constexpr):
        pb = tl.program_id(0); pd = tl.program_id(1)
        offs = pd * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D
        base = pb * T * D
        g = tl.zeros([BLOCK_D], dtype=tl.float32)
        a_next = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in range(T):
            t = T - 1 - i
            p = base + t * D + offs
            dh = tl.load(dh_ptr + p, mask=mask, other=0.0)
            a = tl.load(a_ptr + p, mask=mask, other=1.0)
            g = dh + a_next * g
            if t > 0:
                h_prev = tl.load(h_ptr + p - D, mask=mask, other=0.0)
            else:
                if HAS_H0:
                    h_prev = tl.load(h0_ptr + pb * D + offs, mask=mask, other=0.0)
                else:
                    h_prev = tl.zeros([BLOCK_D], dtype=tl.float32)
            tl.store(db_ptr + p, g, mask=mask)
            tl.store(da_ptr + p, g * h_prev, mask=mask)
            a_next = a
        if HAS_H0:
            tl.store(dh0_ptr + pb * D + offs, a_next * g, mask=mask)


class _Scan(torch.autograd.Function):
    BLOCK_D = 32

    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, h0: Tensor | None) -> Tensor:
        a = a.contiguous(); b = b.contiguous()
        B, T, D = a.shape
        h = torch.empty_like(a)
        has_h0 = h0 is not None
        h0c = h0.contiguous() if has_h0 else a  # a dummy pointer when absent (never read)
        grid = (B, triton.cdiv(D, _Scan.BLOCK_D))
        if T:
            with torch.cuda.device(a.device):   # Triton launches on the current device; the lab's tensors live on cuda:1
                _fwd[grid](a, b, h0c, h, T, D, HAS_H0=has_h0, BLOCK_D=_Scan.BLOCK_D, num_warps=1)
        ctx.save_for_backward(a, h, h0c)
        ctx.has_h0 = has_h0
        return h

    @staticmethod
    def backward(ctx, dh: Tensor):
        a, h, h0c = ctx.saved_tensors
        B, T, D = a.shape
        dh = dh.contiguous()
        da, db = torch.empty_like(a), torch.empty_like(a)
        dh0 = torch.empty(B, D, device=a.device, dtype=a.dtype) if ctx.has_h0 else a
        grid = (B, triton.cdiv(D, _Scan.BLOCK_D))
        if T:
            with torch.cuda.device(a.device):
                _bwd[grid](a, h, h0c, dh, da, db, dh0, T, D, HAS_H0=ctx.has_h0, BLOCK_D=_Scan.BLOCK_D, num_warps=1)
        elif ctx.has_h0:
            dh0 = dh.new_zeros(B, D)
        return da, db, (dh0 if ctx.has_h0 else None)


def fused_scan(a: Tensor, b: Tensor, h0: Tensor | None) -> Tensor:
    """h_t = a_t h_{t-1} + b_t over [B, T, D] fp32 CUDA tensors, h0 [B, D] or None."""
    return _Scan.apply(a.float(), b.float(), None if h0 is None else h0.float())
