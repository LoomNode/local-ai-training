"""The diagonal recurrence h_t = a_t h_{t-1} + b_t as fused kernels (2026-11-05).

Two-pass chunked form, so the walk is parallel over chunks as well as channels (a single lane walking all of T was
latency-bound at ~2 ms per layer; chunks of CH give T/CH times the parallelism):
  forward:  (1) each chunk walks itself from h = 0 and reports (prod a, h at its end);
            (2) a tiny walk over the chunk ends gives the carry into each chunk;
            (3) each chunk walks itself again from its carry and writes h.
  backward: the same three passes in reverse for g_t = dh_t + a_{t+1} g_{t+1}; then db_t = g_t, da_t = g_t h_{t-1},
            dh0 = a_0 g_0.
Exact fp32 arithmetic of the recurrence itself (no closed form, no division). `scan_available()` is False without
CUDA or Triton; callers then use MinGRULayer._scan_pairs_full.
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

CH = 128      # positions per chunk
BLOCK_D = 32  # channels per program


def scan_available(t: Tensor) -> bool:
    return _HAVE_TRITON and t.is_cuda


if _HAVE_TRITON:

    @triton.jit
    def _fwd_ends(a_ptr, b_ptr, A_ptr, H_ptr, T, D, NC, CH: tl.constexpr, BLOCK_D: tl.constexpr):
        """Pass 1: per chunk, (prod a, h at the chunk end from h = 0)."""
        pb = tl.program_id(0); pd = tl.program_id(1); pc = tl.program_id(2)
        offs = pd * BLOCK_D + tl.arange(0, BLOCK_D); mask = offs < D
        base = pb * T * D
        A = tl.full([BLOCK_D], 1.0, dtype=tl.float32); h = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in range(CH):
            t = pc * CH + i
            m = mask & (t < T)
            p = base + t * D + offs
            a = tl.load(a_ptr + p, mask=m, other=1.0); b = tl.load(b_ptr + p, mask=m, other=0.0)
            h = a * h + b; A = A * a
        q = (pb * NC + pc) * D + offs
        tl.store(A_ptr + q, A, mask=mask); tl.store(H_ptr + q, h, mask=mask)

    @triton.jit
    def _carry(A_ptr, H_ptr, h0_ptr, C_ptr, D, NC, HAS_H0: tl.constexpr, BLOCK_D: tl.constexpr):
        """Pass 2: the carry into each chunk, c_0 = h0, c_{i+1} = A_i c_i + H_i."""
        pb = tl.program_id(0); pd = tl.program_id(1)
        offs = pd * BLOCK_D + tl.arange(0, BLOCK_D); mask = offs < D
        if HAS_H0:
            c = tl.load(h0_ptr + pb * D + offs, mask=mask, other=0.0)
        else:
            c = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in range(NC):
            q = (pb * NC + i) * D + offs
            tl.store(C_ptr + q, c, mask=mask)
            A = tl.load(A_ptr + q, mask=mask, other=1.0); H = tl.load(H_ptr + q, mask=mask, other=0.0)
            c = A * c + H

    @triton.jit
    def _fwd_walk(a_ptr, b_ptr, C_ptr, h_ptr, T, D, NC, CH: tl.constexpr, BLOCK_D: tl.constexpr):
        """Pass 3: each chunk from its carry, writing h."""
        pb = tl.program_id(0); pd = tl.program_id(1); pc = tl.program_id(2)
        offs = pd * BLOCK_D + tl.arange(0, BLOCK_D); mask = offs < D
        base = pb * T * D
        h = tl.load(C_ptr + (pb * NC + pc) * D + offs, mask=mask, other=0.0)
        for i in range(CH):
            t = pc * CH + i
            m = mask & (t < T)
            p = base + t * D + offs
            a = tl.load(a_ptr + p, mask=m, other=1.0); b = tl.load(b_ptr + p, mask=m, other=0.0)
            h = a * h + b
            tl.store(h_ptr + p, h, mask=m)

    @triton.jit
    def _bwd_ends(a_ptr, dh_ptr, P_ptr, G_ptr, A0_ptr, T, D, NC, CH: tl.constexpr, BLOCK_D: tl.constexpr):
        """Reverse pass 1: per chunk [s, e], g at s from an incoming 0 (G), prod_{r=s+1..e} a_r (P), and a_s (A0)."""
        pb = tl.program_id(0); pd = tl.program_id(1); pc = tl.program_id(2)
        offs = pd * BLOCK_D + tl.arange(0, BLOCK_D); mask = offs < D
        base = pb * T * D
        g = tl.zeros([BLOCK_D], dtype=tl.float32); P = tl.full([BLOCK_D], 1.0, dtype=tl.float32)
        a_next = tl.full([BLOCK_D], 1.0, dtype=tl.float32)   # a_{t+1} where a real position follows t
        nxt = tl.zeros([BLOCK_D], dtype=tl.float32)          # 1 once a real position follows (0 past the end: nothing flows in)
        a_s = tl.full([BLOCK_D], 1.0, dtype=tl.float32)
        for i in range(CH):
            t = pc * CH + (CH - 1 - i)
            m = mask & (t < T)
            p = base + t * D + offs
            dh = tl.load(dh_ptr + p, mask=m, other=0.0); a = tl.load(a_ptr + p, mask=m, other=1.0)
            g = dh + nxt * a_next * g
            P = tl.where(m, P * (nxt * a_next + (1.0 - nxt)), P)   # product of a over the real positions after t, within the chunk
            a_next = tl.where(m, a, a_next); nxt = tl.where(m, 1.0, nxt)
            a_s = tl.where(m, a, a_s)
        q = (pb * NC + pc) * D + offs
        tl.store(G_ptr + q, g, mask=mask); tl.store(P_ptr + q, P, mask=mask); tl.store(A0_ptr + q, a_s, mask=mask)

    @triton.jit
    def _bwd_carry(P_ptr, G_ptr, A0_ptr, C_ptr, D, NC, BLOCK_D: tl.constexpr):
        """Reverse pass 2: the incoming gradient to each chunk's end, from the chunk after it."""
        pb = tl.program_id(0); pd = tl.program_id(1)
        offs = pd * BLOCK_D + tl.arange(0, BLOCK_D); mask = offs < D
        gin = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in range(NC):
            c = NC - 1 - i
            q = (pb * NC + c) * D + offs
            tl.store(C_ptr + q, gin, mask=mask)
            G = tl.load(G_ptr + q, mask=mask, other=0.0); P = tl.load(P_ptr + q, mask=mask, other=1.0); A0 = tl.load(A0_ptr + q, mask=mask, other=1.0)
            gin = A0 * (G + P * gin)                    # g at the chunk's first position, times its a: what flows into the chunk before

    @triton.jit
    def _bwd_walk(a_ptr, h_ptr, h0_ptr, dh_ptr, C_ptr, da_ptr, db_ptr, dh0_ptr, T, D, NC, HAS_H0: tl.constexpr, CH: tl.constexpr, BLOCK_D: tl.constexpr):
        """Reverse pass 3: each chunk from its incoming gradient, writing da and db (and dh0 from chunk 0)."""
        pb = tl.program_id(0); pd = tl.program_id(1); pc = tl.program_id(2)
        offs = pd * BLOCK_D + tl.arange(0, BLOCK_D); mask = offs < D
        base = pb * T * D
        gin = tl.load(C_ptr + (pb * NC + pc) * D + offs, mask=mask, other=0.0)
        g = tl.zeros([BLOCK_D], dtype=tl.float32)
        a_next = tl.full([BLOCK_D], 1.0, dtype=tl.float32)
        for i in range(CH):
            t = pc * CH + (CH - 1 - i)
            m = mask & (t < T)
            p = base + t * D + offs
            dh = tl.load(dh_ptr + p, mask=m, other=0.0); a = tl.load(a_ptr + p, mask=m, other=1.0)
            # the chunk's last position takes the incoming term (it already carries a_{t+1}); masked positions past
            # the end read dh = 0, a = 1 and pass g through unchanged
            g = dh + tl.where(i == 0, gin, a_next * g)
            if t > 0:
                h_prev = tl.load(h_ptr + p - D, mask=m, other=0.0)
            else:
                if HAS_H0:
                    h_prev = tl.load(h0_ptr + pb * D + offs, mask=mask, other=0.0)
                else:
                    h_prev = tl.zeros([BLOCK_D], dtype=tl.float32)
            tl.store(db_ptr + p, g, mask=m); tl.store(da_ptr + p, g * h_prev, mask=m)
            a_next = a
        if HAS_H0:
            if pc == 0:
                tl.store(dh0_ptr + pb * D + offs, a_next * g, mask=mask)


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, h0: Tensor | None) -> Tensor:
        a = a.contiguous(); b = b.contiguous()
        B, T, D = a.shape
        h = torch.empty_like(a)
        has_h0 = h0 is not None
        h0c = h0.contiguous() if has_h0 else a          # a dummy pointer when absent (never read)
        if T:
            NC = triton.cdiv(T, CH); nd = triton.cdiv(D, BLOCK_D)
            A = torch.empty(B, NC, D, device=a.device, dtype=a.dtype); H = torch.empty_like(A); C = torch.empty_like(A)
            with torch.cuda.device(a.device):          # Triton launches on the current device; the lab's tensors live on cuda:1
                _fwd_ends[(B, nd, NC)](a, b, A, H, T, D, NC, CH=CH, BLOCK_D=BLOCK_D, num_warps=1)
                _carry[(B, nd)](A, H, h0c, C, D, NC, HAS_H0=has_h0, BLOCK_D=BLOCK_D, num_warps=1)
                _fwd_walk[(B, nd, NC)](a, b, C, h, T, D, NC, CH=CH, BLOCK_D=BLOCK_D, num_warps=1)
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
        if T:
            NC = triton.cdiv(T, CH); nd = triton.cdiv(D, BLOCK_D)
            P = torch.empty(B, NC, D, device=a.device, dtype=a.dtype); G = torch.empty_like(P); A0 = torch.empty_like(P); C = torch.empty_like(P)
            with torch.cuda.device(a.device):
                _bwd_ends[(B, nd, NC)](a, dh, P, G, A0, T, D, NC, CH=CH, BLOCK_D=BLOCK_D, num_warps=1)
                _bwd_carry[(B, nd)](P, G, A0, C, D, NC, BLOCK_D=BLOCK_D, num_warps=1)
                _bwd_walk[(B, nd, NC)](a, h, h0c, dh, C, da, db, dh0, T, D, NC, HAS_H0=ctx.has_h0, CH=CH, BLOCK_D=BLOCK_D, num_warps=1)
        elif ctx.has_h0:
            dh0 = dh.new_zeros(B, D)
        return da, db, (dh0 if ctx.has_h0 else None)


def fused_scan(a: Tensor, b: Tensor, h0: Tensor | None) -> Tensor:
    """h_t = a_t h_{t-1} + b_t over [B, T, D] fp32 CUDA tensors, h0 [B, D] or None."""
    return _Scan.apply(a.float(), b.float(), None if h0 is None else h0.float())
