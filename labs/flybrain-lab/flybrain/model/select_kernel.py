"""Slot selection for the sparse key in one kernel (2026-11-05): score all n_key slots of a row from the key matrix's
4 nonzeros per slot (signed picks from x, no 512-wide matmul), threshold, and keep the top-k by 4 rounds of max.

Replaces `relu(x @ proj.T - theta).topk(k)` on the no-gradient selection path; the selected slots are then rescored
differentiably by KeyCode.sparse_code as before. The only difference from torch.topk is which slot wins an exact tie
(this kernel: the lowest index). Ties at a positive score need two slots with identical rows and thresholds; ties at
zero select slots whose value is zero, which read and write nothing.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


def select_available(t: Tensor) -> bool:
    return _HAVE_TRITON and t.is_cuda


if _HAVE_TRITON:

    @triton.jit
    def _select(x_ptr, cols_ptr, signs_ptr, theta_ptr, idx_ptr, N, D, NK: tl.constexpr, CLAWS: tl.constexpr, K: tl.constexpr, RB: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * RB + tl.arange(0, RB)                       # [RB]
        rmask = rows < N
        slots = tl.arange(0, NK)                                 # [NK]
        z = tl.zeros([RB, NK], dtype=tl.float32)
        for c in tl.static_range(CLAWS):
            col = tl.load(cols_ptr + slots * CLAWS + c)          # [NK] int32
            sgn = tl.load(signs_ptr + slots * CLAWS + c)         # [NK]
            xv = tl.load(x_ptr + rows[:, None] * D + col[None, :], mask=rmask[:, None], other=0.0)   # [RB, NK]
            z += xv * sgn[None, :]
        theta = tl.load(theta_ptr + slots)
        s = tl.maximum(z - theta[None, :], 0.0)
        for j in tl.static_range(K):
            best = tl.max(s, axis=1)                             # [RB]
            arg = tl.argmax(s, axis=1)                           # lowest index among ties
            tl.store(idx_ptr + rows * K + j, arg.to(tl.int64), mask=rmask)
            s = tl.where(slots[None, :] == arg[:, None], float("-inf"), s)
            best = best  # the value is recomputed differentiably by the caller


def fused_select(x: Tensor, cols: Tensor, signs: Tensor, theta: Tensor, k: int, rows_per_program: int = 4, num_warps: int = 8) -> Tensor:
    """idx [N, k] int64 of the top-k slots by relu(x . proj_row - theta), for x [N, D] fp32 on CUDA."""
    assert x.dim() == 2
    N, D = x.shape
    n_key, claws = cols.shape
    if n_key & (n_key - 1):
        raise ValueError("fused_select needs a power-of-two number of slots")
    x = x.contiguous().float()
    cols32 = cols.to(torch.int32).contiguous(); signs = signs.float().contiguous(); theta = theta.float().contiguous()
    idx = torch.empty(N, k, dtype=torch.int64, device=x.device)
    if N:
        with torch.cuda.device(x.device):
            _select[(triton.cdiv(N, rows_per_program),)](x, cols32, signs, theta, idx, N, D, NK=n_key, CLAWS=claws, K=k, RB=rows_per_program, num_warps=num_warps)
    return idx
