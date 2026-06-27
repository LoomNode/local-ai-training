"""Project an FP weight matrix to its best per-row (scale, integer-code) representation.

This is post-training quantization of a single matrix: for each output row, find the FP32
scale and integer codes in [-max_code, max_code] that minimize reconstruction MSE
||W_row - scale * code_row||^2. It defines the *representation floor* the ratchet update rule
is bounded by, and serves as the PTQ baseline in the projection-oracle diagnostic.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = torch.finfo(torch.float32).tiny


def project_to_codes(weight: Tensor, max_code: int, *, iters: int = 8) -> tuple[Tensor, Tensor]:
    """Best per-row (code, scale) approximation of ``weight``, minimizing row MSE.

    Alternating minimization seeded from the training init's ``row_max/max_code`` scale:
    with the scale fixed, the optimal integer codes are ``round(W/scale)`` clamped; with the
    codes fixed, the optimal scale is the least-squares ``<W,code>/<code,code>`` per row.
    A few iterations converge (early-out when the codes stop changing).

    Returns ``(code int8 [out,in] in [-max_code,max_code], scale float32 [out])``.
    """
    if max_code < 1:
        raise ValueError("max_code must be >= 1")
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D matrix")
    w = weight.detach().float()
    row_max = w.abs().amax(dim=1)
    scale = (row_max / max_code).clamp_min(_EPS)
    code = torch.round(w / scale[:, None]).clamp(-max_code, max_code)
    for _ in range(iters):
        cc = (code * code).sum(dim=1)
        wc = (w * code).sum(dim=1)
        new_scale = torch.where(cc > 0, wc / cc, scale).clamp_min(_EPS)
        new_code = torch.round(w / new_scale[:, None]).clamp(-max_code, max_code)
        converged = torch.equal(new_code, code)
        scale, code = new_scale, new_code
        if converged:
            break
    return code.to(torch.int8), scale.to(torch.float32)


def reconstruction_mse(weight: Tensor, code: Tensor, scale: Tensor) -> Tensor:
    """Mean squared error of the dequantized ``scale*code`` against ``weight``."""
    approx = code.to(torch.float32) * scale.to(torch.float32)[:, None]
    return (weight.detach().float() - approx).square().mean()
