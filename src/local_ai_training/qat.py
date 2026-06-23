"""STE-QAT control linear.

A quantization-aware-training control: keeps a full-precision master weight (trained by
Adam) and quantizes it in the forward pass with the ratchet's exact per-row quantizer,
using a straight-through estimator. This is deliberately NOT a DiscreteRatchetLinear — it
HAS master weights and exists only as the control that isolates the cost of dropping them.
It is therefore outside audit_no_master_weights' scope.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class QATLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        max_code: int,
        initial_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        if max_code not in (2, 3, 4):
            raise ValueError("max_code must be 2 (quinary), 3 (septenary), or 4 (nonary)")
        self.in_features = in_features
        self.out_features = out_features
        self.max_code = max_code
        if initial_weight is None:
            weight = torch.empty(out_features, in_features, dtype=torch.float32)
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        else:
            if initial_weight.shape != (out_features, in_features):
                raise ValueError(
                    f"initial_weight must have shape {(out_features, in_features)}, "
                    f"got {tuple(initial_weight.shape)}"
                )
            weight = initial_weight.detach().to(dtype=torch.float32).clone()
        self.weight = nn.Parameter(weight)

    def quantized_weight(self) -> Tensor:
        # Matches DiscreteRatchetLinear's quantizer (ratchet.py:317-319): per-row
        # row_max_abs/max_code scale (clamped to finfo.eps), round-to-code, dequantize.
        # NOTE: scale is recomputed LIVE from the current master every forward (the master
        # moves under Adam, so scale must track it) — unlike the ratchet, which freezes scale
        # in a buffer. They coincide only at step 0. Do not "fix" this to a frozen buffer.
        scale = (
            self.weight.detach().abs().amax(dim=1, keepdim=True) / self.max_code
        ).clamp_min(torch.finfo(torch.float32).eps)
        code = torch.round(self.weight / scale).clamp(-self.max_code, self.max_code)
        # Straight-through: forward value is code*scale; gradient is identity to weight,
        # including saturated entries (the whole quantized term is detached).
        return self.weight + (code * scale - self.weight).detach()

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.quantized_weight())
