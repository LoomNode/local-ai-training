"""Int8 master-weight linear layer: stochastic-rounded sign update, no optimizer state.

Iso-state competitor to the ratchet. Persistent state per matrix is one int8 buffer
(``weight_int8``, out x in) plus one FP32 per-output-row scale (``_scale``), frozen at
init -- the same one-byte-per-weight-plus-row-scale budget the ratchet spends on a
packed code+pressure nibble pair. Unlike the ratchet, the int8 value itself *is* the
master weight (no separate code/pressure split, no pressure accumulator); the update
is a stateless, stochastically-rounded sign step. See
``docs/superpowers/specs/2026-09-15-int8-master-iso-state-design.md``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .ratchet import RatchetUpdateStats

_INT8_MAX = 127


def _coarse_bin_counts(values: Tensor, *, bin_width: int = 16) -> dict[int, int]:
    """Value histogram in coarse bins (bin key = the bin's lower bound).

    The int8 grid spans [-127, 127] (255 states) -- far wider than the ratchet's
    3..15-state code, so a per-value histogram would be needlessly fine. Binning by
    ``bin_width`` keeps the reported histogram to a couple dozen buckets.
    """
    flat = values.detach().flatten()
    if flat.numel() == 0:
        return {}
    bins = torch.div(flat.to(torch.int64), bin_width, rounding_mode="floor")
    lo = int(bins.min().item())
    hi = int(bins.max().item())
    counts: dict[int, int] = {}
    for bucket in range(lo, hi + 1):
        count = int((bins == bucket).sum().item())
        if count:
            counts[bucket * bin_width] = count
    return counts


class Int8MasterLinear(nn.Module):
    """Linear layer whose only persistent state is an int8 weight and a row scale.

    Initialization mirrors ``DiscreteRatchetLinear``: the same seeded kaiming-uniform
    reference (so a shared seed yields the same logical FP init across arms), then
    ``scale = row_max_abs / 127`` and ``weight_int8 = round(reference / scale)``. The
    scale is frozen for the module's lifetime (a buffer, never recomputed).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        initial_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features

        if initial_weight is None:
            reference = torch.empty(out_features, in_features, dtype=torch.float32)
            nn.init.kaiming_uniform_(reference, a=5**0.5)
        else:
            if initial_weight.shape != (out_features, in_features):
                raise ValueError(
                    f"initial_weight must have shape {(out_features, in_features)}, "
                    f"got {tuple(initial_weight.shape)}"
                )
            reference = initial_weight.detach().to(dtype=torch.float32)

        row_max = reference.abs().amax(dim=1)
        scale = (row_max / _INT8_MAX).clamp_min(torch.finfo(torch.float32).eps)
        code = torch.round(reference / scale[:, None]).clamp(-_INT8_MAX, _INT8_MAX)
        self.register_buffer("weight_int8", code.to(torch.int8))
        self.register_buffer("_scale", scale)

        # Deliberately non-persistent: exists only between forward and the update that
        # consumes its gradient, exactly like the ratchet's transient effective weight.
        self._effective_weight: Tensor | None = None

    @classmethod
    def from_reference(cls, reference: Tensor) -> Int8MasterLinear:
        if reference.ndim != 2:
            raise ValueError("reference weight must be a matrix")
        return cls(reference.shape[1], reference.shape[0], initial_weight=reference)

    @property
    def scale(self) -> Tensor:
        return self._scale

    @property
    def persistent_state_bytes(self) -> int:
        return (
            self.weight_int8.numel() * self.weight_int8.element_size()
            + self._scale.numel() * self._scale.element_size()
        )

    @property
    def has_pending_gradient(self) -> bool:
        return self._effective_weight is not None and self._effective_weight.grad is not None

    def effective_weight(self) -> Tensor:
        return self.weight_int8.to(dtype=self._scale.dtype) * self._scale[:, None]

    def forward(self, inputs: Tensor) -> Tensor:
        effective = self.effective_weight().to(dtype=inputs.dtype)
        if self.training and torch.is_grad_enabled():
            # Transient FP32 leaf so autograd fills .grad; released by int8_update()
            # (or discard_pending_gradient()) -- never stored as a Parameter.
            effective = effective.detach().requires_grad_(True)
            self._effective_weight = effective
        return F.linear(inputs, effective)

    def discard_pending_gradient(self) -> None:
        self._effective_weight = None

    @torch.no_grad()
    def int8_update(self, lr: float, *, validate: bool = True) -> RatchetUpdateStats:
        """Stochastic-rounded sign step: ``weight_int8 -= round_stochastic(lr * sign(grad))``.

        ``delta = -lr * sign(grad)`` (grid units); the new position
        ``weight_int8 + delta`` is stochastically rounded to an integer via
        ``floor(x + u)`` with ``u ~ Uniform[0, 1)`` drawn from the run-seeded default
        generator on the weight's device (matching ``bucket_pressure``'s stochastic
        path), so the rounding is unbiased in expectation. Clamped to [-127, 127];
        moves that would have left that range are counted as blocked, like the
        ratchet's boundary clicks.
        """
        if self._effective_weight is None or self._effective_weight.grad is None:
            raise RuntimeError("int8-master layer has no pending effective-weight gradient")
        try:
            gradient = self._effective_weight.grad
            if validate and not torch.isfinite(gradient).all():
                raise FloatingPointError("int8-master gradient contains NaN or Inf")
            rms_mean = gradient.float().square().mean(dim=1).sqrt().mean()

            old_code = self.weight_int8.to(torch.float32)
            delta = -float(lr) * torch.sign(gradient).to(torch.float32)
            u = torch.rand(old_code.shape, device=old_code.device, dtype=torch.float32)
            unclamped = torch.floor(old_code + delta + u)
            new_code = unclamped.clamp(-_INT8_MAX, _INT8_MAX)

            blocked_positive = unclamped > _INT8_MAX
            blocked_negative = unclamped < -_INT8_MAX
            positive_moves = new_code > old_code
            negative_moves = new_code < old_code

            self.weight_int8.copy_(new_code.to(torch.int8))
        finally:
            self._effective_weight = None

        stats = RatchetUpdateStats(
            total_weights=self.weight_int8.numel(),
            positive_moves=positive_moves.sum(),
            negative_moves=negative_moves.sum(),
            blocked_positive_moves=blocked_positive.sum(),
            blocked_negative_moves=blocked_negative.sum(),
            gradient_rms_mean=rms_mean,
        )
        return stats.materialize() if validate else stats

    def state_histogram(self, *, bin_width: int = 16) -> dict[str, Any]:
        """Coarse-binned value histogram plus zero/saturated counts, for metrics."""
        values = self.weight_int8
        total = values.numel()
        zero = int((values == 0).sum().item())
        saturated = int((values.abs() == _INT8_MAX).sum().item())
        return {
            "total": total,
            "zero": zero,
            "saturated": saturated,
            "histogram": _coarse_bin_counts(values, bin_width=bin_width),
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"states=255, bias=False"
        )
