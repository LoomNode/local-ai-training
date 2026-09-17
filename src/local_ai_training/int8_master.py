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

_VALID_BITS = (4, 5, 6, 7, 8)


def scheduled_lr(lr: float, lr_final: float, step: int, total_steps: int) -> float:
    """Linear grid-unit-step schedule for the int8-master sign step.

    Returns ``lr`` unchanged when ``lr_final <= 0`` (the default, constant-lr path).
    Otherwise interpolates linearly from ``lr`` at ``step == 0`` to ``lr_final`` at
    ``step == total_steps``, with the fraction clamped to [0, 1] so a step beyond
    ``total_steps`` never overshoots past ``lr_final``.
    """
    if lr_final <= 0:
        return lr
    fraction = step / total_steps if total_steps > 0 else 0.0
    fraction = min(max(fraction, 0.0), 1.0)
    return lr + (lr_final - lr) * fraction


def _coarse_bin_counts(values: Tensor, *, bin_width: int = 16) -> dict[int, int]:
    """Value histogram in coarse bins (bin key = the bin's lower bound).

    The int8 grid spans [-max_value, max_value] (up to 255 states at the default 8
    bits) -- far wider than the ratchet's 3..15-state code, so a per-value histogram
    would be needlessly fine. Binning by ``bin_width`` keeps the reported histogram
    to a couple dozen buckets.
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
    ``scale = row_max_abs / max_value`` (``max_value = 2**(bits-1) - 1``, 127 at the
    default 8 bits) and ``weight_int8 = round(reference / scale)``. The scale is
    frozen at init unless ``live_scale`` is on (see ``_rescale_rows``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        initial_weight: Tensor | None = None,
        live_scale: bool = False,
        bits: int = 8,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if bits not in _VALID_BITS:
            raise ValueError(f"bits must be one of {_VALID_BITS}, got {bits}")
        self.in_features = in_features
        self.out_features = out_features
        self.live_scale = live_scale
        self.bits = bits
        # The grid is [-max_value, max_value] (2*max_value + 1 states); the buffer
        # stays physically int8 regardless of bits -- only the logical range and the
        # reported persistent-byte accounting shrink.
        self.max_value = 2 ** (bits - 1) - 1

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
        scale = (row_max / self.max_value).clamp_min(torch.finfo(torch.float32).eps)
        code = torch.round(reference / scale[:, None]).clamp(-self.max_value, self.max_value)
        self.register_buffer("weight_int8", code.to(torch.int8))
        self.register_buffer("_scale", scale)
        # Registered ONLY when live_scale is on, so a live_scale=False layer's
        # state_dict keys are unchanged (a plain layer can still load such a
        # checkpoint) and a live_scale=True layer's keys match on load. Frozen for
        # the module's lifetime -- it is the fixed reference the live grid delta is
        # held constant against, never recomputed after init.
        if live_scale:
            self.register_buffer("_init_scale", scale.clone())

        # Rows rescaled by the most recent int8_update (reset every call); 0 when
        # live_scale is off. Plain python ints, not buffers -- purely observational.
        self.rows_grown = 0
        self.rows_shrunk = 0

        # Deliberately non-persistent: exists only between forward and the update that
        # consumes its gradient, exactly like the ratchet's transient effective weight.
        self._effective_weight: Tensor | None = None

    @classmethod
    def from_reference(
        cls, reference: Tensor, *, live_scale: bool = False, bits: int = 8
    ) -> Int8MasterLinear:
        if reference.ndim != 2:
            raise ValueError("reference weight must be a matrix")
        return cls(
            reference.shape[1],
            reference.shape[0],
            initial_weight=reference,
            live_scale=live_scale,
            bits=bits,
        )

    @property
    def scale(self) -> Tensor:
        return self._scale

    @property
    def persistent_state_bytes(self) -> int:
        # Logical weight bytes at `bits` bits/weight (numel * bits // 8); the buffer
        # itself stays physically an int8 tensor -- this is the reported footprint a
        # packed `bits`-wide format would need, not the actual in-memory dtype.
        total = (
            self.weight_int8.numel() * self.bits // 8
            + self._scale.numel() * self._scale.element_size()
        )
        if self.live_scale:
            total += self._init_scale.numel() * self._init_scale.element_size()
        return total

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
        path), so the rounding is unbiased in expectation. Clamped to
        [-max_value, max_value]; moves that would have left that range are counted as
        blocked, like the ratchet's boundary clicks.

        When ``live_scale`` is on, the grid delta is scaled by ``init_scale / scale``
        so the *effective* (FP) step stays ``lr * init_scale`` regardless of how many
        times the row has been rescaled -- equal to the plain ``-lr * sign(grad)`` path
        while ``scale == init_scale`` (i.e. before any row ever rescales). The
        ``live_scale=False`` path below is left byte-for-byte identical to the
        pre-lever formula so defaults are bit-identical.
        """
        if self._effective_weight is None or self._effective_weight.grad is None:
            raise RuntimeError("int8-master layer has no pending effective-weight gradient")
        self.rows_grown = 0
        self.rows_shrunk = 0
        try:
            gradient = self._effective_weight.grad
            if validate and not torch.isfinite(gradient).all():
                raise FloatingPointError("int8-master gradient contains NaN or Inf")
            rms_mean = gradient.float().square().mean(dim=1).sqrt().mean()

            old_code = self.weight_int8.to(torch.float32)
            if self.live_scale:
                ratio = (self._init_scale / self._scale)[:, None]
                delta = -float(lr) * ratio * torch.sign(gradient).to(torch.float32)
            else:
                delta = -float(lr) * torch.sign(gradient).to(torch.float32)
            u = torch.rand(old_code.shape, device=old_code.device, dtype=torch.float32)
            unclamped = torch.floor(old_code + delta + u)
            new_code = unclamped.clamp(-self.max_value, self.max_value)

            blocked_positive = unclamped > self.max_value
            blocked_negative = unclamped < -self.max_value
            positive_moves = new_code > old_code
            negative_moves = new_code < old_code

            self.weight_int8.copy_(new_code.to(torch.int8))
            if self.live_scale:
                self._rescale_rows()
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

    @torch.no_grad()
    def _rescale_rows(self) -> None:
        """Grow or shrink each row's block exponent to keep the int8 grid in range.

        Grow: more than 1% of a row's weights sit at +-max -> double the scale and
        halve the integers with stochastic rounding (``floor(w / 2 + u)``), then
        clamp. Shrink: a non-grown row whose max |w| <= max // 4 -> halve the scale
        and double the integers (exact -- ``2 * (max // 4) <= max`` cannot overflow).
        Both preserve the effective (FP) weight within one grid unit. Called only
        when ``live_scale`` is on; sets ``rows_grown``/``rows_shrunk`` for the row
        counts this call rescaled.
        """
        w = self.weight_int8.to(torch.float32)
        sat_frac = (w.abs() == self.max_value).to(torch.float32).mean(dim=1)
        row_max = w.abs().amax(dim=1)
        grow = sat_frac > 0.01
        shrink = (~grow) & (row_max <= self.max_value // 4)

        new_w = w
        new_scale = self._scale
        if bool(grow.any()):
            u = torch.rand(w.shape, device=w.device, dtype=torch.float32)
            grown = torch.floor(w / 2 + u).clamp(-self.max_value, self.max_value)
            new_w = torch.where(grow[:, None], grown, new_w)
            new_scale = torch.where(grow, self._scale * 2, new_scale)
        if bool(shrink.any()):
            shrunk = w * 2
            new_w = torch.where(shrink[:, None], shrunk, new_w)
            new_scale = torch.where(shrink, self._scale / 2, new_scale)

        self.weight_int8.copy_(new_w.to(torch.int8))
        self._scale = new_scale
        self.rows_grown = int(grow.sum().item())
        self.rows_shrunk = int(shrink.sum().item())

    def state_histogram(self, *, bin_width: int | None = None) -> dict[str, Any]:
        """Coarse-binned value histogram plus zero/saturated counts, for metrics.

        ``bin_width`` defaults to ``max(1, (2 * max_value + 1) // 16)`` -- roughly a
        couple dozen buckets across the grid regardless of ``bits``.
        """
        if bin_width is None:
            bin_width = max(1, (2 * self.max_value + 1) // 16)
        values = self.weight_int8
        total = values.numel()
        zero = int((values == 0).sum().item())
        saturated = int((values.abs() == self.max_value).sum().item())
        return {
            "total": total,
            "zero": zero,
            "saturated": saturated,
            "histogram": _coarse_bin_counts(values, bin_width=bin_width),
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"states={2 * self.max_value + 1}, bias=False"
        )
