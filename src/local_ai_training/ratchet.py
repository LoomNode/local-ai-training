"""Master-weight-free discrete linear layers and state audits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RatchetUpdateStats:
    total_weights: int
    positive_moves: int
    negative_moves: int
    blocked_positive_moves: int
    blocked_negative_moves: int
    gradient_rms_mean: float

    @property
    def code_moves(self) -> int:
        return self.positive_moves + self.negative_moves

    @property
    def blocked_moves(self) -> int:
        return self.blocked_positive_moves + self.blocked_negative_moves


@dataclass(frozen=True)
class RatchetAuditReport:
    ratchet_layers: int
    ratchet_weights: int
    ratchet_state_bytes: int
    support_parameter_bytes: int
    violations: tuple[str, ...]


def bucket_pressure(z: Tensor, *, low: float = 0.5, high: float = 1.5) -> Tensor:
    """Convert normalized gradients to integer pressure in descent direction."""
    if not 0 <= low < high:
        raise ValueError("pressure bucket thresholds must satisfy 0 <= low < high")
    magnitude = z.abs()
    bucket = torch.where(
        magnitude >= high,
        torch.full_like(z, 2, dtype=torch.int16),
        torch.where(
            magnitude >= low,
            torch.ones_like(z, dtype=torch.int16),
            torch.zeros_like(z, dtype=torch.int16),
        ),
    )
    return -torch.sign(z).to(torch.int16) * bucket


class DiscreteRatchetLinear(nn.Module):
    """Linear layer with integer codes and pressure, but no master weight parameter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        max_code: int,
        pressure_threshold: int = 8,
        bucket_low: float = 0.5,
        bucket_high: float = 1.5,
        eps: float = 1e-8,
        initial_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if max_code not in (2, 3):
            raise ValueError("max_code must be 2 (quinary) or 3 (septenary)")
        if pressure_threshold <= 0 or pressure_threshold > 127:
            raise ValueError("pressure_threshold must be in [1, 127]")
        if not 0 <= bucket_low < bucket_high:
            raise ValueError("bucket thresholds must satisfy 0 <= low < high")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.max_code = max_code
        self.pressure_threshold = pressure_threshold
        self.bucket_low = bucket_low
        self.bucket_high = bucket_high
        self.eps = eps

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
        scale = (row_max / max_code).clamp_min(torch.finfo(torch.float32).eps)
        code = torch.round(reference / scale[:, None]).clamp(-max_code, max_code)
        self.register_buffer("code", code.to(torch.int8))
        self.register_buffer("pressure", torch.zeros_like(code, dtype=torch.int8))
        self.register_buffer("scale", scale)

        # This is deliberately non-persistent. It exists only between forward and update.
        self._effective_weight: Tensor | None = None

    @classmethod
    def from_reference(
        cls,
        reference: Tensor,
        *,
        max_code: int,
        pressure_threshold: int = 8,
        bucket_low: float = 0.5,
        bucket_high: float = 1.5,
        eps: float = 1e-8,
    ) -> DiscreteRatchetLinear:
        if reference.ndim != 2:
            raise ValueError("reference weight must be a matrix")
        return cls(
            reference.shape[1],
            reference.shape[0],
            max_code=max_code,
            pressure_threshold=pressure_threshold,
            bucket_low=bucket_low,
            bucket_high=bucket_high,
            eps=eps,
            initial_weight=reference,
        )

    @property
    def has_pending_gradient(self) -> bool:
        return self._effective_weight is not None and self._effective_weight.grad is not None

    @property
    def persistent_state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.code, self.pressure, self.scale))

    def effective_weight(self) -> Tensor:
        return self.code.to(dtype=self.scale.dtype) * self.scale[:, None]

    def forward(self, inputs: Tensor) -> Tensor:
        effective = self.effective_weight().to(dtype=inputs.dtype)
        if self.training and torch.is_grad_enabled():
            if self._effective_weight is not None:
                raise RuntimeError("ratchet_update() must be called before reusing this layer")
            effective = effective.detach().requires_grad_(True)
            self._effective_weight = effective
        return F.linear(inputs, effective)

    @torch.no_grad()
    def apply_weight_gradient(self, gradient: Tensor) -> RatchetUpdateStats:
        if gradient.shape != self.code.shape:
            raise ValueError(
                f"gradient must have shape {tuple(self.code.shape)}, got {tuple(gradient.shape)}"
            )
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("ratchet gradient contains NaN or Inf")
        rms = gradient.float().square().mean(dim=1, keepdim=True).sqrt()
        normalized = gradient.float() / (rms + self.eps)
        stats = self.apply_normalized_gradient(normalized)
        return RatchetUpdateStats(
            total_weights=stats.total_weights,
            positive_moves=stats.positive_moves,
            negative_moves=stats.negative_moves,
            blocked_positive_moves=stats.blocked_positive_moves,
            blocked_negative_moves=stats.blocked_negative_moves,
            gradient_rms_mean=float(rms.mean().item()),
        )

    @torch.no_grad()
    def apply_normalized_gradient(self, normalized: Tensor) -> RatchetUpdateStats:
        if normalized.shape != self.code.shape:
            raise ValueError(
                f"normalized gradient must have shape {tuple(self.code.shape)}, "
                f"got {tuple(normalized.shape)}"
            )
        increments = bucket_pressure(
            normalized, low=self.bucket_low, high=self.bucket_high
        ).to(device=self.pressure.device)
        pressure = self.pressure.to(torch.int16) + increments
        code = self.code.to(torch.int16)

        positive_requests = pressure >= self.pressure_threshold
        negative_requests = pressure <= -self.pressure_threshold
        positive_moves = positive_requests & (code < self.max_code)
        negative_moves = negative_requests & (code > -self.max_code)
        blocked_positive = positive_requests & ~positive_moves
        blocked_negative = negative_requests & ~negative_moves

        code = code + positive_moves.to(torch.int16) - negative_moves.to(torch.int16)
        pressure = pressure - positive_requests.to(torch.int16) * self.pressure_threshold
        pressure = pressure + negative_requests.to(torch.int16) * self.pressure_threshold
        pressure = pressure.clamp(-128, 127)

        self.code.copy_(code.to(torch.int8))
        self.pressure.copy_(pressure.to(torch.int8))
        self._validate_state()
        return RatchetUpdateStats(
            total_weights=self.code.numel(),
            positive_moves=int(positive_moves.sum().item()),
            negative_moves=int(negative_moves.sum().item()),
            blocked_positive_moves=int(blocked_positive.sum().item()),
            blocked_negative_moves=int(blocked_negative.sum().item()),
            gradient_rms_mean=0.0,
        )

    def ratchet_update(self) -> RatchetUpdateStats:
        if self._effective_weight is None or self._effective_weight.grad is None:
            raise RuntimeError("ratchet layer has no pending effective-weight gradient")
        try:
            return self.apply_weight_gradient(self._effective_weight.grad)
        finally:
            self._effective_weight = None

    def discard_pending_gradient(self) -> None:
        self._effective_weight = None

    def _validate_state(self) -> None:
        if self.code.dtype != torch.int8 or self.pressure.dtype != torch.int8:
            raise RuntimeError("ratchet code and pressure must remain int8")
        if self.code.min().item() < -self.max_code or self.code.max().item() > self.max_code:
            raise RuntimeError("ratchet code escaped its allowed range")
        if not torch.isfinite(self.scale).all() or torch.any(self.scale <= 0):
            raise RuntimeError("ratchet row scales must remain positive and finite")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"states={2 * self.max_code + 1}, threshold={self.pressure_threshold}, bias=False"
        )


def audit_no_master_weights(
    model: nn.Module, *, raise_on_violation: bool = False
) -> RatchetAuditReport:
    violations: list[str] = []
    ratchet_layers = 0
    ratchet_weights = 0
    ratchet_state_bytes = 0

    for module_name, module in model.named_modules():
        if not isinstance(module, DiscreteRatchetLinear):
            continue
        ratchet_layers += 1
        ratchet_weights += module.code.numel()
        ratchet_state_bytes += module.persistent_state_bytes
        prefix = module_name or "<root>"
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter.is_floating_point() and parameter.ndim >= 2:
                violations.append(f"{prefix}.{parameter_name}: floating matrix parameter")
        if module.code.dtype != torch.int8:
            violations.append(f"{prefix}.code: expected int8, got {module.code.dtype}")
        if module.pressure.dtype != torch.int8:
            violations.append(f"{prefix}.pressure: expected int8, got {module.pressure.dtype}")
        if module.scale.ndim != 1 or module.scale.shape[0] != module.out_features:
            violations.append(f"{prefix}.scale: expected one scale per output row")

    support_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    report = RatchetAuditReport(
        ratchet_layers=ratchet_layers,
        ratchet_weights=ratchet_weights,
        ratchet_state_bytes=ratchet_state_bytes,
        support_parameter_bytes=support_bytes,
        violations=tuple(violations),
    )
    if raise_on_violation and violations:
        raise RuntimeError("ratchet master-weight audit failed: " + "; ".join(violations))
    return report

