"""Master-weight-free discrete linear layers and state audits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def pack_code_pressure(code: Tensor, pressure: Tensor, max_code: int) -> Tensor:
    """Pack signed code (low nibble) and pressure (high nibble) into one uint8.

    Lossless for code in [-max_code, max_code] (max_code <= 4) and pressure in [-7, 7].
    """
    low = (code.to(torch.int16) + max_code) & 0x0F
    high = (pressure.to(torch.int16) + 7) & 0x0F
    return (low | (high << 4)).to(torch.uint8)


def unpack_code_pressure(packed: Tensor, max_code: int) -> tuple[Tensor, Tensor]:
    value = packed.to(torch.int16)
    code = ((value & 0x0F) - max_code).to(torch.int8)
    pressure = ((value >> 4) - 7).to(torch.int8)
    return code, pressure


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
        trainable_scale: bool = False,
        initial_weight: Tensor | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if max_code not in (2, 3, 4):
            raise ValueError("max_code must be 2 (quinary), 3 (septenary), or 4 (nonary)")
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
        self.trainable_scale = trainable_scale

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
        zero_pressure = torch.zeros_like(code, dtype=torch.int8)
        self.register_buffer(
            "packed", pack_code_pressure(code.to(torch.int8), zero_pressure, max_code)
        )
        # One positive FP32 magnitude per output row. Frozen (a buffer) by default; when
        # trainable, stored in log space so AdamW updates can never drive it non-positive.
        if trainable_scale:
            self.log_scale = nn.Parameter(scale.log())
        else:
            self.register_buffer("_scale", scale)

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
        return (
            self.packed.numel() * self.packed.element_size()
            + self.scale.numel() * self.scale.element_size()
        )

    @property
    def code(self) -> Tensor:
        return unpack_code_pressure(self.packed, self.max_code)[0]

    @property
    def pressure(self) -> Tensor:
        return unpack_code_pressure(self.packed, self.max_code)[1]

    @property
    def scale(self) -> Tensor:
        return self.log_scale.exp() if self.trainable_scale else self._scale

    def effective_weight(self) -> Tensor:
        return self.code.to(dtype=self.scale.dtype) * self.scale[:, None]

    def forward(self, inputs: Tensor) -> Tensor:
        effective = self.effective_weight().to(dtype=inputs.dtype)
        if self.training and torch.is_grad_enabled():
            if self._effective_weight is not None:
                raise RuntimeError("ratchet_update() must be called before reusing this layer")
            if self.trainable_scale:
                # Keep scale in the autograd graph (so AdamW trains log_scale) while still
                # retaining the effective-weight gradient that drives the code ratchet.
                effective.retain_grad()
            else:
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
        current_code, current_pressure = unpack_code_pressure(self.packed, self.max_code)
        pressure = current_pressure.to(torch.int16) + increments
        code = current_code.to(torch.int16)

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

        self.packed.copy_(
            pack_code_pressure(code.to(torch.int8), pressure.to(torch.int8), self.max_code)
        )
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
        code, pressure = unpack_code_pressure(self.packed, self.max_code)
        if code.min().item() < -self.max_code or code.max().item() > self.max_code:
            raise RuntimeError("ratchet code escaped its allowed range")
        if pressure.abs().max().item() > 7:
            raise RuntimeError("ratchet pressure escaped the packed nibble range")
        if not torch.isfinite(self.scale).all() or torch.any(self.scale <= 0):
            raise RuntimeError("ratchet row scales must remain positive and finite")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"states={2 * self.max_code + 1}, threshold={self.pressure_threshold}, bias=False"
        )


@dataclass(frozen=True)
class PersistentFootprint:
    """Static byte accounting for the trainable matrices, ratchet vs FP32+AdamW.

    Embeddings and RMSNorm are identical support parameters under both schemes and
    cancel out, so only the ratchet/linear matrices are compared. Transient gradients
    and the eager FP effective weights are excluded; this is persistent state only.
    """

    ratchet_weights: int
    ratchet_matrix_bytes: int
    fp32_master_bytes: int
    fp32_optimizer_bytes: int

    @property
    def fp32_matrix_bytes(self) -> int:
        return self.fp32_master_bytes + self.fp32_optimizer_bytes

    @property
    def reduction_ratio(self) -> float:
        return self.fp32_matrix_bytes / max(self.ratchet_matrix_bytes, 1)


def compare_persistent_footprint(model: nn.Module) -> PersistentFootprint:
    """Count persistent bytes the ratchet matrices need versus FP32 + AdamW.

    FP32 training must keep, per weight, a 4-byte master copy plus AdamW's two
    4-byte moment buffers. The ratchet keeps only its int8 code/pressure and a
    per-row FP32 scale (already summed in ``persistent_state_bytes``).
    """
    weights = 0
    ratchet_bytes = 0
    for module in model.modules():
        if isinstance(module, DiscreteRatchetLinear):
            weights += module.code.numel()
            ratchet_bytes += module.persistent_state_bytes
    return PersistentFootprint(
        ratchet_weights=weights,
        ratchet_matrix_bytes=ratchet_bytes,
        fp32_master_bytes=weights * 4,
        fp32_optimizer_bytes=weights * 8,
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

