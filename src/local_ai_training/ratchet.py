"""Master-weight-free discrete linear layers and state audits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .int8_matmul import (
    quantize_rows,
    scaled_int8_mm,
)


class _RatchetMatmul(torch.autograd.Function):
    """BF16/int8 matmul whose backward exposes the effective-weight gradient."""

    @staticmethod
    def forward(
        ctx,
        inputs: Tensor,
        inputs_int8: Tensor | None,
        inputs_scale: Tensor | None,
        code: Tensor,
        scale: Tensor,
        mode: str,
        fuse_backward_update: bool,
        tile_size: int,
        gradient_sink,
    ):
        input_shape = inputs.shape
        flat_inputs = inputs.flatten(0, -2)
        
        ctx.input_shape = input_shape
        ctx.input_dtype = inputs.dtype
        ctx.mode = mode
        ctx.fuse_backward_update = fuse_backward_update
        ctx.tile_size = tile_size
        ctx.gradient_sink = gradient_sink
        
        if mode == "int8" and inputs_int8 is not None and inputs_scale is not None:
            quantized_inputs = inputs_int8.flatten(0, -2)
            input_scale = inputs_scale.flatten(0, -1)
            ctx.save_for_backward(quantized_inputs, input_scale, code, scale)
            ctx.used_fused_int8 = True
            output = scaled_int8_mm(
                quantized_inputs, code.t().contiguous(), input_scale, scale.float()
            )
        else:
            ctx.save_for_backward(flat_inputs, code, scale)
            ctx.used_fused_int8 = False
            if mode == "fp32":
                effective = code.to(torch.float32) * scale.to(torch.float32)[:, None]
                output = flat_inputs.to(torch.float32) @ effective.t()
            elif mode == "bf16":
                effective = code.to(torch.bfloat16) * scale.to(torch.bfloat16)[:, None]
                output = flat_inputs.to(torch.bfloat16) @ effective.t()
            else:
                quantized_inputs, input_scale = quantize_rows(flat_inputs)
                output = scaled_int8_mm(quantized_inputs, code.t(), input_scale, scale.float())
                
        return output.reshape(*input_shape[:-1], code.shape[0]).to(inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.used_fused_int8:
            quantized_inputs, input_scale, code, scale = ctx.saved_tensors
            flat_inputs_bf16 = quantized_inputs.to(torch.bfloat16) * input_scale[:, None].to(
                torch.bfloat16
            )
            flat_inputs = flat_inputs_bf16  # For places that expect flat_inputs
        else:
            flat_inputs, code, scale = ctx.saved_tensors
            if ctx.mode != "fp32":
                flat_inputs_bf16 = flat_inputs.to(torch.bfloat16)
            
        flat_gradient = grad_output.flatten(0, -2)

        if not ctx.fuse_backward_update:
            if ctx.mode == "fp32":
                gradient_fp32 = flat_gradient.to(torch.float32)
                inputs_fp32 = flat_inputs.to(torch.float32)
                effective = code.to(torch.float32) * scale.to(torch.float32)[:, None]
                grad_input = gradient_fp32 @ effective
                grad_weight_fp32 = gradient_fp32.t() @ inputs_fp32
            else:
                gradient_bf16 = flat_gradient.to(torch.bfloat16)
                inputs_bf16 = (
                    flat_inputs_bf16 if ctx.used_fused_int8 else flat_inputs.to(torch.bfloat16)
                )
                effective = code.to(torch.bfloat16) * scale.to(torch.bfloat16)[:, None]
                grad_input = gradient_bf16 @ effective
                grad_weight = gradient_bf16.t() @ inputs_bf16
                grad_weight_fp32 = grad_weight.float()
            
            ctx.gradient_sink(grad_weight_fp32, None, None)
            grad_scale = None
            if ctx.needs_input_grad[4]:
                grad_scale = (grad_weight_fp32 * code.float()).sum(dim=1)
            return (
                grad_input.reshape(ctx.input_shape).to(ctx.input_dtype),
                None,
                None,
                None,
                grad_scale,
                None,
                None,
                None,
                None,
            )

        out_features, in_features = code.shape

        if ctx.mode == "fp32":
            gradient_fp32 = flat_gradient.to(torch.float32)
            inputs_fp32 = flat_inputs.to(torch.float32)
            effective = code.to(torch.float32) * scale.to(torch.float32)[:, None]
            grad_input = gradient_fp32 @ effective
        else:
            gradient_bf16 = flat_gradient.to(torch.bfloat16)
            inputs_bf16 = (
                flat_inputs_bf16 if ctx.used_fused_int8 else flat_inputs.to(torch.bfloat16)
            )
            effective = code.to(torch.bfloat16) * scale.to(torch.bfloat16)[:, None]
            grad_input = gradient_bf16 @ effective

        grad_scale = (
            torch.zeros(code.shape[0], device=code.device, dtype=torch.float32)
            if ctx.needs_input_grad[4]
            else None
        )
        grad_out_t = flat_gradient.t()

        for tile_start in range(0, code.shape[0], ctx.tile_size):
            tile_end = min(tile_start + ctx.tile_size, code.shape[0])
            grad_out_tile = grad_out_t[tile_start:tile_end, :]

            if ctx.mode == "fp32":
                grad_weight_tile_fp32 = grad_out_tile.to(torch.float32) @ flat_inputs.to(
                    torch.float32
                )
            else:
                inputs_bf16 = (
                    flat_inputs_bf16 if ctx.used_fused_int8 else flat_inputs.to(torch.bfloat16)
                )
                grad_weight_tile = grad_out_tile.to(torch.bfloat16) @ inputs_bf16
                grad_weight_tile_fp32 = grad_weight_tile.float()
            
            ctx.gradient_sink(grad_weight_tile_fp32, tile_start, tile_end)
            
            if grad_scale is not None:
                code_tile = code[tile_start:tile_end, :].float()
                grad_scale[tile_start:tile_end] = (grad_weight_tile_fp32 * code_tile).sum(dim=1)

        return (
            grad_input.reshape(ctx.input_shape).to(ctx.input_dtype),
            None,
            None,
            None,
            grad_scale,
            None,
            None,
            None,
            None,
        )


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


def _ratchet_update_core(
    packed: Tensor,
    normalized: Tensor,
    max_code: int,
    pressure_threshold: int,
    bucket_low: float,
    bucket_high: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pure elementwise ratchet update; torch.compile fuses this into ~1-2 kernels.

    Returns the new packed buffer and the four move-count sums.
    """
    increments = bucket_pressure(normalized, low=bucket_low, high=bucket_high)
    current_code, current_pressure = unpack_code_pressure(packed, max_code)
    pressure = current_pressure.to(torch.int16) + increments
    code = current_code.to(torch.int16)

    positive_requests = pressure >= pressure_threshold
    negative_requests = pressure <= -pressure_threshold
    positive_moves = positive_requests & (code < max_code)
    negative_moves = negative_requests & (code > -max_code)
    blocked_positive = positive_requests & ~positive_moves
    blocked_negative = negative_requests & ~negative_moves

    code = code + positive_moves.to(torch.int16) - negative_moves.to(torch.int16)
    pressure = pressure - positive_requests.to(torch.int16) * pressure_threshold
    pressure = pressure + negative_requests.to(torch.int16) * pressure_threshold
    pressure = pressure.clamp(-128, 127)

    new_packed = pack_code_pressure(code.to(torch.int8), pressure.to(torch.int8), max_code)
    return (
        new_packed,
        positive_moves.sum(),
        negative_moves.sum(),
        blocked_positive.sum(),
        blocked_negative.sum(),
    )


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
        rms_ema_beta: float = 0.0,
        pressure_leak_period: int = 0,
        trainable_scale: bool = False,
        compile_update: bool = False,
        matmul_mode: str = "fp32",
        initial_weight: Tensor | None = None,
        fuse_backward_update: bool = False,
        tile_size: int = 256,
    ) -> None:
        super().__init__()
        self._update_fn = (
            torch.compile(_ratchet_update_core) if compile_update else _ratchet_update_core
        )
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if max_code not in (1, 2, 3, 4, 5, 6, 7):
            raise ValueError(
                "max_code must be in 1..7 (3..15 states); 7 is the 4-bit packing cap"
            )
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
        if not 0.0 <= rms_ema_beta < 1.0:
            raise ValueError("rms_ema_beta must be in [0, 1)")
        self.rms_ema_beta = rms_ema_beta
        if pressure_leak_period < 0:
            raise ValueError("pressure_leak_period must be a non-negative integer")
        self.pressure_leak_period = pressure_leak_period
        self._update_count = 0
        if rms_ema_beta > 0.0:
            # Per-row 2nd-moment EMA (Adam's v at row granularity). Lazily seeded per row
            # from the first step's mean-square (rms_ema==0 => uninitialized), so step 0
            # reproduces the instantaneous rule exactly. 1-D => audit-clean.
            self.register_buffer("rms_ema", torch.zeros(out_features))
        self.trainable_scale = trainable_scale
        self.matmul_mode = matmul_mode
        self.fuse_backward_update = fuse_backward_update
        self.tile_size = tile_size

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
        self._pending_weight_gradient: Tensor | None = None
        
        self._pending_stats_total_weights = 0
        self._pending_stats_positive_moves = 0
        self._pending_stats_negative_moves = 0
        self._pending_stats_blocked_positive_moves = 0
        self._pending_stats_blocked_negative_moves = 0
        self._pending_stats_rms_sum = 0.0

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
        eager_pending = (
            self._effective_weight is not None and self._effective_weight.grad is not None
        )
        return eager_pending or self._pending_weight_gradient is not None

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

    def forward(
        self,
        inputs: Tensor,
        *,
        inputs_int8: Tensor | None = None,
        inputs_scale: Tensor | None = None,
    ) -> Tensor:
        if self.matmul_mode != "fp32" or self.fuse_backward_update:
            return _RatchetMatmul.apply(
                inputs,
                inputs_int8,
                inputs_scale,
                self.code,
                self.scale,
                self.matmul_mode,
                self.fuse_backward_update,
                self.tile_size,
                self._capture_weight_gradient,
            )
        effective = self.effective_weight().to(dtype=inputs.dtype)
        if self.training and torch.is_grad_enabled():
            if self.trainable_scale:
                effective.retain_grad()
            else:
                effective = effective.detach().requires_grad_(True)
            self._effective_weight = effective
        return F.linear(inputs, effective)

    def _capture_weight_gradient(
        self, gradient: Tensor, tile_start: int | None, tile_end: int | None
    ) -> None:
        if not self.fuse_backward_update:
            if self._pending_weight_gradient is not None:
                raise RuntimeError(
                    "ratchet layer has multiple pending effective-weight gradients "
                    "(missing ratchet_update or unsupported weight sharing)"
                )
            self._pending_weight_gradient = gradient
        else:
            assert tile_start is not None and tile_end is not None
            if gradient.shape != (tile_end - tile_start, self.in_features):
                raise ValueError("fused tile gradient shape mismatch")
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("ratchet gradient contains NaN or Inf")
            
            normalized = self._normalize(gradient, tile_start, tile_end)

            packed_tile = self.packed[tile_start:tile_end, :]
            (
                new_packed_tile,
                positive,
                negative,
                blocked_positive,
                blocked_negative,
            ) = self._update_fn(
                packed_tile,
                normalized.to(self.packed.device),
                self.max_code,
                self.pressure_threshold,
                self.bucket_low,
                self.bucket_high,
            )
            self.packed[tile_start:tile_end, :] = new_packed_tile
            
            self._pending_stats_total_weights += (tile_end - tile_start) * self.in_features
            self._pending_stats_positive_moves += int(positive.item())
            self._pending_stats_negative_moves += int(negative.item())
            self._pending_stats_blocked_positive_moves += int(blocked_positive.item())
            self._pending_stats_blocked_negative_moves += int(blocked_negative.item())
            self._pending_stats_rms_sum += float(
                gradient.float().square().mean(dim=1).sqrt().sum().item()
            )

    @torch.no_grad()
    def _normalize(self, gradient: Tensor, row_start: int, row_end: int) -> Tensor:
        grad = gradient.float()
        ms = grad.square().mean(dim=1, keepdim=True)  # [rows, 1] mean-square per row
        if self.rms_ema_beta > 0.0:
            ema = self.rms_ema[row_start:row_end].unsqueeze(1)
            ema = torch.where(
                ema == 0, ms, self.rms_ema_beta * ema + (1.0 - self.rms_ema_beta) * ms
            )
            self.rms_ema[row_start:row_end] = ema.squeeze(1)
            rms = ema.sqrt()
        else:
            rms = ms.sqrt()
        return grad / (rms + self.eps)

    @torch.no_grad()
    def apply_weight_gradient(self, gradient: Tensor) -> RatchetUpdateStats:
        if gradient.shape != self.code.shape:
            raise ValueError(
                f"gradient must have shape {tuple(self.code.shape)}, got {tuple(gradient.shape)}"
            )
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("ratchet gradient contains NaN or Inf")
        rms_mean = float(gradient.float().square().mean(dim=1).sqrt().mean().item())
        normalized = self._normalize(gradient, 0, self.out_features)
        stats = self.apply_normalized_gradient(normalized)
        return RatchetUpdateStats(
            total_weights=stats.total_weights,
            positive_moves=stats.positive_moves,
            negative_moves=stats.negative_moves,
            blocked_positive_moves=stats.blocked_positive_moves,
            blocked_negative_moves=stats.blocked_negative_moves,
            gradient_rms_mean=rms_mean,
        )

    @torch.no_grad()
    def apply_normalized_gradient(self, normalized: Tensor) -> RatchetUpdateStats:
        if normalized.shape != self.code.shape:
            raise ValueError(
                f"normalized gradient must have shape {tuple(self.code.shape)}, "
                f"got {tuple(normalized.shape)}"
            )
        new_packed, positive, negative, blocked_positive, blocked_negative = self._update_fn(
            self.packed,
            normalized.to(self.packed.device),
            self.max_code,
            self.pressure_threshold,
            self.bucket_low,
            self.bucket_high,
        )
        self.packed.copy_(new_packed)
        self._validate_state()
        return RatchetUpdateStats(
            total_weights=self.code.numel(),
            positive_moves=int(positive.item()),
            negative_moves=int(negative.item()),
            blocked_positive_moves=int(blocked_positive.item()),
            blocked_negative_moves=int(blocked_negative.item()),
            gradient_rms_mean=0.0,
        )

    @torch.no_grad()
    def _maybe_leak_pressure(self) -> None:
        # 1st-moment EMA analogue: every `pressure_leak_period`-th update, bleed each nonzero
        # pressure one unit toward zero so stale pressure fades (recent direction dominates).
        # Moving toward zero never enlarges |pressure|, so the nibble range is preserved.
        if self.pressure_leak_period <= 0:
            return
        self._update_count += 1
        if self._update_count % self.pressure_leak_period != 0:
            return
        code, pressure = unpack_code_pressure(self.packed, self.max_code)
        pressure = pressure - torch.sign(pressure).to(pressure.dtype)
        self.packed.copy_(pack_code_pressure(code, pressure, self.max_code))

    def ratchet_update(self) -> RatchetUpdateStats:
        if self.fuse_backward_update:
            self._validate_state()
            stats = RatchetUpdateStats(
                total_weights=self._pending_stats_total_weights,
                positive_moves=self._pending_stats_positive_moves,
                negative_moves=self._pending_stats_negative_moves,
                blocked_positive_moves=self._pending_stats_blocked_positive_moves,
                blocked_negative_moves=self._pending_stats_blocked_negative_moves,
                gradient_rms_mean=self._pending_stats_rms_sum / max(1, self.out_features),
            )
            self._pending_stats_total_weights = 0
            self._pending_stats_positive_moves = 0
            self._pending_stats_negative_moves = 0
            self._pending_stats_blocked_positive_moves = 0
            self._pending_stats_blocked_negative_moves = 0
            self._pending_stats_rms_sum = 0.0
        elif self.matmul_mode != "fp32":
            if self._pending_weight_gradient is None:
                raise RuntimeError("ratchet layer has no pending effective-weight gradient")
            try:
                stats = self.apply_weight_gradient(self._pending_weight_gradient)
            finally:
                self._pending_weight_gradient = None
        else:
            if self._effective_weight is None or self._effective_weight.grad is None:
                raise RuntimeError("ratchet layer has no pending effective-weight gradient")
            try:
                stats = self.apply_weight_gradient(self._effective_weight.grad)
            finally:
                self._effective_weight = None
        self._maybe_leak_pressure()
        return stats

    def discard_pending_gradient(self) -> None:
        if self._pending_weight_gradient is not None:
            self._pending_weight_gradient = None
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
    4-byte moment buffers. The ratchet keeps only one packed uint8 byte (code +
    pressure nibbles) and a per-row FP32 scale (summed in ``persistent_state_bytes``).
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
        if module.packed.dtype != torch.uint8:
            violations.append(f"{prefix}.packed: expected uint8, got {module.packed.dtype}")
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
