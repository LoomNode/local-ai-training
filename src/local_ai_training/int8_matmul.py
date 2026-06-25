"""Autotuned CUDA int8 GEMM with explicit row/column dequantization scales."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


def _configs() -> list[triton.Config]:
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
            num_stages=stages,
            num_warps=warps,
        )
        for bm in (64, 128, 256)
        for bn in (64, 128, 256)
        for bk in (32, 64, 128)
        for stages in (3, 4)
        for warps in (4, 8)
    ]


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _scaled_int8_kernel(
    lhs_ptr,
    rhs_ptr,
    output_ptr,
    lhs_scale_ptr,
    rhs_scale_ptr,
    M,
    N,
    K,
    stride_lm,
    stride_lk,
    stride_rk,
    stride_rn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    lhs_ptrs = lhs_ptr + offs_m[:, None] * stride_lm + offs_k[None, :] * stride_lk
    rhs_ptrs = rhs_ptr + offs_k[:, None] * stride_rk + offs_n[None, :] * stride_rn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        lhs = tl.load(lhs_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        rhs = tl.load(rhs_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        accumulator = tl.dot(lhs, rhs, accumulator, out_dtype=tl.int32)
        lhs_ptrs += BLOCK_K * stride_lk
        rhs_ptrs += BLOCK_K * stride_rk

    offs_om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    lhs_scale = tl.load(lhs_scale_ptr + offs_om, mask=offs_om < M, other=0.0)
    rhs_scale = tl.load(rhs_scale_ptr + offs_on, mask=offs_on < N, other=0.0)
    output = accumulator.to(tl.float32) * lhs_scale[:, None] * rhs_scale[None, :]
    output_ptrs = output_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(output_ptrs, output.to(tl.bfloat16), mask=mask)


def _positive_scale(values: Tensor, *, dimension: int) -> Tensor:
    minimum = torch.finfo(torch.float32).tiny
    return (values.float().abs().amax(dim=dimension) / 127.0).clamp_min(minimum)


_TINY = torch.finfo(torch.float32).tiny


@triton.jit
def _rint_div(x, scale):
    # IEEE round-to-nearest division then round-half-to-even, matching torch's
    # values.float()/scale then .round() bit-for-bit (a plain Triton `/` diverges by
    # a ULP on tie boundaries, flipping a handful of bf16 cases).
    return tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(x.to(tl.float32), scale))


def _block_configs(reduce_block: str, serve_block: str) -> list[triton.Config]:
    # serve_block = the axis whose scale each program owns; reduce_block = the axis
    # the quantization scale reduces over. Kept small to bound autotune memory/compile.
    return [triton.Config({serve_block: s, reduce_block: r}, num_warps=w)
            for s, r, w in ((64, 512, 4), (128, 512, 8), (64, 1024, 8),
                            (128, 256, 4), (256, 256, 8))]


@triton.autotune(configs=_block_configs("BLOCK_K", "BLOCK_M"), key=["M", "K"])
@triton.jit
def _quantize_rows_kernel(
    in_ptr, out_ptr, scale_ptr, col_scale_ptr, M, K,
    stride_im, stride_ik, stride_om, stride_ok, TINY, SEED,
    HAS_COLSCALE: tl.constexpr, STOCHASTIC: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # When HAS_COLSCALE, each value is multiplied by a per-column scale (fp32) before
    # the per-row amax and quantization — the fused grad_input pre-scaling, equivalent to
    # quantize_rows(values.float() * col_scale[None, :]) with no M×N fp32 temp.
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        ptrs = in_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
        m2 = mask_m[:, None] & (offs_k[None, :] < K)
        x = tl.load(ptrs, mask=m2, other=0.0).to(tl.float32)
        if HAS_COLSCALE:
            cs = tl.load(col_scale_ptr + offs_k, mask=offs_k < K, other=0.0)
            x = x * cs[None, :]
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=1))
    scale = tl.maximum(amax / 127.0, TINY)
    tl.store(scale_ptr + offs_m, scale, mask=mask_m)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        m2 = mask_m[:, None] & (offs_k[None, :] < K)
        ptrs = in_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
        x = tl.load(ptrs, mask=m2, other=0.0).to(tl.float32)
        if HAS_COLSCALE:
            cs = tl.load(col_scale_ptr + offs_k, mask=offs_k < K, other=0.0)
            x = x * cs[None, :]
        if STOCHASTIC:
            # Unbiased stochastic rounding: floor(y + u), u ~ U[0,1). E[round(y)] = y, so
            # quantization error has zero mean and does not compound a bias over training.
            y = tl.extra.cuda.libdevice.div_rn(x, scale[:, None])
            offs = offs_m[:, None] * K + offs_k[None, :]
            u = tl.rand(SEED, offs)
            r = tl.floor(y + u)
        else:
            r = _rint_div(x, scale[:, None])
        q = tl.minimum(tl.maximum(r, -127.0), 127.0)
        optr = out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
        tl.store(optr, q.to(tl.int8), mask=m2)


@triton.autotune(configs=_block_configs("BLOCK_M", "BLOCK_N"), key=["M", "K"])
@triton.jit
def _quantize_columns_kernel(
    in_ptr, out_ptr, scale_ptr, M, K,
    stride_im, stride_ik, stride_om, stride_ok, TINY,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < K
    amax = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        m2 = (offs_m[:, None] < M) & mask_n[None, :]
        ptrs = in_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_ik
        x = tl.load(ptrs, mask=m2, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=0))
    scale = tl.maximum(amax / 127.0, TINY)
    tl.store(scale_ptr + offs_n, scale, mask=mask_n)
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        m2 = (offs_m[:, None] < M) & mask_n[None, :]
        ptrs = in_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_ik
        x = tl.load(ptrs, mask=m2, other=0.0)
        q = tl.minimum(tl.maximum(_rint_div(x, scale[None, :]), -127.0), 127.0)
        optr = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_ok
        tl.store(optr, q.to(tl.int8), mask=m2)


def _quantize_rows_fused(
    values: Tensor, col_scale: Tensor | None = None, *, stochastic: bool = False
) -> tuple[Tensor, Tensor]:
    """Fused Triton row quantization; bit-exact with _quantize_rows_reference.

    When col_scale (a 1-D fp32 tensor of length values.shape[1]) is given, each value is
    multiplied by its column scale before quantization, fusing the grad_input pre-scaling.
    When stochastic, rounding is unbiased stochastic rounding (a fresh per-call seed) instead
    of round-half-to-even — used for gradient quant where bias compounds across steps.
    """
    if values.ndim != 2:
        raise ValueError("row quantization requires a 2D tensor")
    m, k = values.shape
    out = torch.empty((m, k), device=values.device, dtype=torch.int8)
    scale = torch.empty((m,), device=values.device, dtype=torch.float32)
    has_colscale = col_scale is not None
    # scale is a valid same-device pointer used as a dummy when no col_scale (never read).
    col_scale_arg = col_scale.contiguous() if has_colscale else scale
    # Host-side RNG (no CUDA sync); 0 when deterministic (unused by the kernel).
    seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if stochastic else 0
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]),)  # noqa: E731
    _quantize_rows_kernel[grid](
        values, out, scale, col_scale_arg, m, k,
        values.stride(0), values.stride(1), out.stride(0), out.stride(1), _TINY, seed,
        HAS_COLSCALE=has_colscale, STOCHASTIC=stochastic,
    )
    return out, scale


def _quantize_columns_fused(values: Tensor) -> tuple[Tensor, Tensor]:
    """Fused Triton column quantization; bit-exact with _quantize_columns_reference."""
    if values.ndim != 2:
        raise ValueError("column quantization requires a 2D tensor")
    m, k = values.shape
    out = torch.empty((m, k), device=values.device, dtype=torch.int8)
    scale = torch.empty((k,), device=values.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(k, meta["BLOCK_N"]),)  # noqa: E731
    _quantize_columns_kernel[grid](
        values, out, scale, m, k,
        values.stride(0), values.stride(1), out.stride(0), out.stride(1), _TINY,
    )
    return out, scale


def _quantize_rows_reference(
    values: Tensor, col_scale: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Torch reference for row quantization. Bit-exact oracle for the fused kernel.

    With col_scale, pre-scales each column before quantizing (the grad_input path).
    """
    if values.ndim != 2:
        raise ValueError("row quantization requires a 2D tensor")
    prepared = values.float()
    if col_scale is not None:
        prepared = prepared * col_scale.float()[None, :]
    scale = _positive_scale(prepared, dimension=1)
    quantized = torch.clamp((prepared / scale[:, None]).round(), -127, 127)
    return quantized.to(torch.int8), scale


def _quantize_columns_reference(values: Tensor) -> tuple[Tensor, Tensor]:
    """Torch reference for column quantization. Bit-exact oracle for the fused kernel."""
    if values.ndim != 2:
        raise ValueError("column quantization requires a 2D tensor")
    scale = _positive_scale(values, dimension=0)
    quantized = torch.clamp((values.float() / scale[None, :]).round(), -127, 127)
    return quantized.to(torch.int8), scale


def quantize_rows(values: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetrically quantize each row and return int8 values plus FP32 scales.

    CUDA-only: the int8 path is GPU-only (see scaled_int8_mm). The torch reference is
    kept as _quantize_rows_reference for tests.
    """
    if not values.is_cuda:
        raise RuntimeError("quantize_rows requires CUDA; the int8 path is GPU-only")
    return _quantize_rows_fused(values)


def quantize_rows_colscaled(
    values: Tensor, col_scale: Tensor, *, stochastic: bool = False
) -> tuple[Tensor, Tensor]:
    """Row-quantize values after per-column scaling, fused in one pass (CUDA-only).

    Bit-exact with quantize_rows(values.float() * col_scale[None, :]) but without the
    M×N fp32 temp. Used by the int8 backward's grad_input path. When stochastic, uses
    unbiased stochastic rounding (for gradient quant, where rounding bias compounds).
    """
    if not values.is_cuda:
        raise RuntimeError("quantize_rows_colscaled requires CUDA; the int8 path is GPU-only")
    if col_scale.ndim != 1 or values.ndim != 2 or col_scale.shape[0] != values.shape[1]:
        raise ValueError("col_scale must be 1-D with length values.shape[1]")
    if col_scale.dtype != torch.float32:
        raise TypeError("col_scale must be float32")
    return _quantize_rows_fused(values, col_scale, stochastic=stochastic)


def quantize_columns(values: Tensor) -> tuple[Tensor, Tensor]:
    if not values.is_cuda:
        raise RuntimeError("quantize_columns requires CUDA; the int8 path is GPU-only")
    amax = values.abs().amax(dim=0)
    scale = torch.clamp(amax / 127.0, min=_TINY)
    out = torch.clamp(torch.round(values / scale), -127.0, 127.0).to(torch.int8)
    return out, scale


def scaled_int8_mm(lhs: Tensor, rhs: Tensor, lhs_scale: Tensor, rhs_scale: Tensor) -> Tensor:
    """Multiply int8 matrices and fuse row/column scale dequantization into BF16 output."""
    if not lhs.is_cuda or not rhs.is_cuda:
        raise RuntimeError("scaled_int8_mm requires CUDA operands")
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError("scaled_int8_mm operands must be 2D")
    if lhs.dtype != torch.int8 or rhs.dtype != torch.int8:
        raise TypeError("scaled_int8_mm operands must be int8")
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError("scaled_int8_mm inner dimensions must match")
    if lhs.device != rhs.device:
        raise ValueError("scaled_int8_mm operands must share a device")
    m, k = lhs.shape
    _, n = rhs.shape
    if lhs_scale.shape != (m,) or rhs_scale.shape != (n,):
        raise ValueError("scaled_int8_mm scales must match output rows and columns")
    for name, scale in (("lhs", lhs_scale), ("rhs", rhs_scale)):
        if scale.device != lhs.device or scale.dtype != torch.float32:
            raise TypeError(f"{name} scale must be FP32 on the operand device")
        if not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError(f"{name} scale must be finite and positive")

    # The kernel addresses both operands through their full (row, col) strides, so a
    # transposed/strided operand (e.g. an int8 weight's `.t()` view) is consumed in
    # place — no contiguous copy. Avoiding that copy is the int8-specific working-set
    # saving: at the frontier a transposed FFN weight copy is hundreds of MiB. Triton
    # still needs one of the two strides to be unit; fall back to a copy otherwise.
    if 1 not in lhs.stride():
        lhs = lhs.contiguous()
    if 1 not in rhs.stride():
        rhs = rhs.contiguous()
    lhs_scale = lhs_scale.contiguous()
    rhs_scale = rhs_scale.contiguous()
    output = torch.empty((m, n), device=lhs.device, dtype=torch.bfloat16)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    _scaled_int8_kernel[grid](
        lhs,
        rhs,
        output,
        lhs_scale,
        rhs_scale,
        m,
        n,
        k,
        lhs.stride(0),
        lhs.stride(1),
        rhs.stride(0),
        rhs.stride(1),
        output.stride(0),
        output.stride(1),
    )
    return output
