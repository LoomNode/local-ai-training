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


def quantize_rows(values: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetrically quantize each row and return int8 values plus FP32 scales."""
    if values.ndim != 2:
        raise ValueError("row quantization requires a 2D tensor")
    scale = _positive_scale(values, dimension=1)
    quantized = torch.clamp((values.float() / scale[:, None]).round(), -127, 127)
    return quantized.to(torch.int8), scale


def quantize_columns(values: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetrically quantize each column and return int8 values plus FP32 scales."""
    if values.ndim != 2:
        raise ValueError("column quantization requires a 2D tensor")
    scale = _positive_scale(values, dimension=0)
    quantized = torch.clamp((values.float() / scale[None, :]).round(), -127, 127)
    return quantized.to(torch.int8), scale


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

    lhs = lhs.contiguous()
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
