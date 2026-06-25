"""Fused int8 operations to eliminate memory-bound quantization passes."""

import torch
import triton
import triton.language as tl
from torch import Tensor

_TINY = torch.finfo(torch.float32).tiny

@triton.jit
def _rint_div(x, scale):
    return tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(x.to(tl.float32), scale))

@triton.jit
def _fused_rmsnorm_quantize_kernel(
    x_ptr, weight_ptr, out_ptr, scale_ptr,
    M, K, stride_x_m, stride_x_k, stride_o_m, stride_o_k,
    eps: tl.constexpr, BLOCK_K: tl.constexpr, TINY: tl.constexpr
):
    row_id = tl.program_id(0)
    x_ptrs = x_ptr + row_id * stride_x_m + tl.arange(0, BLOCK_K) * stride_x_k
    mask = tl.arange(0, BLOCK_K) < K
    
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_K), mask=mask, other=0.0).to(tl.float32)
    
    # RMSNorm
    variance = tl.sum(x * x) / K
    rsqrt = tl.math.rsqrt(variance + eps)
    x_norm = x * rsqrt * w
    
    # Quantize
    amax = tl.max(tl.abs(x_norm))
    scale = tl.maximum(amax / 127.0, TINY)
    tl.store(scale_ptr + row_id, scale)
    
    q = tl.minimum(tl.maximum(_rint_div(x_norm, scale), -127.0), 127.0)
    out_ptrs = out_ptr + row_id * stride_o_m + tl.arange(0, BLOCK_K) * stride_o_k
    tl.store(out_ptrs, q.to(tl.int8), mask=mask)

def fused_rmsnorm_quantize(x: Tensor, weight: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    if not x.is_cuda:
        raise RuntimeError("CUDA required")
    original_shape = x.shape
    x_flat = x.flatten(0, -2)
    M, K = x_flat.shape
    out = torch.empty((M, K), device=x.device, dtype=torch.int8)
    scale = torch.empty((M,), device=x.device, dtype=torch.float32)
    
    BLOCK_K = triton.next_power_of_2(K)
    _fused_rmsnorm_quantize_kernel[(M,)](
        x_flat, weight, out, scale,
        M, K, x_flat.stride(0), x_flat.stride(1), out.stride(0), out.stride(1),
        eps, BLOCK_K, _TINY
    )
    return out.reshape(original_shape), scale.reshape(original_shape[:-1])

@triton.jit
def _fused_gelu_quantize_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K, stride_x_m, stride_x_k, stride_o_m, stride_o_k,
    BLOCK_K: tl.constexpr, TINY: tl.constexpr
):
    row_id = tl.program_id(0)
    x_ptrs = x_ptr + row_id * stride_x_m + tl.arange(0, BLOCK_K) * stride_x_k
    mask = tl.arange(0, BLOCK_K) < K
    
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # GELU (approximate)
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    tanh_inner = tl.extra.cuda.libdevice.tanh(inner)
    x_gelu = 0.5 * x * (1.0 + tanh_inner)
    
    # Quantize
    amax = tl.max(tl.abs(x_gelu))
    scale = tl.maximum(amax / 127.0, TINY)
    tl.store(scale_ptr + row_id, scale)
    
    q = tl.minimum(tl.maximum(_rint_div(x_gelu, scale), -127.0), 127.0)
    out_ptrs = out_ptr + row_id * stride_o_m + tl.arange(0, BLOCK_K) * stride_o_k
    tl.store(out_ptrs, q.to(tl.int8), mask=mask)

def fused_gelu_quantize(x: Tensor) -> tuple[Tensor, Tensor]:
    if not x.is_cuda:
        raise RuntimeError("CUDA required")
    original_shape = x.shape
    x_flat = x.flatten(0, -2)
    M, K = x_flat.shape
    out = torch.empty((M, K), device=x.device, dtype=torch.int8)
    scale = torch.empty((M,), device=x.device, dtype=torch.float32)
    
    BLOCK_K = triton.next_power_of_2(K)
    _fused_gelu_quantize_kernel[(M,)](
        x_flat, out, scale,
        M, K, x_flat.stride(0), x_flat.stride(1), out.stride(0), out.stride(1),
        BLOCK_K, _TINY
    )
    return out.reshape(original_shape), scale.reshape(original_shape[:-1])

@triton.jit
def _fused_transpose_quantize_kernel(
    x_ptr, out_ptr, scale_ptr,
    batch_size, n_head, seq_len, head_dim, channels,
    stride_b, stride_h, stride_s, stride_d,
    stride_o_m, stride_o_k,
    BLOCK_C: tl.constexpr, TINY: tl.constexpr
):
    row_id = tl.program_id(0)
    b = row_id // seq_len
    s = row_id % seq_len
    
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < channels
    
    h = offs_c // head_dim
    d = offs_c % head_dim
    
    x_ptrs = x_ptr + b * stride_b + h * stride_h + s * stride_s + d * stride_d
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    
    # Quantize
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(amax / 127.0, TINY)
    tl.store(scale_ptr + row_id, scale)
    
    q = tl.minimum(tl.maximum(_rint_div(x, scale), -127.0), 127.0)
    out_ptrs = out_ptr + row_id * stride_o_m + offs_c * stride_o_k
    tl.store(out_ptrs, q.to(tl.int8), mask=mask)

def fused_transpose_quantize(x: Tensor) -> tuple[Tensor, Tensor]:
    if not x.is_cuda:
        raise RuntimeError("CUDA required")
    batch_size, n_head, seq_len, head_dim = x.shape
    channels = n_head * head_dim
    M = batch_size * seq_len
    
    out = torch.empty((M, channels), device=x.device, dtype=torch.int8)
    scale = torch.empty((M,), device=x.device, dtype=torch.float32)
    
    BLOCK_C = triton.next_power_of_2(channels)
    _fused_transpose_quantize_kernel[(M,)](
        x, out, scale,
        batch_size, n_head, seq_len, head_dim, channels,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1),
        BLOCK_C, _TINY
    )
    return out, scale

class FusedRMSNormQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float = 1e-5):
        int8_out, scale_out = fused_rmsnorm_quantize(x, weight, eps)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        dummy = torch.empty_like(x)
        return int8_out, scale_out, dummy

    @staticmethod
    def backward(ctx, grad_int8, grad_scale, grad_dummy):
        x, weight = ctx.saved_tensors
        from local_ai_training.int8_backward import rmsnorm_backward
        grad_x, grad_w = rmsnorm_backward(grad_dummy, x, weight, ctx.eps)
        return grad_x, grad_w, None

class FusedGELUQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        int8_out, scale_out = fused_gelu_quantize(x)
        ctx.save_for_backward(x)
        dummy = torch.empty_like(x)
        return int8_out, scale_out, dummy

    @staticmethod
    def backward(ctx, grad_int8, grad_scale, grad_dummy):
        x, = ctx.saved_tensors
        from local_ai_training.int8_backward import gelu_backward
        grad_x = gelu_backward(grad_dummy, x)
        return grad_x

class FusedTransposeQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attended: Tensor):
        int8_out, scale_out = fused_transpose_quantize(attended)
        ctx.save_for_backward(attended)
        batch_size, n_head, seq_len, head_dim = attended.shape
        channels = n_head * head_dim
        dummy = torch.empty(
            (batch_size, seq_len, channels), device=attended.device, dtype=attended.dtype
        )
        return int8_out, scale_out, dummy

    @staticmethod
    def backward(ctx, grad_int8, grad_scale, grad_dummy):
        attended, = ctx.saved_tensors
        batch_size, n_head, seq_len, head_dim = attended.shape
        grad_attended = grad_dummy.view(batch_size, seq_len, n_head, head_dim).transpose(1, 2)
        return grad_attended
