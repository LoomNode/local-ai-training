import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def _rmsnorm_bwd_kernel(
    grad_y_ptr, x_ptr, w_ptr, grad_x_ptr,
    M, K, stride_m, stride_k, eps, BLOCK_K: tl.constexpr
):
    row_id = tl.program_id(0)
    ptrs = row_id * stride_m + tl.arange(0, BLOCK_K) * stride_k
    mask = tl.arange(0, BLOCK_K) < K
    
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    grad_y = tl.load(grad_y_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + tl.arange(0, BLOCK_K), mask=mask, other=0.0).to(tl.float32)
    
    variance = tl.sum(x * x) / K
    rsqrt = tl.math.rsqrt(variance + eps)
    
    dx_hat = grad_y * w
    dvar = tl.sum(dx_hat * x) * (-0.5) * (rsqrt * rsqrt * rsqrt)
    dx = (dx_hat * rsqrt) + (dvar * 2.0 * x / K)
    
    tl.store(grad_x_ptr + ptrs, dx.to(tl.bfloat16), mask=mask)

def rmsnorm_backward(grad_y: Tensor, x: Tensor, w: Tensor, eps: float = 1e-5) -> tuple[Tensor, Tensor]:
    if not x.is_cuda:
        raise RuntimeError("CUDA required")
    x_flat = x.flatten(0, -2)
    grad_y_flat = grad_y.flatten(0, -2)
    M, K = x_flat.shape
    grad_x = torch.empty_like(x_flat)
    
    BLOCK_K = triton.next_power_of_2(K)
    _rmsnorm_bwd_kernel[(M,)](
        grad_y_flat, x_flat, w, grad_x,
        M, K, x_flat.stride(0), x_flat.stride(1), eps, BLOCK_K
    )
    
    # Calculate grad_w using PyTorch native operations instead of Triton atomic_add
    # which is significantly faster.
    variance = (x_flat.float() ** 2).mean(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(variance + eps)
    grad_w = torch.sum(grad_y_flat.float() * x_flat.float() * rsqrt, dim=0)
    
    return grad_x.reshape(x.shape), grad_w.to(w.dtype)

@triton.jit
def _gelu_bwd_kernel(
    grad_y_ptr, x_ptr, grad_x_ptr,
    M, K, stride_m, stride_k, BLOCK_K: tl.constexpr
):
    row_id = tl.program_id(0)
    ptrs = row_id * stride_m + tl.arange(0, BLOCK_K) * stride_k
    mask = tl.arange(0, BLOCK_K) < K
    
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    grad_y = tl.load(grad_y_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    
    sqrt_2_over_pi = 0.7978845608028654
    x_sq = x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x * x_sq)
    tanh_inner = tl.extra.cuda.libdevice.tanh(inner)
    
    term1 = 0.5 * (1.0 + tanh_inner)
    term2 = 0.5 * x * (1.0 - tanh_inner * tanh_inner) * sqrt_2_over_pi * (1.0 + 0.134145 * x_sq)
    grad_x = grad_y * (term1 + term2)
    
    tl.store(grad_x_ptr + ptrs, grad_x.to(tl.bfloat16), mask=mask)

def gelu_backward(grad_y: Tensor, x: Tensor) -> Tensor:
    if not x.is_cuda:
        raise RuntimeError("CUDA required")
    x_flat = x.flatten(0, -2)
    grad_y_flat = grad_y.flatten(0, -2)
    M, K = x_flat.shape
    grad_x = torch.empty_like(x_flat)
    
    BLOCK_K = triton.next_power_of_2(K)
    _gelu_bwd_kernel[(M,)](
        grad_y_flat, x_flat, grad_x,
        M, K, x_flat.stride(0), x_flat.stride(1), BLOCK_K
    )
    return grad_x.reshape(x.shape)
