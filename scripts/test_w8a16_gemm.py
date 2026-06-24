import torch
import triton
import triton.language as tl

# We define block configs for autotuning
def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ]

@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def _w8a16_gemm_kernel(
    a_ptr, b_ptr, b_scale_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b_scale_ptrs = b_scale_ptr + offs_k

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b_int8 = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_bn[None, :] < N), other=0.0)
        b_scale = tl.load(b_scale_ptrs, mask=offs_k < K - k * BLOCK_K, other=0.0)
        
        # Dequantize B on the fly inside SRAM!
        b_bf16 = b_int8.to(tl.bfloat16) * b_scale[:, None].to(tl.bfloat16)
        
        accumulator = tl.dot(a, b_bf16, accumulator)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        b_scale_ptrs += BLOCK_K

    c = accumulator.to(tl.bfloat16)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def w8a16_matmul(a: torch.Tensor, b_int8: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    # A is [M, K], b_int8 is [K, N], b_scale is [K]
    assert a.shape[1] == b_int8.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b_int8.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b_int8.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    _w8a16_gemm_kernel[grid](
        a, b_int8, b_scale, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b_int8.stride(0), b_int8.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def time_fn(name, fn, *args, iters=1000):
    for _ in range(10): fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn(*args)
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / iters
    print(f"{name:30s}: {avg_ms:.3f} ms")

def main():
    # Simulate the backward pass gradient accumulation tile shape:
    # A is grad_out_tile: [tile_size, seq*batch] -> [256, 16384]
    # B is inputs: [seq*batch, hidden_dim] -> [16384, 512]
    M, K, N = 256, 16384, 512
    
    a_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b_int8 = torch.randint(-127, 127, (K, N), device='cuda', dtype=torch.int8)
    b_scale = torch.rand(K, device='cuda', dtype=torch.float32)
    
    # 1. Baseline: PyTorch reconstruction + matmul
    def pytorch_baseline(a, b_i8, b_s):
        b_bf16 = b_i8.to(torch.bfloat16) * b_s[:, None].to(torch.bfloat16)
        return a @ b_bf16

    # Verify correctness
    c_ref = pytorch_baseline(a_bf16, b_int8, b_scale)
    c_custom = w8a16_matmul(a_bf16, b_int8, b_scale)
    max_diff = (c_ref.float() - c_custom.float()).abs().max().item()
    print(f"Max difference between PyTorch and Triton: {max_diff:.6f}")
    
    if max_diff > 1.0:
        print("WARNING: Custom kernel is producing incorrect math! Running benchmark anyway to see raw speed.")

    # Benchmark
    time_fn("PyTorch reconstruction + bf16", pytorch_baseline, a_bf16, b_int8, b_scale)
    time_fn("Triton fused W8A16 gemm", w8a16_matmul, a_bf16, b_int8, b_scale)

if __name__ == "__main__":
    main()
