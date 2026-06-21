# scripts/kernel_prototype/ratchet_forward.py
import torch
import triton
import triton.language as tl


@triton.jit
def _ratchet_forward_kernel(
    x_ptr, packed_ptr, scale_ptr, out_ptr,
    T, N, K, max_code,
    stride_xt, stride_xk,
    stride_pn, stride_pk,
    stride_ot, stride_on,
    BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        x_tile = tl.load(
            x_ptr + offs_t[:, None] * stride_xt + k[None, :] * stride_xk,
            mask=(offs_t[:, None] < T) & (k[None, :] < K), other=0.0,
        ).to(tl.bfloat16)
        p_tile = tl.load(
            packed_ptr + offs_n[:, None] * stride_pn + k[None, :] * stride_pk,
            mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0,
        )
        code = (p_tile & 0x0F).to(tl.int32) - max_code          # [BLOCK_N, BLOCK_K]
        code_bf = code.to(tl.bfloat16)
        acc += tl.dot(x_tile, tl.trans(code_bf))                # [BLOCK_T, BLOCK_N], fp32 accumulate
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc * scale[None, :]
    tl.store(
        out_ptr + offs_t[:, None] * stride_ot + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=(offs_t[:, None] < T) & (offs_n[None, :] < N),
    )


def ratchet_forward(packed: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, max_code: int) -> torch.Tensor:
    assert packed.dtype == torch.uint8 and packed.is_cuda
    T, K = x.shape
    N, Kp = packed.shape
    assert Kp == K
    x = x.to(torch.bfloat16)
    scale = scale.to(torch.float32)
    out = torch.empty((T, N), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_T"]), triton.cdiv(N, meta["BLOCK_N"]))
    _ratchet_forward_kernel[grid](
        x, packed, scale, out, T, N, K, max_code,
        x.stride(0), x.stride(1), packed.stride(0), packed.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=64, BLOCK_N=64, BLOCK_K=32,
    )
    return out
