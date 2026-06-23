# scripts/int8_spike/int8_backward_bench.py
# The real "does TRAINING speed up" test: a full linear step in int8 vs bf16.
# A linear y = x @ W (x:[M,K], W:[K,N]) needs THREE matmuls per step:
#   forward : y      = x @ W                 (contract K)
#   backward: grad_x = grad_y @ W^T          (contract N)
#   backward: grad_W = x^T @ grad_y          (contract M, the token dim = 16384)
# Each int8 GEMM quantizes both operands (per-row LHS, per-col RHS), matmuls int8->int32,
# and dequants in a fused epilogue -> bf16. We time all three and check grad accuracy vs bf16.
import time

import torch
import triton
import triton.language as tl

BF16_PEAK = 71e12
M = 16384
WIDTHS = [768, 2048, 4096, 8192, 12288]


def _cfgs():
    cfgs = []
    for bm in (64, 128, 256):
        for bn in (64, 128, 256):
            for bk in (32, 64, 128):
                for s in (3, 4):
                    for w in (4, 8):
                        cfgs.append(
                            triton.Config(
                                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                                num_stages=s, num_warps=w,
                            )
                        )
    return cfgs


@triton.autotune(configs=_cfgs(), key=["M", "N", "K"])
@triton.jit
def _i8mm(
    a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
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
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_s = tl.load(as_ptr + offs_cm, mask=offs_cm < M, other=0.0)
    b_s = tl.load(bs_ptr + offs_cn, mask=offs_cn < N, other=0.0)
    out = acc.to(tl.float32) * a_s[:, None] * b_s[None, :]
    c_ptrs = c_ptr + scm * offs_cm[:, None] + scn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out.to(tl.bfloat16), mask=mask)


def int8_mm(a_bf16, b_bf16):
    """Quantize both operands (per-row a, per-col b), int8 matmul, fused dequant -> bf16."""
    a_bf16 = a_bf16.contiguous()
    b_bf16 = b_bf16.contiguous()
    Mx, Kx = a_bf16.shape
    Kb, Nx = b_bf16.shape
    a_s = a_bf16.abs().amax(dim=1, keepdim=True) / 127.0
    a_i8 = torch.clamp((a_bf16 / a_s).round(), -127, 127).to(torch.int8)
    b_s = b_bf16.abs().amax(dim=0) / 127.0
    b_i8 = torch.clamp((b_bf16 / b_s).round(), -127, 127).to(torch.int8)
    c = torch.empty((Mx, Nx), device=a_bf16.device, dtype=torch.bfloat16)

    def grid(meta):
        return (triton.cdiv(Mx, meta["BLOCK_M"]) * triton.cdiv(Nx, meta["BLOCK_N"]),)

    _i8mm[grid](a_i8, b_i8, c, a_s.squeeze(1).float().contiguous(), b_s.float().contiguous(),
                Mx, Nx, Kx, a_i8.stride(0), a_i8.stride(1), b_i8.stride(0), b_i8.stride(1),
                c.stride(0), c.stride(1))
    return c


def step_int8(x, w, gy):
    y = int8_mm(x, w)            # forward
    gx = int8_mm(gy, w.t())      # grad_x  = grad_y @ W^T
    gw = int8_mm(x.t(), gy)      # grad_W  = x^T @ grad_y
    return y, gx, gw


def step_bf16(x, w, gy):
    return x @ w, gy @ w.t(), x.t() @ gy


def _time(fn, it=30, wu=10):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / it * 1000


def main():
    dev = "cuda"
    print(f"M(tokens)={M}; full linear STEP (fwd + grad_x + grad_W), int8 vs bf16\n")
    hdr = f"{'K':>6} | {'bf16 ms':>8} | {'int8 ms':>8} | {'spd':>5} | {'gx err':>6} | {'gW err':>6}"
    print(hdr)
    print("-" * 64)
    for w in WIDTHS:
        k = n = w
        x = torch.randn(M, k, device=dev, dtype=torch.bfloat16) * 0.5
        wmat = torch.randn(k, n, device=dev, dtype=torch.bfloat16) * 0.02
        gy = torch.randn(M, n, device=dev, dtype=torch.bfloat16) * 0.1

        step_int8(x, wmat, gy)  # autotune all three shapes
        _, gx, gw = step_int8(x, wmat, gy)
        _, gx_ref, gw_ref = step_bf16(x, wmat, gy)
        gx_err = ((gx - gx_ref).norm() / gx_ref.norm()).item()
        gw_err = ((gw - gw_ref).norm() / gw_ref.norm()).item()

        ms_bf16 = _time(lambda x=x, wmat=wmat, gy=gy: step_bf16(x, wmat, gy))
        ms_i8 = _time(lambda x=x, wmat=wmat, gy=gy: step_int8(x, wmat, gy))
        spd = ms_bf16 / ms_i8
        print(f"{w:>6} | {ms_bf16:>8.3f} | {ms_i8:>8.3f} | {spd:>6.2f}x | "
              f"{100 * gx_err:>6.2f}% | {100 * gw_err:>6.2f}%")


if __name__ == "__main__":
    main()
