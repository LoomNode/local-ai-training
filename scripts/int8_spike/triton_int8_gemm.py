# scripts/int8_spike/triton_int8_gemm.py
# Hand-written, autotuned Triton int8 GEMM (int8 x int8 -> int32 accumulate), benched vs bf16
# cuBLAS. The question: can a tuned custom kernel push int8 past bf16 (>~45% of int8 peak),
# i.e. deliver the 2x the 3090 spec promises -- where the vendor kernels stalled at ~35%?
import time

import torch
import triton
import triton.language as tl

BF16_PEAK = 71e12
INT8_PEAK = 142e12
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
                                num_stages=s,
                                num_warps=w,
                            )
                        )
    return cfgs


@triton.autotune(configs=_cfgs(), key=["M", "N", "K"])
@triton.jit
def _int8_gemm(
    a_ptr, b_ptr, c_ptr, M, N, K,
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
    c_ptrs = c_ptr + scm * offs_cm[:, None] + scn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def int8_gemm(a, b):
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _int8_gemm[grid](a, b, c, M, N, K,
                     a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c


def _time(fn, it=50, wu=15):
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
    # correctness check at small size
    a = torch.randint(-8, 8, (256, 256), device=dev, dtype=torch.int8)
    b = torch.randint(-8, 8, (256, 256), device=dev, dtype=torch.int8)
    ref = (a.float() @ b.float()).to(torch.int32)  # exact for these small magnitudes
    err = (int8_gemm(a, b) - ref).abs().max().item()
    print(f"correctness max|err| = {err} (must be 0)\n")

    print(f"{'K':>6} | {'bf16 ms':>8} {'%pk':>5} | {'triton i8':>10} {'%pk':>5} | {'spd':>5}")
    print("-" * 52)
    for w in WIDTHS:
        k = n = w
        flops = 2 * M * n * k
        a8 = torch.randint(-127, 128, (M, k), device=dev, dtype=torch.int8)
        b8 = torch.randint(-127, 128, (k, n), device=dev, dtype=torch.int8)
        a16 = torch.randn(M, k, device=dev, dtype=torch.bfloat16)
        b16 = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
        int8_gemm(a8, b8)  # trigger autotune for this shape
        ms_bf16 = _time(lambda a16=a16, b16=b16: a16 @ b16)
        ms_i8 = _time(lambda a8=a8, b8=b8: int8_gemm(a8, b8))
        bp = 100 * flops / (ms_bf16 / 1000) / BF16_PEAK
        ip = 100 * flops / (ms_i8 / 1000) / INT8_PEAK
        spd = ms_bf16 / ms_i8
        print(f"{w:>6} | {ms_bf16:>8.3f} {bp:>4.0f}% | {ms_i8:>10.3f} {ip:>4.0f}% | {spd:>4.2f}x")


if __name__ == "__main__":
    main()
