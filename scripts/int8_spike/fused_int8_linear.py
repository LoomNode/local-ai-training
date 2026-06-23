# scripts/int8_spike/fused_int8_linear.py
# End-to-end FORWARD linear: does int8 still win once you pay activation quant + dequant?
# Pipeline: per-token int8 quant of activations (eager) -> Triton int8 GEMM with the dequant
# FUSED into the epilogue (writes bf16, no int32 round-trip) -> compare to bf16 cuBLAS linear.
# This is the honest "is there a real training-forward speedup" test the bare GEMM couldn't answer.
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
def _i8_lin(
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
    a_s = tl.load(as_ptr + offs_cm, mask=offs_cm < M, other=0.0)  # per-token act scale
    b_s = tl.load(bs_ptr + offs_cn, mask=offs_cn < N, other=0.0)  # per-col weight scale
    out = acc.to(tl.float32) * a_s[:, None] * b_s[None, :]        # fused dequant
    c_ptrs = c_ptr + scm * offs_cm[:, None] + scn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out.to(tl.bfloat16), mask=mask)


def fused_int8_linear(x_bf16, w_int8, w_scale):
    M, K = x_bf16.shape
    K2, N = w_int8.shape
    a_s = x_bf16.abs().amax(dim=1, keepdim=True) / 127.0          # per-token quant (eager)
    x_i8 = torch.clamp((x_bf16 / a_s).round(), -127, 127).to(torch.int8)
    c = torch.empty((M, N), device=x_bf16.device, dtype=torch.bfloat16)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    a_s_vec = a_s.squeeze(1).float().contiguous()
    b_s_vec = w_scale.float().contiguous()
    _i8_lin[grid](x_i8, w_int8, c, a_s_vec, b_s_vec,
                  M, N, K, x_i8.stride(0), x_i8.stride(1), w_int8.stride(0), w_int8.stride(1),
                  c.stride(0), c.stride(1))
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
    print(f"M(tokens)={M}; forward linear, int8(quant+gemm+dequant) vs bf16\n")
    print(f"{'K':>6} | {'bf16 ms':>8} | {'int8 ms':>8} | {'speedup':>7} | {'rel err':>8}")
    print("-" * 52)
    for w in WIDTHS:
        k = n = w
        x = torch.randn(M, k, device=dev, dtype=torch.bfloat16) * 0.5
        wmat = (torch.randn(k, n, device=dev, dtype=torch.bfloat16) * 0.02)
        w_scale = wmat.abs().amax(dim=0) / 127.0
        w_i8 = torch.clamp((wmat / w_scale).round(), -127, 127).to(torch.int8)

        fused_int8_linear(x, w_i8, w_scale)  # autotune
        ref = x @ wmat
        out = fused_int8_linear(x, w_i8, w_scale)
        rel = ((out - ref).norm() / ref.norm()).item()

        ms_bf16 = _time(lambda x=x, wmat=wmat: x @ wmat)
        ms_i8 = _time(lambda x=x, w_i8=w_i8, w_scale=w_scale: fused_int8_linear(x, w_i8, w_scale))
        spd = ms_bf16 / ms_i8
        print(f"{w:>6} | {ms_bf16:>8.3f} | {ms_i8:>8.3f} | {spd:>6.2f}x | {100 * rel:>7.2f}%")


if __name__ == "__main__":
    main()
