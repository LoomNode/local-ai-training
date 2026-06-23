# scripts/int8_spike/width_sweep_bench.py
# Does int8's matmul win switch on as the model gets WIDER (larger contraction K)?
# Sweep K = model hidden width, measure bf16 cuBLAS vs int8 _int_mm (cuBLASLt IMMA),
# and report each as % of its own hardware peak. Square-ish GEMM per width: N=K, M=16384 tokens.
import time

import torch

BF16_PEAK = 71e12   # RTX 3090 bf16 tensor TFLOPS (fp32 accum)
INT8_PEAK = 142e12  # int8 tensor TOPS (~2x bf16)
M = 16384           # tokens (batch*seq)
WIDTHS = [768, 2048, 4096, 8192, 12288]  # K = N = hidden width


def _time(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1000  # ms/call


def main():
    dev = "cuda"
    print(f"M(tokens)={M}, square GEMM N=K=width\n")
    hdr = f"{'width(K)':>8} | {'bf16 ms':>8} {'%pk':>5} | {'int8 ms':>8} {'%pk':>5} | {'spd':>5}"
    print(hdr)
    print("-" * 60)
    for w in WIDTHS:
        k = n = w
        flops = 2 * M * n * k
        a8 = torch.randint(-127, 128, (M, k), device=dev, dtype=torch.int8)
        b8 = torch.randint(-127, 128, (k, n), device=dev, dtype=torch.int8)
        a16 = torch.randn(M, k, device=dev, dtype=torch.bfloat16)
        b16 = torch.randn(k, n, device=dev, dtype=torch.bfloat16)

        ms_bf16 = _time(lambda a16=a16, b16=b16: a16 @ b16)
        ms_int8 = _time(lambda a8=a8, b8=b8: torch._int_mm(a8, b8))

        bf16_pct = 100 * flops / (ms_bf16 / 1000) / BF16_PEAK
        int8_pct = 100 * flops / (ms_int8 / 1000) / INT8_PEAK
        print(
            f"{w:>8} | {ms_bf16:>8.3f} {bf16_pct:>5.1f}% | "
            f"{ms_int8:>8.3f} {int8_pct:>5.1f}% | {ms_bf16 / ms_int8:>6.2f}x"
        )


if __name__ == "__main__":
    main()
