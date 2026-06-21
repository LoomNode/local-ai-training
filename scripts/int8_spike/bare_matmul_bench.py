# scripts/int8_spike/bare_matmul_bench.py
# Isolated BARE matmul comparison: int8 matmul kernels vs bf16 cuBLAS, with NO activation
# quantization or dequant in the loop -- pure "is the int8 matmul itself faster?" at our shapes.
import time

import torch

SHAPES = [(768, 768, 16384), (2304, 768, 16384), (3072, 768, 16384)]  # (N, K, T)


def _time(fn, iters=100, warmup=20):
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
    ao_int = None
    try:
        from torchao.kernel.intmm import int_matmul as ao_int  # autotuned bare int8 matmul
    except Exception as e:  # noqa: BLE001
        print(f"(torchao bare int_matmul unavailable: {e})")

    for n, k, t in SHAPES:
        a8 = torch.randint(-127, 128, (t, k), device=dev, dtype=torch.int8)
        b8 = torch.randint(-127, 128, (k, n), device=dev, dtype=torch.int8)
        a16 = torch.randn(t, k, device=dev, dtype=torch.bfloat16)
        b16 = torch.randn(k, n, device=dev, dtype=torch.bfloat16)

        ms_bf16 = _time(lambda: a16 @ b16)
        try:
            ms_intmm = _time(lambda: torch._int_mm(a8, b8))
            intmm_str = f"_int_mm(cuBLASLt IMMA)={ms_intmm:.3f}ms ({ms_bf16 / ms_intmm:.2f}x)"
        except Exception as e:  # noqa: BLE001
            intmm_str = f"_int_mm ERR: {e}"

        line = f"N={n} K={k} T={t}: bf16={ms_bf16:.3f}ms  {intmm_str}"
        if ao_int is not None:
            try:
                ms_ao = _time(lambda: ao_int(a8, b8))
                line += f"  torchao_int_matmul={ms_ao:.3f}ms ({ms_bf16 / ms_ao:.2f}x)"
            except Exception as e:  # noqa: BLE001
                line += f"  torchao_int_matmul ERR: {e}"
        print(line)


if __name__ == "__main__":
    main()
