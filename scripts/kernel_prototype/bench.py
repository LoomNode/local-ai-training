# scripts/kernel_prototype/bench.py
import time

import torch

from local_ai_training.ratchet import pack_code_pressure
from scripts.kernel_prototype.ratchet_forward import ratchet_forward

SHAPES = [(768, 768, 16384), (2304, 768, 16384), (3072, 768, 16384)]
MAX_CODE = 4


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
    for n, k, t in SHAPES:
        code = torch.randint(-MAX_CODE, MAX_CODE + 1, (n, k), dtype=torch.int8, device=dev)
        packed = pack_code_pressure(code, torch.zeros_like(code), MAX_CODE).to(torch.uint8)
        scale = (torch.rand(n, device=dev) + 0.1).to(torch.float32)
        x_bf16 = torch.randn(t, k, device=dev, dtype=torch.bfloat16)
        x_fp32 = x_bf16.to(torch.float32)
        eff_bf16 = (code.to(torch.float32) * scale[:, None]).to(torch.bfloat16)
        eff_fp32 = code.to(torch.float32) * scale[:, None]

        ms_kernel = _time(lambda: ratchet_forward(packed, scale, x_bf16, MAX_CODE))
        ms_bf16 = _time(lambda: x_bf16 @ eff_bf16.t())
        ms_fp32 = _time(lambda: x_fp32 @ eff_fp32.t())

        # peak memory of the eager path (materializes the effective weight) vs kernel
        torch.cuda.reset_peak_memory_stats()
        _ = (code.to(torch.float32) * scale[:, None]).to(torch.bfloat16)
        _ = x_bf16 @ eff_bf16.t()
        mem_eager = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()
        _ = ratchet_forward(packed, scale, x_bf16, MAX_CODE)
        mem_kernel = torch.cuda.max_memory_allocated() / 1e6

        print(f"shape N={n} K={k} T={t}")
        print(f"  kernel     : {ms_kernel:.3f} ms")
        print(f"  bf16-eager : {ms_bf16:.3f} ms   (kernel {ms_bf16 / ms_kernel:.2f}x vs bf16-eager)")
        print(f"  fp32-eager : {ms_fp32:.3f} ms   (kernel {ms_fp32 / ms_kernel:.2f}x vs fp32-eager)")
        print(f"  peak mem   : kernel {mem_kernel:.0f}MB vs eager {mem_eager:.0f}MB")


if __name__ == "__main__":
    main()
