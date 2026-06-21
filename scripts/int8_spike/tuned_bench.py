import time

import torch
from torch import nn

from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

SHAPES = [(768, 768, 16384), (2304, 768, 16384), (3072, 768, 16384)]  # (N, K, T)


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
        x = torch.randn(t, k, device=dev, dtype=torch.bfloat16)

        bf16 = nn.Linear(k, n, bias=False).to(dev).to(torch.bfloat16).eval()
        ms_bf16 = _time(lambda: bf16(x))

        # torchao int8: per-token int8 activations + int8 weight, tuned int8 matmul
        ao = nn.Linear(k, n, bias=False).to(dev).to(torch.bfloat16).eval()
        quantize_(ao, Int8DynamicActivationInt8WeightConfig())
        with torch.no_grad():
            ms_ao = _time(lambda: ao(x))

        print(f"shape N={n} K={k} T={t}")
        print(f"  bf16        : {ms_bf16:.3f} ms")
        print(f"  torchao int8: {ms_ao:.3f} ms   ({ms_bf16 / ms_ao:.2f}x vs bf16)")


if __name__ == "__main__":
    main()
