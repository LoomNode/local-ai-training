# scripts/int8_spike/int4_weight_bench.py
# Bare matmul: optimized int4-WEIGHT x bf16-activation (_weight_int4pack_mm) vs bf16 cuBLAS.
# This is the ratchet's real inference shape: 4-bit codes, bf16 activations, fused dequant.
# Tensor cores still run bf16 here -- this measures the WEIGHT-BANDWIDTH win, not int4 flops.
# (True int4xint4 tensor-core matmul needs int4 activations; no stock PyTorch path -> not tested.)
import time

import torch

SHAPES = [(768, 768, 16384), (2304, 768, 16384), (3072, 768, 16384)]  # (N, K, T)
GROUPSIZE = 128
INNER_K_TILES = 8


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
    for n, k, t in SHAPES:
        x = torch.randn(t, k, device=dev, dtype=torch.bfloat16)
        w16 = torch.randn(n, k, device=dev, dtype=torch.bfloat16)

        # pack w16 -> int4 weights + per-group scales/zeros
        # newer torch expects uint8 [N, K//2], two 4-bit values packed per byte
        lo = torch.randint(0, 16, (n, k // 2), device=dev, dtype=torch.uint8)
        hi = torch.randint(0, 16, (n, k // 2), device=dev, dtype=torch.uint8)
        w_u8 = (hi << 4) | lo
        packed = torch.ops.aten._convert_weight_to_int4pack(w_u8, INNER_K_TILES)
        n_groups = k // GROUPSIZE
        scales_zeros = torch.randn(n_groups, n, 2, device=dev, dtype=torch.bfloat16)

        ms_bf16 = _time(lambda x=x, w16=w16: x @ w16.t())
        try:
            ms_i4 = _time(
                lambda x=x, packed=packed, scales_zeros=scales_zeros: torch._weight_int4pack_mm(
                    x, packed, GROUPSIZE, scales_zeros
                )
            )
            i4_str = f"int4weight={ms_i4:.3f}ms ({ms_bf16 / ms_i4:.2f}x)"
        except Exception as e:  # noqa: BLE001
            i4_str = f"int4weight ERR: {e}"

        print(f"N={n} K={k} T={t}: bf16={ms_bf16:.3f}ms  {i4_str}")


if __name__ == "__main__":
    main()
