# Ratchet Forward Kernel Prototype: Results

## Setup
Triton dequant-matmul (4-bit weights, bf16 activations, fp32 accumulate) vs eager
materialize+matmul, single RTX 3090, nonary-100M layer shapes at a 16384-token batch.

## Numbers
| shape (N x K, T) | kernel ms | bf16-eager ms | kernel vs bf16 | vs fp32 | peak mem k/e |
| --- | ---: | ---: | ---: | ---: | ---: |
| (768 x 768, 16384) | 0.545 | 0.306 | 0.56x | 2.18x | 139MB / 115MB |
| (2304 x 768, 16384) | 1.651 | 1.030 | 0.62x | 2.13x | 250MB / 178MB |
| (3072 x 768, 16384) | 2.209 | 1.257 | 0.57x | 2.17x | 304MB / 208MB |

Correctness: kernel matches bf16-eager within 2e-2 (test passes).

## Verdict
NO-GO / needs tuning -- The custom kernel is slower than the bf16-eager path (around ~0.6x the speed of bf16-eager). This is likely a tuning issue, as the kernel relies on fixed block sizes without autotuning. Given that it successfully beats fp32-eager but underperforms significantly against bf16 cuBLAS, further tuning of memory access patterns and block dimensions is required before recommending building the full backward-compatible kernel.
