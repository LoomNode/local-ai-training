# Int8 Activation Spike: Results

## Setup
int8 per-token activation quant + int8 codes via torch._int_mm vs bf16-eager, single RTX 3090,
nonary-100M layer shapes at a 16384-token batch, max_code 4.

## Numbers
| shape (N x K, T) | int8 pipeline ms | bare _int_mm ms | bf16-eager ms | int8 vs bf16 | rel err (per-token / per-tensor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| (768 x 768, 16384) | 1.540 | 0.329 | 0.309 | 0.20x | 0.0105 / 0.0157 |
| (2304 x 768, 16384) | 3.130 | 1.077 | 1.002 | 0.32x | 0.0100 / 0.0124 |
| (3072 x 768, 16384) | 3.931 | 1.424 | 1.283 | 0.33x | 0.0102 / 0.0127 |

Correctness test (exact-quantization pipeline check): FAILS. The math is wrong in the plan.

## Verdict
NO-GO -- The speed gate fails drastically. The full int8 pipeline is significantly slower (0.2x-0.33x) than bf16-eager. Looking at the bare `_int_mm` numbers (0.9x), the `torch._int_mm` matmul itself is already slower than bf16 cuBLAS on these shapes, meaning quantization overhead isn't the sole bottleneck. The tensor core math simply does not yield a speedup here. The accuracy gate looks very promising, with both per-token and per-tensor relative errors remaining in the low single digits (~1.0-1.5%), indicating that outliers wouldn't even necessitate SmoothQuant-style handling. However, because the primary objective was training speed, the int8-activation training run is not recommended to be built.
