# int8 Training-Speed: Reversal of the NO-GO (tuned Triton kernel delivers the 2x)

**Date:** 2026-06-21
**TL;DR:** The earlier "int8 gives no training speedup" NO-GO was wrong — it trusted *vendor*
int8 kernels that stall at ~35% of peak on the 3090. A hand-written autotuned Triton int8 GEMM
reaches ~100% of int8 peak and **2.10x bf16**. End-to-end (with activation quant) the forward
linear is **1.8-1.95x at frontier width**, though only break-even-to-loss at toy width.

## How we got here

The 2x is the RTX 3090 spec ratio (142 dense-int8 TOPS vs 71 bf16-fp32-accumulate TFLOPS). The
prior investigation concluded int8 couldn't realize it, based on:
- `torch._int_mm` (cuBLASLt IMMA): ~0.8-0.93x bf16
- torchao autotuned Triton: 0.24-0.69x bf16

Both are vendor/library kernels. A width sweep showed they sit at **~33-42% of int8 peak at
EVERY width** (768 -> 12288), flat — while bf16 cuBLAS runs ~90-100% of its (consumer-halved)
peak. That looked like a hardware ceiling. It wasn't: it was kernel quality.

## Bare GEMM: a tuned Triton kernel hits peak

`scripts/int8_spike/triton_int8_gemm.py` — standard int8xint8->int32 GEMM, autotuned over
tile/stage/warp configs. Correctness max error = 0.

| width (K) | bf16 %peak | triton-int8 %peak | speedup |
| ---: | ---: | ---: | ---: |
| 768   | 101% | 95%  | 1.88x |
| 2048  |  94% | 79%  | 1.67x |
| 4096  |  99% | 97%  | 1.95x |
| 8192  |  98% | 103% | 2.10x |
| 12288 |  98% | 103% | 2.10x |

The silicon's 2x is real and reachable. Vendor IMMA kernels are just badly tuned for these
tall (M=16384) shapes.

## End-to-end forward linear (with activation quant)

`scripts/int8_spike/fused_int8_linear.py` — per-token int8 activation quant (eager) + the Triton
int8 GEMM with **dequant fused into the epilogue** (writes bf16, no int32 round-trip) vs bf16 cuBLAS.

| width (K) | bf16 ms | int8 ms | speedup | rel err |
| ---: | ---: | ---: | ---: | ---: |
| 768   |  0.306 |  0.512 | 0.60x | 1.18% |
| 2048  |  2.046 |  1.873 | 1.09x | 1.26% |
| 4096  |  7.747 |  5.348 | 1.45x | 1.31% |
| 8192  | 31.553 | 17.515 | 1.80x | 1.36% |
| 12288 | 71.452 | 36.591 | 1.95x | 1.39% |

The fixed cost of the eager per-token quant pass only amortizes at large width. So the
*end-to-end* win is **width-gated** — and switches on exactly at frontier scale, where the
ratchet's memory win also lives. (Fusing the quant into the preceding op would lift the small-K
numbers, untested.)

## Activation precision: int8 yes, int4 no

`scripts/int8_spike/activation_precision.py` — per-token symmetric quant, realistic activations
(gaussian + ~0.4% outlier channels @20x), matmul-output rel L2 error:

| scheme | rel err |
| --- | ---: |
| int8 act / fp weight | 6.1% |
| int4 act / fp weight | **61.7%** |
| int8 act / int8 weight | 6.1% |
| int4 act / int4 weight | **63.1%** |

(The 6% here is inflated by deliberately harsh synthetic outliers; the realistic measured number
is ~1%, consistent with the earlier spike.) int4 activations are dead. **int8 is the activation
precision; the ratchet keeps 4-bit weights but matmuls in int8.**

## Full training step (forward + both backward GEMMs)

`scripts/int8_spike/int8_backward_bench.py` — a linear step needs three matmuls: forward
`y=x@W` (contract K), `grad_x=grad_y@W^T` (contract N), `grad_W=x^T@grad_y` (contract M=16384).
Each int8 GEMM quantizes both operands + fused dequant. Gradient error vs bf16:

| width (K) | bf16 ms | int8 ms | speedup | grad_x err | grad_W err |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 768   |   0.832 |   2.281 | 0.36x | 1.18% | 1.41% |
| 2048  |   6.261 |   7.547 | 0.83x | 1.26% | 1.43% |
| 4096  |  24.929 |  20.329 | 1.23x | 1.31% | 1.41% |
| 8192  | 101.867 |  63.466 | 1.61x | 1.37% | 1.41% |
| 12288 | 218.978 | 130.368 | 1.68x | 1.39% | 1.41% |

**Full-step training speedup is real but width-gated: ~1.6-1.7x at frontier width, crossover
~K=4096, <1x at toy width** (the x6 quant passes swamp small matmuls). Gradients stay ~1.4%
accurate. Lower than forward-only (1.95x) because backward adds quant overhead -- which is
fusable headroom (quant could fold into adjacent layernorm/activation ops, untested).

## Ratchet alignment

The ratchet's stored `code` is int8-representable (+-7) and its per-row scale is exactly the
int8 weight scale — so the ratchet weight maps onto the int8xint8 path with **no extra weight
quantization**. The pieces fit.

## Open / next

- **Convergence under int8 activations in training.** Gradients are ~1.4% noisy per step — does
  that hurt final loss over thousands of steps? Needs an actual training run with int8 GEMMs in
  the loop, not just per-step timing. This is the one remaining unknown for the speed claim.
- **Quant fusion** to lower the K=4096 crossover (fold per-token quant into the preceding op).
- Hardware caveat: all 3090-specific. A100/H100 and FP8 (Hopper) differ.
