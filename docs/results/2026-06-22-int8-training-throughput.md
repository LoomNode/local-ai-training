# End-to-end int8 training throughput: int8 wins at width ≥2048, bf16 owns width 512

**Date:** 2026-06-22
**GPU:** NVIDIA GeForce RTX 3090, 24,576 MiB (`CUDA_VISIBLE_DEVICES=1`)
**Closes the open question from:** `docs/results/2026-06-22-int8-backward-working-set.md` and the
"STILL UNPROVEN: end-to-end training speedup" caveat on the tuned int8 GEMM.

## CURRENT NUMBERS — read this first (the tables below the fold are the pre-optimization baseline)

This note records a before→after. The final, post-optimization end-to-end training throughput
(bit-exact, after fused quant kernels + backward restructure + Step 2B) is:

| width | int8 vs fp32 | int8 vs bf16 | verdict |
| ---: | ---: | ---: | --- |
| **512** (the 25M research model) | **0.79x** | **0.71x** | **bf16's regime — int8 is slower here** |
| 2048 | 2.07x | 1.07x | int8 wins |
| 4096 | 2.87x | 1.30x | int8 wins |

**Bottom line:** int8 training is the throughput winner at **width ≥ 2048**; at **width 512 it stays
slower than both bf16 and fp32**, and that is *structural* — the GEMM is too small to amortize the
per-step quantize passes, **not** leftover inefficiency (confirmed in `docs/ROADMAP.md`). Do not quote
the 0.56x/0.61x figures below as current — they are the unoptimized path, kept for the before→after
record. The full optimization story is in the **Update** section at the end.

## Question

The bare-GEMM bench showed the custom Triton int8 kernel reaches ~2x bf16. The eager-throughput note
only compared fp32 against the *fp32-effective-weight* ratchet. This measures the **integrated int8
path** (`matmul_mode="int8"`: tuned kernel in the forward and both backward GEMMs) running a full
training step — forward, backward, ratchet update, optimizer — against bf16-ratchet and fp32.

## Method

`scripts/int8_training_throughput.py`: each mode in a fresh process (clean Triton autotuning), random
tokens (throughput is data-independent), 8 warmup steps (excludes autotune/compile), then median of 30
timed steps with explicit CUDA syncs. Batch 64 x block 256 (M = 16,384, compute-bound — where the 2x
should live), checkpointing off to isolate the matmul cost. Width swept to probe the GEMM-size regime.

## Result (pre-optimization baseline — SUPERSEDED by the Update below; see the top banner for current numbers)

| width (n_embd) | fp32 tok/s | bf16 tok/s | int8 tok/s | int8 vs fp32 | int8 vs bf16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 (the 25M research model) | 92,504 | 101,655 | 56,579 | 0.61x | **0.56x** |
| 2048 | 10,009 | 19,166 | 15,838 | 1.58x | 0.83x |
| 4096 | 4,992 | 11,190 | 11,970 | 2.40x | **1.07x** |

Two clear findings:

1. **The int8 speedup does not manifest at the research model size.** At width 512 (the 25M arm) the
   integrated int8 path is **0.61x fp32 and 0.56x bf16** — nearly 2x *slower* than bf16. The bare-GEMM
   2x is swamped by per-step quantization overhead: per-token activation quant in the forward, plus
   gradient and input quant for the two backward GEMMs, are memory-bound elementwise passes that don't
   shrink with the GEMM, and at K=512 the int8 tensor-core advantage is far below the bench's
   8K-12K-wide best case.

2. **int8 only overtakes bf16 above ~4096-width, and only barely (+7%).** The quantization overhead is
   roughly fixed per element while GEMM compute grows with width, so int8 catches up as matmuls get
   large. The crossover is at width ~4096 — far above any model this project trains — and even there
   the margin over bf16 is 1.07x, not 2x.

3. **bf16-ratchet is the practical throughput winner and beats fp32.** At every tested size bf16 is
   1.10x-2.24x fp32 with identical 1-byte persistent storage and no quantization machinery. It is the
   sweet spot: the storage win of the ratchet plus faster-than-fp32 steps from bf16 cuBLAS, with none
   of int8's per-step overhead.

## Implication

The int8 path's only remaining advantage was speed, and end-to-end it is a throughput **liability** at
the sizes that matter (and a memory liability per the corrected sweep). The capacity story belongs to
**bf16-ratchet**: ~21.7B trainable params on one 24 GiB 3090 (vs fp32's ~0.8B), at 1.10x fp32
throughput. int8 is not worth pursuing for training on this hardware unless a future model lives above
~4096-width AND the per-step activation/gradient quantization is itself fused away — a much larger
effort than the bare-GEMM kernel, with at best a single-digit-percent payoff over bf16.

## Update (2026-06-22): the NO-GO was the unoptimized path — int8 now beats bf16 at scale

The numbers above are the *unoptimized* int8 path. A step profile (`scripts/int8_step_profile.py`)
showed the int8 GEMM was only ~16% of the step and ~74% was unfused quantization tax. Two bit-exact
changes fixed it:

1. Fused Triton quantization kernels (commit 21c2628) replacing the ~5-pass torch quantize.
2. int8 backward quantizes the gradient once on the contiguous tensor instead of per-tile on a
   transposed view (commit ddedbc7).

Re-measured end-to-end training throughput (same batch64 x block256, bit-exact vs the table above):

| width | int8 vs fp32 (before → after) | int8 vs bf16 (before → after) |
| ---: | ---: | ---: |
| 512  | 0.61x → 0.79x | 0.56x → 0.70x |
| 2048 | 1.58x → 2.07x | 0.83x → **1.05x** |
| 4096 | 2.40x → 2.87x | 1.07x → **1.28x** |

int8 now **wins** over bf16 at width ≥ 2048 (and 2.1–2.9x fp32), bit-exact. At width 512 bf16-cublas
is still ahead — the small-GEMM regime where the fixed per-step quantization cost is not amortized.

Step 2B (commit 9ad98f1) then fused the grad_input per-column pre-scaling into the row-quant
(`quantize_rows_colscaled`), dropping the M×N FP32 `scaled_gradient` temp. This was a **small** gain —
the temp was a minor slice of the elementwise bucket (most is other casts), so int8 step 181→177 ms:

| width | int8 vs bf16 (after 2A → after 2B) |
| ---: | ---: |
| 512  | 0.70x → 0.71x |
| 2048 | 1.05x → 1.07x |
| 4096 | 1.28x → 1.30x |

Net across the whole effort, int8 went from 0.56x/0.83x/1.07x bf16 (512/2048/4096, unoptimized) to
0.71x/1.07x/1.30x, all bit-exact. int8 is the throughput winner at width ≥2048; width 512 remains
bf16's (the regime where the GEMM is too small to outweigh quantization overhead).

## Limitations

- 3090-specific; consumer bf16/int8 throughput ratios differ from datacenter parts.
- Checkpointing off (isolates matmul cost); enabling it scales forward cost equally across modes.
- Throughput only — accuracy of int8-activation training is a separate axis (int8 activations ~1%
  per-token error in earlier spike work; not re-measured here).
- Width swept by shrinking depth to keep runs short; per-step time is dominated by the per-layer GEMMs,
  which is what the comparison isolates.
