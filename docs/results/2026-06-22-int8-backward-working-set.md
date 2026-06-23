# Phase B: int8 backward working set — transposed-copy removed, but the frontier OOM is the autotuner

**Date:** 2026-06-22
**GPU:** NVIDIA GeForce RTX 3090, 24,576 MiB (`CUDA_VISIBLE_DEVICES=1`)
**Builds on:** `docs/results/2026-06-22-corrected-memory-sweep.md`

## Goal

The corrected sweep showed int8 is a narrow memory liability: it trains to 5120-width while bf16
reaches 6144 on a 24 GiB 3090 (int8 OOMs at 6144). Phase B's target was the int8-specific backward
working set so int8 stops costing that one model size.

## Change (bit-exact, kept)

`scaled_int8_mm` previously forced `.contiguous()` on both GEMM operands even though the Triton kernel
already addresses them through full `(row, col)` strides. The int8 forward therefore materialized a
full transposed int8 copy of every weight via `code.t().contiguous()` — at 6144-width an FFN weight
copy is 6144×24576 = **144 MiB** of transient allocation per layer.

The fix consumes a transposed/strided operand in place (Triton needs one unit stride; falls back to a
copy only if neither stride is unit), and the forward now passes `code.t()` directly. This removes
hundreds of MiB of transient int8 allocator traffic at the frontier and simplifies the path.

**Bit-exactness preserved:** `test_fused_backward_equivalence` (int8) and the int8 GEMM equivalence
tests pass, plus a new `test_scaled_int8_mm_consumes_noncontiguous_operands_bit_exactly` guarding the
strided path. 49 int8+ratchet GPU tests pass; `lat audit` clean.

## Measured effect: the targeted working set was not the frontier blocker

Training-step peaks (observability excluded), before vs after:

| width | bf16 train | int8 train (before) | int8 train (after) |
| ---: | ---: | ---: | ---: |
| 3072 | 2,991.1 | 3,047.8 | 3,047.7 |
| 4096 | 6,792.6 | 6,836.7 | 6,837.2 |
| 6144 | 22,163.1 | **OOM** | **OOM** |

All MiB. The steady-state int8 peak is essentially unchanged — the transposed copy was a brief
transient that did not coincide with the per-step maximum where headroom existed.

At 6144 int8 **still OOMs**, and the failure is a Triton **autotuner launch-time** OOM
(`RuntimeError: Triton Error [CUDA]: out of memory` inside `driver.py` `self.launch`), not the
backward working set. With persistent ratchet state at 20.7 GiB and the step at ~22.2 GiB (bf16 fits),
the autotuner's transient workspace while benchmarking its 108 configs (`_configs()` =
3×3×3×2×2) spikes over the 24 GiB ceiling. The steady int8 working set itself would fit
(~bf16 + tens of MiB).

## Conclusion

The transposed-operand copy is real overhead and worth removing (done, bit-exact), but it is not what
costs int8 the 6144 size. Buying that size back requires reducing the **autotuner's** peak — constrain
the config set, or pre-warm/cache autotuning before the model fills memory — which trades against the
separate tuned-kernel speed work (the 108-config space is what delivers the ~2× int8 GEMM). That is a
deliberate speed-vs-memory decision, deferred rather than taken here.

## Limitations

- 6144 int8 failure is a kernel-launch OOM; the exact autotuner workspace footprint is not separately
  attributed.
- Per-step peaks from `scripts/int8_3072_observability_probe.py` (observability no-op'd); 3 steps,
  batch 2, context 32, seed 1337.
