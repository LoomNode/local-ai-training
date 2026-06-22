# Fused int8 quantization kernel — design

**Date:** 2026-06-22
**Status:** approved, ready for implementation
**Motivation:** `docs/results/2026-06-22-int8-training-throughput.md` (int8 is 0.56x bf16 at the 25M
model) and the step profile (`scripts/int8_step_profile.py`): at width 512 the tuned int8 GEMM is only
~16% of a training step while ~74% is quantization tax — unfused, memory-bound elementwise work that
bf16 never pays.

## Goal

Replace `quantize_rows` / `quantize_columns` in `src/local_ai_training/int8_matmul.py` with fused
single-pass Triton kernels that compute the per-row/per-column `amax`, derive `scale = amax/127`
(clamped to `finfo(float32).tiny`), and emit int8 + fp32 scale with minimal global memory traffic.

**Bit-exact** with the current torch quantization — this is a pure-performance change, provably neutral
to all convergence/quality results. Any change to quantizer *numerics* (rounding mode, scale
granularity, stochastic rounding) is explicitly out of scope and would be a separate, measured
experiment on top of this faster baseline.

## Scope

In scope: the two quantize functions only. Out of scope (later steps): `scaled_int8_mm`, the ratchet
backward structure, the full-FP32 `scaled_gradient` materialization (`ratchet.py:103`), and the
double-quantization of inputs (forward + backward). Folding those in now would entangle the change and
muddy attribution.

## Contract change

The int8 path is CUDA-only end to end: `scaled_int8_mm` raises on CPU (`int8_matmul.py:107`) and
`train.py:113` rejects `matmul_mode="int8"` off CUDA. `quantize_rows`/`quantize_columns` are only ever
called inside the int8 branches of the ratchet backward, so they never see a CPU tensor in production.
The only CPU caller is one unit test.

Therefore `quantize_rows`/`quantize_columns` become **CUDA-only**, raising a clear error on CPU
(consistent with `scaled_int8_mm`). The current torch bodies are preserved verbatim as private
`_quantize_rows_reference` / `_quantize_columns_reference` in the same module, used only by tests as the
bit-exact oracle. No shipped dual code path to drift.

## Kernels

Two kernels in `int8_matmul.py`. Each is inherently two passes over the input — the scale depends on the
whole row/column `amax` before any element can be quantized — which is still ~3 global round-trips vs the
current ~5–6 separate full-tensor kernels (`.float()`, `/scale`, `round`, `clamp`, `.to(int8)`, plus the
`amax` reduction).

- `_quantize_rows_kernel`: one program per block of rows. Pass 1 loops K-tiles accumulating fp32
  `abs`-max → `amax`. `scale = max(amax/127, tiny)`. Pass 2 loops K-tiles storing
  `int8 = clamp(rint(val/scale), -127, 127)`. Reduces over the contiguous axis (coalesced). Stores
  `scale[row]`.
- `_quantize_columns_kernel`: one program per block of columns. Pass 1 loops M-tiles loading
  `(BLOCK_M, BLOCK_N)` (coalesced across columns) and reducing `amax` over M. Pass 2 quantizes. Handles
  the strided reduction without a transpose copy.

Both take full input strides, so the backward's transposed inputs (e.g. `flat_gradient.t()`) are
consumed in place — same principle as the strided-operand fix already landed in `scaled_int8_mm`.

Autotuning: a small `triton.autotune` config set keyed on the reduced/served dims. Keep the config space
modest to avoid the autotuner-memory spikes seen at the frontier.

## Bit-exactness

Match torch op-for-op, all intermediate math in fp32:

1. load → cast to fp32 (exact widening from bf16/fp32).
2. `amax = abs(values).max()` over the quant axis in fp32. Max is order-independent, so reduction order
   cannot diverge from torch.
3. `scale = (amax / 127).clamp_min(torch.finfo(float32).tiny)` (`tiny = 1.1754943508222875e-38`).
4. `q = clamp(rint(values / scale), -127, 127)` using `libdevice.rint` (round-half-to-even, matches
   `torch.round`; **not** `libdevice.round`, which is half-away-from-zero).
5. store as int8 (post-clamp values are integral, so the cast is exact).

The one residual risk is whether Triton's fp32 divide is IEEE round-to-nearest on all inputs. This is
not asserted a priori — the bit-exact tests are the proof. A divergence turns a test red and we fix the
kernel, never the tolerance.

Edge: an all-zero row gives `amax = 0 → scale = tiny → rint(0/tiny) = 0`, matching torch; the
zero-stays-finite property holds.

## Tests (TDD, written first)

1. `test_quantize_rows_fused_matches_reference_bit_exact` — `torch.equal` of both int8 and scale vs
   `_quantize_rows_reference`, across fp32 and bf16 inputs, tile-aligned and ragged shapes (e.g. 67×35),
   including a transposed/non-contiguous input.
2. `test_quantize_columns_fused_matches_reference_bit_exact` — same for columns.
3. zero-input-stays-finite, on CUDA (scale finite and > 0, int8 all zero).
4. `quantize_rows`/`quantize_columns` on a CPU tensor raises (the new contract).
5. existing `test_fused_backward_equivalence` (int8) stays green — the end-to-end oracle.

## Verification

- All of the above plus the full suite green; `lat audit` clean (no master weights introduced).
- Re-run `scripts/int8_step_profile.py` and `scripts/int8_training_throughput.py`: success = the
  elementwise/reduce buckets shrink substantially and the int8 step time falls toward bf16 at width 512
  and ideally below it at width ≥ 2048. Record the before/after in a results note.

## Risks / limitations

- Triton fp32 divide semantics (covered by tests, above).
- Two-pass reload is the floor for a scale-dependent quantizer; a true single-pass would require caching
  the whole row/column in SRAM (infeasible for large K). Accepted.
- 3090-specific throughput numbers.
