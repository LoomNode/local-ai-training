# int8 grad_input: fused scaled row-quant (Step 2B)

**Date:** 2026-06-22
**Status:** approved, ready for implementation
**Builds on:** Steps 1 (commit 21c2628) and 2A (commit ddedbc7).
**Motivation:** after 2A, ~28% of the int8 step is still elementwise — the biggest piece is the
grad_input path's `scaled_gradient = flat_gradient.float() * scale[None, :]`, a full `M×N` FP32
materialization plus a separate quantize pass. int8 still trails bf16 only at width 512 (0.70x).

## Goal

Fold the per-column pre-scaling into the row-quantization kernel so the grad_input gradient is
quantized in one pass, with no `M×N` FP32 temp and no separate elementwise multiply. Bit-exact.

## Change

In the int8 branch of `_RatchetMatmul.backward` (`src/local_ai_training/ratchet.py`), replace:

```python
scaled_gradient = flat_gradient.float() * scale.float()[None, :]
gradient_int8, gradient_scale = quantize_rows(scaled_gradient)
```

with:

```python
gradient_int8, gradient_scale = quantize_rows_colscaled(flat_gradient, scale)
```

`unit_scale` and the `scaled_int8_mm` call are unchanged.

## Kernel (parametrize, don't duplicate)

Extend the existing `_quantize_rows_kernel` with an optional per-column scale: add a `col_scale_ptr`
argument and a `HAS_COLSCALE: tl.constexpr` flag. When set, each loaded value is multiplied by
`col_scale[n]` (loaded per-column-tile, fp32) before the abs/amax and before quantization, in both
passes. The branch is compile-time, so the no-scale path is unchanged. One bit-exact source of truth
for row quantization, two thin entry points:

- `quantize_rows(values)` → calls the kernel with `HAS_COLSCALE=False`.
- `quantize_rows_colscaled(values, col_scale)` → calls it with the scale; validates `col_scale` is a
  1-D fp32 CUDA tensor of length `values.shape[1]`.

The reference gains `_quantize_rows_reference(values, col_scale=None)`: when given a scale it computes
`values.float() * col_scale[None, :]` then the existing reference quantization — the exact bit-for-bit
oracle.

## Bit-exactness

`grad.to(fp32) * scale[n]` is the same single IEEE fp32 multiply as `grad.float() * scale[None, :]`.
Everything after (per-row `amax`, `div_rn`, `rint`, clamp) is the already-proven row path. So
`quantize_rows_colscaled(g, s)` is bit-identical to `quantize_rows(g.float() * s[None, :])`.

## Tests (TDD)

- New `test_quantize_rows_colscaled_matches_reference`: `quantize_rows_colscaled(g, s)` equals
  `_quantize_rows_reference(g, s)` and equals `quantize_rows((g.float()*s[None,:]))` bit-for-bit, across
  fp32 and bf16, tile-aligned and ragged shapes, including a transposed/non-contiguous `g`.
- `quantize_rows` (no scale) results unchanged — existing row tests stay green (regression guard on the
  parametrized kernel).
- `test_fused_backward_equivalence[int8]` stays green — end-to-end oracle.

## Verification

- Targeted + full suite green; `lat audit` clean.
- Re-run `scripts/int8_step_profile.py` (the `scaled_gradient` elementwise should vanish; the M×N FP32
  temp gone) and `scripts/int8_training_throughput.py` at 512 / 2048 / 4096. Success: int8 step drops
  further, width 512 improves most. Record in the throughput results note.

## Risks

- `col_scale` indexing in the kernel must use the same column offsets as the value load (covered by the
  bit-exact test, including ragged/masked tiles).
- 3090-specific throughput numbers.
