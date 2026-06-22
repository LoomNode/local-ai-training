# int8 backward restructure (Step 2A) — quantize the gradient once

**Date:** 2026-06-22
**Status:** approved, ready for implementation
**Builds on:** the fused quantize kernels (commit 21c2628) and
`docs/superpowers/specs/2026-06-22-fused-int8-quantization-design.md`.
**Motivation:** with the fused kernels in place, the profile shows the remaining int8 quant cost is the
backward *structure*: the grad_weight tiling loop calls `quantize_rows` on transposed gradient slices
(`flat_gradient.t()[tile]`) ~6–8× per linear (`tile_size=256`), strided (0.46 ms vs 0.12 ms
contiguous) — hundreds of strided calls per step.

## Goal

Quantize the gradient for grad_weight **once** on the contiguous tensor and reuse it across tiles,
eliminating the per-tile strided quantization. Bit-exact with the current backward.

## Scope

Optimization **A** only, in the int8 branch of the fused path in `_RatchetMatmul.backward`
(`src/local_ai_training/ratchet.py`, ~lines 102–132). Out of scope (Step 2B, later): fusing the
per-column pre-scaling to drop the full-FP32 `scaled_gradient` materialization on the grad_input path.
The grad_input path is untouched here.

## Change

In the int8 branch, before the tiling loop:

```python
grad_cols_int8, grad_cols_scale = quantize_columns(flat_gradient)  # (M, N) int8, per-N scale
```

In the loop's int8 branch, replace the per-tile quantize with slices of that result:

```python
weight_lhs_tile = grad_cols_int8[:, tile_start:tile_end].t()   # (N_tile, M), stride (1, N)
weight_lhs_scale_tile = grad_cols_scale[tile_start:tile_end]
grad_weight_tile = scaled_int8_mm(weight_lhs_tile, weight_rhs, weight_lhs_scale_tile, weight_rhs_scale)
```

`weight_lhs_tile` has a unit stride, so `scaled_int8_mm` consumes it in place (no contiguous copy).
`grad_out_t = flat_gradient.t()` remains for the bf16/fp32 branches, which still tile it; the int8
branch no longer references it.

## Bit-exactness

`quantize_rows(flat_gradient.t()[tile])` and `quantize_columns(flat_gradient)[:, tile]` both reduce
over M per output-feature, giving identical `scale[n] = amax_m(|grad[m,n]|)/127`, and
`round(grad[m,n] / scale[n])` is the same integer in both layouts (only transposed). The fused
`quantize_columns` is already proven bit-exact vs `_quantize_columns_reference`. Therefore every
grad_weight tile is bit-identical to the current code, and `grad_scale` (derived from
`grad_weight_tile_fp32`) is unchanged.

## Tests

- New `test_int8_backward_restructure_matches_quantize_rows_transpose`: assert
  `quantize_columns(grad)[:, a:b].t()` equals `quantize_rows(grad.t()[a:b])` bit-for-bit (int8 and
  scale), the exact identity the restructure relies on. This is the targeted RED→GREEN unit.
- `test_fused_backward_equivalence[int8]` stays green — the end-to-end oracle (fused backward vs eager).
- `test_int8_matmul` stays green.

## Verification

- Targeted test + full suite green; `lat audit` clean.
- Re-run `scripts/int8_step_profile.py` (row-kernel time should collapse) and
  `scripts/int8_training_throughput.py` at widths 512 / 2048 / 4096. Success: int8 step time drops and
  int8 crosses bf16 at width ≥ 2048. Record before/after in the throughput results note.

## Risks

- The strided int8 `.t()` slice into `scaled_int8_mm` is exercised by existing strided-operand tests,
  but the new call site is covered by `test_fused_backward_equivalence`.
- 3090-specific throughput numbers.
