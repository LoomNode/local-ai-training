# Spec: Fused Backward Update for Packed Storage

## Context
Phase 0 memory profiling revealed that the peak memory during `int8` training is dominated by the backward pass. Specifically, computing `grad_W_eff` (the `[out_features, in_features]` gradient in FP32) and the temporaries needed to produce it causes massive transient allocations. Since the integer code and pressure updates are entirely row-independent, we can tile the backward pass.

## Goal
Compute the integer code and pressure updates without ever materializing the full `[out_features, in_features]` FP32 effective-weight gradient matrix. Peak memory should scale with `O(tile_size * in_features)` instead of `O(out_features * in_features)`.

## Equivalence Invariant
The resulting integer state (`code` and `pressure`) must remain bit-identical (or extremely close if FP accumulation order changes) to the current eager `bucket_pressure` logic. 

## Design
Currently, the process happens in two steps:
1. **`loss.backward()`**: `_RatchetMatmul.backward` computes `grad_weight_fp32` and passes it to `ctx.gradient_sink(grad_weight_fp32)`. The gradient is saved in `self._pending_weight_gradient`.
2. **`optimizer.step()` & `ratchet_update()`**: `ratchet_update()` iterates over layers, runs RMS normalization on `_pending_weight_gradient`, converts to integer pressure, updates `packed`, and frees `_pending_weight_gradient`.

### Fused Streaming Pipeline
We will change `ctx.gradient_sink` to instead accept an inline "updater" closure that can stream blocks of `grad_weight_fp32` as they are computed. Actually, instead of pushing the gradient to the sink, we can just push the row-tile computations directly.

Since `grad_weight_fp32 = grad_output.t() @ flat_inputs`, we can tile over the `out_features` dimension of `grad_output.t()`.

For `tile_start` from `0` to `out_features` with step `TILE_SIZE`:
1. Slice `grad_output.t()[tile_start : tile_start + TILE_SIZE, :]`
2. Multiply by `flat_inputs` to get `tile_grad = [TILE_SIZE, in_features]`
3. Slice `self.packed[tile_start : tile_start + TILE_SIZE, :]`
4. Compute `rms = tile_grad.square().mean(dim=1).sqrt()`
5. Normalize `tile_normalized = tile_grad / (rms + eps)`
6. Run `_ratchet_update_core` on the tile
7. Write the new packed rows back to `self.packed`.
8. Accumulate move statistics (positive, negative, blocked).
9. Free `tile_grad` and `tile_normalized`.

Because this performs the ratchet update *during* `loss.backward()`, it effectively bypasses `model.ratchet_update()` for these layers.
Wait, `train.py` calls `model.ratchet_update()` and aggregates `RatchetUpdateStats`. If we update during backward, we must store the stats so that `ratchet_update()` can simply collect them and clear them, instead of doing the computation.

### Integration with `DiscreteRatchetLinear`
- Introduce a flag `fuse_backward_update=False`.
- Add `self._pending_update_stats: RatchetUpdateStats | None = None`.
- If `fuse_backward_update=True`, `_RatchetMatmul.backward` will not materialize the full `grad_weight`. Instead, it will loop over chunks.
- For `bf16` mode, it does `gradient_bf16_tile @ inputs_bf16` -> `tile_grad`.
- For `int8` mode, it quantizes `inputs` once (already done?), then tile-by-tile quantizes `grad_output` and runs `scaled_int8_mm` to get `tile_grad`.
- It invokes a callback `ctx.fused_update(tile_start, tile_grad)`.
- The callback updates `packed` in place and aggregates stats.
- At the end of backward, `ctx.fused_update_finalize()` saves the total stats.
- When `ratchet_update()` is called, it just returns `_pending_update_stats` and resets it.

### Testing Plan
1. Write a standalone equivalence test: create a `DiscreteRatchetLinear`. Feed it `inputs` and `grad_output`.
2. Compute `expected_packed` and `expected_stats` using the standard eager path.
3. Compute `fused_packed` and `fused_stats` using the tiled backward path.
4. `torch.testing.assert_close(fused_packed, expected_packed)`.
5. Verify `max_memory_allocated` drops massively during the test.

## Plan
1. Add `test_fused_backward_equivalence` in `tests/test_ratchet.py`. It should fail or be a stub.
2. Modify `ratchet.py` to support `fuse_backward_update` in `DiscreteRatchetLinear`.
3. Implement the tiling logic in `_RatchetMatmul.backward`.
4. Ensure `train.py` or `model.py` can enable this.
5. Verify tests pass.
