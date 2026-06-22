# Phase 0: Packed Training Memory Probe

## Memory Profile

Using `configs/ratchet_tiny.toml` (256 width) for memory measurements over a single forward-backward pass, breaking down base persistent memory, forward peak, and full step backward peak.

| Mode   | Peak (MB) | Base (MB) | Fwd (MB) | Diff Peak-Fwd (MB) |
|--------|-----------|-----------|----------|--------------------|
| fp32   | 94.68     | 0.49      | 82.29    | 12.39              |
| bf16   | 102.77    | 21.81     | 88.28    | 14.49              |
| int8   | 369.84    | 19.11     | 91.34    | **278.50**         |

### Observations
1. **Forward Pass (`int8`)**: The forward memory is modest because `int8` does not instantiate or hold a full FP `_effective_weight`. The memory consumed is strictly activations.
2. **Backward Peak (`int8`)**: The transient allocation blows up massively during `int8` backprop (278 MB). The backward pass for int8 currently forms the full `grad_W_eff` (in FP32) by allocating several huge int8 quantized matrices and transposes of the inputs and output-gradients (`weight_lhs`, `weight_rhs`), then materializes the full `[out, in]` gradient matrix just to feed `bucket_pressure`.

### The Dominant Transient
The dominant transient is indeed **`grad_W_eff`** and the operations required to materialize it. To realize the 1-byte training peak, we must fuse the backward pass to avoid ever instantiating the full `[out_features, in_features]` FP gradient matrix.

## Scope of Phase 1 (Backward Fusion)
We will tile the gradient computation over output rows.
Instead of computing:
`grad_W_eff = grad_out^T @ inputs`
We will stream `grad_out` row-blocks (or tile output dimensions), compute the gradient for that block, feed it directly to the `bucket_pressure` logic to get the `code` and `pressure` integer updates, apply them to the packed buffer, and immediately discard the tile's FP gradients. Peak backward memory will drop to $O(\text{tile size})$, fully eliminating the $O(O \times I)$ `grad_W_eff` transient.

## Phase 1 Results (Fused Backward)
After implementing the fused backward update and tiling over output rows (including tiling the quantization), the int8 peak memory dropped dramatically from ~370 MB to **134.93 MB**, falling well within range of the FP32/BF16 baselines. Bit-for-bit equivalence with the original eager update was fully preserved across all parameterizations.

| Mode   | Peak (MB) | Base (MB) | Fwd (MB) | Diff Peak-Fwd (MB) |
|--------|-----------|-----------|----------|--------------------|
| fp32   | 95.49     | 18.68     | 88.28    | 7.21               |
| bf16   | 102.74    | 18.68     | 88.28    | 14.46              |
| int8   | **134.93**| 36.68     | 107.45   | **27.48**          |

*Note: The remaining backward overhead primarily comes from the activation-bound `weight_rhs` (`quantize_columns(flat_inputs)`).*

## Phase 2 Results (Activation Checkpointing)
After applying standard `torch.utils.checkpoint` to the `TransformerBlock` and turning it on by default via `ModelConfig(gradient_checkpointing=True)`, the forward peak drops massively and the orthogonal activation memory footprint is largely eliminated.

| Mode         | Peak (MB) | Base (MB) | Fwd (MB) | Diff Peak-Fwd (MB) |
|--------------|-----------|-----------|----------|--------------------|
| fp32 (ckpt)  | 65.21     | 18.68     | 26.89    | 38.32              |
| bf16 (ckpt)  | 74.46     | 18.68     | 26.89    | 47.57              |
| int8 (ckpt)  | **87.48** | 18.68     | 26.89    | **60.59**          |

The fully optimized `int8` backward path with tiled quantization and activation checkpointing peaks at
**87.48 MB**, fully removing the 278 MB backward regression.

> **Correction (interpretation):** the original conclusion here compared int8-checkpointed (87 MB)
> against the *unfused, uncheckpointed* fp32 (95 MB) — apples-to-oranges. The like-for-like number is
> int8-ckpt **87.48** vs fp32-ckpt **65.21**: at this 256-width toy size int8 is still the *highest*,
> because weights don't dominate memory here, so the 1-byte advantage is invisible and only int8's
> overhead shows. This probe proves the **regression is fixed and the update stays bit-exact** — it
> does **not** demonstrate a memory win. For where the win actually appears (and where it doesn't —
> int8 ties/loses to bf16, the 12× is stranded behind activations), see the scaling sweep:
> `2026-06-21-packed-memory-scaling.md`.
