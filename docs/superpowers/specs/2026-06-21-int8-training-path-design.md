# Int8 Training Path Design

**Date:** 2026-06-21
**Status:** Approved for implementation

## Goal

Add an opt-in int8 forward and backward matmul path to ratchet linear layers so a later
matched convergence experiment can isolate whether int8 quantization noise compounds during
training. This design covers only the production wiring and verification. It does not run or
interpret the convergence experiment.

## Scientific Boundary

The ratchet update rule and persistent state do not change. A ratchet matrix continues to
persist only packed `uint8` code/pressure and one explicit FP32 scale per output row. The int8
path changes how the forward activation, input gradient, and effective-weight gradient matmuls
are computed; it does not add a floating-point parameter or persistent tensor that mirrors a
code matrix.

The custom backward must produce the gradient with respect to the effective weight
`W_eff = code * row_scale`. That gradient enters the existing
`apply_weight_gradient -> bucket_pressure -> pressure -> code update` pipeline unchanged.

## Matmul Modes

Add `matmul_mode = "fp32" | "bf16" | "int8"` to `ExperimentConfig` and `ModelConfig`.
The default is `"fp32"`, preserving all current configs, CPU smoke runs, controls, and tests.
The value is parsed from `[training]`, serialized into checkpoint metadata, and threaded into
ratchet linears. FP32-control `nn.Linear` modules retain their existing behavior.

- `fp32`: the existing eager effective-weight path. It remains CPU-capable.
- `bf16`: native BF16 forward and backward matmuls behind the same ratchet dispatch boundary
  used by int8. It requires CUDA.
- `int8`: Triton int8 operands with int32 accumulation and fused BF16 dequantization. It
  requires CUDA.

Setup fails before creating a run directory when BF16 or int8 is selected without CUDA. Int8
uses the error `RuntimeError("int8_matmul requires CUDA; the Triton int8 path is GPU-only")`;
BF16 uses an equally explicit mode-specific error. There is no silent fallback.

Resume validates that the requested `matmul_mode` matches checkpoint experiment metadata.
Changing it mid-run raises an error rather than combining two experimental conditions in one
logical run.

## Production Kernel

Create `src/local_ai_training/int8_matmul.py` from the proven autotuned Triton kernel in
`scripts/int8_spike/int8_backward_bench.py`. The package helper accepts two 2D int8 operands,
explicit FP32 per-row and per-column scales, and returns BF16 after fused dequantization.

The helper validates CUDA placement, ranks, compatible dimensions, scale shapes, contiguity,
and finite positive scales. Quantization clamps scale denominators for all-zero rows or columns
so zero inputs remain finite. Non-tile-multiple dimensions are supported by kernel masks.

## Ratchet Autograd Path

One ratchet matmul autograd function serves the BF16 and int8 modes. It receives activations,
unpacked integer codes, row scales, the selected mode, and a non-tensor temporary gradient sink
owned by the layer. Leading activation dimensions are flattened to
`[tokens, in_features]` and restored on output.

### Forward

BF16 computes:

```text
x_bf16 @ (code_bf16 * row_scale).T
```

Int8 quantizes each activation row symmetrically, consumes `code.T` directly as its integer
right operand, and supplies the ratchet row scale as the right-output-column scale. It does not
requantize the codes or materialize a full effective-weight matrix.

Both paths produce BF16 internally and cast the result back to the input dtype. The surrounding
embedding, normalization, residual, attention, and loss code therefore remains FP32 in both
matched arms; only the linear matmul implementation differs.

### Backward

BF16 backward uses native BF16 matmuls. Int8 backward quantizes the operands and uses the same
Triton int8 matmul helper for both products.

The effective-weight gradient is:

```text
grad_W_eff = grad_output.T @ input
```

It has shape `[out_features, in_features]` and is delivered to the layer's temporary gradient
sink for the unchanged ratchet update.

The input gradient is:

```text
grad_input = (grad_output * row_scale) @ code
```

The row scale lies on the contracted output-feature dimension after transposition, so int8
backward folds it into `grad_output` before quantization. It cannot be represented as an
output-column epilogue scale for this product.

For trainable scales, backward also returns the FP32 scale gradient corresponding to
`W_eff = code * scale`. Fixed scales remain buffers. Temporary effective-weight gradients are
cleared by `ratchet_update()` or `discard_pending_gradient()`. A second training forward before
clearing still raises.

## Matched Comparison Contract

The later convergence experiment compares `bf16` against `int8`, not FP32 against int8. The
int8 path dequantizes to BF16, so a FP32 comparison would combine the FP32-to-BF16 precision
change with int8 quantization. BF16 versus int8 changes only operand quantization.

Matched arms must use the same seed, logical initialization, packed state, support parameters,
batch schedule, evaluation batches, token budget, and all other configuration. FP32 remains
available as a separate absolute-quality reference, not as the quantization-noise control.

## Testing Strategy

Implementation follows strict red-green-refactor cycles.

Configuration and setup tests prove:

- the mode defaults to FP32, validates its three allowed values, and reaches every ratchet layer;
- checkpoint metadata records the mode and resume rejects a mismatch;
- the unchanged CPU smoke/default FP32 path still trains;
- BF16 and int8 fail before model execution or run-directory creation without CUDA.

State and lifecycle tests prove:

- BF16 and int8 add no parameters or persistent code-matrix buffers;
- `lat audit` remains free of violations and reports the same persistent tensor dtypes/bytes;
- pending gradients clear after update/discard and layer reuse before clearing still fails.

CUDA numerical tests prove:

1. BF16 forward and backward match an eager BF16 effective-weight reference.
2. Int8 forward matches the standalone quantized reference computation.
3. The custom backward's `grad_W_eff` matches eager autograd within an int8-appropriate
   tolerance and is consumed by the existing ratchet update.
4. Trainable-scale gradients match autograd through `W_eff = code * scale`.

CUDA edge tests cover all-zero rows, non-tile-multiple dimensions, flattened 3D transformer
inputs, all supported code ranges, and finite outputs and gradients. CUDA-only tests skip when
CUDA is unavailable; CPU configuration, validation, and default-path coverage always run.

A model-level test confirms BF16 and int8 arms built with the same seed have identical packed
state, row scales, and floating-point support parameters before their matmul dispatch differs.

## Error Handling And Reporting

Kernel contract errors identify the invalid device, shape, dtype, or scale. Non-finite loss and
scale handling remains unchanged. The implementation makes no convergence or speed claim.

Gradient quantization is the principal unresolved risk because gradient distributions may be
more outlier-heavy than activations. Task 1 does not preemptively add per-tensor scaling,
stochastic rounding, selective BF16 operations, or other tuning. A later matched convergence
test must expose any divergence or loss gap. Such a result will be preserved and reported rather
than silently tuned away.

## Documentation And Verification

README documentation describes the modes, CUDA restrictions, experimental distinction, and
resume restriction. Before the task-1 implementation is complete, run:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest
UV_CACHE_DIR=/games/ailab/.uv-cache uv run ruff check .
UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat audit --model configs/ratchet_tiny.toml
git diff --check
```

Do not write to or repeat the completed runs under `runs/tiny-shakespeare`. Task 2, the matched
BF16/int8 convergence experiment, is a separate design and execution cycle.
