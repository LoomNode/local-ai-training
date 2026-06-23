# Int8 Training Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in BF16 and Triton-int8 forward/backward matmuls to ratchet linears without changing persistent ratchet state or the update rule.

**Architecture:** Preserve the existing FP32 layer path and add a shared opt-in ratchet autograd boundary. A focused package module owns row/column quantization and the autotuned int8 GEMM; the layer captures the temporary effective-weight gradient from custom backward and feeds it to the existing update unchanged.

**Tech Stack:** Python 3.11, PyTorch custom autograd, Triton, pytest, Ruff, safetensors.

---

### Task 1: Configure and Validate Matmul Modes

**Files:**
- Modify: `src/local_ai_training/config.py`
- Modify: `src/local_ai_training/model.py`
- Modify: `src/local_ai_training/train.py`
- Modify: `src/local_ai_training/checkpoint.py`
- Test: `tests/test_experiment.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: Write failing configuration and setup tests**

Add tests asserting `ExperimentConfig.matmul_mode == "fp32"`, TOML accepts only `fp32`,
`bf16`, and `int8`, the value reaches every `DiscreteRatchetLinear`, CPU FP32 still trains,
and CPU BF16/int8 raise before `run_dir` exists. Add a checkpoint resume test that changes
only `matmul_mode` and expects `ValueError("checkpoint matmul_mode does not match requested run")`.

- [ ] **Step 2: Verify the tests fail for the missing feature**

Run:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest \
  tests/test_experiment.py tests/test_model.py -k 'matmul_mode or cpu_fp32' -v
```

Expected: failures because `matmul_mode` is not defined or parsed.

- [ ] **Step 3: Add the minimal configuration and setup implementation**

Add `matmul_mode: Literal["fp32", "bf16", "int8"] = "fp32"` to both dataclasses; validate
the allowed set; parse it from `[training]`; thread it through `_linear`; and expose it on
`DiscreteRatchetLinear`. Resolve and validate the device before creating `run_dir`:

```python
if config.matmul_mode == "int8" and device.type != "cuda":
    raise RuntimeError("int8_matmul requires CUDA; the Triton int8 path is GPU-only")
if config.matmul_mode == "bf16" and device.type != "cuda":
    raise RuntimeError("bf16 matmul requires CUDA; the BF16 comparison path is GPU-only")
```

Pass `expected_experiment_config=config.to_dict()` to `load_checkpoint` and reject a saved
`matmul_mode` that differs, treating missing legacy metadata as `"fp32"`.

- [ ] **Step 4: Verify focused and existing CPU tests pass**

Run:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest \
  tests/test_experiment.py tests/test_model.py -v
```

Expected: all tests pass; FP32 remains the CPU default.

### Task 2: Productionize the Scaled Int8 GEMM

**Files:**
- Create: `src/local_ai_training/int8_matmul.py`
- Create: `tests/test_int8_matmul.py`

- [ ] **Step 1: Write failing helper-contract and CUDA numerical tests**

Define tests for `quantize_rows`, `quantize_columns`, and `scaled_int8_mm`. CPU tests cover
rank/dtype/shape/device errors and finite positive scales for all-zero inputs. CUDA tests,
skipped when unavailable, compare non-tile-multiple int8 products against FP32 integer matmul
plus explicit scales and verify exact output shape, BF16 dtype, and finite values.

- [ ] **Step 2: Verify tests fail because the module is absent**

Run:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_int8_matmul.py -v
```

Expected: collection failure for missing `local_ai_training.int8_matmul`.

- [ ] **Step 3: Implement quantization helpers and the tuned Triton kernel**

Move the autotuned `_i8mm` kernel from `scripts/int8_spike/int8_backward_bench.py` into the
package module. Implement symmetric row and column quantizers with `clamp_min` denominators,
and `scaled_int8_mm(lhs, rhs, lhs_scale, rhs_scale)` with explicit contract checks and fused
BF16 dequantization.

- [ ] **Step 4: Verify helper tests pass**

Run:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_int8_matmul.py -v
```

Expected: CPU contract tests pass and CUDA numerical tests pass on an available GPU.

### Task 3: Add the Ratchet BF16/Int8 Autograd Boundary

**Files:**
- Modify: `src/local_ai_training/ratchet.py`
- Modify: `tests/test_ratchet.py`

- [ ] **Step 1: Write failing lifecycle and BF16/int8 numerical tests**

Add CPU state tests proving opt-in modes add no parameters/buffers and leave audit bytes
unchanged. Add CUDA tests for 3D forward flatten/restore, all code ranges, BF16 eager parity,
int8 reference parity, captured effective-weight-gradient accuracy, trainable-scale-gradient
parity, zero rows, non-tile dimensions, finite gradients, and update/discard lifecycle.

- [ ] **Step 2: Verify tests fail on the existing FP32-only forward**

Run:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py \
  -k 'matmul or bf16 or int8' -v
```

Expected: failures because opt-in dispatch and temporary gradient capture do not exist.

- [ ] **Step 3: Implement custom autograd and layer dispatch**

Add `_RatchetMatmul(torch.autograd.Function)` with BF16 and int8 branches. Save only tensors
needed for backward. In int8 `grad_input`, multiply `grad_output` by row scale before row
quantization and multiply by raw codes. Compute `grad_effective_weight = grad_output.T @ input`
through the selected backend, send it to `_capture_weight_gradient`, and return the FP32
scale gradient when scale is trainable. Replace the pending effective tensor with a temporary
`_pending_weight_gradient`; keep FP32 behavior and public lifecycle semantics unchanged.

- [ ] **Step 4: Verify ratchet tests and audit behavior pass**

Run:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py -v
UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat audit --model configs/ratchet_tiny.toml
```

Expected: all tests pass and audit reports zero violations with unchanged persistent state.

### Task 4: Prove Matched Models and Resume Safety End to End

**Files:**
- Modify: `tests/test_model.py`
- Modify: `tests/test_experiment.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing matched-arm model test**

Build BF16 and int8 ratchet models with the same seed and assert identical packed buffers,
row scales, token embeddings, normalization parameters, and positional encoding. Assert only
`matmul_mode` differs.

- [ ] **Step 2: Verify the matched-arm test fails before final wiring**

Run:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest \
  tests/test_model.py::test_bf16_and_int8_modes_share_identical_initial_state -v
```

Expected: failure until mode propagation and model comparison behavior are complete.

- [ ] **Step 3: Complete wiring and document the command surface**

Make the smallest wiring corrections exposed by the test. Update README with `[training]
matmul_mode`, default FP32 behavior, CUDA-only BF16/int8 errors, BF16-vs-int8 convergence
control rationale, and resume-mode restriction. Do not add or run a convergence config.

- [ ] **Step 4: Run the complete task-1 verification gates**

Run:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest
UV_CACHE_DIR=/games/ailab/.uv-cache uv run ruff check .
UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat audit --model configs/ratchet_tiny.toml
git diff --check
```

Expected: all four commands exit zero. Report persistent tensor dtypes and byte counts from
the audit. Confirm `runs/tiny-shakespeare` was not modified.

- [ ] **Step 5: Commit task 1**

Stage only task-1 source, tests, plan, and README changes and commit with LoomNode identity and
the repository's co-author trailer. Do not include generated `data/` or `runs/` artifacts.
