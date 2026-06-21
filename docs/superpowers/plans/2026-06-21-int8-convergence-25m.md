# Matched BF16 vs Int8 Convergence 25M Implementation Plan

**Goal:** Run a deterministic, sequential single-GPU comparison that isolates the accuracy
effect of int8 ratchet matmuls over 12,000 text8 training steps.

**Experiment:** Use seed 1337 and two configurations copied from
`configs/scaleup_text8_25m.toml`. The configurations differ only at `matmul_mode`; BF16 runs
first and int8 runs second on the same idle RTX 3090. No artifact directory may be created until
the matched-arm preflight succeeds.

**Tooling:** Python 3.11, PyTorch, pytest, TOML, the existing `local_ai_training` model/data
APIs, and the `lat` CLI.

## Task 1: Add Matched Configurations

**Files:**

- Add `configs/int8_convergence_25m_bf16.toml`
- Add `configs/int8_convergence_25m_int8.toml`

Copy every field and comment from `configs/scaleup_text8_25m.toml`. Add only
`matmul_mode = "bf16"` or `matmul_mode = "int8"` under `[model]`. Confirm a structural TOML
comparison reports no other difference.

## Task 2: Specify Preflight Behavior With Failing Tests

**Files:**

- Add `tests/test_int8_convergence_preflight.py`
- Add `scripts/int8_convergence_preflight.py`

Write focused tests first and observe each fail because the implementation is absent. Cover:

1. Config equality except for `model.matmul_mode`.
2. Failure on any additional config mismatch.
3. Equality of packed ratchet codes, row scales, floating-point support parameters, buffers,
   and positional state after independent seed-1337 construction.
4. Failure for deliberately mismatched codes, scales, support state, or positional state.
5. Equality of complete 12,000-step training batch schedules and 40-batch evaluation schedules.
6. Failure for a deliberately mismatched training or evaluation schedule.
7. Preservation of CPU and all CUDA RNG states around isolated BF16 and int8 matmuls.
8. Failure when a deliberately RNG-consuming matmul is checked.
9. JSON provenance output and a nonzero CLI exit for every mismatch category.

Keep the comparison functions independently testable. The CLI must fully validate both arms
before writing the JSON summary and must not create the run root.

## Task 3: Implement the Preflight

Use the repository's parsed config, model construction, ratchet storage, data schedule, and
matmul APIs. Do not introduce a persistent floating-point mirror of any ratchet matrix. Hash
large schedules incrementally rather than retaining training tensors. Include config paths,
seed, schedule sizes and digests, state comparisons, RNG checks, CUDA device information, and
the current commit in the JSON summary.

Run the focused tests after each red-green cycle, then the complete focused file. Review the
script output and confirm deliberate mismatches exit nonzero.

## Task 4: Run Repository Gates and Preflight

Run:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest
UV_CACHE_DIR=/games/ailab/.uv-cache uv run ruff check .
UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat audit --model configs/int8_convergence_25m_int8.toml
git diff --check
```

Then run the preflight with its JSON output directed outside the not-yet-created run root. Abort
the experiment on any failure. Only after it passes, inspect `nvidia-smi`, select one idle RTX
3090, and record GPU index, UUID, memory, utilization, active processes, and commit.

## Task 5: Execute Sequential Training

Create `runs/int8-convergence-25m/` only after the hard gate passes. Pin both commands to the
selected GPU with `CUDA_VISIBLE_DEVICES=<index>`. Run BF16 to completion before starting int8:

```bash
MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
CUDA_VISIBLE_DEVICES=<index> uv run lat train \
  --config configs/int8_convergence_25m_bf16.toml \
  --dataset-path data/text8/text8 --codes 9 --seed 1337 \
  --output runs/int8-convergence-25m/bf16

MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
CUDA_VISIBLE_DEVICES=<index> uv run lat train \
  --config configs/int8_convergence_25m_int8.toml \
  --dataset-path data/text8/text8 --codes 9 --seed 1337 \
  --output runs/int8-convergence-25m/int8
```

Preserve partial and failed artifacts unchanged. Never resume with altered settings and never
touch `runs/tiny-shakespeare`.

## Task 6: Analyze Matched Steps and Report

Recursively locate each `metrics.csv`, join only exact shared evaluation steps, and require the
sets to be identical from step 0 through 12,000 at intervals of 200. Generate
`runs/int8-convergence-25m/comparison.png` without interpolation.

Write `docs/results/2026-06-21-int8-convergence-25m.md` with provenance, hardware, commands,
preflight evidence, all matched validation losses and signed `int8 - bf16` gaps, final and best
losses, best steps, late-trajectory direction, and the unchanged classification:

- **Tracks:** final gap at most 0.03 nats, no sustained late widening, and no instability.
- **Marginal:** final gap above 0.03 and at most 0.08 nats, or a visibly widening late curve.
- **Fails accuracy gate:** final gap above 0.08 nats, divergence, or instability.

Report cumulative moves, saturation, gradient RMS, and throughput only as diagnostics. State
that these toy widths are below the speed crossover and throughput is not a success criterion.
Document failures and limitations, including the single seed and toy width.

Confirm both artifact trees contain `metrics.csv`, checkpoint JSON, and safetensors. Re-run all
final gates, commit the configs, preflight code/tests, and result note, and confirm the tracked
worktree is clean.
