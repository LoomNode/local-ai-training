# Experiment integrity correction — 2026-09-12

## Status

Implemented and verified on CPU and NVIDIA RTX 3090 GPU 1. The correction preserves the normal
ratchet learning rule while closing frozen-control and resume-integrity gaps.

## Frozen-control exposure

Git history establishes that commit `d11fe8880672333b9159695cfd1e45192f13dae6` introduced the fused
backward/update backend and passed `fuse_backward_update=True` for every ratchet linear. In that path,
packed code/pressure and update statistics were changed during autograd backward. The training loop's
`weight_mode == "frozen"` gate ran only after `loss.backward()`, so discarding pending gradients could
not undo those mutations. Baseline/current affected revision
`9dcc15f6580459c31aab7658b846a037abe9e9cf` contains that ancestor.

The existing test `test_frozen_control_never_moves_codes` asserted only that the reported cumulative
move count was zero. It did not compare packed tensors, row scales, optional RMS EMA, pressure-leak
counters, or pending update statistics before and after training.

This provenance identifies affected source revisions. Historical result files and their reported
numbers remain valid records of what was observed. A historical artifact should be attributed to this
exposure only when its producing source revision is known to contain the defect; absence of commit
provenance is not evidence that the artifact is invalid.

## Required corrected behavior

- Frozen controls must leave packed code/pressure, row scales, optional row RMS EMA, leak counters,
  and pending ratchet statistics identical while floating-point support parameters may learn.
- Resume must validate tokenizer kind and canonical serialized subword tokenizer identity before
  changing model, optimizer, RNG, or metrics. Equivalent tokenizer JSON is accepted; a same-size but
  different tokenizer is rejected.
- Version-2 checkpoints save CPU RNG, active CUDA RNG, the selected seed, and per-module leak counters.
  RNG is restored immediately before continuation.
- Training continuation requires an explicit matching selected `run_seed` in both checkpoint
  versions, without an override. Original version-1 checkpoints omitted that provenance and reject
  training continuation even when model operations are deterministic, because the selected seed
  controls the batch schedule. Generation continues to support version 1 and version 2.

A synthetic or migrated version-1 checkpoint with explicit selected-seed provenance is otherwise
conditional on the run's stochastic features. Int8-backward stochastic rounding draws its Triton
seed from CPU RNG, so missing CUDA RNG by itself does not disqualify an otherwise compatible
int8-backward checkpoint. Resume must reject when active stochastic CUDA behavior, such as GPU
dropout, needs CUDA RNG state that the checkpoint does not contain.

## Verification record

Environment: Python 3.12.13, PyTorch 2.12.1+cu130, NVIDIA RTX 3090 selected as GPU 1. The CPU run
executed in a sandbox without GPU access; the GPU run executed with host GPU access. Both used
the GPU-1 selector required by `AGENTS.md` and the same command below.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  MPLCONFIGDIR=/tmp/lat-mpl TRITON_CACHE_DIR=/tmp/lat-triton \
  TORCHINDUCTOR_CACHE_DIR=/tmp/lat-inductor UV_CACHE_DIR=/games/ailab/.uv-cache \
  .venv/bin/pytest -q tests

UV_CACHE_DIR=/games/ailab/.uv-cache .venv/bin/ruff check .
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 UV_CACHE_DIR=/games/ailab/.uv-cache \
  .venv/bin/lat audit --model configs/ratchet_tiny.toml
git diff --check
```

Observed final results:

- CPU: 251 passed, 52 CUDA skips, and 17 warnings in 8.61 seconds. Fourteen warnings were inherited
  Torch JIT deprecations; three were CUDA/NVML initialization warnings caused by the
  inaccessible sandbox device.
- GPU 1: 303 passed with 14 inherited Torch JIT deprecation warnings in 159.63 seconds. This includes
  frozen eager embedding, fused BF16/int8, compiled-update and gradient-checkpointing coverage;
  tokenizer mutation-order checks; exact uninterrupted/resumed dropout and off-boundary pressure
  leakage; CUDA dropout plus int8-backward equivalence; and version-1/version-2 generation behavior.
- Ruff: all checks passed. `git diff --check`: no whitespace errors.
- Master-weight audit: 9 ratchet layers, 401,536 ratchet weights, 411,012 ratchet-state bytes,
  35,840 support-parameter bytes, and zero violations.

The immutable affected-revision GPU baseline passed its original 280 tests with 14 inherited warnings
in 679.61 seconds. That baseline establishes that the old suite was green; the final 303-test GPU
result includes the new integrity regressions and must not be confused with the historical count.
