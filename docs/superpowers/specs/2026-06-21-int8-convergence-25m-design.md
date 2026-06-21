# Int8 Convergence 25M Design

**Date:** 2026-06-21
**Status:** Approved for specification

## Goal

Determine whether the numerical noise introduced by int8 forward and backward matmuls
compounds over 12,000 training steps into worse validation loss, or washes out relative to a
matched BF16 control.

This is an accuracy gate, not a speed experiment. The existing 25M text8 model has linear
contraction widths of approximately 512–2048, below the measured K≈4096 int8 speed crossover.
Int8 is expected to provide no speedup and may be slower because quantization overhead does not
amortize at these widths. Throughput is diagnostic context only and is not a success criterion.

## Experimental Arms

Run exactly two ratchet arms:

- BF16 control: `matmul_mode = "bf16"`
- Int8 treatment: `matmul_mode = "int8"`

Do not run a new FP32 arm. Existing text8 FP32 evidence provides absolute-quality context, but
the comparison of record is BF16 versus int8 because both produce BF16 linear outputs and differ
only in operand quantization.

Create two checked-in TOML files derived from `configs/scaleup_text8_25m.toml`. They retain:

- block size 256;
- 8 layers, 8 heads, embedding width 512, and zero dropout;
- batch size 64;
- 12,000 steps;
- evaluation every 200 steps over 40 fixed batches;
- support learning rate 0.0003;
- pressure threshold 8 and bucket thresholds 0.5/1.5;
- seed 1337;
- automatic CUDA device selection.

The TOMLs differ in exactly one field: `matmul_mode`. Explicit configs make the experimental
condition reviewable and preserve provenance with the artifacts.

## Matched-Arm Preflight

Before training, run a deterministic preflight that fails unless all conditions hold:

1. Parsed config dictionaries differ only at `matmul_mode`.
2. Seed 1337 produces identical packed codes, row scales, floating-point support parameters,
   and positional state in both models.
3. Train and validation batch schedules are identical.
4. Isolated BF16 and int8 matmul calls do not consume or change PyTorch CPU or CUDA RNG state.

The int8 implementation uses deterministic rounding rather than stochastic rounding. Dropout
is zero. Therefore the only permitted source of arm divergence is the numerical result of the
linear matmul. Any discovered RNG consumption is a confound to fix before starting either run.

## Execution Protocol

Inspect `nvidia-smi` immediately before execution and choose one idle RTX 3090. Run the two arms
sequentially on that same GPU to avoid contention and device-to-device variance. Record the GPU
identity, relevant process state, commit SHA, and exact commands in the results note.

Use the pinned text8 corpus already associated with the 25M experiment. Write artifacts under:

```text
runs/int8-convergence-25m/bf16/
runs/int8-convergence-25m/int8/
```

The run root is ignored. Never write to or repeat the completed arms under
`runs/tiny-shakespeare`.

Both modes are CUDA-only and must fail before creating artifacts if CUDA is unavailable. Preserve
failed or divergent runs and their exact configurations. Do not change scales, rounding,
precision placement, seed, schedules, or token budget after observing results.

## Metrics And Analysis

Treat each arm's `metrics.csv` as the source of record. At every shared evaluation step report:

- BF16 validation loss;
- int8 validation loss;
- signed gap `int8 - bf16`;
- final validation-loss gap;
- best validation loss and its step for each arm;
- whether the gap is stable, narrows, or widens over training;
- cumulative code moves, move percentage, saturation, gradient RMS, and throughput as
  diagnostics.

Do not add per-step reference-gradient instrumentation. Faithful measurement would add BF16
reference matmuls, synchronization, and memory pressure that could perturb the experiment.
Existing code-move, saturation, and gradient-RMS metrics provide low-cost diagnostic context;
validation-loss trajectories are the deliverable.

Generate a comparison plot under `runs/int8-convergence-25m/`. The plot and tabulated metrics
must use matched evaluation steps without interpolation.

## Predeclared Decision Gate

Classify the result using thresholds fixed before inspecting either run:

- **Tracks:** final validation-loss gap ≤0.03 nats, no sustained late-training widening, and no
  instability or non-finite loss.
- **Marginal:** final gap >0.03 and ≤0.08 nats, or a visibly widening late-training curve.
- **Fails accuracy gate:** final gap >0.08 nats, clear divergence, or instability.

The 0.03-nat threshold is calibrated to an observed state-count effect in this repository:
nonary versus septenary text8 best validation differed by approximately 0.036 nats. A gap above
0.08 nats is therefore materially larger than that established effect size.

Do not revise these thresholds after seeing results. Report exact values and trajectories even
when they complicate the classification. Marginal or failing results stop escalation to a wide
run. Do not silently tune them away with per-tensor scales, stochastic rounding, selective BF16,
seed changes, or schedule changes; those belong to a separate follow-up design.

A result classified as Tracks permits, but does not itself execute, task 3: a separate
frontier-width experiment combining convergence and the measured speed regime.

## Reporting

Write `docs/results/2026-06-21-int8-convergence-25m.md` containing:

- hypothesis and predeclared gate;
- commit, hardware, corpus, commands, and artifact paths;
- proof that config/init/schedules/RNG preflight passed;
- matched-step loss table and comparison plot reference;
- final and best losses, gap trajectory, and classification;
- code-move, saturation, gradient-RMS, and throughput diagnostics;
- any interruption, failure, or divergence without discarded evidence;
- limitations and follow-up boundary.

State explicitly that throughput at toy width is not evidence for or against int8 speed at
frontier width.

## Limitations

- **Single seed:** one matched seed establishes a first deterministic gate, not a population-level
  estimate or statistical equivalence.
- **Horizon:** 12,000 steps is thousands of updates and adequate for a first compounding-noise
  test, but cannot rule out degradation that emerges only at substantially longer horizons.
- **Width:** toy-width gradient-noise behavior may not predict frontier width. Wider contraction
  dimensions average more terms and can change relative quantization noise.
- **Hardware:** execution and throughput observations are specific to an RTX 3090.
- **Necessary, not sufficient:** passing this gate is required before a frontier-width claim but
  does not establish frontier convergence, speed, or cross-hardware behavior.

## Scope Boundary

This specification ends after the matched BF16/int8 runs are analyzed and the results note is
committed. It does not implement quantization remedies and does not launch a wide model. Task 3,
if justified, requires its own brainstorm, specification, and plan.
