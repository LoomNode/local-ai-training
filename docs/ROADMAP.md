# Roadmap

This file is the high-level map for the master-weight-free ratchet training work. It is meant to be
easy for future agents to scan before choosing a branch or proposing a new experiment.

Detailed result writeups live in `docs/results/`. Designs and implementation plans live in
`docs/superpowers/specs/` and `docs/superpowers/plans/`.

## Current Position

The project has shown that ratchet matrices can learn without persistent FP32/BF16 master weights.
The active question is no longer "can the update rule learn at all?" It is:

1. How far does the quality scale?
2. Can the low-bit persistent state become a peak-memory or speed win during training?
3. Which hardware mappings make the ratchet representation useful beyond eager PyTorch?

Do not claim this eager implementation accelerates training. The current reliable claim is
master-weight-free persistent state with auditable byte counts.

## Active Gates

### A. Experiment integrity

Baseline/current affected revision `9dcc15f` contains the fused frozen-control exposure introduced
by `d11fe88`: all ratchet linears selected fused backward updates, those updates mutated state during
`loss.backward()`, and the frozen gate ran only afterward. The existing regression test checked a
zero reported move counter, not tensor or counter invariance. Historical runs and measurements remain
records; absent source-revision provenance, do not infer that a particular artifact was affected.

The approved correction freezes packed state, scales, optional EMA, leak counters, and pending stats;
validates canonical tokenizer identity before resume mutation; and writes version-2 reproducible
training checkpoints. Training resume requires explicit selected-seed provenance in either format;
original version-1 checkpoints omitted it and remain generation-only because the seed also selects
the batch schedule. The correction passed the full CPU and RTX 3090 suites on 2026-09-12, including
CUDA BF16/int8 frozen-state coverage and dropout/int8-backward split-resume equivalence. See
`docs/results/2026-09-12-experiment-integrity.md` for commands and exact results.


### B. Ampere int8 training convergence

The Ampere/3090 int8 path is implemented. Further backend work follows the integrity gate above.

Current evidence:

- Hand-written Triton int8 GEMM reached near-peak int8 throughput on RTX 3090.
- End-to-end int8 training kernels are width-gated: promising at frontier widths, not toy widths.
- `matmul_mode = "int8"` exists behind the ratchet linear path.
- Matched-init tests and audit checks passed for the wired path.

Remaining experiment:

- Run matched `bf16` vs `int8` convergence at width `>=4096`.
- Measure both final loss and real tokens/sec.
- Decide whether the roughly 1.4% per-step gradient error washes out or compounds.

References:

- `docs/superpowers/specs/2026-06-21-int8-training-path-design.md`
- `docs/results/2026-06-21-int8-tuned-kernel-reversal.md`
- `docs/results/2026-06-24-int8-per-token-speed.md`

## Completed Evidence

### Subword Embedding Result

Result: `docs/results/2026-06-26-subword-sparse-embedding-ab.md`.

The 25M subword model trained without persistent floating-point matrix masters. It still has
floating-point support parameters such as RMSNorm and row scales. Across seeds 1337 and 1338, the
ratcheted input embedding matched or slightly beat the FP32 embedding while adding no persistent
floating-point embedding matrix.

### Trainability

Historical runs show discrete ratchet matrices learning with no persistent FP master weights.
The stronger attribution to frozen/FP32 controls is being revalidated after discovery of the frozen
fused-update exposure. Preserve the historical results; qualify attribution by source provenance.

### State Count And Quality

Text8 25M runs showed a clean monotonic state-count dial:

- 5/7/9-state codes improve as states increase.
- Gains taper.
- States alone do not close the FP32 gap.
- The implemented state range now spans ternary through 15-state codes (`max_code` 1..7), matching
  the 4-bit nibble cap.

### QAT De-confounding

Result: `docs/results/2026-06-22-qat-deconfounding.md`.

An STE-QAT control kept an FP32 master plus Adam while quantizing to the same few-state grid. This
split the FP32-to-ratchet gap into:

- few-state quantization cost, which was small (`<=0.055` nats);
- master-weight-free update cost, which owned 76-82% of the gap.

Conclusion: the update rule is the quality ceiling, not merely the number of states.

### Adaptive Per-row Scale

Result: `docs/results/2026-06-23-adaptive-scale-ratchet.md`.

Trainable per-row scale closed about 26% of the master-free gap at 5 states, where saturation was
worst. It was null at 7/9 states.

Conclusion: scale adaptation can rescale saturated rows, but it cannot un-saturate rail-pinned
codes. The remaining lever is the update rule / code movement.

### Iso-memory

MB-for-MB comparisons are the right axis for "parameters per GB":

- Ratchet wins at small budgets.
- FP32 wins around the larger tested budget.
- Caveat: large eager ratchets were undertrained at fixed step budgets, and some convergence runs
  stopped early.

### Packed Persistent State

Code and pressure are packed into one `uint8` byte per weight, plus explicit FP32 row scales. This is
lossless for the current integer state and raises the persistent-state reduction from roughly 6x to
roughly 12x versus FP32 matrix + optimizer state.

### Eager Throughput

The eager ratchet path is close to FP32 throughput in the current small models. The earlier "2.3x
slower" result was confounded by GPU contention.

Important boundary:

- update fusion helps the isolated update;
- end-to-end speed needs the matmul path to exploit low-bit state;
- eager FP materialization is still a training-time cost.

### Int8 Speed Reversal And Correction

The old int8 NO-GO was too broad. Vendor int8 kernels underperformed on RTX 3090, but a hand-written
Triton int8 GEMM reached the expected silicon path.

Corrected conclusion:

- Bare int8 GEMM can beat bf16 on Ampere.
- End-to-end integrated int8 training was not faster than dense bf16 at fittable widths.
- The real advantage at width 4096 was memory: dense bf16 OOMed, while the ratchet path trained.
- `int8_backward` converged on par with bf16 over seeds 1337/1338/1339, but was slower per token than
  the plain int8 path and remains default-off.

Reference: `docs/results/2026-06-24-int8-per-token-speed.md`.

### Assistant-scale 1B feasibility

Status: 5k screen reproduced on corrected source (`5567c51`, rewritten to `1c6f3b6` at push, identical tree) with the historical batch-96 config:
final validation 2.368821 versus the historical 2.3648, artifact audit, both generation modes, and
the exact split/resume gate passed. See [provenance reruns](results/2026-09-13-provenance-reruns.md)
and the [rerun tracker](results/2026-09-12-rerun-tracker.md).

Goal: test whether the current subword + checkpoint + generate stack can train a low-billion
master-weight-free model toward a local-assistant-scale base model.

Protocol / Results:

- The 5k-step gate was successful.
- The run successfully fit on an RTX 3090. VRAM was highly optimized by creating `configs/rtx3090_optimized_1b_5k.toml` (`batch_size=96`, `support_learning_rate=0.00075`), allowing it to reach ~7660 tok/s.
- Validation loss descended cleanly to 2.3648 without OOM (2026-09-13 rerun: 2.368821 final, 2.375096 last-four mean).
- Checkpointing, generation, and metrics all functioned correctly at scale.

Next Action: Promote to a 30k+ continuation of the same recipe or a slightly larger 1-2B sibling screen.

Reference: `docs/HANDOFF-assistant-scale-1b.md`.

## Backlog

### 1. Algorithmic Update-rule Improvements

Priority: primary quality direction.

Why:

- QAT de-confounding says master-weight-free update dynamics own most of the quality gap.
- Adaptive scale says scale is not the main remaining ceiling.

First hypothesis:

- Add a momentum or variance analogue to pressure accumulation, still without FP matrix masters.
- Treat this as the master-weight-free counterpart to Adam's first/second moments.

Other possible levers:

- adaptive `pressure_threshold`;
- smarter bucket boundaries;
- reducing blocked-move pressure wind-up at the rails.

Suggested protocol:

- Screen at the 5k-step-equivalent token budget first; effects have flipped sign before
  about step 3000 in older batch-size-fixed runs.
- Only run 30k confirmation after the 5k screen is clear.
- Keep audit clean and report new persistent state bytes separately.

### 2. Scale And Iso-memory Re-run

Why:

- Existing iso-memory evidence is useful but partially confounded by undertraining and stopped runs.
- The ratchet's value should be tested where extra persistent capacity matters.

Suggested protocol:

- Use `compile_update`.
- Give larger ratchet arms enough steps to converge.
- Run on dedicated GPUs to avoid contention.
- Compare at equal persistent memory, equal token budget, and clearly reported wall-clock/tok/s.

### 3. Peak-memory / No-materialization Work

Why:

- A fused tiled backward/update backend exists and avoids a full persistent FP master matrix.
- Peak memory still includes floating-point support tensors, activations, and backend workspaces.

Next target:

- Re-measure the existing fused path after the integrity correction.
- Attribute remaining peak memory before proposing further kernels.

Reference: `docs/superpowers/specs/2026-06-21-fused-backward-memory.md`.

### 4. Blackwell Low-bit Format Probe

Priority: deferred until the Ampere int8 convergence gate is resolved.

Hardware: RTX 5060 Ti 16GB is available.

Why:

- Blackwell may expose tensor-core compute formats such as FP8/FP6/FP4 and block or microscaled
  variants.
- These formats may help reduce or replace the eager FP32/BF16 transient effective-weight path.

Boundary:

- This is not about shrinking ratchet persistent storage. The ratchet already stores code+pressure
  in one byte per weight plus row scales.
- The question is whether Blackwell-native low-bit formats can compute the ratchet effective matrix,
  activations, or gradients with less transient memory or better throughput without hurting
  convergence.

Suggested branch:

- `hardware/blackwell-probe`
- `backend/blackwell-fp4-probe`

First actions:

- Record driver, CUDA, PyTorch, Triton/CUTLASS support, compute capability, and visible low-bit dtype
  support.
- Run narrow microbenchmarks against the 3090 Ampere paths.
- Do not fork the project or create a long-lived Blackwell-only copy.

### 5. BitNet b1.58 Evaluation Review

Why:

- BitNet is a useful external low-bit reference point.
- The existing `src/local_ai_training/bitnet_*.py` harness should be reviewed before using its
  results as evidence.

Goal:

- Produce a clean apples-to-apples baseline for pretrained BitNet b1.58 inference and qualitative
  generation.

### 6. Large-vocab Sparse Embedding Work

Status:

- `RatchetEmbedding` exists and is viable at byte/subword scale.
- The 8K enwik8 subword proof suggests ratcheting the embedding costs nothing at that scale.

Open problem:

- At 30k-50k vocab, each batch touches only a small subset of rows.
- Sparse gradients need row-local update behavior, stale-row handling, and clear observability.

Rules:

- Keep the tokenizer shared.
- Keep the output `lm_head` as a ratchet matrix.
- Report embedding persistent state separately from linears, norms, and optimizer state.

### 7. CPU Launch Overhead

Priority: deferred.

Current evidence:

- The live 25M enwik8 run used about one CPU core and was GPU-bound.
- CPU launch overhead is not the bottleneck at current scale.

Potential levers, in order:

1. Move corpus indexing/batch construction onto GPU.
2. Gate per-step host syncs such as NaN checks and metrics `.item()`.
3. Try `torch.compile(mode="reduce-overhead")` or CUDA graphs only after the step is capture-safe.

### 8. Int4 Accuracy Experiment

Priority: deferred until int8 convergence and backend lessons are clear.

Why:

- The ratchet code is already 4-bit-representable on the weight side.
- Naive int4 activations were too inaccurate in existing probes.

Possible directions:

- int4 weights with int8 activations;
- better int4 activation schemes using per-group scales or outlier handling.

Gate:

- Must improve throughput or memory without breaking convergence versus the int8 baseline.

## Branching Guidance

Use one repo. Avoid architecture forks.

Good branch names:

- `backend/ampere-int8-training`
- `hardware/blackwell-probe`
- `backend/blackwell-fp4-probe`
- `experiment/no-materialization-update`
- `experiment/update-rule-momentum`

Bad branch names:

- `3090-version`
- `5060ti-version`
- `blackwell-fork`
- `ampere-fork`

Split by capability and experiment question, not by GPU ownership. Merge useful backend boundaries
back into the shared project once they are proven.
