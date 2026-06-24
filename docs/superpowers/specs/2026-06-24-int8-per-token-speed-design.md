# Per-token speed phase for master-weight-free int8 training

**Date:** 2026-06-24
**Status:** Design — approved, pending implementation plan

## Thesis

You quantize anyway. So if master-weight-free int8 ratchet training is also *faster per
token* than the conventional path (train bf16, then post-training quantize), it is strictly
better on time-to-a-quantized-model: cheaper steps **and** no separate quantization pass,
because the codes already *are* the quantized weights.

Now that learning is proven — the momentum update rule closes ~84% of the master-free gap
(`docs/results/2026-06-23-update-rule-momentum.md`) and int8 converges on par with bf16
(`docs/results/2026-06-21-int8-convergence-25m.md`, +0.0077 nats at 5k) — the eager
proof-of-concept can be turned into a per-token speed result. This phase measures and, where a
cheap bottleneck blocks it, removes that bottleneck.

## Win condition

**Headline claim:** an int8-ratchet training step is lower wall-clock than a *plain bf16 dense*
training step (the bf16-then-PTQ workflow's training cost) at the largest width where both fit
on one RTX 3090, with convergence held equal.

Two honesty constraints fixed by earlier work this cycle:

- **Baseline is plain bf16 *dense*, not the bf16 ratchet.** The throughput harness's current
  `bf16` mode is a *ratchet* (`max_code=2`, materializes an FP effective weight + runs the
  update). That is the wrong, easier opponent. The real bf16-then-PTQ baseline is a normal
  dense bf16 model (`max_code=None`) with bf16 weights + fp32 Adam state — no ratchet overhead,
  fast matmul.
- **Sustained throughput, never per-step-synced.** The stale `2026-06-24` throughput table
  used the per-step-`synchronize()` method (measures isolated latency, over-penalizes
  kernel-heavy int8). The corrected sustained method (one sync at each end of the timed block,
  committed `2613410`) is mandatory. Re-measurement under it already flipped width-512 int8
  from "42.7k tok/s, 2.6x slower than bf16" to "110.5k tok/s, ~1.09x faster than the bf16
  ratchet."

## Convergence ("held equal")

Reuse the existing width-512 parity result (+0.0077 nats int8 vs bf16 at 5k, inside the
predeclared ≤0.03 "Tracks" gate). The int8 per-element gradient noise (~1.4%/step) is a
precision effect that does not materially worsen with width, so width-512 parity is taken to
carry to frontier width. A frontier-width convergence re-confirmation is an optional, cheap
follow-on — explicitly out of scope here.

## Memory reframes the frontier

Plain bf16 dense stores ~10 B/param (bf16 weights 2 B + fp32 Adam m,v 8 B); the int8 ratchet
stores ~1–2 B/param with no optimizer state on the matrices. So bf16-dense OOMs *earlier* than
the ratchet. The claim therefore splits by width:

- **Where bf16-dense fits (expected up to ~2048):** head-to-head **per-token** comparison —
  the wall-clock crossover.
- **Where bf16-dense OOMs (expected ~4096):** not a per-token race — a **memory win**: the
  ratchet trains a model bf16-dense physically cannot. Reported as a distinct, stronger claim.

No external/"online" throughput numbers are used: different hardware (A100/H100) and
model/framework/batch make them incomparable for a wall-clock-per-token claim on a 3090. If a
reference ceiling is wanted where bf16-dense will not fit, a **bf16 roofline** (matmul FLOPs ÷
the 3090's measured bf16 tensor-core throughput) may be cited, clearly labelled theoretical —
not a measured opponent.

## Measurement protocol

- **Modes per width:** `fp32_dense` (reference), `bf16_dense` (the baseline; `max_code=None`,
  bf16 autocast), `int8_ratchet` (the arm). `bf16_ratchet` may be retained for continuity with
  the old table. The plain `bf16_dense` mode is a new addition to
  `scripts/int8_training_throughput.py` (today only `fp32` runs `max_code=None`).
- **Method:** sustained tok/s (one sync each end of the timed block), warmup excludes Triton
  autotune/compile, fresh process per mode for clean autotuning.
- **Widths:** 512 / 1024 / 2048 / 4096, batch scaled to fit (e.g. 64 / 32 / 16 / 8).
- **Isolation:** each measured point runs on one **uncontended** GPU. The second 3090 may run a
  *different* (width, mode) point in parallel, never the same one. No tensor parallel — it
  injects inter-GPU comms into the measured quantity and is a separate "scale out" story.
- **Recorded per (width × mode):** sustained tok/s, ms/step, peak MB, and OOM where it occurs.
- **Derived:** the **crossover width** where `int8_ratchet` ms/step ≤ `bf16_dense` ms/step.

## Phases

### Phase 1 — Harness prep + honest baseline (also surfaces the CPU-bound red flag)
- Add the `bf16_dense` mode to `scripts/int8_training_throughput.py`.
- Run the sustained sweep at a starter pair of widths (512, 2048) on the *current* code to
  record the baseline state and confirm/deny the CPU-bound (sync-serialized) hypothesis.
- Discover the `bf16_dense` OOM boundary across widths.

### Phase 2 — Remove the CPU bottleneck first, verify the GEMM
The eager ratchet is launch-bound: per step it reads 6+ stats to CPU via `.item()`
(`ratchet.py` ~487–492, 518, 549–552), serializing the GPU. Measuring a launch-bound
implementation is not an honest per-token-*compute* comparison, so this is done **before** the
headline measurement, with before/after numbers to prove it.
- Accumulate move/blocked/RMS stats as on-GPU tensors during the step; sync to CPU (`.item()`)
  only when a metrics row is actually written (eval cadence), not every step.
- **TDD:** equivalence test — reported stats are identical between per-step-sync and
  cadence-sync over a multi-step run.
- **Verify** the in-model autotuned int8 GEMM (`scaled_int8_mm`, already a hand-written
  `@triton.autotune` `tl.dot` kernel — not a vendor `_int_mm`) hits a healthy % of int8 peak at
  frontier width; only act if it is stalling.
- Re-measure the starter widths: report "CPU-bound X tok/s → sync-free Y tok/s."

### Phase 3 — Headline sweep + crossover
- Full sustained sweep 512→4096, all modes, with the sync-free code.
- Report the per-token crossover width (`int8_ratchet` ≤ `bf16_dense`) and the OOM frontier
  (where `bf16_dense` dies and the ratchet trains on) as the memory claim.
- Reuse the width-512 convergence parity for the quality half.

### Phase 4 — Record
- New `docs/results/2026-06-24-int8-per-token-speed.md` leading with the corrected current
  numbers (dense baseline, sustained method).
- Mark the stale `docs/results/2026-06-24-int8-training-throughput-final.md` table superseded
  (lead-with-current-data convention; do not delete the old numbers, append/super­sede).
- Correct `docs/ROADMAP.md:33` and `docs/README.md:43`: replace "speedup only manifests at
  width ≥2048–4096 / bf16 owns width 512" with the measured crossover against the **dense**
  baseline under the **sustained** method.

## Success criterion

A defensible one-liner backed by the sweep and the reused parity:

> "At width ≥ X on one RTX 3090, master-weight-free int8 training is faster per token than plain
> bf16 — and since the codes are already the quantized model, it is strictly faster to a
> quantized model than bf16-then-PTQ. Beyond width Y, plain bf16 dense will not fit at all,
> while the ratchet trains on."

with X (crossover) and Y (bf16-dense OOM) reported from measurement.

## Guardrails (invariants)

- `audit_no_master_weights` stays clean — no FP/BF16 Parameter mirroring a code matrix; the
  sync-removal change touches only metric accounting, not stored state.
- Never claim packed sub-byte storage; the int8 path stores int8 codes, not 2.32/2.81-bit.
- Do not overwrite existing `runs/` (convergence runs, the six tiny-shakespeare arms). Write
  benchmark artifacts under git-ignored `runs/int8-throughput`.
- Every reported number names its **baseline kind** (dense vs ratchet) and **method**
  (sustained vs per-step-synced). Distinguish per-token claims from memory claims.
- 3090-specific; note the GPU/UUID. A100/H100/FP8 differ.

## Out of scope (YAGNI)

- Tensor parallel / multi-GPU sharding.
- Frontier-width convergence re-confirmation (reuse width-512 parity).
- Packed sub-byte storage kernels.
- Activation-quant fusion beyond what already exists — only pursued if Phase 2 measurement
  shows it is the binding constraint after the syncs are gone.
- int4 (its own accuracy experiment).
