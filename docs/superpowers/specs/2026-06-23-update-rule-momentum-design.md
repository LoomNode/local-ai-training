# Update-rule momentum: temporal EMA of both moments (master-weight-free)

**Date:** 2026-06-23
**Status:** Approved design, pending implementation plan
**Predecessors:** `docs/results/2026-06-22-qat-deconfounding.md`,
`docs/results/2026-06-23-adaptive-scale-ratchet.md`

## Background

Two independent results pinned the ratchet's remaining quality gap on the **pressure/bucket update
rule**, not the state count, the scale, or the bit width:

- **QAT de-confounding:** an STE-QAT control (FP32 master + Adam, same few states) recovers most of
  FP32's quality; master-weight-free training owns **76–82%** of the FP32→ratchet gap.
- **Adaptive per-row scale:** a trainable scale closes only ~26% of the gap at 5 states and nothing
  at 7/9 — it rescales saturated codes but cannot un-saturate them. The lever is the update rule.

So the question this experiment tests: **does adding Adam-style smoothing to the update rule — with
no new per-weight state — close a meaningful part of the master-free gap?**

## The gap in the current rule

The current rule (`_ratchet_update_core`, `ratchet.py`) per step:

1. Gradient w.r.t. the effective weight is RMS-normalized **per output row**
   (`rms = grad.square().mean(dim=1, keepdim=True).sqrt()`; `normalized = grad / (rms + eps)`).
2. `bucket_pressure` maps each `normalized` value to an integer increment in {−2,−1,0,+1,+2}
   (thresholds 0.5 / 1.5), signed downhill.
3. `pressure += increment` (an **undecayed** running sum, a signed 4-bit integer in the high nibble).
4. When `|pressure| ≥ pressure_threshold` (default 8), `code` moves one step (unless at the
   `±max_code` rail — a blocked move), and `threshold` is subtracted from pressure.

Cast against Adam, the rule **already has** a per-row adaptive denominator (the per-row RMS, a `v`-like
scaling) — but **instantaneously**, recomputed each step with no temporal smoothing. And the pressure
accumulator (the `m`-like numerator) is an undecayed sum, also with no smoothing. **What it lacks is
temporal EMA of both moments.** This experiment adds exactly that, at per-row-or-cheaper cost.

## Non-negotiable constraint: no new per-weight state

Adding even one byte per weight doubles the ratchet's persistent training footprint (1 → 2
bytes/weight) and erodes the memory thesis that justifies the approach. **No new per-weight state is
permitted.** New state may only be per-row (`out_features` scalars per matrix — same order as the
existing FP32 scale, negligible vs the weight matrix) or none. `audit_no_master_weights` must stay
violation-free; per-row 1-D buffers are permitted (the audit flags only floating *matrix* params,
ndim ≥ 2).

## Arm A — leaky pressure (temporal EMA of the 1st moment)

Replace the undecayed `pressure += increment` with a **leaky integrator**: pressure decays toward
zero over time so recent gradient direction dominates and stale pressure fades — the master-weight-free
analogue of Adam's first-moment EMA.

- **Integer-friendly decay.** Pressure is a ~4-bit signed integer (range ~[−7, 8]); multiplicative
  decay is too coarse. Instead **bleed one unit toward zero every `K` steps**:
  every `K`-th step, `pressure -= sign(pressure)` (only for `pressure != 0`), applied before the
  increment. `K` is the decay knob (the "β" analogue): large `K` → weak forgetting (≈ current rule),
  small `K` → strong forgetting. Swept as the arm's hyperparameter.
- **State cost: zero** — modifies the dynamics of the existing pressure nibble only.
- The leak applies to the persistent pressure; the move/threshold logic is unchanged.

## Arm B — EMA per-row RMS (temporal EMA of the 2nd moment)

Replace the instantaneous per-row RMS denominator with an **EMA** of the per-row mean-square — exactly
Adam's `v`, at row granularity:

- Maintain a persistent per-row buffer `rms_ema` (`out_features` FP32 per matrix). Each step:
  `ms = grad.square().mean(dim=1)`; `rms_ema = β·rms_ema + (1−β)·ms`; `normalized = grad / (sqrt(rms_ema) + eps)`.
- `β` is the smoothing knob (e.g. 0.9 / 0.99), swept. At `β = 0` this reduces exactly to the current
  rule (a regression check).
- **State cost: per-row only** — `out_features` FP32 per matrix, audit-clean (1-D), initialized on the
  first step from that step's `ms` (so step 0 matches the current rule).

A and B compose: **A+B** is the full per-row Adam analogue (temporal EMA of both moments), still
master-weight-free.

## Wiring

Both arms are opt-in behaviour of `DiscreteRatchetLinear` / `_ratchet_update_core`, selected by config
+ CLI so the baseline path is bit-unchanged when off. Proposed surface (final names pinned in the
plan): ratchet config gains `pressure_leak_period` (A; `0`/`None` = off) and `rms_ema_beta`
(B; `0` = off, current behaviour). The existing `compile_update`/fused-backward paths must keep working
(the leak and EMA live inside the elementwise update core / its normalization).

## Experiment (screening)

- **States:** 5 (where the master-free gap and adaptive-scale signal both live); optionally 7 as a
  second point.
- **Arms:** current-code `frozen5` baseline (re-run for code-version consistency), `A` (leaky), `B`
  (EMA-RMS), `A+B`. Existing `QAT5` (1.1675 best-so-far @5k; 1.0274 @30k) is the FP-master ceiling.
- **Budget:** 5k steps (the validated screening floor — effects can show the wrong sign before
  ~step 3000), seed 1337, `scaleup_text8_25m_5k.toml`, iso-everything otherwise.
- **Hyperparameters:** small sweep of `K` (A) and `β` (B) — a handful of values each, screened at 5k.
- **Output:** `runs/text8-25m-updaterule/` (existing runs untouched).
- **Success:** any arm closes a meaningful fraction of the `frozen5 → QAT5` gap at 5k, measured as
  best-so-far val loss at step 5000 (iso-step). A positive screen graduates to a 30k confirmation;
  a null is itself informative (the rule's smoothing is not the lever).

## Invariants & guardrails

- `lat audit` violation-free for every arm (per-row 1-D buffers permitted; no FP/bf16 matrix
  Parameter mirroring a code matrix; still no per-weight FP state).
- Baseline path bit-unchanged when both knobs are off (`pressure_leak_period` off, `rms_ema_beta=0`)
  — covered by an equivalence test.
- HF dataset revision stays pinned; existing runs preserved (new output dir).
- Single seed (1337), 25M, 5k screen: a trainability/attribution test, not a converged-scale claim.

## Limitations

- Integer leak (A) is a coarse EMA; the 4-bit pressure range bounds the achievable smoothing. If A is
  null but the hypothesis still seems alive, a finer pressure representation would be a *separate*
  (memory-costing) follow-up, explicitly out of scope here.
- B adds per-row FP state — cheap and audit-clean, but not literally zero; reported honestly.
- Screening only; magnitudes are one-seed until a 30k confirmation of any positive arm.
