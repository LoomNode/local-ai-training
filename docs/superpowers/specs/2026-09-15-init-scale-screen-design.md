# Init-scale multiplier screen — design

**Date:** 2026-09-15
**Status:** approved (user: "queue the init-scale screen after the ladder finishes")
**Predecessors:** `docs/results/2026-09-14-momentum-confirmation.md`,
`docs/results/2026-09-14-copyable-mask.md`, `docs/results/2026-06-23-adaptive-scale-ratchet.md`

## Background

Checkpoint analysis of the 30k momentum-study arms (seed 1337) shows the trained FP32/QAT solution
does not fit the ratchet's grid: QAT and FP32 end with 3% of codes at magnitude ≥5 and 54% at ≤1 when
quantized by the live row_max/7 rule, with row scales 4.4× (median; 2.7–10.7×) their init values,
while the frozen-scale ratchet ends with 57% at ≥5, 20% saturated, and half the matrix RMS. The FP32
solution post-training-quantized to 15 states loses 0.02 nats with a live scale, 1.57 nats inside the
ratchet's frozen init grid, and 0.44 / 0.17 nats at 4× / 8× that grid. Hypothesis: the ratchet's
0.08-nat gap to QAT is mostly representational headroom, not update-rule information.

## Question

At the 5k screen (`configs/scaleup_text8_25m_5k.toml`, seed 1337, 15 codes), does starting every
row's scale at a multiple k of the current rule (`row_max_abs / max_code`) lower best-so-far
validation loss versus k = 1? Which k, and what does it do to saturation and the code histogram?

## Mechanism

One opt-in knob, `scale_multiplier` (float, default 1.0), in `DiscreteRatchetLinear` and
`RatchetEmbedding`: at construction, `scale = row_max * scale_multiplier / max_code` and the codes
are re-quantized against it (`round(reference / scale)`, clamped). Bit-identical at 1.0. Nothing
else changes: same packed uint8 state, same update rule, same audit. Threaded exactly like
`stochastic_bucket`: `ModelConfig`, `ExperimentConfig` (`[ratchet] scale_multiplier`), CLI
`--scale-multiplier`, checkpoint resume defaults, generation loader.

## Arms

k ∈ {1, 2, 3, 4, 6, 8}, six cells, all `--weight-mode ratchet --codes 15 --seed 1337`, 5k config.
References: the momentum grid's `plain` (1.1671) and `qat` (1.1250) cells at the same config, read
from `runs/momentum-grid-2026-09-13/results.json`. The k = 1 cell must reproduce the reference plain
cell's best-so-far exactly (the 25M/256-token model is bit-repeatable); the summary reports this
identity check and fails loudly if it does not hold.

## Metrics

Per cell: best-so-far validation loss and step, final validation loss, the step-200…5000 trace, and
from the last metrics row: `saturated_percent`, `zero_percent`, `move_percent`, and the code
histogram tails (share at |code| ≥ 5 and ≤ 1). Summary: winner, best − plain, fraction of the
plain − QAT gap closed, and the trace so an early-transient sign flip is visible.

## Decision rule

Advances if the best cell beats plain by more than 0.005 nats at 5k (same margin as the lever
screen). If it advances, the next step is a 30k three-seed confirmation of the winning k against the
existing `runs/momentum-30k-2026-09-13/` baselines, and the rail-driven scale-growth rule becomes
the follow-up mechanism. If no cell advances, headroom alone is not the answer and the scale-growth
rule is tested directly.

## Execution

`scripts/init_scale_screen.py`, modeled on `scripts/update_rule_screen.py`, with `--gpus 0,1` and a
`--queue-gpu N` child mode (round-robin via `momentum_grid.assign_queues`), each card serial under
the thermal guard. Output `runs/init-scale-screen-2026-09-15/`. Launched after the gap ladder
finishes (a watcher waits for the ladder's queue children to exit). About 18 minutes per cell on a
dedicated card.

## Non-goals

Trainable or growing scales (separate design), other seeds or budgets, other state counts, the
embedding arm, throughput.
