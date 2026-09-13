# Closing the master-weight-free gap: momentum grid and converged confirmation

**Date:** 2026-09-13
**Status:** Approved design, pending implementation plan
**Predecessors:** `docs/results/2026-06-22-qat-deconfounding.md`,
`docs/results/2026-06-23-update-rule-momentum.md`,
`docs/results/2026-06-25-projection-oracle-state-count.md`,
`docs/results/2026-09-13-provenance-reruns.md`

## Background

The QAT de-confounding study showed that master-weight-free training, not the low state count,
owns most of the ratchet's quality gap to FP32: at 30k steps each ratchet arm trails its matched
QAT arm (same states, FP32 master + Adam) by about 0.08 to 0.11 nats, and the gap is an asymptote
that more steps do not close.

The update-rule momentum study then added temporal EMA to both moments of the rule, with no new
per-weight state: a leak on accumulated pressure (`pressure_leak_period`, the first-moment
analogue) and an EMA on the per-row RMS denominator (`rms_ema_beta`, the second-moment analogue).
At the 5k screen, one seed, five states, the combined arm (leak 16, beta 0.99) closed about 84% of
the gap, landing roughly 0.04 nats above the QAT ceiling. The study's own conclusion was that a
30k confirmation was the natural next step.

That confirmation was never run. Only two values of each knob were tried, and the study's own
failure case (leak 4 collapsed learning) shows the leak optimum lies between the values probed.
Every later recipe ran with the leak off, and the 2026-09-13 text8 rerun ran with both knobs off.
The default state count has since moved to 15 codes, which has no 30k baseline of any kind.

So the repository's headline gap is the gap of the rule without momentum. This experiment
finishes the job: sweep the two knobs, then confirm the winner at convergence across seeds
against the FP-master ceiling.

## Question

At 15 codes and convergence (30k steps, three seeds), does the momentum update rule bring the
master-weight-free ratchet to within noise of its matched QAT arm? If yes, the master-weight-free
penalty is recoverable with no per-weight state and the repository's headline flips. If no, the
converged residual is the number the next update-rule idea has to beat.

## Constraints

- No new mechanism and no source change to the update rule. Both knobs already exist, are opt-in,
  and leave the baseline bit-identical when off. New code is limited to run drivers and configs.
- No new per-weight state. The EMA buffer is per-row FP32 and audit-clean; `lat audit` must
  report zero violations on every arm, and persistent bytes are reported including the EMA
  buffer (the counter was corrected on 2026-09-13).
- Matched arms share seed, logical initialization, batch schedule, evaluation batches, and token
  budget. Only the arm-defining knobs differ.
- Corrected source only (`948ff62` or later). Fresh runs, no resume, no retries with changed
  settings, failed runs preserved.
- The 25M text8 model at 256 tokens has head size 64 and is bit-repeatable with the default
  attention backend; `deterministic_attention` is not required and is left off.

## Phase 1: knob grid at the 5k screen

- Config: `configs/scaleup_text8_25m_5k.toml`, text8 (existing SHA256-verified download), seed
  1337, 15 codes, ratchet weight mode.
- Grid: `pressure_leak_period` in {8, 12, 16, 24, 32} by `rms_ema_beta` in {0.9, 0.99, 0.999},
  15 arms. Plus two reference arms at the same budget: plain ratchet (both knobs off) and QAT
  (`--weight-mode qat`, 15 codes), 17 runs.
- Metric: best-so-far validation loss at step 5000, plus the step-200 to step-5000 trace so the
  momentum study's "positive from step 200, no transient trap" finding can be checked at 15 codes.
- Selection: the cell with the lowest best@5k. If the top cells are within 0.005 nats, prefer
  the larger leak period (weaker forgetting, closer to the baseline dynamics) and report the tie.
- Budget: about 18 minutes per run on a dedicated RTX 3090; longer while sharing the card with
  the user's other workloads. All 17 runs on one card, serial, under the thermal guard.

## Phase 2: converged confirmation at 30k, three seeds

- Config: `configs/scaleup_text8_25m_30k.toml`, seeds 1337, 1338, 1339, 15 codes.
- Four arms per seed: momentum ratchet (Phase 1 winner), plain ratchet, QAT ceiling, FP32.
  Twelve runs, about 1.75 hours each on a dedicated card; split across both cards with per-card
  queues and the thermal guard, as in the provenance reruns.
- Metrics: best validation loss, final validation loss, last-four-evaluation mean, saturation
  percentage, code-move statistics, persistent state bytes, and the full per-evaluation trace.
- Comparisons, per seed and as a three-seed mean with sample standard deviation: momentum minus
  QAT (the headline), momentum minus plain ratchet (the recovered gap), QAT minus FP32 (the
  few-states cost), and the fraction of the plain-minus-QAT gap recovered.

## Decision criteria

- **Flipped:** three-seed mean momentum-minus-QAT at or below 0.03 nats, and momentum below plain
  ratchet on every seed. Report the master-weight-free penalty as recovered at this scale; the
  fine-tuning pilot becomes the next project.
- **Partial:** momentum beats plain ratchet on every seed but sits more than 0.03 nats above QAT
  at 30k with a flattening tail. Report the converged residual and hand it to the backlog's next
  update-rule levers (adaptive pressure threshold, rail wind-up).
- **Not confirmed:** the 5k advantage shrinks or reverses by 30k on any seed. Report it as a
  screening artifact and record the trace.

Whatever the outcome, no claim about speed, about scales other than 25M text8, or about exact
reproduction of the June numbers (the loop now uses BF16 autocast).

## Outputs

- `runs/momentum-grid-2026-09-13/`: driver, manifest with source commit, config and dataset
  hashes, per-run logs, metrics, checkpoints, `results.json`, and a validation-loss plot.
- `runs/momentum-30k-2026-09-13/`: same layout, per-card queues, thermal-guard status.
- `docs/results/2026-09-13-momentum-grid.md` after Phase 1 and
  `docs/results/2026-09-14-momentum-confirmation.md` after Phase 2, each with the tables above
  and explicit limits. Roadmap and docs index updated from the actual numbers.

## Non-goals

New update-rule mechanisms, changes to defaults, state counts other than 15, other corpora or
model sizes, throughput measurement, and the fine-tuning pilot. Each of those is a separate
design if Phase 2 warrants it.
