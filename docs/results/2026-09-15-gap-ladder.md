# Gap versus scale: the master-weight-free penalty grows from 7M to 99M while the few-states cost vanishes — 2026-09-15

**Date:** 2026-09-14 to 2026-09-15
**Spec:** `docs/superpowers/specs/2026-09-14-gap-ladder-design.md`
**Driver:** `scripts/gap_ladder.py`; runs in `runs/gap-ladder-2026-09-14/` (7M and 99M rungs),
25M rung reused from `runs/momentum-30k-2026-09-13/`
**Status:** Converged, 30,000 steps, 15 codes, matched arms. Three seeds at 7M and 25M, two at
99M. Verdict **growing**.

## Outcome

The plain ratchet's gap to its matched QAT arm (same 15 states, FP32 master + AdamW) widens with
model size, and the cost of 15 states with a master shrinks to nothing:

| Rung | Ratchet weights | Seeds | plain − QAT | QAT − FP32 | plain − FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7M | 7,381,440 | 3 | 0.0527 ± 0.0081 | 0.0222 ± 0.0029 | 0.0749 ± 0.0052 |
| 25M | 25,179,648 | 3 | 0.0778 ± 0.0020 | 0.0103 ± 0.0007 | 0.0881 ± 0.0013 |
| 99M | 99,111,168 | 2 | 0.0845 ± 0.0044 | −0.0010 ± 0.0014 | 0.0834 ± 0.0030 |

Best validation loss in nats per character, mean ± sample SD over seeds. The 7M-to-99M change in
plain − QAT is +0.032, above the spec's 0.02 threshold, so the driver's verdict is "growing". The
total plain − FP32 penalty is nearly flat (0.075 → 0.088 → 0.083) because its two parts move in
opposite directions: at 7M nearly a third of it is the state count, at 99M none of it is.

Two readings follow. First, the state count is never the bottleneck at these sizes, and by 99M
fifteen states with a live scale match FP32 outright. Second, the master-weight-free update rule
is the whole penalty at 99M and the penalty does not fade with scale; it is largest where the
memory saving would be largest. Combined with the copyable-mask split
(`docs/results/2026-09-14-copyable-mask.md`) and the checkpoint analysis behind the init-scale
screen (`docs/superpowers/specs/2026-09-15-init-scale-screen-design.md`), the working hypothesis
is representational headroom: the ratchet freezes each row's scale at init while the trained
solution's row scales grow about four-fold.

## Per-seed results

| Rung | Seed | Plain ratchet | QAT | FP32 |
| --- | ---: | ---: | ---: | ---: |
| 7M | 1337 | 1.1163 | 1.0576 | 1.0376 |
| 7M | 1338 | 1.1170 | 1.0610 | 1.0400 |
| 7M | 1339 | 1.1096 | 1.0661 | 1.0406 |
| 25M | 1337 | 1.0595 | 0.9827 | 0.9722 |
| 25M | 1338 | 1.0615 | 0.9849 | 0.9741 |
| 25M | 1339 | 1.0615 | 0.9815 | 0.9720 |
| 99M | 1337 | 1.0223 | 0.9410 | 0.9410 |
| 99M | 1338 | 1.0311 | 0.9435 | 0.9455 |

The ratchet is the noisiest trainer at every rung: its seed spread is 0.007 at 7M and 0.009 at
99M against 0.005 or less for FP32. Every arm's best is within the last 2,000 steps except 99M
FP32 seed 1337 (step 28,000), so no arm was still descending steeply at 30k, but the 99M rung is
undertrained by any absolute standard: FP32 gains only 0.030 nats from four times the parameters
at this budget, with no learning-rate schedule. The rung compares matched arms at a fixed budget,
not converged models.

## Protocol and provenance

- Configs: `configs/ladder_text8_7m_30k.toml` (6 layers, 8 heads, width 320) and
  `configs/ladder_text8_99m_30k.toml` (14 layers, 12 heads, width 768) are copies of
  `configs/scaleup_text8_25m_30k.toml` (8 layers, 8 heads, width 512) with only the geometry
  changed: batch 64, 256-character windows, 30,000 steps, 491,520,000 characters per arm, support
  learning rate 0.0003, 15 codes, no RMS EMA, no leak. text8 SHA256 `6e890197…`.
- Source commit `c8daed7` for the launch; the `--continue` relaunches below ran commit `2a5e183`,
  which changes only the driver's planning code, never the training loop. `lat audit` reports zero
  violations on both rung configs (its weight count uses the config's placeholder vocabulary, so
  the table above quotes the counts the runs themselves logged).
- Arms per seed: plain ratchet, QAT, FP32, sharing seed, logical init, batch schedule, evaluation
  batches, and token budget. The 25M rung is the momentum confirmation's plain, QAT, and FP32
  arms, unchanged.
- The 99M rung ran two seeds, not three. The design called for three; the first two ratchet arms
  ran at about 1,900 steps per hour on shared cards (15 and 18 hours each), so the third seed was
  dropped before it started to keep the study inside a day. Seed 1339 at 99M has no runs.
- Execution: one queue per card under `scripts/thermal_guard.py`, both cards shared throughout
  with flybrain-lab training jobs and, on GPU 1, the user's llama-server (about 21 GB) and a
  50-minute scoring job of ours. The 25M model is bit-repeatable on this backend (head size 64,
  256-token windows); the 99M model shares that head size and window, so sharing changed its
  wall-clock only. The 7M model has head size 40 and its repeatability was not checked, so
  sharing could in principle have perturbed its numbers as well as its wall-clock.
- Wall-clock per run (CDT): 99M ratchet 15.2 and 18.0 h; 99M QAT 4.2 and 5.3 h; 99M FP32 5.1 and
  4.0 h; 7M ratchet 0.8 to 1.6 h; 7M QAT and FP32 0.3 to 0.5 h. Total elapsed from launch to
  the last run about 29 hours.
- Queue surgery, all recorded in `runs/gap-ladder-2026-09-14/relaunch.log`: the trim to two 99M
  seeds was applied by relaunching each card's queue at an arm boundary with a pinned plan
  (`--continue --rung-seeds 99m=1337,1338`); two later rebalances spread the 7M tail over both
  cards, each time leaving the card's live run to finish on its own as an orphaned process. Three
  incidents left no trace in any result: a preview of the plan without `--dry-run` moved the live
  seed-1337 ratchet directory aside for 71 seconds (moved back, manifest entry removed, process
  unaffected); a queue child that had just started a run it was about to lose truncated that
  run's stdout log on the other card (the run's `metrics.csv` and checkpoint are complete; the
  first 22,400 steps of console output for 99M QAT seed 1337 are gone); and a stray empty log for
  the dropped 99M seed 1339 was created in the seconds before that child was stopped. Runs that
  finished as orphans have no EXIT line in the queue logs; completeness is judged from
  `metrics.csv`.
- Guard: no thermal stop, no OOM, no restart. `results.json` and `comparison.png` are in the run
  directory.

## What this does and does not establish

- Establishes: at text8, 256-character windows, 30k steps, the ratchet-versus-QAT gap grows from
  about 0.05 at 7M to about 0.08 at 99M, and the QAT-versus-FP32 gap shrinks from 0.02 to zero.
- Does not establish: behaviour beyond 99M or at longer contexts, anything about converged 99M
  models, a throughput claim, or three-seed statistics at 99M.
