# Schedule search at an iso 30k budget: annealing the sign step to zero is the best master-weight-free recipe so far — 2026-09-18

**Driver:** `runs/int8-schedule-2026-09-17/run_schedule.py`; results in `stage1-results.json`
(and `results-30k.json` once the last seed lands)
**Predecessor:** `docs/results/2026-09-17-int8-levers.md`
**Status:** **IN PROGRESS, paused by agreement.** Stage 1 (five schedules, seed 1337, 30k) is
complete. Stage 2 has seeds 1337 and 1338 for the winner; **seed 1339 was still running when the
lab was paused on 2026-09-18** and the full-validation scoring has not been run. Every number below
is the training loop's best-so-far validation loss on the 40 fixed eval windows, not the
full-split score used for headline comparisons elsewhere. See "Finishing this" at the end.

## Question

The levers study annealed the int8 sign step from 1.0 to 0.25 and gained 0.018 at 30k, with all
three seeds still improving at the final step and the endpoint picked off a 5k screen that had
already mis-ranked this family twice. This searches endpoint and shape at the **same 30k budget**.
No budget extension: 30k on 25M text8 is already about 5.5 epochs, the FP32/QAT baselines overfit
past that (`docs/results/2026-09-15-gap-ladder.md`), and a longer run would shrink the gap by
degrading the baselines rather than improving the recipe.

## Stage 1: five schedules, seed 1337, 30k

| Schedule (grid units, from lr 1.0) | Best | Move % at 30k |
| --- | ---: | ---: |
| linear → 0.25 (previous champion) | 1.0253 | 24.9 |
| linear → 0.1 | 1.0198 | 10.0 |
| linear → 0.05 | 1.0199 | 5.0 |
| **linear → 0** | **1.0166** | 0.0 |
| cosine → 0 | 1.0182 | 0.0 |

- **The endpoint matters up to a point, then stops mattering.** 0.25 → 0.1 is worth 0.0055;
  0.1 → 0.05 is worth nothing (0.0001, far inside the run-to-run spread).
- **Reaching zero is worth more than the last halving of the endpoint.** 0.05 → 0 is worth 0.0033,
  and only the zero endpoint reaches a 0% move rate. A stochastically-rounded sign step never stops
  jittering while any step remains, so the weights are always displaced from their own optimum by
  rounding noise; driving the step to zero is what lets them settle. This is the same mechanism the
  bit-width result pointed at from the other side (grid spacing sets the noise floor).
- **Shape matters less than arrival.** Cosine led linear the whole way (−0.008 at both 20k and 25k)
  and lost 0.0016 by 30k: its flat tail spends the last few thousand steps at a step barely above
  zero instead of at zero.
- The schedule knob that makes "→ 0" expressible did not exist before this study: `lr_final = 0`
  meant "constant" (`--int8-lr-schedule`, commit `e6a3b6c`). The best cell here was previously
  unreachable.

## Stage 2: the winner across seeds (incomplete)

| Seed | linear → 0 | linear → 0.25 | constant step |
| --- | ---: | ---: | ---: |
| 1337 | 1.0166 | 1.0253 | 1.0440 |
| 1338 | 1.0193 | 1.0283 | 1.0451 |
| 1339 | *in flight when paused* | 1.0271 | 1.0437 |

Two-seed mean 1.0180. Both seeds gain 0.009 on the previous champion and about 0.026 on the
constant step, and both reach their best at step 30,000 exactly. State at 30k is unchanged from
every other int8 arm: 25,327,212 persistent bytes, 1.02% saturated, 0.33% at zero.

For orientation only, and **not** a headline number until the full-split scoring runs: the levers
study's loop-best and full-split figures differed by 0.0006, so a two-seed loop mean of 1.0180
implies roughly 1.019 on the full split, which would put the gap to QAT (0.9834) near 0.035 and the
recovered fraction of the plain-ratchet gap near 55%. Treat as an estimate, not a result.

## Finishing this

1. Let `runs/int8-schedule-2026-09-17/lin0.0-seed1339` finish (or rerun it: the driver's `cmd()`
   builds the exact command; the run is `--weight-mode int8master --int8-lr 1.0
   --int8-lr-schedule linear --int8-lr-final 0.0` on `configs/scaleup_text8_25m_30k.toml`).
2. Score all three on the full validation split, as every other arm was scored:
   `PYTHONPATH=src .venv/bin/python scripts/copyable_mask_eval.py --root runs/int8-schedule-2026-09-17
   --output runs/int8-schedule-2026-09-17/fullval --arms lin0.0 --seeds 1337 1338 1339 --device cuda:0`
3. Replace the estimate above with the measured three-seed mean and SD, the differences against
   the constant step, the plain ratchet, QAT and FP32, and the per-bin split; update the roadmap.

## Limits

One model size and corpus (25M text8, 15-code references), one seed per stage-1 cell, two seeds so
far for the winner. The schedule is a fraction of the step budget, so nothing here transfers to a
different budget without re-screening. Runs are not bit-repeatable
(`docs/results/2026-09-15-repeatability.md`). No speed or memory claim beyond the counted bytes.
