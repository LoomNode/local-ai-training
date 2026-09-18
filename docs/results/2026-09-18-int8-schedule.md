# Schedule search at an iso 30k budget: annealing the sign step to zero is the best master-weight-free recipe so far — 2026-09-18

**Driver:** `runs/int8-schedule-2026-09-17/run_schedule.py`; results in `stage1-results.json`
(and `results-30k.json` once the last seed lands)
**Predecessor:** `docs/results/2026-09-17-int8-levers.md`
**Status:** Complete. Stage 1 (five schedules, seed 1337, 30k), stage 2 (three seeds for the
winner) and the full-validation scoring are all in. The lab was paused by agreement immediately
after; the follow-up studies named in `docs/ROADMAP.md` are not started.

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

Loop best-so-far, all three reaching their best at step 30,000 exactly:

| Seed | linear → 0 | linear → 0.25 | constant step |
| --- | ---: | ---: | ---: |
| 1337 | 1.0166 | 1.0253 | 1.0440 |
| 1338 | 1.0193 | 1.0283 | 1.0451 |
| 1339 | 1.0187 | 1.0271 | 1.0437 |

Full validation split (39,062 windows), the comparison every headline in this repository uses:

| Arm, 30k, three seeds | Full validation | Persistent bytes |
| --- | ---: | ---: |
| plain ratchet | 1.0618 ± 0.0005 | 25,327,212 |
| int8, constant step | 1.0451 ± 0.0012 | 25,327,212 |
| int8, linear → 0.25 | 1.0275 ± 0.0010 | 25,327,212 |
| **int8, linear → 0** | **1.0181 ± 0.0010** | 25,327,212 |
| QAT (FP master + Adam, 15 states) | 0.9834 ± 0.0013 | — |
| FP32 | 0.9732 ± 0.0007 | — |

| Difference (full validation) | 1337 | 1338 | 1339 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| linear → 0 − linear → 0.25 | −0.0093 | −0.0097 | −0.0093 | **−0.0095 ± 0.0002** |
| linear → 0 − constant step | −0.0285 | −0.0279 | −0.0246 | −0.0270 ± 0.0021 |
| linear → 0 − plain ratchet | −0.0451 | −0.0439 | −0.0423 | −0.0438 ± 0.0014 |
| linear → 0 − QAT | +0.0335 | +0.0335 | +0.0371 | **+0.0347 ± 0.0021** |
| linear → 0 − FP32 | +0.0433 | +0.0448 | +0.0465 | +0.0449 ± 0.0016 |

**The master-weight-free penalty at one byte per weight is 0.035 nats, down from the ratchet's
0.078: 56% ± 2% of the gap recovered**, by a stateless integer update with no optimizer state and
no floating master. The advance over the ratchet holds in every match-length bin, and the residual
to QAT is spread the same way (bin 0 +0.019, bin 1 +0.039, bins 2-3 +0.039, bins 4-7 +0.022,
copyable 8+ bytes +0.022). State at 30k is unchanged from every other int8 arm: 1.02% saturated,
0.33% at zero, 25,327,212 persistent bytes.

## Limits

One model size and corpus (25M text8, 15-code references), one seed per stage-1 cell, two seeds so
far for the winner. The schedule is a fraction of the step budget, so nothing here transfers to a
different budget without re-screening. Runs are not bit-repeatable
(`docs/results/2026-09-15-repeatability.md`). No speed or memory claim beyond the counted bytes.
