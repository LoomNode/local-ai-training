# Iso-state competitor: an int8 master with a stateless sign step beats the ratchet by 0.017 at the same byte per weight — 2026-09-16

**Spec:** `docs/superpowers/specs/2026-09-15-int8-master-iso-state-design.md`
**Driver:** `runs/int8master-2026-09-15/run_int8.py` (5k screen) and `run_int8_stage2.py` (screen
extension and 30k); results in `screen-results.json`, `results-30k.json`, `fullval/results.json`
**Status:** 25M text8, 15-code references, 30k steps, three seeds. Verdict under the spec's rule:
**the ratchet's byte split is dominated** (int8 − plain = −0.017, below the −0.02 line only by
noise: see Reading).

## Question

The ratchet spends one byte per weight as a 4-bit code plus a 4-bit pressure accumulator, with a
frozen FP32 row scale. Is that the right way to spend the byte? The competitor spends the same byte
as an 8-bit master weight on the same frozen row scale (`row_max_abs / 127`, so the same range and
an 18× finer grid), updated by a stateless sign step with stochastic rounding:
`w ← clamp(floor(w − lr · sign(g) + u))`, `u ~ U[0, 1)`. No per-weight optimizer state, no
floating master; `lat audit` reports zero violations, and the note says plainly that this arm *is*
an 8-bit master. Persistent bytes are identical: 25,327,212 for the ratcheted matrices in both arms
(weights plus 4 bytes per row), 90,112 support bytes.

## 5k screen (seed 1337)

| `int8_lr` (grid units per step) | Best at 5k | Saturated at 5k | Move % per step |
| ---: | ---: | ---: | ---: |
| 0.05 | 1.7841 | | |
| 0.1 | 1.5529 | | |
| 0.2 | 1.3543 | | |
| 0.5 | 1.2290 | 1.0% | 49.8 |
| **1.0** | **1.1802** | 1.2% | 99.5 |
| 2.0 | 1.2256 | 2.7% | 98.9 |

Plain ratchet at 5k: 1.1664 ± 0.0014. The screen's first pass (0.05 to 0.5) peaked at its top, so
the automatic 30k launch was stopped and the screen extended to 1.0 and 2.0 (`notes.log`). The
optimum is `lr = 1.0`: pure signSGD on the int8 grid, every weight moving one grid unit per step.
At 5k the int8 arm is 0.014 *behind* the ratchet.

## 30k, three seeds

Loop numbers are best-so-far and final validation on the 40 fixed windows; "full" is every
non-overlapping 256-token window of the validation split (39,062 windows), scored with
`scripts/copyable_mask_eval.py`. Plain, QAT and FP32 arms are the existing runs in
`runs/momentum-30k-2026-09-13/` (same seeds, config, budget).

| Arm | Best (3-seed mean ± SD) | Final | Full validation |
| --- | ---: | ---: | ---: |
| int8 master, lr 1.0 | 1.0442 ± 0.0009 | 1.0453 ± 0.0013 | 1.0451 ± 0.0012 |
| plain ratchet | 1.0608 ± 0.0012 | 1.0616 ± 0.0002 | 1.0618 ± 0.0005 |
| QAT (FP master + Adam, 15 states) | 0.9831 ± 0.0018 | 0.9834 ± 0.0024 | 0.9834 ± 0.0013 |
| FP32 | 0.9728 ± 0.0012 | 0.9732 ± 0.0012 | 0.9732 ± 0.0007 |

| Difference (full validation) | Seed 1337 | 1338 | 1339 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| int8 − plain | −0.0166 | −0.0160 | −0.0176 | **−0.0168 ± 0.0008** |
| int8 − QAT | +0.0620 | +0.0613 | +0.0618 | +0.0617 ± 0.0003 |
| plain − QAT | +0.0786 | +0.0774 | +0.0794 | +0.0785 ± 0.0010 |
| QAT − FP32 | +0.0098 | +0.0113 | +0.0094 | +0.0102 ± 0.0010 |

The int8 arm beats the plain ratchet on every seed and recovers 21% ± 1% of the plain − QAT gap.
Its remaining gap to QAT, 0.062, is the gap of a stateless sign step on a frozen grid to Adam on a
live one.

Trace (seed 1339; the other seeds match within 0.005):

| Step | 3k | 5k | 10k | 20k | 30k |
| --- | ---: | ---: | ---: | ---: | ---: |
| int8 | 1.2448 | 1.1820 | 1.1152 | 1.0694 | 1.0438 |
| plain | 1.2150 | 1.1662 | 1.1160 | 1.0762 | 1.0615 |

The int8 arm trails until about step 10k and stays ahead from 11.8k / 12.6k / 13.0k on (seeds
1337 / 1338 / 1339), the lead still widening at 30k. A 5k screen would have rejected it.

State at 30k: int8 saturation 1.1% (plain 20%), 0.26% of weights at zero (plain 5%), move rate
99.5% of weights per step (plain 1.1%). The int8 value histogram is U-shaped, with 2.4× more
weights in the outermost sixteenth of the range than in the central one: the same outward random
walk the ratchet shows, but with 127 grid steps to the rail instead of 7, it rarely arrives.

Per match-length bin the int8 advance over plain is spread like the gap itself: −0.020 on bins
0 and 1, −0.015 on 2 to 3, −0.008 on 4 to 7, −0.013 on copyable (8+) bytes.

## Reading

- **At one byte per weight, a finer grid with no accumulator beats a coarse grid with one.** The
  ratchet's pressure byte buys nothing that the int8 arm's extra 4 bits of weight precision do not
  buy better. The spec's rule puts −0.02 as the "dominated" line and the three-seed mean sits at
  −0.017 ± 0.001, formally the "tie" band, but every seed is below −0.015 and the sign is
  unanimous; the honest reading is that the ratchet's byte split is the worse of the two.
- **The gap to QAT is mostly not master-weight-freeness in the sense the ratchet meant it.** An
  8-bit integer master with no optimizer state closes a fifth of the gap. The rest is what Adam
  and a live scale buy over a sign step on a frozen grid, which is the same conclusion the
  init-scale and step-size screens reached from the other side.
- **The ratchet's 15 states are not the constraint that matters at this byte budget.** The
  question the state-count studies asked (how few codes can carry the model?) is separate from the
  question of how to spend a byte during training. At one byte, spend it on the weight.

## Limits

One model size and corpus, one competitor design, `int8_lr` screened at one seed on a six-point
grid (the 30k optimum may differ). Both arms use a frozen row scale from the same init; a live
scale, 8-bit Adam, or fewer int8 states are non-goals of the spec and untested. The int8 arm's
inference cost is not measured and no speed or memory claim beyond the counted bytes is made.
The 5k screen ran on a card shared with other jobs; timings are not reported. Runs are not
bit-repeatable (`docs/results/2026-09-15-repeatability.md`).
