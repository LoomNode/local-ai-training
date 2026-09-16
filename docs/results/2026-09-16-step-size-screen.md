# Step-size and bucket screen: finer steps and rarer moves are both worse at 5k — 2026-09-16

**Driver:** `runs/step-size-screen-2026-09-15/run_screen.py` (configs under `configs/`, results in
`results.json`, manifest with source commit `7d70231`)
**Status:** Screen, 5,000 steps, seed 1337, 15 codes, six cells serial on one shared card. Verdict
**no advance**; the plain ratchet's default step and bucket settings are the best of everything tried.

## Question

The init-scale screen showed the frozen row scale is the ratchet's step size and that any multiplier
above 1 is worse. This screen asks the other direction: does a *finer* step (multiplier below 1) or a
*more consistent* move (higher bucket thresholds, so fewer and better-supported code moves) get the
ratchet closer to its QAT ceiling?

## Cells

Every cell is the default 25M text8 screening config with one or two `[ratchet]` knobs changed.
`k` is `scale_multiplier`; `b<low>-<high>` sets `bucket_low` and `bucket_high` (defaults 0.5 / 1.5).

| Cell | Best validation | Best step | Saturated at 5k | Codes at 0 at 5k | Move % at 5k |
| --- | ---: | ---: | ---: | ---: | ---: |
| plain (reference, 4 repeats) | 1.1664 ± 0.0014 | 5000 | 19.8% | 4.9% | 1.01 |
| k0.5 | 1.2298 | 5000 | 24.0% | 4.3% | 0.82 |
| k0.7 | 1.1871 | 5000 | 21.3% | 4.6% | 0.93 |
| b1-2 | 1.1702 | 5000 | 16.2% | 6.1% | 0.42 |
| b1-3 | 1.1736 | 5000 | 15.5% | 6.4% | 0.29 |
| b1.5-3 | 1.2136 | 5000 | 13.5% | 6.8% | 0.13 |
| k0.7-b1-2 | 1.1985 | 5000 | 19.5% | 5.2% | 0.36 |

Reference statistics are the k = 1 cell of the init-scale screen; the mean and spread are from the
four plain repeats in `docs/results/2026-09-15-repeatability.md`. The advance bar is 0.005 below
1.1664. No cell reaches 1.1664, let alone the bar. The best cell, b1-2, is 0.004 worse than plain.

## Reading

- **Finer steps lose more than they gain.** Halving the step (k0.5) costs 0.063 nats at 5k and
  raises saturation from 20% to 24%: the grid gets finer but the box gets smaller, and more rows hit
  the rails. k0.7 is a milder version of the same trade (−0.021).
- **Rarer moves do not mean better moves.** Raising the bucket thresholds cuts the move rate from
  1.0% to 0.13% per step and does reduce rail pile-up (13.5% saturated at b1.5-3), but the loss gets
  worse in step with the move rate. The ratchet is not moving too often; the moves it makes are
  about as good as the bucketed gradient allows.
- **The two levers do not combine.** k0.7-b1-2 is worse than either knob alone.
- Together with the init-scale screen, this brackets the default in both directions: scale 0.5 and
  0.7 are worse, scale 2 and above are worse, and every bucket setting tried is worse. The gap to QAT
  is not an untuned step size.

## What follows

The ratchet's own knobs are exhausted at this budget. The remaining comparison is the iso-state
int8-master arm (`docs/superpowers/specs/2026-09-15-int8-master-iso-state-design.md`): the same
byte per weight, but an 8-bit master with a stateless sign step instead of a 4-bit code plus 4-bit
pressure. That result is in `docs/results/2026-09-16-int8-master-iso-state.md`.

## Limits

One seed, 5,000 steps, text8 at 25M, 15 codes. Cells ran serially on a card shared with other
workloads (about 25 to 60 minutes each); throughput is not reported. Screened against the measured
run-to-run spread, not an identity check.
