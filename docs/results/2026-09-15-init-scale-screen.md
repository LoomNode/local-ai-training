# Init-scale multiplier screen: a larger frozen scale is a larger step, and every multiplier above 1 is worse — 2026-09-15

**Spec:** `docs/superpowers/specs/2026-09-15-init-scale-screen-design.md`
**Driver:** `scripts/init_scale_screen.py`; runs in `runs/init-scale-screen-2026-09-15/`
**Status:** Screen, 5,000 steps, seed 1337, 15 codes, six cells, one card. Verdict **no advance**; the
headroom hypothesis is rejected in the form tested.

## Outcome

| Scale multiplier k | Best validation | Best step | Saturated at 5k | Codes at 0 at 5k | Move % at 5k |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.1646 | 5000 | 19.8% | 4.9% | 1.01 |
| 2 | 1.6784 | 5000 | 16.9% | 5.9% | 0.94 |
| 3 | 2.3670 | 4000 | 15.9% | 6.7% | 1.26 |
| 4 | 2.3905 | 2800 | 14.9% | 7.6% | 1.34 |
| 6 | 2.4087 | 3800 | 16.4% | 9.0% | 1.04 |
| 8 | 2.4094 | 4800 | 15.3% | 11.3% | 1.40 |

The reference plain cell of the momentum grid is 1.1671; four repeats of that cell (this screen's k = 1
among them) give 1.1664 ± 0.0014 (see `docs/results/2026-09-15-repeatability.md`), so the advance bar is
about 0.005 below 1.166. No cell approaches it. Doubling the scale costs half a nat; from k = 3 on the
model barely leaves the unigram regime (about 2.4 nats) in 5,000 steps.

## Reading

- **The row scale is the ratchet's learning rate.** A code move is one scale unit, so multiplying
  the scale multiplies the step. The model cannot exploit range it can only reach in coarse steps,
  and the flip noise at k ≥ 3 swamps the signal.
- **Rail pile-up is diffusion, not lack of range.** With eight times the range, 15% of codes still
  end saturated after 5,000 steps, versus 20% at k = 1: the codes random-walk outward until they
  meet whatever rail exists. A rule that grows a row's scale when it saturates would follow the walk
  outward and coarsen every step on the way. It is not worth building.
- The checkpoint analysis behind the spec stands: the trained FP32/QAT solution has row scales about
  4× the init scale and 54% of codes at |code| ≤ 1, and it cannot be represented in the ratchet's
  grid. But giving the ratchet that range from the start does not let it find that solution; the
  obstacle is the step, not the box.

## What follows

The same evidence says the ratchet's step is too coarse and too frequent. The knobs for that exist
already: a multiplier below 1 (finer steps, more saturation) and the bucket thresholds (more
consistent gradient per move). They are screened in `runs/step-size-screen-2026-09-15/`.

## Limits

One seed, 5,000 steps, text8 at 25M, 15 codes; screened against a measured run-to-run spread rather
than an identity check (the identity check the driver was written with cannot pass on this backend,
see the repeatability note). Six cells serial on a shared card, about 25 minutes each.
