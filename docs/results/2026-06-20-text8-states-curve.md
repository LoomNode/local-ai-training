# text8 30k: States-vs-Quality Curve (5/7/9) and the Iso-Memory Reframing

## Setup

25M-parameter model on text8 (100M chars), seed 1337, extended to 30,000 steps (fp32/5/7
resumed from their 12k checkpoints, nonary run fresh). Goal: find the ratchet arms' true
plateau and test whether adding states closes the gap to FP32. Best validation loss:

| Arm | bits/wt | best val | bits/char | gap to FP32 | saturation |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | — | 0.9726 | 1.40 | — | — |
| Nonary (9) | 3.17 | 1.1065 | 1.60 | +0.134 | 29.8% |
| Septenary (7) | 2.81 | 1.1434 | 1.65 | +0.171 | 35.9% |
| Quinary (5) | 2.32 | 1.2096 | 1.74 | +0.237 | 46.2% |

(Nonary's job was killed at step 29,400 but had fully plateaued; 1.1065 is its floor.)

## Findings

1. **Monotonic bits->quality dial.** More states -> lower loss AND lower saturation
   (46% -> 36% -> 30%), all the way to 9 states. A mid-run guess that returns were
   diminishing was wrong: nonary pulled away from septenary and its lead *widened* over
   training (+0.026 at 6k -> +0.041 at 24k matched steps).
2. **The gap to FP32 closes with states:** 0.237 -> 0.171 -> 0.134. Per-step gains do taper
   (-0.066 then -0.037 per +2 states), so pure state-scaling likely asymptotes somewhat
   above FP32 — states alone probably will not reach parity.
3. **At iso-parameters, FP32 wins.** All arms are 25M params; FP32 is clearly best. The
   earlier Shakespeare "septenary ties FP32" was an overfitting artifact, confirmed here:
   with enough data not to memorize, FP32 descends past the ratchets.

## The reframing: iso-parameters is the wrong axis

The thesis is *most parameters per GB*, so the comparison should be iso-**memory**, not
iso-**parameters**. Per-parameter persistent training state:

- FP32 + AdamW: ~12 bytes/param (4 master + 8 moment buffers).
- Ratchet (packed): ~0.8-0.9 bytes/param (log2(states) code bits + ~4-bit pressure; scale
  negligible). No optimizer state on the matrices.

So in a fixed memory budget the ratchet fits ~13-15x more parameters. The per-parameter
quality tax measured above (~0.13 nats for nonary) is small; a 13x parameter increase
should buy far more loss reduction than that. Each state tier (5/7/9 bits) is a useful point
on a bits-vs-capacity Pareto frontier — pick the tier that maximizes quality for a given
memory budget.

The decisive untested experiment is therefore the **MB-for-MB** comparison: a small FP32
model vs a much larger ratchet model at equal packed memory. It is runnable eagerly (train
to measure loss, compute memory as if packed) and is set up in
`configs/iso_fp32_*.toml` and `configs/iso_nonary_100m.toml`.

Artifacts under ignored `runs/text8-25m/`.
