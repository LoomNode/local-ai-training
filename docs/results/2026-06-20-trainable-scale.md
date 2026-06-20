# Trainable Per-Row Scale Experiment

## Question

Fixed-scale ratchet weights saturate heavily (quinary ~57.6%, septenary ~47.3% of weights
pinned at a code boundary). Does making the per-row FP32 scale trainable — giving each row a
continuous magnitude knob — relieve that saturation without hurting validation loss?

## Protocol

Opt-in `trainable_scale = true` (`[ratchet]`) stores each row's scale as a log-space
`nn.Parameter` that AdamW trains; the discrete code/pressure ratchet is unchanged. Three
quinary and three septenary arms were trained for 2,000 steps with matched seeds 1337, 1338,
1339, identical to the fixed-scale `configs/ratchet_tiny.toml` runs except for the flag.
`lat audit` reported zero violations for both configs: one scale per row is O(rows), not a
per-weight master copy, so the master-weight-free invariant holds.

Artifacts are under ignored `runs/trainable-scale-quinary/` and
`runs/trainable-scale-septenary/`.

## Results (3-seed means)

| Arm | Final validation loss | Final saturation |
| --- | ---: | ---: |
| Quinary, fixed scale (baseline) | 1.8266 | 57.6% |
| Quinary, trainable scale | 1.8276 | 56.5% |
| Septenary, fixed scale (baseline) | 1.7941 | 47.3% |
| Septenary, trainable scale | 1.7894 | 45.9% |

Per-seed saturation spread was ~0.5 pp (quinary) and ~0.9 pp (septenary).

## Interpretation

The result is a **null**, and it reproduces across both code counts and six seeds.
Validation loss is unchanged (quinary +0.001, septenary -0.005 nats — both negligible), and
saturation drops only ~1.1 pp (quinary) and ~1.4 pp (septenary), at the edge of the per-seed
noise. The scales clearly trained — loss held, codes still moved (~5.1-5.6M cumulative
moves per run) — but the optimizer did not use the added magnitude freedom to pull weights
off the code boundaries.

The conclusion is that saturation here is **intrinsic to the pressure/bucket ratchet update
dynamics, not a dynamic-range limitation**: extra per-row magnitude headroom does not relieve
boundary pinning. Trainable scales therefore add floating-point trainable state for no
practical benefit, so `trainable_scale` should remain off by default. If saturation is worth
attacking later, the lever is the update rule itself (`pressure_threshold`, bucket
thresholds), not the scales.

The cumulative move counts (~5.1-5.6M) are also the first honest full-run totals; the earlier
"~49,849" figure was a 20-sample sum, ~100x lower than the true per-run total.
