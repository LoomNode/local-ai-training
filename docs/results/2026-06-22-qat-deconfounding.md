# De-confounding the ratchet: few states vs master-weight-free

**Date:** 2026-06-23 (runs), design 2026-06-22
**GPUs:** 2× NVIDIA GeForce RTX 3090 (CUDA 0 + 1), seed 1337, text8 25M.
**Spec:** `docs/superpowers/specs/2026-06-22-qat-deconfounding-design.md`
**Plan:** `docs/superpowers/plans/2026-06-22-qat-deconfounding.md`

## Question

Every ratchet quality number to date was FP32-vs-ratchet, which flips **two** switches at
once: (1) **few weight states** (quinary 5 / septenary 7 / nonary 9 levels) and (2)
**master-weight-free training** (no FP latent weight, no optimizer state; codes move by the
pressure/bucket rule). The ~0.13–0.23 nat gap could not be attributed. This adds the missing
control — **STE-QAT**: an arm that quantizes to the *same* few states but **keeps** an FP32
master weight + Adam, updated through a straight-through estimator — to fill the 2×2 and split
the gap.

## Method

`QATLinear` (`src/local_ai_training/qat.py`, `weight_mode="qat"`) uses the ratchet's **exact**
per-row quantizer (`scale = (|W|.amax(dim=1)/max_code).clamp_min(finfo.eps)`,
`code = round(W/scale).clamp(±max_code)`, effective `code*scale`) with a pure straight-through
gradient (round passes identity; no clamp on saturated entries). Shared `kaiming_uniform_(a=√5)`
init means FP32, QAT, and ratchet start from one logical FP init per matrix at each seed; at step
0 QAT's code equals the ratchet's. QAT trains exactly like the FP32 control (AdamW over all
params, no `ratchet_update`).

**Iso-everything**, all seven arms re-run fresh 0→30000 under one current HEAD: same config
(`scaleup_text8_25m_30k.toml` — n_embd 512 / 8L / 8H / block 256 / batch 64, lr 3e-4,
pressure_threshold 8), data, eval (every 200 steps, 40 batches), token budget, seed 1337.

**Why all seven were re-run.** The plan's reproduce-check re-ran ratchet-quinary under current
code and compared to the stored `runs/text8-25m/` trajectory: step 0 was **bit-exact**, but the
trajectory drifted ~1.7e-2 nats by step 200. Investigation traced this to commit `d11fe88`
(fused backward + activation checkpointing): fp32 ratchet layers now route through the tiled
`_RatchetMatmul` backward instead of a monolithic `F.linear`, a **benign float reassociation**
(same algorithm, different accumulation order), not an algorithmic change. Small, but to keep the
2×2 free of a code-version confound, every arm — FP32, ratchet{5,7,9}, QAT{5,7,9} — was produced
by the identical current codebase, into `runs/text8-25m-qat/` (the stored `runs/text8-25m/` arms
are preserved, not overwritten).

## Result

Best validation loss (nats/token) over 30000 steps:

| arm | best val | @step |
| --- | ---: | ---: |
| FP32 | **0.9727** | 30000 |
| QAT-9 | 0.9968 | 29400 |
| QAT-7 | 1.0060 | 30000 |
| QAT-5 | 1.0274 | 29800 |
| ratchet-9 | 1.1060 | 29800 |
| ratchet-7 | 1.1525 | 29800 |
| ratchet-5 | 1.2014 | 29600 |

### The 2×2 decomposition

| states | FP32 | QAT | ratchet | few-states (FP32→QAT) | master-free (QAT→ratchet) | total (FP32→ratchet) | master-free share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.9727 | 1.0274 | 1.2014 | +0.0547 | **+0.1741** | +0.2288 | **76%** |
| 7 | 0.9727 | 1.0060 | 1.1525 | +0.0333 | **+0.1465** | +0.1798 | **81%** |
| 9 | 0.9727 | 0.9968 | 1.1060 | +0.0241 | **+0.1092** | +0.1333 | **82%** |

## Headline attribution

**Master-weight-free training — not few states — owns the gap.** At every state count, QAT (which
keeps masters but quantizes to the same few levels) recovers most of FP32's quality: QAT-9 is only
+0.024 nats behind FP32, QAT-5 only +0.055. But each ratchet arm trails its **matched** QAT by
+0.11 to +0.17 nats — and that master-weight-free penalty is **76–82% of the total FP32→ratchet
gap**, rising with state count.

Two consequences:

1. **Few states is cheap; the update rule is expensive.** Discretizing to 5/7/9 levels with a
   master weight costs ≤0.055 nats. The cost lives almost entirely in dropping the master and
   moving codes by the pressure/bucket rule. This points the remaining quality lever squarely at
   the **update rule**, consistent with the roadmap's open question #3 (the per-bit gains taper;
   the rule, not the state count, is the ceiling). Improving the pressure/bucket rule, not adding
   states, is where FP32-parity would come from.

2. **More states widens the master-free share, not narrows it.** As states increase the few-states
   cost shrinks (+0.055→+0.024) but the master-free cost shrinks more slowly (+0.174→+0.109), so
   its *share* of the gap grows (76%→82%). A higher-resolution code does not buy back the
   master-weight-free penalty.

## Is the master-free cost a slowdown or a floor?

The natural follow-up: master-weight-free training is a *hardware* win (no FP32 master, no
optimizer moments), so the fair question is not "how much worse at iso-steps" but "how many extra
steps does the ratchet need to reach the same loss." We answer this from the existing
`metrics.csv` trajectories — no re-runs. For each loss level **L** both arms pass through, the
step-multiplier is `step_ratchet(L) / step_qat(L)`.

**In the region above the ratchet's floor, it is a bounded slowdown — and it shrinks with state
count:**

| states | ratchet step-multiplier (both arms still descending) |
| ---: | --- |
| 9 | ~1.9× |
| 7 | ~2.8–3.2× |
| 5 | ~4–6× |

**Below its own floor, no step budget closes the gap.** Each ratchet arm asymptotes *above* its
matched QAT, and within 30000 steps never reaches QAT's loss. QAT passed each ratchet arm's *best*
loss in roughly the first 10% of training:

| states | ratchet floor | QAT floor | QAT reaches ratchet floor @step |
| ---: | ---: | ---: | ---: |
| 5 | 1.2014 | 1.0274 | ~2900 |
| 7 | 1.1525 | 1.0060 | ~2600 |
| 9 | 1.1060 | 0.9968 | ~2400 |

The step-multiplier diverges as the ratchet nears its floor (9-state: 1.9× → 2.3× → 3.3× → ∞),
the signature of an **asymptote gap, not a speed gap**.

**Tail slopes confirm more compute will not buy it back.** Over steps 25000–30000 both arms have
flattened, but at every state count QAT's residual descent is *steeper* than the matched ratchet's
(qat9 −0.0020 vs ratchet9 −0.0012 nats/1k; qat7 −0.0021 vs ratchet7 −0.0007; qat5 −0.0023 vs
ratchet5 −0.0013). The ratchet has plateaued harder while QAT is still pulling away, so the gap is
stable-to-widening. A back-of-envelope on the best case (9-state, 0.109-nat gap, ratchet tail
slope ~0.0012/1k) puts closing the gap on the ratchet's own steam at ~90000 further steps — 3× the
run — and that assumes a flat slope while QAT, descending faster, keeps receding. The gap never
closes.

**Conclusion: the master-free penalty is a structural quality floor, not a compute-tradeable
slowdown.** The ~2×–5× step-multiplier applies only to the easy losses above the floor; below it,
the master-weight-free update rule converges to a worse asymptote. The only lever that moved the
floor here was adding states (1.20 → 1.15 → 1.11), not adding steps — reinforcing that the
pressure/bucket update rule, not the step budget, is the ceiling.

## Limitations

- **Single seed (1337), 25M/30k.** A trainability/attribution test, not a converged-scale claim.
  The master-free penalty is large (0.11–0.17) relative to plausible seed noise, so the direction
  is robust; the exact magnitudes are one-seed. A multi-seed repeat is the natural follow-up,
  especially if anyone wants confidence intervals on the 76–82% split.
- **3090-specific runtime only** affects how the arms were produced (the `d11fe88` reassociation),
  not the quality conclusion.
- QAT is an absmax/BitNet-style STE control matched to the ratchet's quantizer; it is not a tuned
  SOTA QAT recipe. It is the right *control* (isolates the master weight), not a quality ceiling
  claim for QAT itself.
- The iso-quality tail analysis reads the 25000–30000 slopes as "both plateaued, QAT no slower,"
  which is robust to the per-row noise; the ~90000-step extrapolation to close the gap is
  illustrative arithmetic, not a guarantee. The qualitative finding (gap stable-to-widening, not a
  waitable slowdown) is what the data supports.
