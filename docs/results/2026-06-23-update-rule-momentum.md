# Update-rule momentum: temporal EMA of the moments closes ~84% of the master-free gap (screening)

**Date:** 2026-06-23 (run), analysis 2026-06-24
**GPU:** NVIDIA GeForce RTX 3090 (CUDA 1), seed 1337, text8 25M, 5 states.
**Spec:** `docs/superpowers/specs/2026-06-23-update-rule-momentum-design.md`
**Plan:** `docs/superpowers/plans/2026-06-23-update-rule-momentum.md`
**Status:** **Screening result (5k steps, single seed), not converged.** Strongly positive; a 30k
confirmation of the combined arm is the natural next step.

## Question

The QAT de-confounding (`docs/results/2026-06-22-qat-deconfounding.md`) showed master-weight-free
training owns 76–82% of the ratchet's quality gap, and the adaptive-scale screen
(`docs/results/2026-06-23-adaptive-scale-ratchet.md`) showed scale adaptation only rescues the
worst-saturated regime. Both pointed the remaining lever at the **pressure/bucket update rule**.

Cast against Adam, the rule already has a per-row adaptive denominator (the per-row RMS, a `v`-like
term) but **instantaneously**, with no temporal smoothing; and its pressure accumulator (the `m`-like
numerator) is an undecayed sum, also unsmoothed. The hypothesis: **add temporal EMA to both moments —
with no new per-weight state — and recover the master-free gap.**

## Method

Two opt-in knobs, both off by default (baseline bit-identical; equivalence-tested):

- **Arm A — leaky pressure (`pressure_leak_period=K`):** 1st-moment EMA. Every K-th update, bleed each
  nonzero pressure one unit toward zero so stale pressure fades. **Zero new state.**
- **Arm B — EMA per-row RMS (`rms_ema_beta=β`):** 2nd-moment EMA. The normalization denominator
  becomes an EMA of the per-row mean-square (Adam's `v` at row granularity). **Per-row state only**
  (`out_features` FP32 per matrix; audit-clean 1-D, no per-weight state).

Iso-everything at 5 states, 5k steps, seed 1337 (`configs/scaleup_text8_25m_5k.toml`). Baseline is a
**current-code `frozen5`** (re-run, because the current code differs from the QAT-dir runs by benign
float reassociation — bit-identical at step 0, drifting ~0.09 nats by step 5k; the matched-code
baseline is the valid comparison). The `QAT5` arm (old code) is the FP-master ceiling reference.

## Result

Best-so-far validation loss at step 5000 vs the matched `frozen5` baseline (1.4265); QAT5 ceiling
≈ 1.1675 (old code, so % is approximate — the absolute Δ is the solid number):

| arm | knobs | best@5k | Δ vs frozen5 | ~% of gap |
| --- | --- | ---: | ---: | ---: |
| **AB** | leak16 + β0.99 | **1.2100** | **+0.2166** | **~84%** |
| B_beta0p99 | β0.99 | 1.2305 | +0.1960 | ~76% |
| B_beta0p9 | β0.90 | 1.2335 | +0.1930 | ~74% |
| A_leak16 | leak16 | 1.3008 | +0.1258 | ~49% |
| A_leak4 | leak4 | 1.7224 | −0.2959 | −114% |

## Headline

**Temporal EMA of the update rule's moments recovers the master-weight-free penalty, and the two
halves compose.** The 2nd-moment EMA (Arm B) alone closes ~74–76%; the 1st-moment leak (Arm A) at the
right rate closes ~49%; **combined (A+B) ~84%**, landing only ~0.04 nats above the QAT ceiling from a
0.26-nat starting gap. This directly confirms the QAT + adaptive-scale thesis: the master-free cost
was the **update rule**, and it is largely **recoverable** — not a fundamental floor of code-based
training — with no new per-weight state.

`A_leak4` cratering (−114%) is the mechanism check: leaking every 4 steps destroys accumulated
pressure before it can move a code, so learning collapses. The knob does exactly what the model says;
the win at K=16 and the failure at K=4 bracket a real, tunable optimum.

## When the effect appears (screening budget for this class)

Tracing `AB − frozen5` every step: the effect is **positive from step 200 and never crosses back**,
peaking around step ~1200 (the EMA most helps the steep-descent phase) and settling to +0.20 by 5k.
**No transient trap** — unlike the adaptive-scale arms, which showed the *wrong sign* before ~step
3000. So the safe screening floor is **intervention-dependent**: momentum/EMA changes are reliably
callable by **~2k steps**; only saturation/scale-class changes need the full 5k. Future update-rule
sweeps can screen shorter.

## Limitations

- **Screening, single seed (1337), 5k.** Direction is robust (large, monotonic, mechanism-consistent,
  trap-free); the AB arm warrants a 30k confirmation for a converged magnitude.
- **% gap uses the old-code QAT5 ceiling** — treat the percentages as approximate; the absolute Δ vs
  the matched-code `frozen5` (+0.13 to +0.22 nats) is the trustworthy quantity.
- Only a coarse K/β grid was swept (leak {4,16}, β {0.9,0.99}); A_leak4 shows the leak optimum is
  between, and the true best (K, β) is unprobed.
- Arm B adds per-row FP state — cheap and audit-clean, but not literally zero; reported honestly.
