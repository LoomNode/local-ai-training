# Adaptive per-row scale vs the master-weight-free gap (screening result)

**Date:** 2026-06-23
**GPUs:** 2× NVIDIA GeForce RTX 3090 (CUDA 0 + 1), seed 1337, text8 25M.
**Spec:** `docs/superpowers/specs/2026-06-23-adaptive-scale-ratchet-design.md`
**Plan:** `docs/superpowers/plans/2026-06-23-adaptive-scale-ratchet.md`
**Status:** **Screening result, not converged.** The trainable arms were stopped at 8.8k–9.2k
steps once the effect and its state-gradient were unambiguous (see "Budget"); the 30k baselines
are the existing `runs/text8-25m-qat/` arms. A converged confirmation (re-run `ada5` to 30k) is the
optional follow-up.

## Question

The QAT de-confounding (`docs/results/2026-06-22-qat-deconfounding.md`) showed master-weight-free
training owns 76–82% of the ratchet's quality gap, and a checkpoint analysis traced that penalty to
**frozen-scale saturation**: the ratchet freezes each output row's FP32 scale at init, so 46/35/29%
of codes (at 5/7/9 states) pile up against the `±max_code` rail. QAT, by contrast, recomputes scale
live every forward. Hypothesis: letting the per-row scale **adapt** during training relieves
saturation and closes part of the gap — with the largest effect where saturation is worst (5 states),
tapering toward 9.

## Method

Reuse the existing `trainable_scale=True` path (`--trainable-scale`): each row's scale becomes an
AdamW-trained `log_scale` `nn.Parameter` (log space keeps it positive). This is **audit-clean** — a
1-D per-row scale with optimizer state is permitted persistent state, not a master-weight mirror
(`audit_no_master_weights` flags only floating *matrix* params). No new mechanism; no change to the
pressure/bucket code update. Iso-everything vs the frozen-scale ratchet and QAT arms already in
`runs/text8-25m-qat/`: same config (`scaleup_text8_25m_30k.toml`), data, eval schedule, seed 1337.
Arms written to `runs/text8-25m-adascale/` (existing runs untouched).

A diagnostic confirmed the mechanism is live, not a dead flag: on a controlled run the trainable
scale moved **7.15% in 300 steps** (frozen moved 0.00%), in the optimizer and affecting loss.

## Result

Best-so-far validation loss at **step 5,000** (iso-step; ada arms are past this point):

| states | frozen | ada (trainable) | QAT ceiling | ada − frozen | % of gap to QAT closed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **5** | 1.3302 | **1.2875** | 1.1675 | **−0.0427 (ada better)** | **~26%** |
| 7 | 1.2669 | 1.2664 | 1.1447 | −0.0006 (tied) | ~0% |
| 9 | 1.2049 | 1.2075 | 1.1359 | +0.0026 (ada worse) | ~−4% |

The one **converged (30k)** point we have, 7 states, agrees: `ada7` best 1.1544 vs `frozen7` best
1.1525 — a tie, no gap-closure at 7 states even at full budget.

### When the effect appears

Tracing `ada5 − frozen5` every 200 steps: the durable signal **stabilizes at ~step 3,200** and holds
a steady +0.03 to +0.05 lead through 7,800 with no reversals. Crucially there is a **transient trap at
steps 1,400–3,000 where ada5 looks *worse*** (frozen5 descends faster, then plateaus higher). A screen
at step 2,000 would report the wrong sign. **5,000 steps is the safe screening floor; 2,000 is not.**

## Headline

**Adaptive scale helps only where saturation is worst.** The effect is a clean monotonic gradient —
**+26% gap-closure at 5 states → ~0% at 7 → slightly negative at 9** — tracking saturation (46% → 35%
→ 29%) exactly. Trainable scale relieves the most-saturated regime (5 states) but does nothing at 7+
and never approaches the QAT ceiling.

The mechanism explains the ceiling: trainable scale **rescales** saturated entries but does not
**un-saturate** them — a code pinned at the rail stays pinned; only its magnitude changes. So it helps
when the saturated entries are simply the wrong size (5 states, coarse grid) and not when the problem
is that too many weights need finer resolution than the grid provides (7/9 states). This is consistent
with the QAT finding that the remaining lever is **code resolution / the update rule**, not the scale.

## Limitations

- **Screening, not converged.** Trainable arms stopped at 8.8k (ada5) / 9.2k (ada9); only ada7 ran to
  30k. The 5-state lead is stable for 4,600 consecutive steps but is not a converged magnitude — it
  could compound or wash out by 30k. Re-running `ada5` to 30k on an unloaded machine is the
  confirmation.
- **Single seed (1337), 25M.** Direction is robust (large, monotonic, mechanism-consistent); exact
  magnitudes are one-seed.
- **De-scoped sweep.** The plan's 11/13/15-state tail was dropped once the 5/7/9 gradient was clear —
  the hypothesis predicts the least effect at higher state counts, and the 7/9 null bears that out.
- Trainable scale adds two FP32 AdamW moments per output row per matrix — negligible vs a master
  weight, but not literally zero extra state.
