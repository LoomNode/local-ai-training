# Int8-master levers: sign-step decay, live row scale, fewer bits — design

**Date:** 2026-09-16
**Status:** Approved (user: "queue all 3"), implementation on `feat/int8-levers`
**Predecessors:** `docs/results/2026-09-16-int8-master-iso-state.md`,
`docs/results/2026-09-15-init-scale-screen.md`, `docs/results/2026-09-16-step-size-screen.md`

## Question

The int8 master with a stateless sign step beats the ratchet by 0.017 at one byte per weight and
sits 0.062 above QAT. Three zero-state levers remain untested on it: (1) the sign step never
anneals; (2) the row scale is frozen at init, the handicap the PTQ probe and the QAT scale
analysis both point at; (3) the byte itself may be more than the update needs. Do any of them
move the int8 arm toward QAT, and how few bits does the stateless update survive on?

## Levers (all opt-in; defaults leave `int8master` bit-identical to the 2026-09-15 runs)

1. **Decay.** `int8_lr_final` (default 0.0 = constant). When > 0, the grid-unit step at step `t`
   of `T` is `lr + (lr_final - lr) * t / T` (linear). Stochastic rounding already realizes
   fractional steps, so no new mechanism. `[int8master] lr_final`, `--int8-lr-final`. The loop
   passes the scheduled lr into `int8_master_update(lr=...)`; resume is exact because the lr is a
   function of the step.
2. **Live row scale (block exponent).** `int8_live_scale` (default off). After each update, per
   row: if more than 1% of the row's weights sit at ±max, double the scale and halve the integers
   with stochastic rounding (`floor(w / 2 + u)`); if the row's max |w| ≤ max / 4, halve the scale
   and double the integers (exact). The effective weight is preserved within one grid unit. The
   sign step is held constant in *effective* units: grid delta = `lr * init_scale / scale`, so a
   grown row takes fractional grid steps and a shrunk row larger ones (the init-scale and step-size
   screens showed the default effective step is near optimal in both directions; the live scale
   must change the range, not the step). One extra persistent FP32 per row (`_init_scale`),
   counted in persistent bytes. `[int8master] live_scale`, `--int8-live-scale`.
3. **Fewer bits.** `int8_bits` in {4, 5, 6, 7, 8} (default 8). `max = 2^(bits-1) - 1`; init scale
   `row_max / max`; clamp and saturation at ±max; the buffer stays int8 but persistent bytes are
   reported as `weights * bits / 8` (logical, stated as such in the note). `[int8master] bits`,
   `--int8-bits`.

Every lever is stateless per weight; `lat audit` must report zero violations in every arm.

## Arms and budget (GPU 0 only; GPU 1 holds the user's llama-server)

- Screen, 5k, seed 1337, 15-code references, lr 1.0 (the 2026-09-15 optimum; reference
  `screen-lr1.0` best 1.1802, 5k bar 0.005): decay `lr_final` ∈ {0.5, 0.25, 0.1}; live scale on;
  bits ∈ {6, 4}. Six cells, serial under the thermal guard.
- 30k stage (the 5k screen is known to mislead for this family: the int8 arm itself only passes
  the ratchet near step 12k): seed 1337 for the best decay cell, the live-scale cell, and their
  combination (three runs); then seeds 1338 and 1339 for the best of the three at 30k. Compared
  with `runs/int8master-2026-09-15/int8-lr1.0-seed*` (plain int8), and the plain / QAT / FP32
  arms in `runs/momentum-30k-2026-09-13/`. Full-validation scoring with
  `scripts/copyable_mask_eval.py`. Bits cells are reported at 5k only (informational: where the
  stateless update breaks).

## Decision rule

Three-seed mean of the best recipe minus plain int8 (full validation): below −0.01, the lever is
real and the recipe becomes the int8 default for future work; the gap to QAT is the new headline.
Otherwise the levers are reported as exhausted and the frozen-grid sign step's 0.062 stands.

## Outputs

`runs/int8-levers-2026-09-16/` (driver, manifest, per-run dirs, `screen-results.json`,
`results-30k.json`, `fullval/`), `docs/results/2026-09-17-int8-levers.md`, roadmap and docs index.

## Non-goals

8-bit Adam, any per-weight state, other model sizes or corpora, throughput.
