# Zero-state update-rule levers at the 5k screen, 15 codes: stochastic bucketing and pressure-weighted effective weights — neither advances — 2026-09-14

**Date:** 2026-09-14
**Spec:** `docs/superpowers/specs/2026-09-14-update-rule-levers-design.md`
**Status:** Screening result, one seed, 5,000 steps. **No cell advances to the 30k confirmation.**
**Predecessor:** `docs/results/2026-09-14-momentum-confirmation.md` (converged residual to QAT
0.073 ± 0.006 nats at 25M / 15 codes; state count is not the bottleneck).

## Outcome

Two opt-in levers that add no persistent state were screened against the momentum grid's
plain and QAT references (same config, seed, and training source, not rerun):

| Cell | Best@5k | Saturated % | Cumulative code moves | vs plain |
| --- | ---: | ---: | ---: | ---: |
| QAT reference | 1.1250 | 0 | 0 | −0.0421 |
| `pw0.5` (pressure weight λ = 0.5) | 1.1652 | 23.5 | 908,494,895 | −0.0019 |
| `stoch-pw0.5` (both, λ = 0.5) | 1.1668 | 22.5 | 1,037,595,221 | −0.0003 |
| plain reference | 1.1671 | 19.9 | 1,091,815,262 | 0 |
| `stoch` (stochastic bucketing) | 1.1735 | 18.9 | 1,181,735,934 | +0.0064 |
| `pw1.0` (λ = 1.0) | 1.2331 | 28.7 | 737,361,751 | +0.0660 |
| `stoch-pw1.0` (both, λ = 1.0) | 1.2346 | 27.0 | 869,220,675 | +0.0675 |

The best cell, λ = 0.5, is 0.0019 below plain, under the 0.005 advancement bar and inside
what one seed resolves; the driver reports `advances: false` and 4.6% of the plain-minus-QAT
gap closed. Neither lever, alone or combined, is a candidate for the 30k confirmation.

## What each lever did

- **Stochastic bucketing** (round the gradient bucket stochastically so sub-threshold
  gradients accumulate pressure in expectation). It does what it was built to do: 8% more
  code moves than plain and a faster start (below plain at every evaluation through step
  2,000, by up to 0.110 at step 600 and 0.064 at step 1,000). It then loses that lead and finishes 0.0064 above
  plain. The extra moves are noise the ratchet cannot use: recovering small gradients in
  expectation trades a dead zone for random walk. The dead-zone hypothesis is not supported
  at this budget.
- **Pressure-weighted effective weight** (the forward pass sees `code + λ · pressure / 8`).
  At λ = 0.5 the trace is plain's within 0.007 at every evaluation from step 2,000, with 17%
  fewer moves and saturation up from 19.9% to 23.5%: the fractional term pushes effective
  weights outward more than it helps them settle. At λ = 1.0 it is harmful from the first
  evaluation (2.42 versus plain 2.32 at step 200; 1.97 versus 1.52 at step 1,000), because
  up to 0.875 of a code step now swings with the pressure counter every update. Combining
  with stochastic bucketing changes nothing at either λ.

Traces (validation loss):

| Step | 200 | 400 | 600 | 1000 | 2000 | 3000 | 4000 | 5000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plain | 2.3151 | 2.0330 | 1.7992 | 1.5234 | 1.2826 | 1.2189 | 1.1867 | 1.1671 |
| stoch | 2.2456 | 1.9563 | 1.6888 | 1.4597 | 1.2759 | 1.2229 | 1.1915 | 1.1735 |
| pw0.5 | 2.3084 | 2.0344 | 1.7915 | 1.4934 | 1.2780 | 1.2163 | 1.1868 | 1.1652 |
| pw1.0 | 2.4215 | 2.3302 | 2.1800 | 1.9749 | 1.5925 | 1.3727 | 1.2757 | 1.2331 |
| stoch-pw0.5 | 2.3175 | 2.0587 | 1.8186 | 1.5054 | 1.2775 | 1.2206 | 1.1891 | 1.1668 |
| stoch-pw1.0 | 2.4175 | 2.3146 | 2.1879 | 1.9684 | 1.5701 | 1.3602 | 1.2783 | 1.2346 |

## Protocol and provenance

- `configs/scaleup_text8_25m_5k.toml` (SHA256 `18f53b47…`, byte-identical to the primary
  checkout's copy the references used), text8 (SHA256 `6e890197…`), seed 1337, 15 codes,
  ratchet weight mode, both momentum knobs off. Cells differ from plain only in
  `--stochastic-bucket` and `--pressure-weight`.
- Source: branch `feat/update-rule-levers`, commit `0119ec1` (levers, plumbing, driver
  `scripts/update_rule_screen.py`) in worktree `.worktrees/update-rule` with its own venv;
  the references ran on `592129d`, whose training package is identical apart from the two
  opt-in levers, which are off in the references and bit-identical off (tested). A later
  commit on this branch threads the two settings through `RatchetEmbedding` (the screen's
  configs do not ratchet the embedding; unaffected).
- `lat audit` with both levers on: zero violations, persistent bytes unchanged (tested).
- Execution: five cells serially on GPU 1 under the thermal guard, sharing the card with the
  gap-ladder study's 99M ratchet arm throughout (about 18k tokens per second versus 85k on a
  free card); all five exit codes 0, no guard events. The 25M model at head size 64 is
  bit-repeatable with the default attention backend, so sharing changed wall-clock only.
  Manifest and `results.json` in `runs/update-rule-screen-2026-09-14/`.

## Limitations

One seed, 5,000 steps, one λ pair, one corpus and model size; a screen that rules the levers
out as gap-closers at this scale, not a converged measurement of their effect. The stochastic
cell's early lead could be tuned (a smaller unit, or annealing the noise), but the momentum
study already showed early leads at this scale do not survive to 30k.

## What this changes

Both levers stay in the code as opt-in, off-by-default settings, retired as gap-closers at
25M / 15 codes. Together with the momentum confirmation, the update-rule levers that add no
per-weight state have now been tried in both moments (leak, EMA denominator), at the bucket
(stochastic rounding), and in the forward pass (fractional pressure), and none moves the
converged residual by more than a few hundredths of its size. The remaining backlog levers
(adaptive pressure threshold, rail wind-up) are of the same family; the honest prior is that
the residual belongs to per-weight information the byte does not hold, which is what the
gap-versus-scale ladder (`runs/gap-ladder-2026-09-14/`) will size.
