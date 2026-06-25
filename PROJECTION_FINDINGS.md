# Projection-oracle diagnostic — result (2026-06-25, 512/5k, text8, seed 1337)

| measurement              | val loss |
|--------------------------|----------|
| FP32 (ceiling)           | 1.1166   |
| Projection / PTQ (floor) | 1.3285   |
| Ratchet-trained          | 1.4398   |

Representation cost (FP32->PTQ): +0.212 nats (irreducible by update rule; only states/scale lower it)
Optimization gap (PTQ->ratchet): +0.111 nats (update-rule levers' ceiling; ratchet WORSE than PTQ)

## Verdict
- Remaining FP32 gap is ~2/3 representation, ~1/3 optimization.
- Update-rule work (stochastic bucketing / Lion) addresses ONLY the ~0.11-nat optimization third.
- Master-free training currently LOSES to PTQ at equal bits (1.44 vs 1.33) -> closing the
  optimization gap means reaching PARITY with PTQ, not beating it.
- CAVEAT: 5k screen; ratchet converges slower than FP32 -> 0.11 is an upper bound at matched
  short budget. A 30k converged re-run (both FP32 PTQ floor and ratchet) would firm the split.

## Implication for #2 (update rule)
Pursue update-rule levers only if ~0.11 nats (at this scale) is worth it. The bigger lever for the
FP32 gap is representation (more states / better-fit scale), which is a different experiment.

## Note
Built outside the brainstorm->spec->plan flow on user's "just try it". project_to_codes + 3 tests
green; scripts/projection_oracle.py. FP32 checkpoint serializes max_code=0 (not None) -- loader
needs expected_max_code=0.

## STATE-COUNT SWEEP (2026-06-25, 512/5k, text8, seed 1337) — full result
codes max_code  FP32   PTQ-floor  ratchet  rep_cost  opt_gap
   3      1     1.117    2.142     1.665    +1.025   -0.477   <- ratchet CRUSHES PTQ (ternary)
   5      2     1.117    1.329     1.440    +0.212   +0.111
   7      3     1.117    1.179     1.232    +0.063   +0.053
   9      4     1.117    1.145     1.210    +0.028   +0.065
  11      5     1.117    1.135     1.184    +0.018   +0.050
  13      6     1.117    1.129     1.168    +0.012   +0.039
  15      7     1.117    1.125     1.166    +0.008   +0.041

### Findings (these REFRAME the quinary-only read)
1. Representation floor collapses with states: rep_cost 1.025 -> 0.008. By max_code>=4, PTQ of
   FP32 is ~FP32 quality. Adding states solves representation.
2. At high state counts the residual gap is ~ALL optimization: codes15 total gap 0.049 = 0.008
   representation + 0.041 optimization (84% opt). INVERTS quinary (which was ~2/3 representation).
   => update-rule levers (stochastic bucketing / Lion) now have a VALIDATED target (~0.04 nats).
3. Master-free training BEATS PTQ decisively at ternary (codes3: 1.665 vs 2.142, -0.477) but
   loses modestly at codes>=5. Thesis "master-free beats train-then-quantize" is TRUE at extreme
   low-bit, FALSE at moderate bit. Low-bit/ternary = BitNet b1.58 territory = the strongest thread.

### Strategic read for "go from there"
- Ternary/low-bit is where master-free is a clear win -> connect to BitNet b1.58 (ROADMAP stream).
- Quinary+ : master-free's value is MEMORY (train where FP master won't fit), not quality.
- #2 update-rule work: validated ~0.04-nat target at high states; chase if worthwhile.
CAVEAT: 5k screen, ratchet undertrained vs FP32; opt_gap may shrink at 30k convergence.

## ISO-MEMORY SWEEP (2026-06-25, ~7.3MB matched, 512-ish/5k, text8, seed 1337) — FULL RANGE 3..15
arm        codes  bits  params  val(5k)   Δ from prev
ternary      3    1.58  37.4M   1.9088    --
quinary      5    2.32  25.2M   1.3241    -0.585
septenary    7    2.81  20.7M   1.2422    -0.082   <- end of steep descent (elbow 1)
nonary       9    3.17  18.6M   1.2071    -0.035
codes11     11    3.46  16.1M   1.1835    -0.024   <- plateau entry (elbow 2) = chosen default
codes13     13    3.70  15.3M   1.1775    -0.006   <- noise floor
codes15     15    3.91  14.6M   1.1753    -0.002

### Finding: at equal STORAGE, more bits-per-weight wins MONOTONICALLY through 15 — no turnover.
The extra params low-bit buys do NOT pay for the crude resolution -> ternary is the WORST use of a
memory budget despite most params (opposite of the "low-bit = more capability/GB" hypothesis). The
curve has two elbows: codes 7 ends the steep descent; codes 11 enters the plateau (11->13 is only
-0.006, 13->15 -0.002 = noise). CRUCIAL: codes 7/9/11/13/15 all pack into the SAME 4-bit nibble
(nibble cap = max_code 7 = 15 states), so between them more states is FREE — identical storage and
compute. That makes 11 the value-optimal default: plateau entry, captures essentially all
achievable quality (-0.064 below 7) at zero storage cost. 7 is only the iso-param quality-per-bit
*knee* (where rep_cost +0.063 ~ opt_gap +0.053, the rep->opt crossover) — it never won on absolute
trained val on EITHER axis (iso-param ratchet val also drops monotonically 1.232->1.166 to codes 15).
CAVEAT: ternary 37M is most undertrained at 5k, but the 0.7-nat gap is far beyond undertraining
(iso-param ternary-25M was already 1.665). Ranking robust.

### Decision: codes 11 = new default (cli.py --codes 5->7->11). Plateau entry; free quality over 7/9
at the same 4-bit nibble; 13/15 add only noise. codes 7 retained as the conservative/BitNet-adjacent
knee. Ternary's value is narrow: beats PTQ + true 1.58-bit IF you specifically need that bit-width,
but NOT the params/GB winner.
