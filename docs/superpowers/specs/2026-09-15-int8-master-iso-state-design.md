# Iso-state competitor: int8 master weight with stochastic rounding and a stateless sign update — design

**Date:** 2026-09-15
**Status:** approved (user: "queue up what gives us more info to go on")
**Predecessors:** `docs/results/2026-09-14-momentum-confirmation.md`, `docs/results/2026-09-15-gap-ladder.md`,
the init-scale screen (`runs/init-scale-screen-2026-09-15`, negative: larger scale = larger step = worse)

## Question

The ratchet spends one byte of persistent training state per weight as a 4-bit code plus a 4-bit
pressure accumulator, with a frozen per-row FP32 scale. Is that the right use of the byte? The
simplest alternative with the same budget is an 8-bit master weight, per-row scale frozen at init,
updated by a stateless sign rule with stochastic rounding. Same bytes during training, same bytes
at inference, no optimizer moments, no FP master. If it matches or beats the ratchet at 25M / text8 /
30k, the code-plus-pressure split is dominated by a simpler scheme; if the ratchet wins, the 0.08-nat
gap to QAT is the price of the byte budget rather than of the ratchet's design.

## Mechanism (`Int8MasterLinear`, new module `src/local_ai_training/int8_master.py`)

- Persistent state per matrix: `weight_int8` (int8 buffer, out × in) and `_scale` (FP32 per output
  row). Initialization from the same seeded kaiming-uniform reference every other arm uses:
  `scale = row_max_abs / 127`, `weight_int8 = round(reference / scale)` clamped to [-127, 127]. The
  represented range (±row_max) is therefore identical to the ratchet's at init; the grid is 18×
  finer. Nothing recomputes the scale later (frozen, as in the ratchet).
- Forward: `effective = weight_int8.float() * scale[:, None]` as a temporary FP32 tensor that
  requires grad; `F.linear(x, effective)` under the training loop's autocast. The module keeps the
  temporary until the update consumes its `.grad`, then releases it (the repository's rule for
  temporary FP effective weights).
- Update (called where the loop calls the ratchet update, once per step, after `backward`):
  `delta = -int8_lr * sign(grad)` in grid units, `weight_int8 = clamp(floor(weight_int8 + delta + u), -127, 127)`
  with `u ~ Uniform[0, 1)` drawn from the run-seeded generator on the weight's device (stochastic
  rounding; expectation equals the fractional step). `int8_lr` is a float in grid units per step
  (default 0.1); no per-weight state of any kind. Blocked moves at ±127 are counted like the
  ratchet's. Gradients are zeroed/released after the update.
- Stats per step, matching the ratchet's columns so `metrics.csv` compares: `code_moves` (weights
  whose int8 value changed), `move_percent`, `blocked_*_moves`, `saturated_percent` (|w| = 127),
  `zero_percent`, and the value histogram in coarse bins. `persistent_state_bytes` = weights + 4 ×
  rows.
- Audit: `audit_no_master_weights` must report zero violations (the int8 buffer is not a floating
  Parameter); the note will state plainly that this arm *is* an 8-bit master, which is the point.

## Wiring

`weight_mode = "int8master"`: `ModelConfig.int8_master: bool`, `_linear` dispatch (every matrix the
ratchet arm ratchets, including `lm_head`), `train_run` calls the int8 update on every step in this
mode (and `discard_pending_gradients` semantics for the frozen path are not needed), `ExperimentConfig`
gains `[int8master] lr` (`int8_lr`), CLI `--weight-mode int8master --int8-lr`, checkpoint resume
defaults, `load_for_generation` rebuilds the module kind from the saved weight_mode, README one
paragraph. Support parameters (embedding, norms) train with AdamW exactly as in every other arm.

## Arms and budget

1. 5k screen, seed 1337, 15-code ratchet references from `runs/repeatability-2026-09-15` (plain@5k
   mean 1.1664, sd 0.0014): `int8_lr` ∈ {0.05, 0.1, 0.2, 0.5}. Four runs, about 18 minutes each.
2. 30k, seeds 1337/1338/1339, `configs/scaleup_text8_25m_30k.toml`, the screen's best `int8_lr`,
   compared with the existing plain ratchet, QAT, and FP32 arms in `runs/momentum-30k-2026-09-13/`
   (same seeds, config, budget). Final scoring on the full validation split with
   `scripts/copyable_mask_eval.py`'s scorer in addition to the loop's best-so-far.

## Decision rule

int8-master minus plain-ratchet, three-seed mean: below −0.02 the ratchet's byte split is dominated;
above +0.02 the ratchet's accumulator earns its half byte; between, a tie, and the byte budget itself
is the cost. Whatever the outcome, no claim about speed or memory beyond the counted bytes.

## Non-goals

Live scales, optimizer moments in fewer bits, 8-bit Adam, other state counts, other corpora.
