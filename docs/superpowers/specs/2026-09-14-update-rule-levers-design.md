# Zero-state update-rule levers: stochastic bucketing and pressure-weighted effective weights

**Date:** 2026-09-14
**Status:** Approved design (user: "line up the next tests"); implemented on branch
`feat/update-rule-levers` in worktree `.worktrees/update-rule`
**Predecessor:** `docs/results/2026-09-14-momentum-confirmation.md`: at 25M/15 codes/30k the plain
ratchet sits 0.077 nats above matched QAT, QAT is within 0.010 of FP32, and temporal EMA of the
moments recovers 6%. The state count is not the bottleneck; the information the update rule
discards is.

## Hypothesis

The ratchet discards two things Adam keeps. (1) Sub-bucket gradient magnitude: a normalized
gradient below 0.5 contributes nothing, ever, so weights with persistently small gradients form
a dead zone. (2) Sub-code position: a weight that has accumulated 7 of 8 units of pressure
toward its next code is, to the forward pass, identical to one at 0. Both can be recovered with
no new persistent state.

- **Lever 1, stochastic bucketing (`stochastic_bucket`, bool, default off).** Replace the
  deterministic round-to-nearest bucket with stochastic rounding of the same quantity. With
  `unit = bucket_high - bucket_low`, the increment magnitude is
  `min(2, floor(|z| / unit + u))` with `u ~ U[0, 1)` drawn fresh per weight per update, so
  `E[increment] = min(|z| / unit, 2)`-ish and small gradients still accumulate pressure in
  expectation. Sign is `-sign(z)` as today; `z == 0` gives 0. With the flag off the rule is
  bit-identical to today's (`|z| >= low` → 1, `|z| >= high` → 2, which for the default
  0.5/1.5 is round-to-nearest with unit 1). The noise comes from the global torch RNG on the
  packed tensor's device (`torch.rand_like`), which the checkpoint already saves and restores.
- **Lever 2, pressure-weighted effective weight (`pressure_weight`, float λ ≥ 0, default 0).**
  The forward pass uses `scale × (code + λ × pressure / pressure_threshold)` instead of
  `scale × code`. Pressure lives in [-7, 7] and the threshold is 8, so the fractional term is
  in (-0.875 λ, 0.875 λ): a weight partway to its next code is partway there in the forward
  pass too. The backward uses the same effective weight for `grad_input` and for the row-scale
  gradient; the ratchet update itself is unchanged. Zero new persistent state: the nibble
  already exists. Supported in `fp32` and `bf16` matmul modes; `int8` mode with λ > 0 raises
  `ValueError` at construction (the int8 GEMM consumes integer codes). Generation from a
  checkpoint must apply the same λ, so the setting is stored in checkpoint metadata like
  `rms_ema_beta` and read back by `generate.py`.

## Constraints (repository invariants)

- No new FP/BF16 `Parameter` mirroring a code matrix; `audit_no_master_weights` zero
  violations with both levers on; `persistent_state_bytes` unchanged by either lever.
- Both levers opt-in and off by default; with both off every existing test and the update rule
  are bit-identical to today's.
- Transient effective-weight tensors are released after each step, as today.
- Plumbing mirrors `rms_ema_beta`: `[ratchet]`/`[training]` TOML field as appropriate,
  `ExperimentConfig` field and allowlist, `model_config` pass-through, `ModelConfig` field,
  `RatchetGPT._linear` kwargs, `DiscreteRatchetLinear.__init__` kwargs, `lat train`
  `--stochastic-bucket` / `--pressure-weight`, `_RESUME_CONFIG_DEFAULTS`, `generate.py`
  model construction, README flag list.

## Screen (Phase 1)

`scripts/update_rule_screen.py`: `configs/scaleup_text8_25m_5k.toml`, text8, seed 1337, 15
codes, ratchet weight mode, cells `stoch` (lever 1), `pw0.5`, `pw1.0` (lever 2 at λ = 0.5,
1.0), `stoch-pw0.5`, `stoch-pw1.0` (both). References are the grid's `plain` (1.1671) and
`qat` (1.1250) cells from `runs/momentum-grid-2026-09-13/results.json`, same config, seed,
and training source, copied into this study's `results.json` rather than rerun. Metric: best
validation at 5k, plus trace; selection: lowest best@5k; a cell must beat plain by more than
0.005 nats to advance. Runs go through the thermal guard on whichever card has capacity and
may share a card with the gap-ladder study (the 25M model is bit-repeatable at this shape).
The driver runs the worktree's own `lat` (`.venv/bin/lat` under the worktree), reads the
dataset from the primary checkout's `data/text8/text8`, and writes to an absolute `--root`
under the primary checkout's `runs/` (default `/games/ailab/local-ai-training/runs/update-rule-screen-2026-09-14`).

## Confirmation (Phase 2, only if a cell advances)

Winner at `configs/scaleup_text8_25m_30k.toml`, seeds 1337–1339, three runs, compared against
the twelve matched baselines in `runs/momentum-30k-2026-09-13/` (plain, QAT, FP32 per seed).
Flipped / partial / not confirmed by the same criteria as the momentum confirmation
(winner minus QAT ≤ 0.03 with winner below plain on every seed; below plain every seed but
> 0.03 above QAT; else not confirmed).

## Non-goals

New per-weight state, changes to defaults, int8-mode support for lever 2, throughput claims.
