# Experiment integrity implementation plan

Implement the user-approved four-part plan against baseline `9dcc15f`.

## Constraints

- Preserve datasets, checkpoints, historical result records, and unrelated work.
- Ratchet state must not acquire floating-point master matrices.
- Training/benchmark commands use `CUDA_VISIBLE_DEVICES=1`.
- Legacy checkpoints remain usable for generation. Reject training continuation when
  required reproducibility state is missing; do not add an override.
- No long training runs or learning-rule changes.

## Tasks

1. Freeze ratchet codes, pressure, scales, EMA and leak counters before backward,
   while preserving support-parameter gradients. Clear discarded gradients/stats.
2. Validate tokenizer kind and canonical serialized subword identity before resume
   mutates model, optimizer, RNG, or metrics. Preserve character vocabulary checks.
3. Write version-2 checkpoints with CPU/CUDA training RNG, selected seed and per-module
   leak counters. Restore immediately before continuation. Validate settings governing
   stochastic behavior/leak scheduling. Accept version 1 only when missing state is
   irrelevant; keep generation compatible with both versions.
4. Reconcile README, documentation index, roadmap and stale handoffs. Explain packed
   storage and floating-point support state. Add dated frozen-control evidence caveat,
   attributing historical exposure only when source revisions establish it.

## Acceptance

- Actual frozen tensors/counters remain identical across training; support tensors learn.
- Mismatched tokenizers of equal size fail before mutation; equivalent JSON succeeds.
- Uninterrupted/resumed tensors, optimizer, RNG and leak counters match, including
  non-boundary leak splits, dropout and available stochastic CUDA coverage.
- Legacy checkpoint acceptance/rejection and both generation formats are covered.
- Run pytest, Ruff, master-weight audit and diff whitespace checks, separating baseline
  failures and unavailable GPU checks from changes verified in this work.
