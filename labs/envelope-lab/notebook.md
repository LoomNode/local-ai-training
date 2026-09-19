# Research notebook (sticky across agents/sessions)

Long-form notes: **[notes/](notes/)**. **Returning later → [notes/HANDOFF.md](notes/HANDOFF.md).**

## Campaign 1 notes

### Baseline
- 60s smoke: DEPTH=6 N_EMBD=384 plain · ~30.1M · val_bpb≈6.92 · ok envelope (**do not use as champion**)
- 300s plain same knobs: val_bpb≈**10.90** · ok envelope (fair budget)

### Campaign 1 queue results (300s solo, 2026-08-02)
- memory+shallow: ~10.91 discard*
- memory+thin MLP: crash/skipped
- **plain deeper/narrower 8×320: ~10.43** — best 300s bpb
- memory slots=32: ~10.84
- \*keep-logic wrongly compared to 60s smoke — fix before next campaign

### Ideas next
1. Shared recurrent core (true folding) — see notes/
2. Token-side capacity vs stack size
3. LR sweep after fair 300s champion locked

### Fractalsearch lessons applied
- One coherent idea per trial
- Never rewrite best from scratch — knobs + train.best.py snapshot
- Fixed time budget; metric is king; envelope is veto
