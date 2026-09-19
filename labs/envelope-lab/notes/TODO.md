# Lab TODO

## Now (Campaign 1 — solo GPU1)

- [x] Seed Campaign 1 queue + run solo 300s loop (2026-08-02) — **queue finished**
- [x] Write architecture notes + HANDOFF (2026-08-02)
- [ ] Review Campaign 1: among 300s runs m3 had best bpb (~10.43); all marked discard vs 60s smoke
- [ ] Fix keep-logic: budget-aware champion (same TIME_BUDGET only)
- [ ] Set `train.best.py` from best fair 300s keep (probably re-keep m3 or re-run baseline)
- [ ] `systemctl --user start qwen-server` when you want KAT chat on GPU1 again

## Next architecture (post–Campaign 1)

- [ ] Implement **shared recurrent block** mode (`N_UNIQUE_BLOCKS` × `N_RECURRENCE`) — real parameter folding
- [ ] A/B: stacked L=6 vs shared 2×3 or 1×6 under **same 300s budget + envelope**
- [ ] Track **params in embed vs core** and **unique vs effective depth** in trial logs
- [ ] Prototype **token-side capacity** growth with **smaller stack** (see `tokens-as-parameters.md`)
- [ ] Optional: `BLOCK_TYPE = stacked | memory | shared_recurrent` knob for the loop

## Product / lab harness

- [ ] `ailab loop` journal + queue already; polish auto-append clarity in UI/shell
- [ ] Optional fractalsearch-style **dashboard** (read-only scoreboard over `runs.jsonl`)
- [ ] When solo is solid: document dual turbo as optional only
- [ ] Campaign 2 write-up once folding A/B exists

## Later / bigger

- [ ] SSM/Mamba as alternate **small computer** over token memory (orthogonal to folding)
- [ ] Quantization campaign (bit compression) separate from architecture density
- [ ] Scale modular specialists (many total params, few active) only after reuse thesis is tested small

## Ideas parked (not tasks yet)

- Fold **pretrained** checkpoints (merge redundant layers + adapters) — harder eval
- Colibri-style paging of **token tables** rather than layer shards
- Full RSI agent that edits solution files freely (fractalsearch-style `solutions/`)
