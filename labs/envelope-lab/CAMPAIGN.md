# Campaign 1 — Size efficiency under solo GPU1

**Status:** active  
**Started:** 2026-08-02  
**GPU mode:** **solo only** (dedicated AI card = GPU 1). Do not use dual until solo is proven.

## Research question

> Can we beat a plain small GPT **baseline** on `val_bpb` under the same fixed train-time budget and hardware envelope by changing **architecture / capacity allocation** (not by spending more wall-clock)?

Hypothesis class (stolen spirit of fractalsearch + your weight+memory idea):

- **Intelligence density** matters more than raw depth stacking.
- **Weight + memory blocks** (soft slot memory) may buy quality per param.
- **Shallower + memory**, **thinner MLP + deeper**, **width vs depth** tradeoffs should be measured, not assumed.

## Frozen law (never edit)

| Rule | Value |
|------|--------|
| Train budget per trial | fixed (`ENVELOPE_TIME_BUDGET`, default 300s) |
| Metric primary | **lower `val_bpb`** |
| Envelope | `peak_vram_mb ≤ 22000`, `decode_tps ≥ 15` |
| Data / tokenizer / eval | `prepare.py` only |
| GPU | **solo → GPU 1 only** for this campaign |

## Success criteria

1. **Baseline locked:** one `keep` row for plain block, DEPTH=6, N_EMBD=384, USE_MEMORY_BLOCK=0, **300s** budget.  
2. **At least one keep** that improves `val_bpb` vs that baseline **or** matches bpb with clearly fewer `num_params_M` (density win) while still `envelope: ok`.  
3. Log every run to `results.tsv` + `runs.jsonl` + `notebook.md`.

## What the agent may change

- Knobs in `train.py` between `AGENT_KNOBS_BEGIN/END`
- Especially: `USE_MEMORY_BLOCK`, `MEMORY_SLOTS`, `DEPTH`, `N_EMBD`, `N_HEAD`, `MLP_MULT`, `LR`, `BATCH_SIZE`
- Notes in `notebook.md`

## What the agent must not change

- `prepare.py`, `lab_config.py`, eval metric definition
- Claiming dual GPU for this campaign without human toggle

## Human workflow (solo)

```text
ailab
  discuss → /cue N
  /loop --mode solo --budget 300
```

Or unattended after cues are queued:

```bash
ailab loop --mode solo --budget 300
```

## Fractalsearch patterns we steal

- Immutable harness / metric  
- Copy-forward best (`train.best.py`)  
- Per-run log records (`runs.jsonl`)  
- Human steers via markdown (`CAMPAIGN.md` + `program.md`)  
- Notebook for sticky ideas across sessions  
- Loop until human stops (when running unattended)

## Architecture north star (not fully in knobs yet)

See **[notes/](notes/)**:

- Parameter inefficiency / folding: [notes/parameter-inefficiency-and-folding.md](notes/parameter-inefficiency-and-folding.md)
- Tokens as parameters: [notes/tokens-as-parameters.md](notes/tokens-as-parameters.md)
- Lab TODO: [notes/TODO.md](notes/TODO.md)

Campaign 1 tests density + weak memory blocks. **Shared recurrent core + token-side capacity** are the next real tests of the thesis.
