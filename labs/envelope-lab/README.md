# envelope-lab

Hardware-envelope autoresearch lab for the dual RTX 3090 box.

**KAT (or any coding agent) directs research** by editing `train.py` under a frozen envelope.  
**Goal:** intelligence density under device law — not unbounded scale.

## Layout

| Location | Contents |
|----------|----------|
| `/games/ailab/envelope-lab/` | Code, `program.md`, `CAMPAIGN.md`, `results.tsv` |
| `/games/ailab/envelope-lab/notes/` | Architecture theses, **TODO**, research notes |
| `/vault/ailab/envelope-lab/cache/` | Corpus + token bins |
| `/vault/ailab/envelope-lab/runs/` | Trial logs, `runs.jsonl`, journal |

### Notes index

- **[notes/HANDOFF.md](notes/HANDOFF.md)** — **resume here**  
- [notes/TODO.md](notes/TODO.md) — lab task list  
- [notes/parameter-inefficiency-and-folding.md](notes/parameter-inefficiency-and-folding.md) — fold weights + memory  
- [notes/tokens-as-parameters.md](notes/tokens-as-parameters.md) — unify tokenizer capacity with weights

## Quick start

```bash
cd /games/ailab/envelope-lab
uv sync

export ENVELOPE_MODE=solo            # GPU 1 only (dedicated AI)
export ENVELOPE_AI_GPU=1
export ENVELOPE_TRAIN_GPU=0          # dual turbo only
export ENVELOPE_CACHE=/vault/ailab/envelope-lab/cache
export ENVELOPE_RUNS=/vault/ailab/envelope-lab/runs

uv run python prepare.py
./scripts/run_trial.sh               # 5 min default; or ENVELOPE_TIME_BUDGET=60
```

### Lab console (`ailab`)

Prefer the unified CLI (talk to KAT, goals, trials):

```bash
/games/ailab/ailab-cli/bin/ailab status
/games/ailab/ailab-cli/bin/ailab goal add "…"
/games/ailab/ailab-cli/bin/ailab chat
/games/ailab/ailab-cli/bin/ailab trial --budget 60
```

Or point an agent at **`program.md`** and let it loop.

## Modes

| Mode | Behavior |
|------|----------|
| `solo` (**default**) | **GPU 1 only** — dedicated AI card |
| `dual` (turbo **toggle**) | **Both GPUs** — train GPU 0 + KAT on GPU 1 |

Default is always the AI card. Turbo is opt-in: `ailab trial --mode dual` or `ENVELOPE_MODE=dual`.

## Frozen vs editable

| File | Who touches it |
|------|----------------|
| `prepare.py` | Frozen — envelope, data, eval, report |
| `lab_config.py` | Frozen — solo/dual routing |
| `train.py` | **Agent edits** |
| `program.md` | Human research policy |

## Metrics (every trial)

Printed at end of `run.log` (grep-friendly):

- `val_bpb` — lower better (fixed tokenizer)
- `peak_vram_mb` — must be ≤ 22000 to keep
- `decode_tps` — must be ≥ 15 to keep as serve candidate
- `envelope` — `ok` \| `soft_vram` \| `fail_vram` \| `fail_tps`

## Reference

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (local copy under `/games/ailab/References/autoresearch-master`).
