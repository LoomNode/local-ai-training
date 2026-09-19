# envelope-lab — research program (KAT directs)

You are the research agent for **hardware-envelope autoresearch**.

**Active campaign:** read `CAMPAIGN.md` first. Campaign 1 = size efficiency on **solo GPU1**.

## Mission

Maximize **useful model quality under this machine’s law**, not unbounded scale.
Prefer **intelligence density** (quality per param / VRAM) when primary metric ties.

```
maximize  quality (val_bpb lower is better, for now)
subject to
  peak_vram_mb  ≤ 22000   (hard)
  decode_tps    ≥ 15      (hard for "keep" as a serve candidate)
  train trial   = fixed TIME_BUDGET (default 300s wall train time)
  solo mode must always work (dual GPU is optional turbo only)
```

Long-term hypotheses you may explore (after a solid baseline):
- richer blocks that combine **fixed weights + writable memory**
- modular specialists / sticky experts (less thrash)
- intelligence **density** (quality per VRAM · bandwidth · time) — not raw param count

You do **not** need to beat KAT-Coder at agentic coding in v0. You need a working keep/discard loop under the envelope.

## Layout

| Path | Role |
|------|------|
| `/games/ailab/envelope-lab/` | **Code** (git here) |
| `prepare.py` | **FROZEN** — envelope, data, eval, report |
| `train.py` | **YOU EDIT** — model, optim, hparams, arch |
| `lab_config.py` | **FROZEN** — solo/dual device routing |
| `program.md` | **HUMAN** policy (this file); propose edits, don't silently rewrite goals |
| `results.tsv` | Experiment log |
| `/vault/ailab/envelope-lab/cache/` | Token bins, corpus |
| `/vault/ailab/envelope-lab/runs/` | `run.log` per trial |

## Setup (once per campaign)

1. Agree a run tag (e.g. `aug2`).
2. Work on a branch if git is in use: `envelope/<tag>`.
3. Read `README.md`, `prepare.py`, `train.py`, `lab_config.py`.
4. Ensure data: `uv run python prepare.py`
5. Init `results.tsv` header if missing.
6. **First trial = unmodified baseline.**

## How to run a trial

**GPU toggle:** default **solo = GPU 1**. **dual = turbo both cards** (opt-in).

| Mode | GPUs in play | Trial train on | AI / KAT |
|------|----------------|----------------|----------|
| **solo** (**default**) | **1 only** | GPU 1 | same card (unload if tight) |
| **dual** (turbo toggle) | **both** | GPU 0 | GPU 1 |

```bash
ailab trial              # solo / GPU1
ailab trial --mode dual  # turbo both
ailab trial --mode dual --bg
```

```bash
cd /games/ailab/envelope-lab
export ENVELOPE_MODE=solo          # or dual
export ENVELOPE_AI_GPU=1
export ENVELOPE_TRAIN_GPU=0        # dual only
export ENVELOPE_CACHE=/vault/ailab/envelope-lab/cache
export ENVELOPE_RUNS=/vault/ailab/envelope-lab/runs

# optional short debug: ENVELOPE_TIME_BUDGET=60
./scripts/run_trial.sh
# preferred: ailab trial
```

Or manually:

```bash
uv run python train.py > "$ENVELOPE_RUNS/run.log" 2>&1
grep -E '^(val_bpb|peak_vram_mb|decode_tps|envelope|num_params_M):' "$ENVELOPE_RUNS/run.log"
```

## What you CAN do

- Edit **`train.py` only** (architecture, LR, depth, batch, blocks, weight+memory ideas, etc.).
- Commit each experiment (if git available) before/after as you prefer; log every run in `results.tsv`.

## What you CANNOT do

- Modify `prepare.py` or `lab_config.py`.
- Change the eval metric definition or tokenizer without human approval (breaks comparability).
- Install new pip packages unless the human approves (stick to `pyproject.toml`).
- “Win” by ignoring envelope failures (`envelope: fail_vram` / `fail_tps`).

## Keep / discard rules

After each run, read `val_bpb`, `peak_vram_mb`, `decode_tps`, `envelope`.

| status | When |
|--------|------|
| **keep** | `envelope` is `ok` or `soft_vram`, and `val_bpb` improved vs best keep (or first baseline). Prefer simpler if tie. |
| **discard** | Worse bpb, or complexity with tiny gain, or you choose not to retain. |
| **crash** | OOM, exception, no metrics. |
| **illegal** | `envelope: fail_vram` or `fail_tps` — **never keep** as champion even if bpb looks great. |

Simplicity: deleting code with equal/better bpb is a win. Ugly +0.001 bpb is not.

## results.tsv columns (tab-separated)

```
commit	val_bpb	peak_vram_mb	decode_tps	envelope	status	description
```

Example:

```
commit	val_bpb	peak_vram_mb	decode_tps	envelope	status	description
baseline	3.120000	2100.0	120.0	ok	keep	baseline DEPTH=6
a1b2c3d	3.050000	2200.0	110.0	ok	keep	LR 6e-4
```

Use `baseline` as commit if not using git yet.

## Experiment loop

LOOP:

1. Inspect last `results.tsv` rows and current `train.py`.
2. Form a hypothesis (one change when possible).
3. Edit `train.py`.
4. Run `./scripts/run_trial.sh` (redirected log — don't flood context).
5. Grep metrics from `$ENVELOPE_RUNS/run.log`.
6. Append `results.tsv` with keep/discard/crash/illegal.
7. If discard, revert `train.py` to last keep (git checkout or manual).
8. Repeat.

## First command to the human

After setup, confirm baseline run completed and `results.tsv` has one `keep` row, then continue overnight-style iteration.
