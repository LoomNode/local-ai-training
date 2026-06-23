# Adaptive-per-row-scale Ratchet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable and run a state-count sweep (5→15) of frozen-scale vs trainable-per-row-scale ratchet arms to measure how much of the master-weight-free quality gap is the frozen-scale saturation artifact.

**Architecture:** No new training mechanism. Reuse the existing `trainable_scale` flag (AdamW-trained `log_scale`). Two small enabling code changes — relax the `max_code` range to the 4-bit nibble cap, and expose `--codes {11,13,15}` plus a `--trainable-scale` flag on the CLI — then run 9 iso-everything arms and write a results note.

**Tech Stack:** Python, PyTorch, `uv`, pytest. Single package `src/local_ai_training/`.

## Global Constraints

- `max_code` valid range is `2..7` inclusive (7 = 15 states = 4-bit nibble cap, since codes pack as `code + max_code ≤ 14 < 16`). `max_code=8` must raise.
- `lat audit` must report zero violations for every ratchet arm. A 1-D per-row trainable scale (`log_scale`, `ndim==1`) is permitted; only floating *matrix* params (`ndim >= 2`) are violations.
- Preserve existing runs: all new artifacts go to `runs/text8-25m-adascale/`. Never write to `runs/text8-25m/` or `runs/text8-25m-qat/`.
- Iso-everything for runs: config `configs/scaleup_text8_25m_30k.toml`, seed 1337, 30000 steps, eval every 200 steps / 40 batches. HF dataset revision stays pinned (inherited from config).
- `max_code = (codes - 1) // 2` is the existing codes→max_code mapping (11→5, 13→6, 15→7). Do not change it.
- `uv` commands use the shared cache: prefix with `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache`.

---

### Task 1: Relax `max_code` range to the 4-bit nibble cap

**Files:**
- Modify: `src/local_ai_training/ratchet.py:285-286` (DiscreteRatchetLinear `max_code` check)
- Modify: `src/local_ai_training/qat.py:27-28` (QATLinear `max_code` check)
- Test: `tests/test_ratchet.py`, `tests/test_qat.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `DiscreteRatchetLinear(..., max_code=m)` and `QATLinear(..., max_code=m)` accept `m in {2,3,4,5,6,7}` and raise `ValueError` for `m=8`. Packing/unpacking is unchanged and already correct for `max_code ≤ 7`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_ratchet.py`:

```python
import pytest
from local_ai_training.ratchet import DiscreteRatchetLinear


@pytest.mark.parametrize("max_code", [2, 3, 4, 5, 6, 7])
def test_ratchet_accepts_max_code_up_to_nibble_cap(max_code):
    layer = DiscreteRatchetLinear(8, 4, max_code=max_code)
    assert layer.max_code == max_code
    assert layer.code.abs().max().item() <= max_code


def test_ratchet_rejects_max_code_above_nibble_cap():
    with pytest.raises(ValueError):
        DiscreteRatchetLinear(8, 4, max_code=8)
```

In `tests/test_qat.py`:

```python
import pytest
from local_ai_training.qat import QATLinear


@pytest.mark.parametrize("max_code", [2, 3, 4, 5, 6, 7])
def test_qat_accepts_max_code_up_to_nibble_cap(max_code):
    layer = QATLinear(8, 4, max_code=max_code)
    assert layer.max_code == max_code


def test_qat_rejects_max_code_above_nibble_cap():
    with pytest.raises(ValueError):
        QATLinear(8, 4, max_code=8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py::test_ratchet_accepts_max_code_up_to_nibble_cap tests/test_qat.py::test_qat_accepts_max_code_up_to_nibble_cap -v`
Expected: FAIL — `max_code=5/6/7` currently raises `ValueError` ("max_code must be 2 ... 3 ... 4").

- [ ] **Step 3: Relax the validation in `ratchet.py`**

Replace `src/local_ai_training/ratchet.py:285-286`:

```python
        if max_code not in (2, 3, 4):
            raise ValueError("max_code must be 2 (quinary), 3 (septenary), or 4 (nonary)")
```

with:

```python
        if max_code not in (2, 3, 4, 5, 6, 7):
            raise ValueError(
                "max_code must be in 2..7 (5..15 states); 7 is the 4-bit packing cap"
            )
```

- [ ] **Step 4: Relax the validation in `qat.py`**

Replace `src/local_ai_training/qat.py:27-28`:

```python
        if max_code not in (2, 3, 4):
            raise ValueError("max_code must be 2 (quinary), 3 (septenary), or 4 (nonary)")
```

with:

```python
        if max_code not in (2, 3, 4, 5, 6, 7):
            raise ValueError(
                "max_code must be in 2..7 (5..15 states); 7 is the 4-bit packing cap"
            )
```

- [ ] **Step 5: Run the new tests and the existing ratchet/qat suites**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py tests/test_qat.py -v`
Expected: PASS (new tests pass; no existing test regresses).

- [ ] **Step 6: Commit**

```bash
git add src/local_ai_training/ratchet.py src/local_ai_training/qat.py tests/test_ratchet.py tests/test_qat.py
git commit -m "feat: allow max_code up to 7 (15 states, 4-bit nibble cap)"
```

---

### Task 2: Expose `--codes {11,13,15}` and a `--trainable-scale` flag on the CLI

**Files:**
- Modify: `src/local_ai_training/cli.py:34` (train `--codes` choices), `:63` (audit `--codes` choices), add `--trainable-scale` to train, apply it in the train handler, import `replace`
- Test: `tests/test_cli.py` (create if absent; otherwise extend the existing CLI test module)

**Interfaces:**
- Consumes: `ExperimentConfig` (from `config.py`), `dataclasses.replace`.
- Produces: `lat train --codes {5,7,9,11,13,15}` accepted; `lat train --trainable-scale` sets `config.trainable_scale = True` before `train_run`, producing a model whose `DiscreteRatchetLinear` layers carry a 1-D `log_scale` `nn.Parameter`. `lat audit --codes {5,7,9,11,13,15}` accepted. Parser is reachable as `build_parser()`.

- [ ] **Step 1: Inspect the current parser construction**

Run: `grep -n "def build_parser\|def main\|argparse.ArgumentParser\|add_subparsers" src/local_ai_training/cli.py`
If there is no `build_parser()` function returning the parser, the Step 3 change includes extracting one (construct + return the parser; have `main` call `parser = build_parser()`). The tests below import `build_parser`.

- [ ] **Step 2: Write the failing tests**

Create or extend `tests/test_cli.py`:

```python
import pytest
from local_ai_training.cli import build_parser


@pytest.mark.parametrize("codes", [5, 7, 9, 11, 13, 15])
def test_train_accepts_extended_codes(codes):
    args = build_parser().parse_args(["train", "--codes", str(codes)])
    assert args.codes == codes


def test_train_trainable_scale_flag_defaults_off_and_can_enable():
    parser = build_parser()
    assert parser.parse_args(["train"]).trainable_scale is False
    assert parser.parse_args(["train", "--trainable-scale"]).trainable_scale is True


@pytest.mark.parametrize("codes", [11, 13, 15])
def test_audit_accepts_extended_codes(codes):
    args = build_parser().parse_args(["audit", "--codes", str(codes)])
    assert args.codes == codes
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_cli.py -v`
Expected: FAIL — `--codes 11` rejected by `choices=(5,7,9)`; `--trainable-scale` unrecognized; possibly `ImportError: build_parser` (drives the extract refactor).

- [ ] **Step 4: Expand `--codes` choices, add `--trainable-scale`, extract `build_parser` if needed**

In `src/local_ai_training/cli.py`:

Change the train `--codes` line:

```python
    train.add_argument("--codes", type=int, choices=(5, 7, 9), default=5)
```

to:

```python
    train.add_argument("--codes", type=int, choices=(5, 7, 9, 11, 13, 15), default=5)
```

Add, immediately after the train `--weight-mode` argument block:

```python
    train.add_argument("--trainable-scale", dest="trainable_scale", action="store_true")
```

Change the audit `--codes` line:

```python
    audit.add_argument("--codes", type=int, choices=(5, 7, 9), default=5)
```

to:

```python
    audit.add_argument("--codes", type=int, choices=(5, 7, 9, 11, 13, 15), default=5)
```

If the parser is built inline in `main`, wrap its construction in `def build_parser() -> argparse.ArgumentParser:` that returns the fully-built `parser`, and replace the inline construction in `main` with `parser = build_parser()`.

- [ ] **Step 5: Apply the flag in the train handler**

Ensure `replace` is imported at the top of `src/local_ai_training/cli.py`:

```python
from dataclasses import asdict, replace
```

In the `train` command handler, after `config = ExperimentConfig.from_toml(args.config)` and before the `train_run(...)` call, add:

```python
        if args.trainable_scale:
            config = replace(config, trainable_scale=True)
```

`config.trainable_scale` already flows through `config.model_config(...)` → `ModelConfig.trainable_scale` → `_linear` → `DiscreteRatchetLinear(trainable_scale=...)`; no further plumbing is needed.

- [ ] **Step 6: Run the CLI tests plus an audit smoke check**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_cli.py -v`
Expected: PASS.

Then confirm a trainable-scale model is audit-clean and exposes `log_scale` (also exercises Task 1's `max_code=7`):

```bash
MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run python -c "
from dataclasses import replace
from local_ai_training.config import ExperimentConfig
from local_ai_training.model import build_seeded_model
from local_ai_training.ratchet import audit_no_master_weights, DiscreteRatchetLinear
c = replace(ExperimentConfig.from_toml('configs/ratchet_tiny.toml'), trainable_scale=True)
m = build_seeded_model(c.model_config(vocab_size=65), max_code=7, seed=1337)
rep = audit_no_master_weights(m, raise_on_violation=True)
has_log_scale = any(hasattr(l, 'log_scale') for l in m.modules() if isinstance(l, DiscreteRatchetLinear))
print('violations:', rep.violations, 'has_log_scale:', has_log_scale)
assert rep.violations == () and has_log_scale
"
```

Expected: `violations: () has_log_scale: True`.

- [ ] **Step 7: Commit**

```bash
git add src/local_ai_training/cli.py tests/test_cli.py
git commit -m "feat: CLI --codes {11,13,15} and --trainable-scale flag"
```

---

### Task 3: Run driver, execute the 9-arm sweep, and write the results note

**Files:**
- Create: `runs/text8-25m-adascale/run_adascale.sh` (driver; `runs/` is git-ignored, so this is an uncommitted execution artifact)
- Create: `docs/results/2026-06-23-adaptive-scale-ratchet.md` (committed)

**Interfaces:**
- Consumes: `lat train --codes {5..15} --weight-mode ratchet [--trainable-scale]` from Tasks 1–2.
- Produces: per-arm `runs/text8-25m-adascale/<arm>/seed-1337/metrics.csv` + `checkpoint.safetensors`; a committed results note with frozen→trainable gap-closure and saturation tables.

This task is an experiment run, not a TDD unit. Its acceptance test is that every arm reaches step 30000 and `lat audit` stays clean.

- [ ] **Step 1: Confirm dataset and idle GPUs**

Run: `ls data/text8/text8 && nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`
Expected: dataset path exists; identify two idle GPUs (GPU_A, GPU_B). If `data/text8/text8` is absent, run `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat dataset --which text8` first.

- [ ] **Step 2: Write the driver script**

Create `runs/text8-25m-adascale/run_adascale.sh` (modeled on `runs/text8-25m-qat/run_all7.sh`):

```bash
#!/usr/bin/env bash
# Adaptive-scale ratchet sweep: frozen{11,13,15} + trainable{5,7,9,11,13,15}.
# Tests whether trainable per-row scale relieves frozen-scale saturation.
# Usage: bash run_adascale.sh GPU_ID name:codes:mode ...
#   mode = "ada" (--trainable-scale) or "frozen" (omit the flag)
set -euo pipefail
cd /games/ailab/local-ai-training
CFG=configs/scaleup_text8_25m_30k.toml
DS=data/text8/text8
OUT=runs/text8-25m-adascale
GPU=$1; shift

run() { # name codes mode
  local name=$1 codes=$2 mode=$3
  local flag=""
  [ "$mode" = "ada" ] && flag="--trainable-scale"
  echo "=== [$(date +%H:%M:%S)] START $name (codes=$codes $mode) GPU $GPU ==="
  env CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
    uv run lat train --config "$CFG" --dataset-path "$DS" \
    --codes "$codes" --weight-mode ratchet $flag --seed 1337 \
    --output "$OUT/$name/seed-1337" > "$OUT/$name.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE  $name GPU $GPU ==="
}

for spec in "$@"; do
  IFS=: read -r name codes mode <<< "$spec"
  run "$name" "$codes" "$mode"
done
echo "=== GPU $GPU COMPLETE [$(date +%H:%M:%S)] ==="
```

- [ ] **Step 3: Launch the sweep across both GPUs (background)**

Substitute the GPU_A / GPU_B indices from Step 1:

```bash
cd /games/ailab/local-ai-training
mkdir -p runs/text8-25m-adascale
nohup bash runs/text8-25m-adascale/run_adascale.sh GPU_A \
  frozen11:11:frozen frozen13:13:frozen frozen15:15:frozen ada5:5:ada ada7:7:ada \
  > runs/text8-25m-adascale/gpuA.out 2>&1 &
nohup bash runs/text8-25m-adascale/run_adascale.sh GPU_B \
  ada9:9:ada ada11:11:ada ada13:13:ada ada15:15:ada \
  > runs/text8-25m-adascale/gpuB.out 2>&1 &
```

Expected: two background drivers; ~110 min/arm → ~8–9h wall (5 arms on GPU_A, 4 on GPU_B).

- [ ] **Step 4: Wait for completion and verify every arm reached step 30000**

After both drivers print "COMPLETE":

```bash
MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run python -c "
import csv, glob
for f in sorted(glob.glob('runs/text8-25m-adascale/*/seed-1337/metrics.csv')):
    rows = list(csv.DictReader(open(f)))
    last = int(rows[-1]['step']); arm = f.split('/')[2]
    best = min(float(r['validation_loss']) for r in rows)
    print(f'{arm:10s} last {last}  best val {best:.4f}')
    assert last == 30000, f'{arm} did not reach 30000'
"
```

Expected: all 9 arms report `last 30000`.

- [ ] **Step 5: Compute the gap-closure + saturation table**

```bash
MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run python - <<'PYEOF'
import csv, torch
from safetensors import safe_open

mc = {5:2, 7:3, 9:4, 11:5, 13:6, 15:7}

def best_val(path):
    rows = list(csv.DictReader(open(path)))
    return min(float(r["validation_loss"]) for r in rows)

def mean_sat(path, max_code):
    sats = []
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.endswith("packed"):
                code = (f.get_tensor(k).to(torch.int16) & 0x0F) - max_code
                sats.append((code.abs() == max_code).float().mean().item() * 100)
    return sum(sats) / len(sats)

def arm(base):
    return (f"runs/text8-25m-adascale/{base}/seed-1337/metrics.csv",
            f"runs/text8-25m-adascale/{base}/seed-1337/checkpoint.safetensors")

print("states | frozen val | ada val | gap-closed | frozen sat% | ada sat%")
for s in (5, 7, 9, 11, 13, 15):
    if s in (5, 7, 9):  # frozen baseline reused from the QAT experiment dir
        fcsv = f"runs/text8-25m-qat/ratchet{s}/seed-1337/metrics.csv"
        fck  = f"runs/text8-25m-qat/ratchet{s}/seed-1337/checkpoint.safetensors"
    else:
        fcsv, fck = arm(f"frozen{s}")
    acsv, ack = arm(f"ada{s}")
    fv, av = best_val(fcsv), best_val(acsv)
    fsat, asat = mean_sat(fck, mc[s]), mean_sat(ack, mc[s])
    print(f"{s:6d} | {fv:10.4f} | {av:7.4f} | {fv-av:+10.4f} | {fsat:10.1f} | {asat:7.1f}")
PYEOF
```

Expected: a 6-row table. Hypothesis: positive `gap-closed` at every state count, largest at 5 states, with `ada sat%` well below `frozen sat%`.

- [ ] **Step 6: Confirm one trainable arm is audit-clean**

```bash
MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat audit --model configs/scaleup_text8_25m_30k.toml --codes 5 --vocab-size 27
```

Expected: JSON report with `"violations": []`. (Audit builds a fresh frozen model; this confirms the relaxed `max_code` path stays clean. The trainable-scale audit was already verified in Task 2 Step 6.)

- [ ] **Step 7: Write the results note**

Create `docs/results/2026-06-23-adaptive-scale-ratchet.md`. Fill the table from Step 5 and the per-state curves. Required sections, mirroring `docs/results/2026-06-22-qat-deconfounding.md`:

- **Question** — restate the saturation hypothesis and that this isolates the frozen-scale contribution to the master-weight-free gap.
- **Method** — `trainable_scale=True` (AdamW `log_scale`), iso-everything vs frozen `runs/text8-25m-qat/ratchet{5,7,9}` and the new frozen{11,13,15}; QAT{5,7,9} as the FP-master ceiling; GPUs/UUIDs noted.
- **Result** — the Step 5 gap-closure + saturation table; per-state best val loss.
- **Headline** — how much of the master-free gap the frozen scale explained (frozen→ada improvement vs the QAT→ratchet penalty from the prior note), and the state-count gradient (does the gain shrink as saturation falls).
- **Limitations** — single seed (1337), 25M/30k; trainable scale adds two FP32 AdamW moments per output row per matrix (negligible vs a master weight, not literally zero); one mechanism only.

- [ ] **Step 8: Commit the results note**

```bash
git add docs/results/2026-06-23-adaptive-scale-ratchet.md
git commit -m "docs: adaptive-scale ratchet results — frozen-scale share of the master-free gap"
```

---

## Verification

- `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest` green (new max_code + CLI tests pass, no regressions).
- `uv run lat audit --model configs/scaleup_text8_25m_30k.toml --codes 5 --vocab-size 27` reports `"violations": []`.
- `uv run ruff check` on changed files (`src/local_ai_training/ratchet.py`, `qat.py`, `cli.py`, the test files) + `git diff --check`.
- All 9 arms reach step 30000; the results note reports the gap-closure + saturation table.
- Existing runs untouched: `runs/text8-25m/` and `runs/text8-25m-qat/` unchanged; new artifacts only under `runs/text8-25m-adascale/`.
