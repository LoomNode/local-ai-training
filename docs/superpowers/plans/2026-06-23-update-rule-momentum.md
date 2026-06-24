# Update-rule Momentum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two opt-in, master-weight-free temporal-EMA smoothers to the ratchet update rule — leaky pressure (1st moment) and EMA per-row RMS (2nd moment) — and screen whether they close the master-free quality gap at 5 states.

**Architecture:** Both knobs default off and leave the baseline path bit-identical. Arm B replaces the two duplicated per-row RMS normalizations with one `_normalize` helper backed by an optional per-row EMA buffer. Arm A applies a per-step integer leak to the packed pressure inside `ratchet_update()` (the single per-step entry point), never touching the compiled `_ratchet_update_core` or the fused per-tile path.

**Tech Stack:** Python, PyTorch, `uv`, pytest. Single package `src/local_ai_training/`.

## Global Constraints

- **No new per-weight state.** New state may only be per-row (`out_features` scalars per matrix) or none. The packed code+pressure byte stays one byte/weight.
- `lat audit` must report zero violations for every arm: it flags only floating *matrix* params (`ndim >= 2`); 1-D per-row buffers are permitted. No FP/bf16 matrix Parameter mirroring a code matrix.
- **Baseline bit-unchanged when off:** `rms_ema_beta == 0.0` and `pressure_leak_period == 0` must reproduce the current update exactly (equivalence tests required).
- `_validate_state()` must keep passing: pressure stays within `abs <= 7` (the leak only moves pressure toward zero, never enlarges it).
- Both knobs live in the `[ratchet]` config section and as `lat train` CLI flags, mirroring the existing `trainable_scale` plumbing (`config.py` → `ModelConfig` → `DiscreteRatchetLinear`).
- Screening protocol: 5 states (`--codes 5`), seed 1337, `configs/scaleup_text8_25m_5k.toml`, 5000 steps, into `runs/text8-25m-updaterule/` (existing runs untouched). HF dataset revision stays pinned (inherited from config).
- `uv` commands use the shared cache: prefix with `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache`.

---

### Task 1: Arm B — EMA per-row RMS (`rms_ema_beta`)

**Files:**
- Modify: `src/local_ai_training/config.py` (add `rms_ema_beta` field, `[ratchet]` allowed key, validation, `model_config` pass-through)
- Modify: `src/local_ai_training/model.py` (`ModelConfig.rms_ema_beta`, pass in `_linear`)
- Modify: `src/local_ai_training/ratchet.py` (`DiscreteRatchetLinear.__init__` param + conditional `rms_ema` buffer; new `_normalize` helper; use it at the two RMS sites `:423` and `:452`)
- Modify: `src/local_ai_training/cli.py` (`--rms-ema-beta` on `train`, apply via `replace`)
- Test: `tests/test_ratchet.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: existing `DiscreteRatchetLinear`, `ExperimentConfig`, `dataclasses.replace`.
- Produces: `DiscreteRatchetLinear(..., rms_ema_beta: float = 0.0)`; method `self._normalize(grad: Tensor, row_start: int, row_end: int) -> Tensor` returning the normalized gradient (and updating `self.rms_ema[row_start:row_end]` in place when `rms_ema_beta > 0`). Config/CLI key `rms_ema_beta`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_ratchet.py`:

```python
def test_rms_ema_beta_zero_matches_instantaneous_normalization():
    torch.manual_seed(0)
    ref = torch.randn(6, 8)
    base = DiscreteRatchetLinear(8, 6, max_code=2, initial_weight=ref.clone())
    ema = DiscreteRatchetLinear(8, 6, max_code=2, rms_ema_beta=0.0, initial_weight=ref.clone())
    grad = torch.randn(6, 8)
    n_base = base._normalize(grad, 0, 6)
    n_ema = ema._normalize(grad, 0, 6)
    assert torch.equal(n_base, n_ema)  # beta=0 is bit-identical to the current rule


def test_rms_ema_first_step_matches_instantaneous_then_smooths():
    torch.manual_seed(0)
    layer = DiscreteRatchetLinear(8, 6, max_code=2, rms_ema_beta=0.9)
    g1 = torch.randn(6, 8)
    # first step: EMA seeds from this step's mean-square, so identical to instantaneous
    rms1 = g1.float().square().mean(dim=1, keepdim=True).sqrt()
    assert torch.allclose(layer._normalize(g1, 0, 6), g1.float() / (rms1 + layer.eps))
    # second step: denominator is the EMA, NOT this step's rms
    g2 = torch.randn(6, 8) * 5.0
    ms1 = g1.float().square().mean(dim=1)
    ms2 = g2.float().square().mean(dim=1)
    expected_ema = 0.9 * ms1 + 0.1 * ms2
    expected = g2.float() / (expected_ema.unsqueeze(1).sqrt() + layer.eps)
    assert torch.allclose(layer._normalize(g2, 0, 6), expected)


def test_rms_ema_buffer_is_per_row_and_audit_clean():
    layer = DiscreteRatchetLinear(8, 6, max_code=2, rms_ema_beta=0.9)
    assert layer.rms_ema.shape == (6,)  # one scalar per output row
    assert layer.rms_ema.ndim == 1
    assert audit_no_master_weights(nn.Sequential(layer)).violations == ()
```

In `tests/test_cli.py`:

```python
def test_train_rms_ema_beta_flag_defaults_zero_and_parses():
    parser = build_parser()
    assert parser.parse_args(["train"]).rms_ema_beta == 0.0
    assert parser.parse_args(["train", "--rms-ema-beta", "0.9"]).rms_ema_beta == 0.9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py -k rms_ema tests/test_cli.py -k rms_ema -v`
Expected: FAIL — `_normalize` and `rms_ema` do not exist; `--rms-ema-beta` unrecognized.

- [ ] **Step 3: Add the `rms_ema_beta` constructor param, buffer, and `_normalize` helper**

In `DiscreteRatchetLinear.__init__` (after the existing `self.eps = eps` / `self.trainable_scale = ...` block), add the parameter to the signature (`rms_ema_beta: float = 0.0,`) and:

```python
        if not 0.0 <= rms_ema_beta < 1.0:
            raise ValueError("rms_ema_beta must be in [0, 1)")
        self.rms_ema_beta = rms_ema_beta
        if rms_ema_beta > 0.0:
            # Per-row 2nd-moment EMA (Adam's v at row granularity). Lazily seeded per row
            # from the first step's mean-square (rms_ema==0 => uninitialized), so step 0
            # reproduces the instantaneous rule exactly. 1-D => audit-clean.
            self.register_buffer("rms_ema", torch.zeros(out_features))
```

Add the helper method (near `apply_normalized_gradient`):

```python
    def _normalize(self, gradient: Tensor, row_start: int, row_end: int) -> Tensor:
        grad = gradient.float()
        ms = grad.square().mean(dim=1, keepdim=True)  # [rows, 1] mean-square per row
        if self.rms_ema_beta > 0.0:
            ema = self.rms_ema[row_start:row_end].unsqueeze(1)
            ema = torch.where(ema == 0, ms, self.rms_ema_beta * ema + (1.0 - self.rms_ema_beta) * ms)
            self.rms_ema[row_start:row_end] = ema.squeeze(1)
            rms = ema.sqrt()
        else:
            rms = ms.sqrt()
        return grad / (rms + self.eps)
```

- [ ] **Step 4: Route the two RMS sites through `_normalize`**

In `_capture_weight_gradient` (the fused tile path), replace the two lines:

```python
            rms = gradient.float().square().mean(dim=1, keepdim=True).sqrt()
            normalized = gradient.float() / (rms + self.eps)
```

with a single call to the helper:

```python
            normalized = self._normalize(gradient, tile_start, tile_end)
```

`rms` is no longer in scope there, so update the stat line a few lines down. Replace:

```python
            self._pending_stats_rms_sum += float(rms.sum().item())
```

with an independent recompute of the (instantaneous) per-row RMS, so the reported stat is unchanged regardless of the EMA:

```python
            self._pending_stats_rms_sum += float(
                gradient.float().square().mean(dim=1).sqrt().sum().item()
            )
```

In `apply_weight_gradient` (the eager path), replace:

```python
        rms = gradient.float().square().mean(dim=1, keepdim=True).sqrt()
        normalized = gradient.float() / (rms + self.eps)
        stats = self.apply_normalized_gradient(normalized)
        return RatchetUpdateStats(
            ...
            gradient_rms_mean=float(rms.mean().item()),
        )
```

with:

```python
        rms_mean = float(gradient.float().square().mean(dim=1).sqrt().mean().item())
        normalized = self._normalize(gradient, 0, self.out_features)
        stats = self.apply_normalized_gradient(normalized)
        return RatchetUpdateStats(
            total_weights=stats.total_weights,
            positive_moves=stats.positive_moves,
            negative_moves=stats.negative_moves,
            blocked_positive_moves=stats.blocked_positive_moves,
            blocked_negative_moves=stats.blocked_negative_moves,
            gradient_rms_mean=rms_mean,
        )
```

- [ ] **Step 5: Plumb `rms_ema_beta` through config and model**

`config.py`: add `rms_ema_beta: float = 0.0` to `ExperimentConfig`; add `"rms_ema_beta"` to the `[ratchet]` allowed-keys set; in `model_config(...)` add `rms_ema_beta=self.rms_ema_beta,`. `model.py`: add `rms_ema_beta: float = 0.0` to `ModelConfig` and pass `rms_ema_beta=config.rms_ema_beta` in `_linear`'s `DiscreteRatchetLinear(...)` call.

- [ ] **Step 6: Add the CLI flag**

`cli.py`: after the `--trainable-scale` argument add:

```python
    train.add_argument("--rms-ema-beta", dest="rms_ema_beta", type=float, default=0.0)
```

In the train handler, alongside the existing `if args.trainable_scale:` block:

```python
        if args.rms_ema_beta:
            config = replace(config, rms_ema_beta=args.rms_ema_beta)
```

- [ ] **Step 7: Run the tests**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py tests/test_qat.py tests/test_cli.py -q`
Expected: PASS (new tests pass; no regression — the beta=0 path is bit-identical).

- [ ] **Step 8: Commit**

```bash
git add src/local_ai_training/config.py src/local_ai_training/model.py src/local_ai_training/ratchet.py src/local_ai_training/cli.py tests/test_ratchet.py tests/test_cli.py
git commit -m "feat: arm B — EMA per-row RMS (rms_ema_beta), 2nd-moment temporal smoothing"
```

---

### Task 2: Arm A — leaky pressure (`pressure_leak_period`)

**Files:**
- Modify: `src/local_ai_training/config.py` (`pressure_leak_period` field, `[ratchet]` allowed key, validation, `model_config` pass-through)
- Modify: `src/local_ai_training/model.py` (`ModelConfig.pressure_leak_period`, pass in `_linear`)
- Modify: `src/local_ai_training/ratchet.py` (`__init__` param + `self._update_count = 0`; `_maybe_leak_pressure()`; call it once per `ratchet_update()` before returning)
- Modify: `src/local_ai_training/cli.py` (`--pressure-leak-period` on `train`, apply via `replace`)
- Test: `tests/test_ratchet.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `pack_code_pressure` / `unpack_code_pressure`, `ratchet_update()`.
- Produces: `DiscreteRatchetLinear(..., pressure_leak_period: int = 0)`; private `self._maybe_leak_pressure() -> None` that, every `pressure_leak_period`-th `ratchet_update`, moves each nonzero pressure one unit toward zero and repacks. Config/CLI key `pressure_leak_period`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_ratchet.py`:

```python
def _force_pressure(layer, value):
    # set every weight's pressure to `value`, codes unchanged, via the packing helpers
    code, _ = unpack_code_pressure(layer.packed, layer.max_code)
    pressure = torch.full_like(code, value, dtype=torch.int8)
    layer.packed.copy_(pack_code_pressure(code, pressure, layer.max_code))


def test_pressure_leak_period_zero_never_leaks():
    layer = DiscreteRatchetLinear(8, 4, max_code=2, pressure_leak_period=0)
    _force_pressure(layer, 5)
    for _ in range(10):
        layer._maybe_leak_pressure()
    _, pressure = unpack_code_pressure(layer.packed, layer.max_code)
    assert int(pressure.min()) == 5 and int(pressure.max()) == 5  # untouched


def test_pressure_leak_fires_every_k_and_moves_toward_zero():
    layer = DiscreteRatchetLinear(8, 4, max_code=2, pressure_leak_period=3)
    _force_pressure(layer, 5)
    for _ in range(3):  # fires on the 3rd call (count 1,2,3 -> leak at 3)
        layer._maybe_leak_pressure()
    _, pressure = unpack_code_pressure(layer.packed, layer.max_code)
    assert int(pressure.max()) == 4  # one unit toward zero, exactly once


def test_pressure_leak_moves_negative_toward_zero_and_never_enlarges():
    layer = DiscreteRatchetLinear(8, 4, max_code=2, pressure_leak_period=1)
    _force_pressure(layer, -2)
    layer._maybe_leak_pressure()
    _, pressure = unpack_code_pressure(layer.packed, layer.max_code)
    assert int(pressure.min()) == -1  # toward zero, |pressure| shrank
    layer._validate_state()  # still within the nibble range
```

In `tests/test_cli.py`:

```python
def test_train_pressure_leak_period_flag_defaults_zero_and_parses():
    parser = build_parser()
    assert parser.parse_args(["train"]).pressure_leak_period == 0
    assert parser.parse_args(["train", "--pressure-leak-period", "4"]).pressure_leak_period == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py -k leak tests/test_cli.py -k leak -v`
Expected: FAIL — `_maybe_leak_pressure`, `pressure_leak_period`, and the CLI flag do not exist.

- [ ] **Step 3: Add the constructor param, counter, and leak method**

In `__init__`, add `pressure_leak_period: int = 0,` to the signature and:

```python
        if pressure_leak_period < 0:
            raise ValueError("pressure_leak_period must be a non-negative integer")
        self.pressure_leak_period = pressure_leak_period
        self._update_count = 0
```

Add the method:

```python
    @torch.no_grad()
    def _maybe_leak_pressure(self) -> None:
        # 1st-moment EMA analogue: every `pressure_leak_period`-th update, bleed each nonzero
        # pressure one unit toward zero so stale pressure fades (recent direction dominates).
        # Moving toward zero never enlarges |pressure|, so the nibble range is preserved.
        if self.pressure_leak_period <= 0:
            return
        self._update_count += 1
        if self._update_count % self.pressure_leak_period != 0:
            return
        code, pressure = unpack_code_pressure(self.packed, self.max_code)
        pressure = pressure - torch.sign(pressure).to(pressure.dtype)
        self.packed.copy_(pack_code_pressure(code, pressure, self.max_code))
```

- [ ] **Step 4: Call the leak once per `ratchet_update()`**

`ratchet_update()` has three early-return branches (fused; non-fp32 eager; fp32 eager). Restructure so each computes `stats` and falls through to a single tail that applies the leak and returns. Concretely, replace the three `return stats` / `return self.apply_weight_gradient(...)` sites so the method ends with:

```python
        self._maybe_leak_pressure()
        return stats
```

For the two eager branches, capture the result first: `stats = self.apply_weight_gradient(self._pending_weight_gradient)` (inside the existing `try/finally`), then fall through to the shared `_maybe_leak_pressure()` + `return stats` tail. The leak must run in all three modes exactly once per step.

- [ ] **Step 5: Plumb `pressure_leak_period` through config and model**

`config.py`: add `pressure_leak_period: int = 0` to `ExperimentConfig`; add `"pressure_leak_period"` to the `[ratchet]` allowed-keys set; in `model_config(...)` add `pressure_leak_period=self.pressure_leak_period,`. `model.py`: add `pressure_leak_period: int = 0` to `ModelConfig` and pass `pressure_leak_period=config.pressure_leak_period` in `_linear`.

- [ ] **Step 6: Add the CLI flag**

`cli.py`: after `--rms-ema-beta` add:

```python
    train.add_argument("--pressure-leak-period", dest="pressure_leak_period", type=int, default=0)
```

In the train handler:

```python
        if args.pressure_leak_period:
            config = replace(config, pressure_leak_period=args.pressure_leak_period)
```

- [ ] **Step 7: Run the tests + audit**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest tests/test_ratchet.py tests/test_cli.py -q`
Expected: PASS.

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat audit --model configs/scaleup_text8_25m_5k.toml --codes 5 --vocab-size 27`
Expected: `"violations": []` (both knobs off by default; audit path unaffected).

- [ ] **Step 8: Commit**

```bash
git add src/local_ai_training/config.py src/local_ai_training/model.py src/local_ai_training/ratchet.py src/local_ai_training/cli.py tests/test_ratchet.py tests/test_cli.py
git commit -m "feat: arm A — leaky pressure (pressure_leak_period), 1st-moment temporal smoothing"
```

---

### Task 3: Screening run + results note

**Files:**
- Create: `runs/text8-25m-updaterule/run_updaterule.sh` (driver; git-ignored execution artifact)
- Create: `docs/results/2026-06-23-update-rule-momentum.md` (committed)

**Interfaces:**
- Consumes: `lat train --codes 5 [--rms-ema-beta B] [--pressure-leak-period K]` from Tasks 1–2.
- Produces: per-arm `runs/text8-25m-updaterule/<arm>/seed-1337/metrics.csv`; a committed results note with the step-5000 gap table.

This is an experiment run, not a TDD unit. Acceptance: every arm reaches step 5000 and `lat audit` stays clean.

- [ ] **Step 1: Confirm dataset, GPUs, and coordinate with the int8 worktree**

Run: `ls data/text8/text8 && nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`
Expected: dataset exists; pick idle GPU(s). **The box crashed under dual-3090 load on 2026-06-23 (PSU/VRM); the `.worktrees/codex` int8-convergence work also wants the GPUs.** Coordinate: prefer a single GPU here, or confirm the other worktree is idle, before launching. If `data/text8/text8` is absent, run `... uv run lat dataset --which text8` first.

- [ ] **Step 2: Write the driver script**

Create `runs/text8-25m-updaterule/run_updaterule.sh`:

```bash
#!/usr/bin/env bash
# Update-rule momentum screen: frozen5 baseline + leaky-pressure (A) + EMA-RMS (B) + A+B.
# 5-state, 5k steps, seed 1337. Usage: bash run_updaterule.sh GPU_ID name:flags ...
#   flags is a |-separated list of extra `lat train` args, or "-" for none.
set -euo pipefail
cd /games/ailab/local-ai-training
CFG=configs/scaleup_text8_25m_5k.toml
DS=data/text8/text8
OUT=runs/text8-25m-updaterule
GPU=$1; shift

run() { # name  flags
  local name=$1 flags=$2
  [ "$flags" = "-" ] && flags=""
  echo "=== [$(date +%H:%M:%S)] START $name ($flags) GPU $GPU ==="
  env CUDA_VISIBLE_DEVICES="$GPU" MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
    uv run lat train --config "$CFG" --dataset-path "$DS" \
    --codes 5 --weight-mode ratchet --seed 1337 ${flags//|/ } \
    --output "$OUT/$name/seed-1337" > "$OUT/$name.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE  $name GPU $GPU ==="
}

for spec in "$@"; do
  IFS=: read -r name flags <<< "$spec"
  run "$name" "$flags"
done
echo "=== GPU $GPU COMPLETE [$(date +%H:%M:%S)] ==="
```

- [ ] **Step 3: Launch the screen (single GPU, serial — light load given the crash risk)**

Substitute the idle GPU index. Sweep a couple of values for each knob:

```bash
cd /games/ailab/local-ai-training
mkdir -p runs/text8-25m-updaterule
nohup bash runs/text8-25m-updaterule/run_updaterule.sh GPU_IDX \
  frozen5:- \
  A_leak4:"--pressure-leak-period|4" \
  A_leak16:"--pressure-leak-period|16" \
  B_beta0p9:"--rms-ema-beta|0.9" \
  B_beta0p99:"--rms-ema-beta|0.99" \
  AB:"--pressure-leak-period|16|--rms-ema-beta|0.99" \
  > runs/text8-25m-updaterule/run.out 2>&1 &
```

Expected: 6 arms, ~5000 steps each (~20 min/arm at ~67k tok/s), ~2h serial on one GPU.

- [ ] **Step 4: Verify every arm reached step 5000 and audit a trained arm**

After the driver prints "COMPLETE":

```bash
MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run python -c "
import csv, glob
for f in sorted(glob.glob('runs/text8-25m-updaterule/*/seed-1337/metrics.csv')):
    rows=list(csv.DictReader(open(f))); arm=f.split('/')[2]
    last=int(rows[-1]['step']); best=min(float(r['validation_loss']) for r in rows)
    print(f'{arm:12s} last {last}  best {best:.4f}'); assert last==5000, arm
"
```

Expected: all 6 arms report `last 5000`.

- [ ] **Step 5: Build the gap table (vs frozen5 baseline + QAT5 ceiling)**

```bash
MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run python - <<'PYEOF'
import csv
def bsf(path, step=5000):
    return min(float(r['validation_loss']) for r in csv.DictReader(open(path)) if int(r['step'])<=step)
qat5 = bsf('runs/text8-25m-qat/qat5/seed-1337/metrics.csv')          # FP-master ceiling
base = bsf('runs/text8-25m-updaterule/frozen5/seed-1337/metrics.csv')  # current-code baseline
print(f"QAT5 ceiling @5k: {qat5:.4f}   frozen5 baseline @5k: {base:.4f}   gap: {base-qat5:.4f}")
print("arm           best@5k   vs frozen5   % of gap closed")
import glob
for f in sorted(glob.glob('runs/text8-25m-updaterule/*/seed-1337/metrics.csv')):
    arm=f.split('/')[2]
    if arm=='frozen5': continue
    v=bsf(f); imp=base-v; pct=100*imp/(base-qat5) if base>qat5 else 0
    print(f"{arm:12s} {v:.4f}   {imp:+.4f}    {pct:+.0f}%")
PYEOF
```

Expected: a table; hypothesis is one or more of A/B/AB shows a positive `% of gap closed`.

- [ ] **Step 6: Write the results note**

Create `docs/results/2026-06-23-update-rule-momentum.md`, mirroring `docs/results/2026-06-23-adaptive-scale-ratchet.md`:
- **Question** — does temporal EMA smoothing of the update rule's moments (A: 1st, B: 2nd) close the master-free gap, with no per-weight state?
- **Method** — leaky pressure (`pressure_leak_period`) and EMA per-row RMS (`rms_ema_beta`); both off => baseline bit-identical; iso-everything 5-state 5k seed 1337; QAT5 ceiling + current-code frozen5 baseline.
- **Result** — the Step 5 table; per-arm best@5k and gap-closure; note the K/β swept.
- **Headline** — which (if any) arm moved the gap, A vs B vs A+B; honest null if flat.
- **Limitations** — screening (5k), single seed, integer-leak coarseness (Arm A bounded by the 4-bit pressure range), B adds per-row FP state (cheap, not zero). A positive arm graduates to a 30k confirmation.

- [ ] **Step 7: Commit the results note**

```bash
git add docs/results/2026-06-23-update-rule-momentum.md
git commit -m "docs: update-rule momentum screen — temporal EMA of the moments vs the master-free gap"
```

---

## Verification

- `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest` green (new equivalence + behavior tests pass; baseline path bit-identical when both knobs off).
- `uv run lat audit --model configs/scaleup_text8_25m_5k.toml --codes 5 --vocab-size 27` reports `"violations": []` (and stays clean with `rms_ema_beta`/`pressure_leak_period` set — per-row buffer is 1-D).
- `uv run ruff check` on changed files + `git diff --check`.
- All 6 arms reach step 5000; the results note reports the step-5000 gap table.
- Existing runs untouched: only `runs/text8-25m-updaterule/` is written.
