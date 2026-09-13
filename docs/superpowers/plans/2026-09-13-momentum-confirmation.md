# Momentum Grid and Converged Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sweep the two existing momentum knobs of the ratchet update rule at the 5k screen, then confirm the winner at 30k across three seeds against plain-ratchet, QAT-ceiling, and FP32 arms, and report whether the master-weight-free gap closes.

**Architecture:** No source change to the update rule. Two committed driver scripts under `scripts/` build matched `lat train` commands, run them under the existing thermal guard (promoted from `runs/` into `scripts/` so it is versioned), record a manifest with source and data hashes, and summarize `metrics.csv` files into tables. Results notes are written from the summaries.

**Tech Stack:** Python 3.12, PyTorch, `uv`, pytest, the `lat` CLI, `nvidia-smi` telemetry via `scripts/thermal_guard.py`.

**Spec:** `docs/superpowers/specs/2026-09-13-momentum-confirmation-design.md`

## Global Constraints

- No source change to `src/local_ai_training/ratchet.py` or any update-rule code; both knobs (`pressure_leak_period`, `rms_ema_beta`) already exist and default off.
- No new per-weight state. `lat audit` reports zero violations on every arm; persistent bytes are reported including the RMS-EMA buffer.
- Matched arms share seed, logical initialization, batch schedule, evaluation batches, and token budget; only the arm-defining knobs differ.
- Corrected source only (`948ff62` or later); fresh runs, no resume, no retries with changed settings; failed runs preserved in place.
- Phase 1: `configs/scaleup_text8_25m_5k.toml`, seed 1337, 15 codes, leak in {8, 12, 16, 24, 32} by beta in {0.9, 0.99, 0.999}, plus plain-ratchet and QAT references: 17 runs, one card, serial.
- Phase 2: `configs/scaleup_text8_25m_30k.toml`, seeds 1337/1338/1339, 15 codes, arms momentum/plain/QAT/FP32: 12 runs, two per-card queues.
- Decision thresholds: "flipped" is a three-seed mean momentum-minus-QAT at or below 0.03 nats with momentum below plain on every seed.
- `deterministic_attention` stays off (head size 64 at 256 tokens is repeatable).
- Commands run from the repo root with `.venv/bin/python` / `.venv/bin/lat`; `runs/` is git-ignored, so drivers and tests live in `scripts/` and `tests/`.
- Commits end with the session's attribution lines.

---

### Task 1: Promote the thermal guard into `scripts/`

**Files:**
- Create: `scripts/thermal_guard.py` (copy of `runs/integrity-monitor-2026-09-13/thermal_guard.py`)
- Create: `tests/test_thermal_guard.py` (copy of `runs/integrity-monitor-2026-09-13/test_thermal_guard.py`)
- Modify: `scripts/thermal_guard.py` lines defining `HELPER_DIR`, `THERMAL_EVENTS_PATH`, `lock_path`, `status_path`

**Interfaces:**
- Consumes: nothing new.
- Produces: `scripts.thermal_guard.run_training(cmd: Sequence[object], log: IO[object], env: Mapping[str, str], gpu: int) -> subprocess.CompletedProcess` (unchanged signature) and `scripts.thermal_guard.guard_dir() -> Path`, the directory holding locks, status files, and `thermal-events.jsonl`, taken from the `LAT_GUARD_DIR` environment variable and defaulting to `runs/guard`.

- [ ] **Step 1: Copy the files**

```bash
cp runs/integrity-monitor-2026-09-13/thermal_guard.py scripts/thermal_guard.py
cp runs/integrity-monitor-2026-09-13/test_thermal_guard.py tests/test_thermal_guard.py
```

- [ ] **Step 2: Write the failing test for the configurable guard directory**

Append to `tests/test_thermal_guard.py`:

```python
def test_guard_dir_defaults_to_runs_guard_and_honours_env(monkeypatch, tmp_path):
    from scripts import thermal_guard

    monkeypatch.delenv("LAT_GUARD_DIR", raising=False)
    assert thermal_guard.guard_dir() == Path("runs/guard").resolve()
    assert thermal_guard.lock_path(1) == Path("runs/guard").resolve() / "training-gpu-1.lock"
    monkeypatch.setenv("LAT_GUARD_DIR", str(tmp_path))
    assert thermal_guard.guard_dir() == tmp_path.resolve()
    assert thermal_guard.status_path(0) == tmp_path.resolve() / "status-gpu-0.json"
```

Also change every `import thermal_guard` / `from thermal_guard import ...` in the copied test to `from scripts import thermal_guard` / `from scripts.thermal_guard import ...`, and add `from pathlib import Path` if missing.

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_thermal_guard.py -q`
Expected: FAIL with `AttributeError: module 'scripts.thermal_guard' has no attribute 'guard_dir'`

- [ ] **Step 4: Make the directory configurable**

In `scripts/thermal_guard.py` replace

```python
HELPER_DIR = Path(__file__).resolve().parent
```

and the three uses (`THERMAL_EVENTS_PATH`, `lock_path`, `status_path`) with:

```python
def guard_dir() -> Path:
    """Directory for locks, per-GPU status files, and thermal-events.jsonl."""
    directory = Path(os.environ.get("LAT_GUARD_DIR", "runs/guard")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def thermal_events_path() -> Path:
    return guard_dir() / "thermal-events.jsonl"


def lock_path(gpu: int) -> Path:
    return guard_dir() / f"training-gpu-{gpu}.lock"


def status_path(gpu: int) -> Path:
    return guard_dir() / f"status-gpu-{gpu}.json"
```

Then replace every remaining `THERMAL_EVENTS_PATH` reference with `thermal_events_path()` (there is one, in `_append_thermal_event`). Leave all thresholds, sampling, admission, and stop logic untouched.

- [ ] **Step 5: Run the tests and lint**

Run: `.venv/bin/python -m pytest tests/test_thermal_guard.py -q && .venv/bin/ruff check scripts/thermal_guard.py tests/test_thermal_guard.py`
Expected: all tests PASS (the original 11 plus the new one), ruff clean. Fix any ruff findings in the copied file (line length 100, import order) without changing behavior.

- [ ] **Step 6: Commit**

```bash
git add scripts/thermal_guard.py tests/test_thermal_guard.py
git commit -m "scripts: version the per-GPU thermal guard with a configurable state dir"
```

---

### Task 2: Phase 1 driver `scripts/momentum_grid.py`

**Files:**
- Create: `scripts/momentum_grid.py`
- Test: `tests/test_momentum_grid.py`

**Interfaces:**
- Consumes: `scripts.thermal_guard.run_training`, `local_ai_training.plotting.plot_comparison`.
- Produces:
  - `@dataclass(frozen=True) Arm(name: str, weight_mode: str, leak: int, beta: float)`
  - `grid_arms() -> list[Arm]` (17 arms: `plain`, `qat`, and `leak{K}-beta{B}` for the grid)
  - `arm_command(arm: Arm, *, config: Path, dataset: Path, output: Path, seed: int, lat: Path) -> list[str]`
  - `best_so_far(metrics_csv: Path) -> dict` with keys `best`, `best_step`, `final`, `trace` (list of `(step, validation_loss)`)
  - `summarize(root: Path) -> dict` writing and returning `results.json`
  - `main(argv: list[str] | None = None) -> int` with flags `--root`, `--gpu`, `--dataset`, `--config`, `--seed`, `--dry-run`, `--summarize-only`

- [ ] **Step 1: Write the failing tests**

`tests/test_momentum_grid.py`:

```python
import csv
import json
from pathlib import Path

from scripts.momentum_grid import Arm, arm_command, best_so_far, grid_arms, summarize


def test_grid_has_seventeen_unique_matched_arms():
    arms = grid_arms()
    assert len(arms) == 17
    assert len({arm.name for arm in arms}) == 17
    assert Arm("plain", "ratchet", 0, 0.0) in arms
    assert Arm("qat", "qat", 0, 0.0) in arms
    momentum = [arm for arm in arms if arm.weight_mode == "ratchet" and arm.leak]
    assert sorted({arm.leak for arm in momentum}) == [8, 12, 16, 24, 32]
    assert sorted({arm.beta for arm in momentum}) == [0.9, 0.99, 0.999]
    assert len(momentum) == 15


def test_arm_command_carries_only_the_arm_defining_knobs(tmp_path):
    common = dict(config=Path("c.toml"), dataset=Path("d"), output=tmp_path, seed=1337, lat=Path("lat"))
    plain = arm_command(Arm("plain", "ratchet", 0, 0.0), **common)
    momentum = arm_command(Arm("leak16-beta0.99", "ratchet", 16, 0.99), **common)
    qat = arm_command(Arm("qat", "qat", 0, 0.0), **common)
    for cmd in (plain, momentum, qat):
        assert cmd[:2] == ["lat", "train"] and "--codes" in cmd and cmd[cmd.index("--codes") + 1] == "15"
        assert cmd[cmd.index("--seed") + 1] == "1337"
    assert "--pressure-leak-period" not in plain and "--rms-ema-beta" not in plain
    assert momentum[momentum.index("--pressure-leak-period") + 1] == "16"
    assert momentum[momentum.index("--rms-ema-beta") + 1] == "0.99"
    assert qat[qat.index("--weight-mode") + 1] == "qat"
    assert "--deterministic-attention" not in momentum


def _write_metrics(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "validation_loss", "saturated_percent"])
        writer.writeheader()
        for step, loss in rows:
            writer.writerow({"step": step, "validation_loss": loss, "saturated_percent": 10.0})


def test_best_so_far_tracks_minimum_and_final(tmp_path):
    _write_metrics(tmp_path / "metrics.csv", [(0, 3.0), (200, 1.5), (400, 1.6), (600, 1.55)])
    result = best_so_far(tmp_path / "metrics.csv")
    assert result["best"] == 1.5 and result["best_step"] == 200 and result["final"] == 1.55
    assert result["trace"] == [(0, 3.0), (200, 1.5), (400, 1.6), (600, 1.55)]


def test_summarize_ranks_arms_and_reports_gap_fraction(tmp_path):
    _write_metrics(tmp_path / "plain" / "metrics.csv", [(0, 3.0), (5000, 1.40)])
    _write_metrics(tmp_path / "qat" / "metrics.csv", [(0, 3.0), (5000, 1.20)])
    _write_metrics(tmp_path / "leak16-beta0.99" / "metrics.csv", [(0, 3.0), (5000, 1.25)])
    result = summarize(tmp_path)
    assert result["winner"] == "leak16-beta0.99"
    assert abs(result["arms"]["leak16-beta0.99"]["gap_closed"] - 0.75) < 1e-9
    assert json.loads((tmp_path / "results.json").read_text())["winner"] == "leak16-beta0.99"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_momentum_grid.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.momentum_grid'`

- [ ] **Step 3: Write the driver**

`scripts/momentum_grid.py`:

```python
"""Phase 1 of the momentum confirmation: grid the two update-rule knobs at the 5k screen.

Spec: docs/superpowers/specs/2026-09-13-momentum-confirmation-design.md. Runs 17 matched arms
serially on one GPU under the thermal guard, records a manifest, and summarizes results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.thermal_guard import run_training

REPO = Path(__file__).resolve().parent.parent
LEAKS = (8, 12, 16, 24, 32)
BETAS = (0.9, 0.99, 0.999)
CODES = 15


@dataclass(frozen=True)
class Arm:
    name: str
    weight_mode: str
    leak: int
    beta: float


def grid_arms() -> list[Arm]:
    arms = [Arm("plain", "ratchet", 0, 0.0), Arm("qat", "qat", 0, 0.0)]
    arms += [Arm(f"leak{k}-beta{b}", "ratchet", k, b) for k in LEAKS for b in BETAS]
    return arms


def arm_command(
    arm: Arm, *, config: Path, dataset: Path, output: Path, seed: int, lat: Path
) -> list[str]:
    cmd = [
        str(lat), "train", "--config", str(config), "--dataset-path", str(dataset),
        "--seed", str(seed), "--codes", str(CODES), "--weight-mode", arm.weight_mode,
        "--output", str(output),
    ]
    if arm.leak:
        cmd += ["--pressure-leak-period", str(arm.leak)]
    if arm.beta:
        cmd += ["--rms-ema-beta", str(arm.beta)]
    return cmd


def best_so_far(metrics_csv: Path) -> dict:
    trace = []
    with metrics_csv.open() as handle:
        for row in csv.DictReader(handle):
            trace.append((int(row["step"]), float(row["validation_loss"])))
    best_step, best = min(trace, key=lambda item: item[1])
    return {"best": best, "best_step": best_step, "final": trace[-1][1], "trace": trace}


def summarize(root: Path) -> dict:
    arms = {}
    for directory in sorted(p for p in root.iterdir() if (p / "metrics.csv").exists()):
        arms[directory.name] = best_so_far(directory / "metrics.csv")
    plain, qat = arms.get("plain"), arms.get("qat")
    for name, record in arms.items():
        if plain and qat and name not in ("plain", "qat"):
            gap = plain["best"] - qat["best"]
            record["gap_closed"] = (plain["best"] - record["best"]) / gap if gap else None
    candidates = [n for n in arms if n not in ("plain", "qat")]
    winner = min(candidates, key=lambda n: arms[n]["best"]) if candidates else None
    result = {"winner": winner, "arms": arms}
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, args: argparse.Namespace, arms: list[Arm]) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    payload = {
        "source_commit": commit, "git_status": status, "config": str(args.config),
        "config_sha256": _sha256(args.config), "dataset_sha256": _sha256(args.dataset),
        "seed": args.seed, "codes": CODES, "gpu": args.gpu, "arms": [asdict(a) for a in arms],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "runs/momentum-grid-2026-09-13")
    parser.add_argument("--gpu", type=int, default=0, help="physical GPU index")
    parser.add_argument("--config", type=Path, default=REPO / "configs/scaleup_text8_25m_5k.toml")
    parser.add_argument("--dataset", type=Path, default=REPO / "data/text8/text8")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    arms = grid_arms()
    if args.summarize_only:
        print(json.dumps(summarize(args.root)["winner"]))
        return 0
    lat = REPO / ".venv/bin/lat"
    commands = {
        arm.name: arm_command(
            arm, config=args.config, dataset=args.dataset, output=args.root / arm.name,
            seed=args.seed, lat=lat,
        )
        for arm in arms
    }
    if args.dry_run:
        for name, cmd in commands.items():
            print(name, " ".join(cmd))
        return 0
    if (args.root / "manifest.json").exists():
        raise SystemExit(f"{args.root} already has a manifest; refusing to rerun into it")
    args.root.mkdir(parents=True)
    _manifest(args.root, args, arms)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    with (args.root / "study.log").open("a") as study:
        for name, cmd in commands.items():
            print(f"START {name}", file=study, flush=True)
            with (args.root / f"{name}.log").open("w") as log:
                result = run_training(cmd, log=log, env=env, gpu=args.gpu)
            print(f"EXIT {name} {result.returncode}", file=study, flush=True)
            if result.returncode != 0:
                print(f"STOPPED: {name} failed", file=study, flush=True)
                return 1
    summary = summarize(args.root)
    print("winner:", summary["winner"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: `run_training` sets `CUDA_VISIBLE_DEVICES` itself if the copied guard does; check `scripts/thermal_guard.py` for how the env is passed to the child and do not set it twice with conflicting values. If the guard already pins the device from its `gpu` argument, drop `CUDA_VISIBLE_DEVICES` from `env` above.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_momentum_grid.py -q && .venv/bin/ruff check scripts/momentum_grid.py tests/test_momentum_grid.py`
Expected: 4 passed, ruff clean.

- [ ] **Step 5: Dry-run against the real config**

Run: `.venv/bin/python -m scripts.momentum_grid --dry-run | head -3`
Expected: three lines beginning `plain`, `qat`, `leak8-beta0.9`, each a full `lat train` command with `--codes 15` and the text8 dataset path.

- [ ] **Step 6: Commit**

```bash
git add scripts/momentum_grid.py tests/test_momentum_grid.py
git commit -m "scripts: momentum knob grid driver for the 5k screen"
```

---

### Task 3: Phase 2 driver `scripts/momentum_30k.py`

**Files:**
- Create: `scripts/momentum_30k.py`
- Test: `tests/test_momentum_30k.py`

**Interfaces:**
- Consumes: `scripts.momentum_grid.Arm`, `scripts.momentum_grid.arm_command`, `scripts.momentum_grid.best_so_far`, `scripts.thermal_guard.run_training`.
- Produces:
  - `confirmation_arms(leak: int, beta: float, seeds: tuple[int, ...]) -> list[tuple[Arm, int]]` (12 `(arm, seed)` pairs; arm names `momentum`, `plain`, `qat`, `fp32`; run directory name `f"{arm.name}-seed{seed}"`)
  - `assign_queues(pairs, gpus: tuple[int, ...]) -> dict[int, list[tuple[Arm, int]]]` (round-robin by pair order so each card gets every arm type)
  - `summarize(root: Path) -> dict` with per-seed `best`/`final`/`last_four_mean`, three-seed `mean`/`sd`, and `gaps`: `momentum_minus_qat`, `momentum_minus_plain`, `qat_minus_fp32`, `fraction_recovered`, plus `verdict` in {`flipped`, `partial`, `not_confirmed`}
  - `main(argv)` with flags `--root`, `--leak`, `--beta`, `--gpus`, `--queue-gpu` (internal: run one card's queue), `--dry-run`, `--summarize-only`

- [ ] **Step 1: Write the failing tests**

`tests/test_momentum_30k.py`:

```python
import csv
from pathlib import Path

from scripts.momentum_30k import assign_queues, confirmation_arms, summarize


def test_confirmation_arms_are_four_per_seed_and_fp32_has_no_codes_knobs():
    pairs = confirmation_arms(16, 0.99, (1337, 1338, 1339))
    assert len(pairs) == 12
    names = sorted({arm.name for arm, _ in pairs})
    assert names == ["fp32", "momentum", "plain", "qat"]
    momentum = next(arm for arm, _ in pairs if arm.name == "momentum")
    assert (momentum.leak, momentum.beta, momentum.weight_mode) == (16, 0.99, "ratchet")
    fp32 = next(arm for arm, _ in pairs if arm.name == "fp32")
    assert fp32.weight_mode == "fp32" and fp32.leak == 0 and fp32.beta == 0.0


def test_assign_queues_gives_every_card_every_arm_type():
    pairs = confirmation_arms(16, 0.99, (1337, 1338, 1339))
    queues = assign_queues(pairs, (0, 1))
    assert sorted(len(q) for q in queues.values()) == [6, 6]
    for queue in queues.values():
        assert {arm.name for arm, _ in queue} == {"fp32", "momentum", "plain", "qat"}


def _write(path: Path, losses):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "validation_loss"])
        writer.writeheader()
        for index, loss in enumerate(losses):
            writer.writerow({"step": index * 200, "validation_loss": loss})


def test_summarize_reports_gaps_and_verdict(tmp_path):
    for seed in (1337, 1338, 1339):
        _write(tmp_path / f"fp32-seed{seed}" / "metrics.csv", [3.0, 1.00, 0.98, 0.97, 0.97, 0.97])
        _write(tmp_path / f"qat-seed{seed}" / "metrics.csv", [3.0, 1.10, 1.05, 1.02, 1.02, 1.02])
        _write(tmp_path / f"plain-seed{seed}" / "metrics.csv", [3.0, 1.30, 1.20, 1.16, 1.16, 1.16])
        _write(tmp_path / f"momentum-seed{seed}" / "metrics.csv", [3.0, 1.15, 1.06, 1.04, 1.04, 1.04])
    result = summarize(tmp_path)
    gaps = result["gaps"]
    assert abs(gaps["momentum_minus_qat"]["mean"] - 0.02) < 1e-9
    assert abs(gaps["momentum_minus_plain"]["mean"] + 0.12) < 1e-9
    assert abs(gaps["fraction_recovered"]["mean"] - (0.12 / 0.14)) < 1e-9
    assert result["verdict"] == "flipped"
    assert result["per_seed"][1337]["momentum"]["last_four_mean"] == 1.04
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_momentum_30k.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.momentum_30k'`

- [ ] **Step 3: Write the driver**

`scripts/momentum_30k.py`:

```python
"""Phase 2 of the momentum confirmation: converged 30k, three seeds, four matched arms.

Spec: docs/superpowers/specs/2026-09-13-momentum-confirmation-design.md. Splits 12 runs across
two per-card queues (each a subprocess of this script with --queue-gpu) under the thermal guard.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from scripts.momentum_grid import CODES, Arm, arm_command, best_so_far
from scripts.thermal_guard import run_training

REPO = Path(__file__).resolve().parent.parent
SEEDS = (1337, 1338, 1339)
FLIP_THRESHOLD = 0.03


def confirmation_arms(leak: int, beta: float, seeds: tuple[int, ...]) -> list[tuple[Arm, int]]:
    arms = [
        Arm("momentum", "ratchet", leak, beta), Arm("plain", "ratchet", 0, 0.0),
        Arm("qat", "qat", 0, 0.0), Arm("fp32", "fp32", 0, 0.0),
    ]
    return [(arm, seed) for seed in seeds for arm in arms]


def assign_queues(pairs, gpus: tuple[int, ...]) -> dict[int, list[tuple[Arm, int]]]:
    queues: dict[int, list[tuple[Arm, int]]] = {gpu: [] for gpu in gpus}
    for index, pair in enumerate(pairs):
        queues[gpus[index % len(gpus)]].append(pair)
    return queues


def _run_name(arm: Arm, seed: int) -> str:
    return f"{arm.name}-seed{seed}"


def _command(arm: Arm, seed: int, root: Path, config: Path, dataset: Path) -> list[str]:
    cmd = arm_command(
        arm, config=config, dataset=dataset, output=root / _run_name(arm, seed), seed=seed,
        lat=REPO / ".venv/bin/lat",
    )
    if arm.weight_mode == "fp32":  # FP32 control takes no code count
        index = cmd.index("--codes")
        del cmd[index:index + 2]
    return cmd


def _stats(values: list[float]) -> dict:
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def summarize(root: Path) -> dict:
    per_seed: dict[int, dict[str, dict]] = {}
    for directory in sorted(p for p in root.iterdir() if (p / "metrics.csv").exists()):
        name, seed = directory.name.rsplit("-seed", 1)
        record = best_so_far(directory / "metrics.csv")
        losses = [loss for _, loss in record["trace"]]
        record["last_four_mean"] = statistics.fmean(losses[-4:])
        per_seed.setdefault(int(seed), {})[name] = record
    seeds = sorted(s for s, arms in per_seed.items() if {"momentum", "plain", "qat", "fp32"} <= arms.keys())
    best = lambda arm: [per_seed[s][arm]["best"] for s in seeds]  # noqa: E731
    momentum, plain, qat, fp32 = best("momentum"), best("plain"), best("qat"), best("fp32")
    gaps = {
        "momentum_minus_qat": _stats([m - q for m, q in zip(momentum, qat, strict=True)]),
        "momentum_minus_plain": _stats([m - p for m, p in zip(momentum, plain, strict=True)]),
        "qat_minus_fp32": _stats([q - f for q, f in zip(qat, fp32, strict=True)]),
        "fraction_recovered": _stats([(p - m) / (p - q) for m, p, q in zip(momentum, plain, qat, strict=True)]),
    } if seeds else {}
    verdict = None
    if seeds:
        beats_plain = all(m < p for m, p in zip(momentum, plain, strict=True))
        if beats_plain and gaps["momentum_minus_qat"]["mean"] <= FLIP_THRESHOLD:
            verdict = "flipped"
        elif beats_plain:
            verdict = "partial"
        else:
            verdict = "not_confirmed"
    result = {"seeds": seeds, "per_seed": per_seed, "gaps": gaps, "verdict": verdict}
    (root / "results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def _run_queue(gpu: int, root: Path, pairs, config: Path, dataset: Path) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with (root / f"queue-gpu{gpu}.log").open("a") as queue_log:
        for arm, seed in pairs:
            name = _run_name(arm, seed)
            print(f"START {name}", file=queue_log, flush=True)
            with (root / f"{name}.log").open("w") as log:
                result = run_training(_command(arm, seed, root, config, dataset), log=log, env=env, gpu=gpu)
            print(f"EXIT {name} {result.returncode}", file=queue_log, flush=True)
            if result.returncode != 0:
                print(f"STOPPED: {name} failed", file=queue_log, flush=True)
                return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "runs/momentum-30k-2026-09-13")
    parser.add_argument("--leak", type=int, required=False)
    parser.add_argument("--beta", type=float, required=False)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--queue-gpu", type=int, help="internal: run this card's queue only")
    parser.add_argument("--config", type=Path, default=REPO / "configs/scaleup_text8_25m_30k.toml")
    parser.add_argument("--dataset", type=Path, default=REPO / "data/text8/text8")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize_only:
        print(json.dumps(summarize(args.root)["verdict"]))
        return 0
    if args.leak is None or args.beta is None:
        parser.error("--leak and --beta (the Phase 1 winner) are required")
    gpus = tuple(int(g) for g in args.gpus.split(","))
    queues = assign_queues(confirmation_arms(args.leak, args.beta, SEEDS), gpus)
    if args.queue_gpu is not None:
        return _run_queue(args.queue_gpu, args.root, queues[args.queue_gpu], args.config, args.dataset)
    if args.dry_run:
        for gpu, pairs in queues.items():
            for arm, seed in pairs:
                print(gpu, " ".join(_command(arm, seed, args.root, args.config, args.dataset)))
        return 0
    if (args.root / "manifest.json").exists():
        raise SystemExit(f"{args.root} already has a manifest; refusing to rerun into it")
    args.root.mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        "source_commit": commit, "leak": args.leak, "beta": args.beta, "codes": CODES, "seeds": SEEDS,
        "queues": {gpu: [[asdict(arm), seed] for arm, seed in pairs] for gpu, pairs in queues.items()},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    children = [
        subprocess.Popen(
            [sys.executable, "-m", "scripts.momentum_30k", "--root", str(args.root), "--leak", str(args.leak),
             "--beta", str(args.beta), "--gpus", args.gpus, "--queue-gpu", str(gpu)],
            cwd=REPO,
        )
        for gpu in gpus
    ]
    codes = [child.wait() for child in children]
    summary = summarize(args.root)
    print("verdict:", summary["verdict"])
    return 0 if all(code == 0 for code in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_momentum_30k.py tests/test_momentum_grid.py -q && .venv/bin/ruff check scripts/ tests/`
Expected: 7 passed, ruff clean (wrap any line over 100 characters).

- [ ] **Step 5: Dry-run**

Run: `.venv/bin/python -m scripts.momentum_30k --leak 16 --beta 0.99 --dry-run | head -4`
Expected: four lines, alternating GPU 0 and 1, the first being `momentum-seed1337` with `--pressure-leak-period 16 --rms-ema-beta 0.99`, and an `fp32-seed1337` line without `--codes`.

- [ ] **Step 6: Commit**

```bash
git add scripts/momentum_30k.py tests/test_momentum_30k.py
git commit -m "scripts: 30k three-seed momentum confirmation driver with per-card queues"
```

---

### Task 4: Run Phase 1 and write the grid note

**Files:**
- Create: `docs/results/2026-09-13-momentum-grid.md`
- Modify: `docs/README.md` (reading order), `docs/ROADMAP.md` (backlog item 1 status)

- [ ] **Step 1: Confirm the card and launch**

Run: `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`
Pick the card with the most free memory (the guard needs 8,000 MiB free for three consecutive samples). Then:

```bash
LAT_GUARD_DIR=runs/momentum-grid-2026-09-13/guard nohup .venv/bin/python -m scripts.momentum_grid --gpu <N> > runs/momentum-grid-2026-09-13.launch.log 2>&1 &
```

- [ ] **Step 2: Monitor to completion**

Watch `runs/momentum-grid-2026-09-13/study.log` for `EXIT <name> <code>` lines; any nonzero code stops the driver (the failed run's directory and log stay in place; inspect before relaunching, never relaunch into the same root). Expect about 18 minutes per arm on a dedicated card, longer when shared.

- [ ] **Step 3: Summarize and audit**

Run: `.venv/bin/python -m scripts.momentum_grid --summarize-only` and, for the winner, `.venv/bin/lat audit --model configs/scaleup_text8_25m_5k.toml --codes 15` (zero violations). Read `results.json`.

- [ ] **Step 4: Write `docs/results/2026-09-13-momentum-grid.md`**

Sections, all numbers from `results.json`: Outcome (winner cell, best@5k, gap fraction versus the plain and QAT references), Protocol (source commit, config and dataset hashes from `manifest.json`, seed, codes, card, sharing conditions), a 5-by-3 table of best@5k by leak and beta, a table of the two references, the step-200-onward trace check (does the momentum advantage appear early and never reverse, as in June), Limitations (one seed, 5k, shared card if applicable), and the Phase 2 launch command with the chosen knobs.

- [ ] **Step 5: Link and commit**

Add the note to `docs/README.md` under "Reading order for results" after the provenance reruns entry, and set `docs/ROADMAP.md` backlog item 1 status to "grid complete, 30k confirmation running". Then:

```bash
git add docs/results/2026-09-13-momentum-grid.md docs/README.md docs/ROADMAP.md
git commit -m "results: momentum knob grid at the 5k screen"
```

---

### Task 5: Run Phase 2 and write the confirmation note

**Files:**
- Create: `docs/results/2026-09-14-momentum-confirmation.md`
- Modify: `docs/ROADMAP.md` (Current Position and backlog item 1), `docs/README.md`, `README.md` (status paragraph if the verdict is "flipped")

- [ ] **Step 1: Launch both queues**

```bash
LAT_GUARD_DIR=runs/momentum-30k-2026-09-13/guard nohup .venv/bin/python -m scripts.momentum_30k --leak <K> --beta <B> --gpus 0,1 > runs/momentum-30k-2026-09-13.launch.log 2>&1 &
```

Use the Phase 1 winner. If one card is busy with the user's other work, pass `--gpus <free card>` and accept a serial 21-hour run rather than contending.

- [ ] **Step 2: Monitor to completion**

Watch `runs/momentum-30k-2026-09-13/queue-gpu0.log` and `queue-gpu1.log` for `EXIT` lines and the guard's `status-gpu-*.json` for temperature. Expect about 1.75 hours per run per card, roughly 10.5 hours total with both cards.

- [ ] **Step 3: Summarize, audit, and plot**

Run: `.venv/bin/python -m scripts.momentum_30k --summarize-only`; `.venv/bin/lat audit` as in Task 4; and `.venv/bin/python -c "from local_ai_training.plotting import plot_comparison; plot_comparison('runs/momentum-30k-2026-09-13', 'runs/momentum-30k-2026-09-13/validation.png')"` (if `plot_comparison` expects a different layout, write a 20-line matplotlib script in the run directory instead and say so in the note).

- [ ] **Step 4: Write `docs/results/2026-09-14-momentum-confirmation.md`**

Sections: Outcome (the verdict word from `results.json` and the three-seed mean momentum-minus-QAT with SD, stated in one sentence), Protocol and provenance (commit, hashes, cards, guard events, any failures preserved), a per-seed table with best, final, and last-four mean for all four arms, a gaps table (momentum minus QAT, momentum minus plain, QAT minus FP32, fraction recovered; mean and SD), saturation and persistent-bytes per arm including the EMA buffer, tail-slope check (last 5k steps of each momentum and QAT arm: still descending, flat, or crossing), Limitations (single model size and corpus, BF16 autocast loop, no throughput claim), and What this changes (if "flipped": the master-weight-free penalty is recoverable at this scale and the fine-tuning pilot is next; if "partial": the residual number for the next update-rule lever; if "not confirmed": the screening artifact and the trace).

- [ ] **Step 5: Update roadmap, docs index, README, and commit**

`docs/ROADMAP.md` Current Position: replace "How far does the quality scale?" context with the verdict and the number. Backlog item 1: mark done and point at the note. `docs/README.md` reading order and, if flipped, the "Status at a glance" bullet. `README.md`: if flipped, one sentence in the intro that the momentum rule recovers the master-weight-free gap at 25M; otherwise no change.

```bash
git add docs/results/2026-09-14-momentum-confirmation.md docs/ROADMAP.md docs/README.md README.md
git commit -m "results: converged three-seed momentum confirmation"
```

---

## Self-review

- Spec coverage: Phase 1 grid and references (Task 2, 4); Phase 2 arms, queues, metrics, comparisons, verdict thresholds (Task 3, 5); constraints on no source change, audit, matched arms, fresh runs, failed-run preservation (Global Constraints, drivers refuse to rerun into an existing root, nonzero exits stop queues); outputs and docs (Tasks 4, 5); thermal guard reuse (Task 1).
- Placeholder scan: none; every code step has code, every doc step lists its sections and data sources.
- Type consistency: `Arm(name, weight_mode, leak, beta)` and `arm_command(arm, *, config, dataset, output, seed, lat)` are defined in Task 2 and consumed unchanged in Task 3; `best_so_far` returns `best`, `best_step`, `final`, `trace` in both consumers; `run_training(cmd, log, env, gpu)` matches the existing guard signature.
