# Group Learning C: Yardstick and Spine Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put a training yardstick into the runner (fixed-window scores, train seconds, VRAM, joules, a contention flag) and use it for the first like-for-like cost comparison: the 16M hybrid, the RoPE transformer and the zero-head local-attention cell, one card, one recipe, bf16, 16,000 steps.

**Architecture:** Every change to `scripts/run_sparsela.py` and `scripts/cross_eval.py` is a new flag that defaults off; with the flags absent both scripts write what they wrote before. Two instruments with testable arithmetic live in `flybrain/meter.py` (`GpuMeter`: energy, hottest sample, contention, the 85 C stop; `SplitClock`: train seconds against eval and save seconds) and `fixed_windows` joins `batch_windows`. `scripts/run_spine.sh` launches an arm, `scripts/spine_report.py` renders the table and applies the two decision rules, and the GPU tasks are exact commands with the observations to expect.

**Tech Stack:** Python 3.11, PyTorch 2.14, nvidia-smi, bash, `systemd-run --user`, git worktree, pytest, uv. All tests run on CPU.

**Spec:** docs/superpowers/specs/2026-10-03-group-learning-design.md (sections 1.5, 1.6, Run rules)

**Placement note:** `fixed_windows` lives in `flybrain/data/text.py` next to `batch_windows` (the spec now says so too; it first listed `files.py`), so this plan does not depend on the probes plan. Cold means the mean over a whole window and warm the mean over positions `seq_len // 2` onward, everywhere in this plan.

## Global Constraints

- Branch `feat/group-learning`. Never push. Commit with the repo's configured identity; end each commit message with the Co-Authored-By trailer your session specifies.
- With every new flag absent, scripts/run_sparsela.py and scripts/cross_eval.py behave and write exactly as before.
- Tests are CPU-only and must not need a GPU, the network or the real corpus.
- GPU 1: run `systemctl --user stop qwen-server.service` before a GPU job and `systemctl --user start qwen-server.service` when the queue is empty. GPU 0 only between 22:00 and 05:00.
- Before any GPU command check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`; do not start on a card that has another compute process.
- The power cap stays at 150 W. Kill jobs by explicit PID only (never `pkill -f`).
- New run names only; the runner skips a name whose results JSON exists.
- Do not edit README.md or notes/RESULTS.md. Do not start fleet.service.

## Review Focus

1. **nvidia-smi prints `[N/A]`, a wrong field count, nothing, or fails** (driver busy, binary missing): `parse` raises `ValueError`, the thread counts a missing sample, 60 s without a good one sets `telemetry_lost`, and the run carries on. Pinned by Task 2 `test_parse_refuses_anything_it_cannot_read` and Task 3 `test_sixty_seconds_without_a_sample_marks_telemetry_lost_and_nothing_else`, `test_a_sampler_that_always_fails_leaves_zeros_and_a_live_run`.
2. **A second compute process appears mid-run** (the model server reloading 19.5 GB, as on 2026-09-17): `contended` turns true and stays true; the run is not stopped; the run's own PID and processes on the other card never count. Pinned by Task 3 `test_a_second_process_sets_contended_for_good_and_never_stops_the_run` and Task 2 `test_parse_picks_the_run_card_and_drops_the_run_itself`; Task 11 checks the card per GPU before launch.
3. **`--meter` on CPU, and every flag absent**: no thread, no nvidia-smi call, no meter fields on CPU; a legacy command line writes the legacy JSON and checkpoint (argparse puts the new flags in `vars(args)`, so they must be stripped) with the same weights and the same RNG state. Pinned by Task 6 `test_cpu_meter_is_a_no_op_that_still_gets_the_timing_fields`, `test_with_every_flag_absent_the_json_and_checkpoint_are_the_legacy_ones`, `test_yardstick_flags_leave_training_untouched`.
4. **`--eval-every` larger than `--steps`, or dividing it**: only the final eval fires, once, with no duplicate row at the last step; the legacy `curve` keeps `steps // 6`. Pinned by Task 6 `test_fixed_eval_cadence`.
5. **Windows**: a region too short for N windows (`load_bytes` slices past the end silently) must raise before any model is built or any checkpoint loaded; and a fixed-window score is not comparable to the runner's random-window `val_bpb` (the 16M winner reads 0.7638 on the runner's 128 windows and 0.8313 on the first 128 fixed windows of the same bytes), so the GPU regression feeds the runner's own windows through the new scorer. Pinned by Task 1 `test_fixed_windows_need_n_times_seq_len_plus_one_bytes`, Task 6 `test_a_fixed_region_the_corpus_cannot_supply_fails_before_any_training`, Task 7 `test_a_region_too_short_for_the_windows_fails_before_any_checkpoint_loads`, and Task 10 regression A.

## File Structure

| File | Change |
|---|---|
| `flybrain/data/text.py` | modify: `fixed_windows` |
| `flybrain/meter.py` | new: `Sample`, `parse`, `nvidia_smi_sampler`, `format_uuid`, `device_uuid`, `GpuMeter`, `SplitClock` |
| `scripts/run_sparsela.py` | modify: `eval_fixed`, `parse_args`, five flags, the arm loop, `recorded_args`, `make_meter`, `fixed_row`, `save_ckpt` |
| `scripts/cross_eval.py` | modify: `evaluate_fixed`, `--fixed` |
| `scripts/run_spine.sh` | new: one spine arm |
| `scripts/spine_report.py` | new: the table and the two decision rules |
| `tests/test_text.py`, `tests/test_meter.py`, `tests/test_yardstick.py`, `tests/test_cross_eval_fixed.py`, `tests/test_spine_script.py`, `tests/test_spine_report.py` | tests: 72 new, 144 collected before this plan, 216 after |
| `.gitignore`, `notebook.md`, `results/*` | `.worktrees/`; Run 53; the regression, spine and score outputs |

Task order: 1 first. 2, 3, 4 build `meter.py` in three steps. 5 needs 1; 6 needs 1 to 5; 7 needs 1; 8 needs 6; 9 is independent. 10 (GPU) needs 1 to 7. 11 (the seed-0 launch) needs 1 to 10. 12 scores what 11 produced, 13 runs only what 12's rules ask for, 14 is the write-up.

---

### Task 1: fixed_windows

**Files:**
- Modify: `flybrain/data/text.py` (add `fixed_windows` after `batch_windows`)
- Test: `tests/test_text.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fixed_windows(data: Tensor, seq_len: int, n: int) -> tuple[Tensor, Tensor]`. `x[i] = data[i*seq_len:(i+1)*seq_len]`, `y[i]` is the same window one byte ahead, both are copies, no RNG. `ValueError` when `n < 1` or `data` has fewer than `n*seq_len + 1` elements.

- [ ] **Step 1: Write the failing test**

Replace in `tests/test_text.py`:
```python
import torch

from flybrain.data.text import batch_windows, bytes_tensor, load_bytes, ngram_bpb, split_bytes
```

with:
```python
import pytest
import torch

from flybrain.data.text import batch_windows, bytes_tensor, fixed_windows, load_bytes, ngram_bpb, split_bytes
```

Append to `tests/test_text.py`:
```python
def test_fixed_windows_are_exact_slices_and_use_no_rng():
    data = bytes_tensor("abcdefghijklmnopqrstuvwxyz")
    x, y = fixed_windows(data, seq_len=5, n=4)
    assert x.shape == y.shape == (4, 5) and x.dtype == data.dtype
    for i in range(4):
        assert torch.equal(x[i], data[i * 5 : (i + 1) * 5]) and torch.equal(y[i], data[i * 5 + 1 : (i + 1) * 5 + 1])
    state = torch.get_rng_state()
    again = fixed_windows(data, 5, 4)
    assert torch.equal(state, torch.get_rng_state())                 # the global generator was not touched
    assert torch.equal(again[0], x) and torch.equal(again[1], y)     # and there is nothing random to differ
    x[0, 0] = 99
    assert int(data[0]) == ord("a")                                  # windows are copies, not views of the corpus


def test_fixed_windows_need_n_times_seq_len_plus_one_bytes():
    data = bytes_tensor("abcdefghij")                                # 10 bytes
    x, y = fixed_windows(data, seq_len=3, n=3)                       # needs exactly 3 * 3 + 1 = 10
    assert y[2].tolist() == data[7:10].tolist()
    with pytest.raises(ValueError, match="need 13 bytes for 4 windows of 3"):
        fixed_windows(data, seq_len=3, n=4)
    with pytest.raises(ValueError, match="data has 0"):
        fixed_windows(data[:0], seq_len=3, n=1)                      # a --val-offset past the end of the file reads nothing
    with pytest.raises(ValueError, match="n >= 1"):
        fixed_windows(data, seq_len=3, n=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_text.py -q`
Expected: FAIL with "cannot import name 'fixed_windows'"

- [ ] **Step 3: Write minimal implementation**

Replace in `flybrain/data/text.py`:
```python
    return x, y
```

with:
```python
    return x, y


def fixed_windows(data: Tensor, seq_len: int, n: int) -> tuple[Tensor, Tensor]:
    """n non-overlapping windows from the start of data: x[i] = data[i*seq_len:(i+1)*seq_len], y[i] one byte ahead. No RNG."""
    if n < 1 or seq_len < 1:
        raise ValueError(f"need n >= 1 and seq_len >= 1, got n={n}, seq_len={seq_len}")
    need, have = n * seq_len + 1, int(data.numel())
    if have < need:
        raise ValueError(f"need {need} bytes for {n} windows of {seq_len} (n*seq_len+1), data has {have}")
    x = data[: n * seq_len].reshape(n, seq_len).clone()
    y = data[1:need].reshape(n, seq_len).clone()
    return x, y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_text.py -q`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add flybrain/data/text.py tests/test_text.py
git commit -m "Data: fixed_windows, n non-overlapping windows from the start of a region, no RNG"
```

---

### Task 2: nvidia-smi parsing and the physical card

**Files:**
- Create: `flybrain/meter.py` (first third: `Sample`, `parse`, `nvidia_smi_sampler`, `format_uuid`, `device_uuid`)
- Test: `tests/test_meter.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Sample(power_w, power_limit_w, temp_c, mem_used_mib, other_pids: tuple[int, ...])` (frozen dataclass); `parse(gpu_csv: str, apps_csv: str, gpu_uuid: str, own_pid: int) -> Sample` (`ValueError` on anything unreadable); `nvidia_smi_sampler(gpu_uuid: str, own_pid: int, timeout: float = 5.0) -> Callable[[], Sample]`; `format_uuid(raw) -> str`; `device_uuid(device: torch.device) -> str`; `GPU_QUERY`, `APPS_QUERY` (the two argv lists).

How this box prints (checked 2026-10-03 with read-only queries): `nvidia-smi --query-gpu=uuid,power.draw,power.limit,temperature.gpu,memory.used --format=csv,noheader,nounits` gives one line per card, fields joined by `, `, for example `GPU-51164b62-fd30-253a-5738-55844d46af9c, 27.35, 150.00, 43, 19553` (GPU 1; GPU 0 is `GPU-cdb4714c-560c-9a81-6420-f07eb59b5a5d`). `--query-compute-apps=gpu_uuid,pid,used_memory` gives `GPU-51164b62-fd30-253a-5738-55844d46af9c, 1179204, 19526` per process and prints nothing when no process runs; `used_memory` and its alias `used_gpu_memory` are both accepted. A UUID is `GPU-` plus lowercase 8-4-4-4-12 hex. `torch.cuda.get_device_properties(device).uuid` is typed `str` in torch 2.14's stubs; its exact spelling was not checked on a card while writing this plan (no CUDA context was opened on a busy GPU), so `format_uuid` accepts a str with or without `GPU-` or dashes and a `uuid.UUID`, and Task 10 checks the result against nvidia-smi.

- [ ] **Step 1: Write the failing test**

Create `tests/test_meter.py`:
```python
"""Yardstick instruments: nvidia-smi parsing, GPU meter accounting, train/eval clock. No GPU, no nvidia-smi, no network."""

from __future__ import annotations

import subprocess
import uuid

import pytest
import torch

from flybrain import meter
from flybrain.meter import Sample, format_uuid, parse

# Literal output of the two queries on the lab box (2026-10-03): GPU 1 is serving a model, GPU 0 has another job.
GPU0, GPU1 = "GPU-cdb4714c-560c-9a81-6420-f07eb59b5a5d", "GPU-51164b62-fd30-253a-5738-55844d46af9c"
GPU_CSV = f"{GPU0}, 30.16, 150.00, 45, 3211\n{GPU1}, 27.35, 150.00, 43, 19553\n"
APPS_CSV = f"{GPU0}, 1531073, 2534\n{GPU1}, 1179204, 19526\n"


def test_parse_picks_the_run_card_and_drops_the_run_itself():
    got = parse(GPU_CSV, APPS_CSV, GPU1, own_pid=4242)
    assert got == Sample(power_w=27.35, power_limit_w=150.0, temp_c=43.0, mem_used_mib=19553.0, other_pids=(1179204,))
    assert parse(GPU_CSV, APPS_CSV, GPU1, own_pid=1179204).other_pids == ()      # the run's own process is not contention
    assert parse(GPU_CSV, "", GPU1, own_pid=4242).other_pids == ()                # nvidia-smi prints nothing when no app runs
    assert parse(GPU_CSV, APPS_CSV, GPU0, own_pid=4242).other_pids == (1531073,)  # a process on the other card is not ours to count
    assert parse(GPU_CSV, f"{GPU1}, 7, 1\n{GPU1}, 5, 1\n", GPU1, own_pid=4242).other_pids == (5, 7)


@pytest.mark.parametrize("gpu_csv, apps_csv, why", [
    (f"{GPU1}, [N/A], 150.00, 43, 19553\n", "", "power.draw"),
    (f"{GPU1}, 27.35, [Not Supported], 43, 19553\n", "", "power.limit"),
    (f"{GPU1}, 27.35, 150.00, nan, 19553\n", "", "temperature.gpu"),
    (f"{GPU1}, 27.35, 150.00, 43\n", "", "expected 5 fields"),
    (f"{GPU0}, 30.16, 150.00, 45, 3211\n", "", "is not in the nvidia-smi output"),
    ("", "", "is not in the nvidia-smi output"),
    ("No devices were found\n", "", "expected 5 fields"),
    (GPU_CSV, f"{GPU1}, [N/A], 19526\n", "pid"),
    (GPU_CSV, f"{GPU1}, 1179204\n", "expected 3 fields"),
])
def test_parse_refuses_anything_it_cannot_read(gpu_csv, apps_csv, why):
    with pytest.raises(ValueError, match=why):
        parse(gpu_csv, apps_csv, GPU1, own_pid=1)


def test_sampler_runs_the_two_documented_queries(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=GPU_CSV if "--query-gpu" in argv[1] else APPS_CSV)

    monkeypatch.setattr(meter.subprocess, "run", fake_run)
    got = meter.nvidia_smi_sampler(GPU1, own_pid=4242)()
    assert calls == [["nvidia-smi", "--query-gpu=uuid,power.draw,power.limit,temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
                     ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]]
    assert got.other_pids == (1179204,)


def test_format_uuid_takes_every_spelling_and_prints_what_nvidia_smi_prints():
    bare = GPU1.removeprefix("GPU-")
    for raw in (GPU1, bare, GPU1.upper(), bare.replace("-", ""), uuid.UUID(bare)):
        assert format_uuid(raw) == GPU1
    for bad in ("", "GPU-1234", bare[:-1], bare + "0", "zz" + bare[2:]):
        with pytest.raises(ValueError, match="not a device uuid"):
            format_uuid(bad)


def test_device_uuid_reads_the_physical_card_not_the_index(monkeypatch):
    class Props:
        uuid = uuid.UUID(GPU1.removeprefix("GPU-"))

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: Props())
    assert meter.device_uuid(torch.device("cuda:1")) == GPU1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_meter.py -q`
Expected: FAIL with "cannot import name 'meter' from 'flybrain'"

- [ ] **Step 3: Write minimal implementation**

Create `flybrain/meter.py`:
```python
"""Yardstick instruments: GPU energy, temperature and contention, and a train/eval wall clock.

The nvidia-smi queries and the 85 C stop follow local-ai-training/scripts/thermal_guard.py; nothing here imports it."""

from __future__ import annotations

import csv
import math
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import torch

GPU_QUERY = ["nvidia-smi", "--query-gpu=uuid,power.draw,power.limit,temperature.gpu,memory.used", "--format=csv,noheader,nounits"]
APPS_QUERY = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]


@dataclass(frozen=True)
class Sample:
    power_w: float
    power_limit_w: float
    temp_c: float
    mem_used_mib: float
    other_pids: tuple[int, ...]  # compute processes on this card other than the run itself


def _rows(text: str, width: int, what: str) -> list[list[str]]:
    rows = [[f.strip() for f in r] for r in csv.reader(text.splitlines(), skipinitialspace=True) if r and any(f.strip() for f in r)]
    for r in rows:
        if len(r) != width:
            raise ValueError(f"nvidia-smi {what}: expected {width} fields, got {len(r)}: {r}")
    return rows


def _num(raw: str, cast: Callable, field: str):
    try:
        v = cast(raw)
    except ValueError:
        raise ValueError(f"nvidia-smi gave {raw!r} for {field}") from None
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError(f"nvidia-smi gave {raw!r} for {field}")
    return v


def parse(gpu_csv: str, apps_csv: str, gpu_uuid: str, own_pid: int) -> Sample:
    """One Sample for gpu_uuid from the two nvidia-smi outputs. ValueError on a missing card, a wrong field count or a
    non-numeric field such as '[N/A]'."""
    row = next((r for r in _rows(gpu_csv, 5, "gpu query") if r[0] == gpu_uuid), None)
    if row is None:
        raise ValueError(f"{gpu_uuid} is not in the nvidia-smi output")
    others = {_num(r[1], int, "pid") for r in _rows(apps_csv, 3, "compute-apps query") if r[0] == gpu_uuid}
    return Sample(_num(row[1], float, "power.draw"), _num(row[2], float, "power.limit"), _num(row[3], float, "temperature.gpu"),
                  _num(row[4], float, "memory.used"), tuple(sorted(others - {own_pid})))


def nvidia_smi_sampler(gpu_uuid: str, own_pid: int, timeout: float = 5.0) -> Callable[[], Sample]:
    """A sampler that runs the two queries; any failure raises, and the meter counts that as a missing sample."""
    def sample() -> Sample:
        gpu = subprocess.run(GPU_QUERY, check=True, capture_output=True, text=True, timeout=timeout).stdout
        apps = subprocess.run(APPS_QUERY, check=True, capture_output=True, text=True, timeout=timeout).stdout
        return parse(gpu, apps, gpu_uuid, own_pid)
    return sample


def format_uuid(raw) -> str:
    """nvidia-smi's spelling of a device uuid: 'GPU-' + lowercase 8-4-4-4-12 hex. Takes a str (with or without 'GPU-' or
    dashes) or a uuid.UUID."""
    h = str(raw).strip().lower().removeprefix("gpu-").replace("-", "")
    if not re.fullmatch(r"[0-9a-f]{32}", h):
        raise ValueError(f"not a device uuid: {raw!r}")
    return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def device_uuid(device: torch.device) -> str:
    """The physical card behind a CUDA device as nvidia-smi prints it, e.g. GPU-51164b62-fd30-253a-5738-55844d46af9c.
    Unlike an index it survives CUDA_VISIBLE_DEVICES and a different device order."""
    return format_uuid(torch.cuda.get_device_properties(device).uuid)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_meter.py -q`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add flybrain/meter.py tests/test_meter.py
git commit -m "Meter: parse nvidia-smi output and name the physical card by its UUID"
```

---

### Task 3: GpuMeter

**Files:**
- Modify: `flybrain/meter.py` (imports, one constant, append `GpuMeter`)
- Test: `tests/test_meter.py`

**Interfaces:**
- Consumes: `Sample`, `nvidia_smi_sampler` from Task 2.
- Produces: `GpuMeter(gpu_uuid: str, own_pid: int, interval: float = 1.0, max_temp_c: float = 85.0, sampler: Callable[[], Sample] | None = None)` with `ingest(sample, t) -> None`, `fail(t) -> None`, `check() -> None` (raises `RuntimeError` once any sample reached `max_temp_c`), `summary() -> dict` with keys `joules`, `mean_power_w`, `power_limit_w`, `max_temp_c`, `contended`, `samples`, `telemetry_lost`, `start()`, `stop()`, and the attribute `gpu_uuid`; `TELEMETRY_TIMEOUT_S = 60.0`. Joules are the new sample's power times the time since the previous sample; `max_temp_c` in the summary is the hottest sample, not the limit.

- [ ] **Step 1: Write the failing test**

Replace in `tests/test_meter.py`:
```python
import subprocess
import uuid
```

with:
```python
import subprocess
import time
import uuid
```

Replace in `tests/test_meter.py`:
```python
from flybrain.meter import Sample, format_uuid, parse
```

with:
```python
from flybrain.meter import GpuMeter, Sample, format_uuid, parse
```

Append to `tests/test_meter.py`:
```python
def s(power=100.0, temp=60.0, others=(), limit=150.0) -> Sample:
    return Sample(power_w=power, power_limit_w=limit, temp_c=temp, mem_used_mib=1000.0, other_pids=tuple(others))


def test_joules_are_power_times_seconds():
    m = GpuMeter(GPU1, own_pid=1)
    for t in range(11):
        m.ingest(s(power=100.0), float(t))
    out = m.summary()
    assert out["joules"] == pytest.approx(1000.0) and out["mean_power_w"] == pytest.approx(100.0) and out["samples"] == 11
    assert (out["power_limit_w"], out["max_temp_c"], out["contended"], out["telemetry_lost"]) == (150.0, 60.0, False, False)


def test_joules_stay_within_5_percent_of_mean_power_times_seconds_on_a_ramp():
    m = GpuMeter(GPU1, own_pid=1)
    for t in range(11):  # 100 W rising to 200 W over 10 s, sampled at 1 Hz: the exact integral is 1500 J
        m.ingest(s(power=100.0 + 10.0 * t), float(t))
    out = m.summary()
    assert abs(out["joules"] - 1500.0) / 1500.0 < 0.05
    assert abs(out["joules"] - out["mean_power_w"] * 10.0) < 1e-9


def test_a_second_process_sets_contended_for_good_and_never_stops_the_run():
    m = GpuMeter(GPU1, own_pid=1)
    m.ingest(s(), 0.0)
    assert m.summary()["contended"] is False
    m.ingest(s(others=(4242,)), 1.0)
    m.ingest(s(), 2.0)                      # it left again: the flag stays, the run was shared for a while
    assert m.summary()["contended"] is True
    m.check()                               # contention is recorded, not punished


def test_the_limit_trips_at_or_above_it_and_check_raises_from_then_on():
    m = GpuMeter(GPU1, own_pid=1, max_temp_c=85.0)
    m.ingest(s(temp=84.9), 0.0)
    m.check()
    m.ingest(s(temp=85.0), 1.0)
    with pytest.raises(RuntimeError, match="85"):
        m.check()
    m.ingest(s(temp=60.0), 2.0)             # it cooled down; the run was already aborted
    with pytest.raises(RuntimeError):
        m.check()
    assert m.summary()["max_temp_c"] == 85.0


def test_sixty_seconds_without_a_sample_marks_telemetry_lost_and_nothing_else():
    m = GpuMeter(GPU1, own_pid=1)
    m.ingest(s(), 0.0)
    m.fail(30.0); m.fail(59.9)
    assert m.summary()["telemetry_lost"] is False
    m.fail(60.0)
    assert m.summary()["telemetry_lost"] is True
    m.ingest(s(), 61.0)                     # the card came back: the flag says the gap was not measured
    m.check()
    assert m.summary()["telemetry_lost"] is True
    cold = GpuMeter(GPU1, own_pid=1)        # never a good sample: the count starts at the first failure
    cold.fail(100.0); cold.fail(159.0)
    assert cold.summary()["telemetry_lost"] is False
    cold.fail(160.0)
    assert cold.summary()["telemetry_lost"] is True and cold.summary()["samples"] == 0


def test_the_thread_samples_survives_a_failing_sampler_and_stops():
    calls = {"n": 0}

    def flaky() -> Sample:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ValueError("nvidia-smi gave '[N/A]' for power.draw")
        return s()

    m = GpuMeter(GPU1, own_pid=1, interval=0.01, sampler=flaky)
    m.start()
    deadline = time.monotonic() + 5.0
    while m.summary()["samples"] < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    m.stop()
    assert m.summary()["samples"] >= 3 and m._thread is not None and not m._thread.is_alive()
    with pytest.raises(RuntimeError, match="already started"):
        m.start()


def test_a_sampler_that_always_fails_leaves_zeros_and_a_live_run():
    def dead() -> Sample:
        raise FileNotFoundError("nvidia-smi")

    m = GpuMeter(GPU1, own_pid=1, interval=0.01, sampler=dead)
    m.start(); time.sleep(0.1); m.stop()
    assert m.summary() == {"joules": 0.0, "mean_power_w": 0.0, "power_limit_w": 0.0, "max_temp_c": 0.0,
                           "contended": False, "samples": 0, "telemetry_lost": False}
    m.check()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_meter.py -q`
Expected: FAIL with "cannot import name 'GpuMeter'"

- [ ] **Step 3: Write minimal implementation**

Replace in `flybrain/meter.py`:
```python
import csv
import math
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import torch
```

with:
```python
import csv
import math
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
```

Replace in `flybrain/meter.py`:
```python
APPS_QUERY = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]
```

with:
```python
APPS_QUERY = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]
TELEMETRY_TIMEOUT_S = 60.0
```

Append to `flybrain/meter.py`:
```python
class GpuMeter:
    """Energy, hottest sample and contention for one physical card while a run trains.

    All accounting is in ingest(), which is plain arithmetic; start()/stop() only drive it from a daemon thread. The one
    condition that stops a run is over-temperature, and only when the run calls check(). A sampler that fails for 60 s
    sets telemetry_lost and the run carries on."""

    def __init__(self, gpu_uuid: str, own_pid: int, interval: float = 1.0, max_temp_c: float = 85.0,
                 sampler: Callable[[], Sample] | None = None) -> None:
        self.gpu_uuid, self.own_pid, self.interval, self.limit_c = gpu_uuid, own_pid, float(interval), float(max_temp_c)
        self._sampler = sampler or nvidia_smi_sampler(gpu_uuid, own_pid)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t_first: float | None = None   # first and last good sample
        self._t_last: float | None = None
        self._t_fail: float | None = None    # where the 60 s count starts when no sample has succeeded yet
        self._last_power = 0.0
        self._power_limit = 0.0
        self._peak_c = 0.0
        self.joules = 0.0
        self.samples = 0
        self.contended = False
        self.tripped = False
        self.telemetry_lost = False

    def ingest(self, sample: Sample, t: float) -> None:
        """Account one sample taken at monotonic time t: joules += power x time since the previous sample."""
        with self._lock:
            if self._t_last is None:
                self._t_first = t
            else:
                self.joules += sample.power_w * max(0.0, t - self._t_last)
            self._t_last, self._last_power = t, sample.power_w
            self.samples += 1
            self._power_limit = max(self._power_limit, sample.power_limit_w)
            self._peak_c = max(self._peak_c, sample.temp_c)
            self.contended = self.contended or bool(sample.other_pids)
            self.tripped = self.tripped or sample.temp_c >= self.limit_c

    def fail(self, t: float) -> None:
        """No sample could be taken at time t; 60 s after the last good one (or the start) the meter reports telemetry_lost."""
        with self._lock:
            ref = self._t_last if self._t_last is not None else self._t_fail
            if ref is None:
                ref = self._t_fail = t
            if t - ref >= TELEMETRY_TIMEOUT_S:
                self.telemetry_lost = True

    def check(self) -> None:
        """Raise RuntimeError once any sample has reached max_temp_c; the run calls this every few steps."""
        with self._lock:
            tripped, peak = self.tripped, self._peak_c
        if tripped:
            raise RuntimeError(f"GPU {self.gpu_uuid} reached {peak:.0f} C (limit {self.limit_c:.0f} C): run aborted")

    def summary(self) -> dict:
        """The per-arm fields. max_temp_c is the hottest sample, not the limit; all zeros when samples == 0."""
        with self._lock:
            span = (self._t_last - self._t_first) if self._t_last is not None else 0.0
            mean = self.joules / span if span > 0 else self._last_power
            return {"joules": self.joules, "mean_power_w": mean, "power_limit_w": self._power_limit, "max_temp_c": self._peak_c,
                    "contended": self.contended, "samples": self.samples, "telemetry_lost": self.telemetry_lost}

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("meter already started")
        self._t_fail = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="gpu-meter", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                self.ingest(self._sampler(), time.monotonic())
            except Exception:  # nvidia-smi missing, hung, or printing [N/A]: a missing sample, never a dead run
                self.fail(time.monotonic())
            if self._stop.wait(self.interval):
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 10.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_meter.py -q`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add flybrain/meter.py tests/test_meter.py
git commit -m "Meter: GpuMeter with joules, contention, the 85 C stop and a 60 s telemetry-lost flag"
```

---

### Task 4: SplitClock

**Files:**
- Modify: `flybrain/meter.py` (imports, append `SplitClock`)
- Test: `tests/test_meter.py`

**Interfaces:**
- Consumes: nothing from the earlier tasks.
- Produces: `SplitClock(sync: Callable[[], None] = no-op, now: Callable[[], float] = time.perf_counter)` with `start()`, the context manager `boundary()`, and the attributes `train_seconds`, `eval_seconds`. Inside a `with clock.boundary():` block `train_seconds` is already closed up to that moment; `sync` is called at both edges of a boundary and at `start()`, never per step.

- [ ] **Step 1: Write the failing test**

Replace in `tests/test_meter.py`:
```python
from flybrain.meter import GpuMeter, Sample, format_uuid, parse
```

with:
```python
from flybrain.meter import GpuMeter, Sample, SplitClock, format_uuid, parse
```

Append to `tests/test_meter.py`:
```python
def test_split_clock_keeps_evals_out_of_train_seconds_and_syncs_only_at_boundaries():
    t, syncs = [0.0], []
    clock = SplitClock(sync=lambda: syncs.append(t[0]), now=lambda: t[0])
    clock.start()
    t[0] = 10.0                              # ten seconds of steps
    with clock.boundary():
        assert clock.train_seconds == 10.0   # already closed inside the block, so a curve row can read it
        t[0] = 13.0                          # three seconds of eval
    t[0] = 20.0                              # seven more of steps
    with clock.boundary():
        t[0] = 21.0
    assert (clock.train_seconds, clock.eval_seconds) == (17.0, 4.0)
    assert syncs == [0.0, 10.0, 13.0, 20.0, 21.0]


def test_split_clock_counts_a_failed_eval_before_the_error_propagates():
    t = [0.0]
    clock = SplitClock(now=lambda: t[0])
    clock.start()
    with pytest.raises(RuntimeError):
        with clock.boundary():
            t[0] = 5.0
            raise RuntimeError("eval blew up")
    assert clock.eval_seconds == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_meter.py -q`
Expected: FAIL with "cannot import name 'SplitClock'"

- [ ] **Step 3: Write minimal implementation**

Replace in `flybrain/meter.py`:
```python
import csv
import math
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
```

with:
```python
import csv
import math
import re
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
```

Append to `flybrain/meter.py`:
```python
class SplitClock:
    """Wall time split into training steps and boundaries (evals, saves). The GPU is synced at a boundary's two edges only,
    never per step, so the clock does not slow training. With start() before the loop and every eval or save inside
    boundary(): train_seconds + eval_seconds is the time since start()."""

    def __init__(self, sync: Callable[[], None] = lambda: None, now: Callable[[], float] = time.perf_counter) -> None:
        self._sync, self._now = sync, now
        self.train_seconds = 0.0
        self.eval_seconds = 0.0
        self._mark = 0.0

    def _tick(self) -> float:
        self._sync()
        t = self._now()
        dt, self._mark = t - self._mark, t
        return dt

    def start(self) -> None:
        self._sync()
        self._mark = self._now()

    @contextmanager
    def boundary(self) -> Iterator[None]:
        self.train_seconds += self._tick()  # inside the block train_seconds is already up to date
        try:
            yield
        finally:
            self.eval_seconds += self._tick()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_meter.py -q`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add flybrain/meter.py tests/test_meter.py
git commit -m "Meter: SplitClock keeps evals and saves out of train_seconds, syncing only at boundaries"
```

---

### Task 5: eval_fixed in the runner

**Files:**
- Modify: `scripts/run_sparsela.py` (add `eval_fixed` before `_CMASK`)
- Test: `tests/test_yardstick.py`

**Interfaces:**
- Consumes: `fixed_windows` (Task 1); `_amp(args, device)` and the imports already in the runner.
- Produces: `eval_fixed(model, xf, yf, args, device) -> {"bpb": cold, "bpb_warm": warm}`. `xf`, `yf` are `[N, T]`; windows are scored in batches of `args.batch` under the runner's own `_amp` context; cold is the mean over all positions, warm the mean over positions `T // 2` onward; any `N` (a short last batch is fine); the model is left in train mode; no RNG.

- [ ] **Step 1: Write the failing test**

Create `tests/test_yardstick.py`:
```python
"""The yardstick's scorer inside the runner: fixed windows, cold and warm bits/byte. CPU only, no corpus."""

from __future__ import annotations

import argparse
import math

import pytest
import torch
from torch.nn import functional as F

from flybrain.data.text import fixed_windows
from flybrain.model.hybrid import GRUHybrid


def _tiny():
    torch.manual_seed(0)
    return GRUHybrid(d_model=16, backbone="mingru", where="none", n_layers=1, d_ff=32)


def test_eval_fixed_is_the_per_position_mean_cold_and_its_second_half_warm():
    from scripts.run_sparsela import eval_fixed
    m = _tiny()
    xf, yf = fixed_windows(torch.randint(0, 256, (7 * 16 + 1,)), 16, 7)   # 7 windows in batches of 3: the last batch is short
    out = eval_fixed(m, xf, yf, argparse.Namespace(batch=3, amp=False), torch.device("cpu"))
    assert m.training                                                    # left in train mode, like the runner's other evals
    m.eval()
    with torch.no_grad():                                                 # reference: one window at a time
        ce = torch.stack([F.cross_entropy(m(xf[i:i + 1])[0][0], yf[i], reduction="none") for i in range(7)])
    assert out["bpb"] == pytest.approx(float(ce.mean()) / math.log(2), abs=1e-5)
    assert out["bpb_warm"] == pytest.approx(float(ce[:, 8:].mean()) / math.log(2), abs=1e-5)
    assert out["bpb_warm"] != pytest.approx(out["bpb"], abs=1e-6)         # the two numbers really are different cuts
    again = eval_fixed(m, xf, yf, argparse.Namespace(batch=7, amp=False), torch.device("cpu"))
    assert again["bpb"] == pytest.approx(out["bpb"], abs=1e-6)            # the batch size does not change the score


def test_eval_fixed_uses_no_global_rng():
    from scripts.run_sparsela import eval_fixed
    m = _tiny()
    xf, yf = fixed_windows(torch.randint(0, 256, (4 * 16 + 1,)), 16, 4)
    state = torch.get_rng_state()
    eval_fixed(m, xf, yf, argparse.Namespace(batch=2, amp=False), torch.device("cpu"))
    assert torch.equal(state, torch.get_rng_state())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_yardstick.py -q -x`
Expected: FAIL with "cannot import name 'eval_fixed' from 'scripts.run_sparsela'"

- [ ] **Step 3: Write minimal implementation**

Replace in `scripts/run_sparsela.py`:
```python
_CMASK: dict = {}
```

with:
```python
@torch.no_grad()
def eval_fixed(model, xf, yf, args, device) -> dict:
    """Score fixed windows (xf, yf: [N, T] from fixed_windows) in batches of args.batch under the training amp context.
    cold = mean bits/byte over every position; warm = mean over positions T // 2 onward. No RNG, any N."""
    model.eval(); T = xf.size(1)
    tot = torch.zeros(T, dtype=torch.float64, device=device)
    for i in range(0, xf.size(0), args.batch):
        x, y = xf[i:i + args.batch].to(device), yf[i:i + args.batch].to(device)
        with _amp(args, device):
            logits, _ = model(x)
        tot += F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1), reduction="none").view(-1, T).double().sum(0)
    model.train()
    per_pos = tot / xf.size(0) / math.log(2)
    return {"bpb": float(per_pos.mean()), "bpb_warm": float(per_pos[T // 2:].mean())}


_CMASK: dict = {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_yardstick.py -q`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add scripts/run_sparsela.py tests/test_yardstick.py
git commit -m "Runner: eval_fixed scores fixed windows, cold and warm, in batches under the training amp context"
```

---

### Task 6: The runner's yardstick flags

**Files:**
- Modify: `scripts/run_sparsela.py` (imports; the argparse block moves into `parse_args` and gains five flags; new helpers; the arm loop)
- Test: `tests/test_yardstick.py`

**Interfaces:**
- Consumes: `fixed_windows` (Task 1); `GpuMeter`, `SplitClock`, `device_uuid` (Tasks 2 to 4); `eval_fixed` (Task 5).
- Produces: flags `--val-fixed N`, `--val-offset B`, `--eval-every K`, `--meter`, `--ckpt-every K`, all default off; `parse_args(argv=None)` (refuses `--val-offset` or `--eval-every` without `--val-fixed`, `--val-fixed` with `--task copy`, and negative values); `YARDSTICK_FLAGS`, `yardstick_on(args)`, `recorded_args(args)`, `make_meter(args, device)`, `fixed_row(...)`, `save_ckpt(...)`, `CHECK_EVERY = 50`.
- Per-arm JSON fields, written only when at least one yardstick flag is on: `train_seconds`, `eval_seconds`, `bytes_seen` (steps x batch x seq_len), `peak_vram_bytes` (0 on CPU; the peak counter is reset at the start of each arm), and, with `--val-fixed`, `curve_fixed` (rows `{"step", "train_seconds", "bpb", "bpb_warm"}`) and `val_fixed` (`{"offset", "n_windows", "bpb", "bpb_warm"}`, the final row); and, with `--meter` on CUDA, `joules`, `mean_power_w`, `power_limit_w`, `max_temp_c`, `contended`, `samples`, `telemetry_lost`, `gpu_uuid`. The existing `seconds` keeps its expression and its position (after the final eval and save). The legacy `curve` and its `steps // 6` cadence are untouched. Checkpoints from `--ckpt-every K` go to `results/ckpt/<name>_<arm>.step<N>.pt` (N is the step reached) holding `{"arm", "args", "step", "state", "opt", "sched"}`; save only, there is no resume path.

What the loop below guarantees, so a timer cannot drift: `clock.start()` runs once before step 0; every legacy eval, fixed eval and checkpoint save runs inside `with boundary():`, and so do the final legacy eval, the final fixed eval and the final `--save`; the CUDA syncs are in `SplitClock` and happen only at those edges; the meter starts before the loop, is checked every `CHECK_EVERY` steps and stopped in `finally`; nothing new draws from the global RNG or touches the per-step data seeds. The last step is scored by the final fixed eval, not by a periodic one, so `curve_fixed` has no duplicate row. A flagged run records the five flags in its JSON and checkpoints; an unflagged run records exactly the arguments it always did.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_yardstick.py`:
```python
# ---- the runner's yardstick flags: tiny CPU runs through main(), no corpus beyond a file written here ----

GOLDEN_ARM_KEYS = ["val_bpb", "val_bpb_first_q", "val_bpb_last_q", "val_bpb_copyable", "copyable_frac", "val_bpb_noncopyable",
                   "arm", "seconds", "trainable_scalars", "curve", "flops_per_token_recurrent"]  # results/m20_linux_depth4local_s0.json
NEW_ARM_KEYS = {"train_seconds", "eval_seconds", "bytes_seen", "peak_vram_bytes", "curve_fixed", "val_fixed", "joules", "mean_power_w",
                "power_limit_w", "max_temp_c", "contended", "samples", "telemetry_lost", "gpu_uuid"}
SMALL = ["--task", "code", "--train-bytes", "3000", "--val-bytes", "4000", "--d-model", "16", "--n-layers", "2", "--d-ff", "32",
         "--n-key", "64", "--top-k", "4", "--key-conv", "2", "--normalize", "--chunk", "8", "--batch", "2", "--seq-len", "32",
         "--steps", "6", "--lr", "0.003", "--arms", "mingru_sparse_post", "--save", "--seed", "0", "--device", "cpu", "--run-name", "tiny"]
FIXED = ["--val-fixed", "4", "--val-offset", "110000"]  # 4 windows of 32 bytes, past the 3000 train and the val bytes at 103000


def _corpus(tmp_path):
    path = tmp_path / "tiny.bytes"
    path.write_bytes((b"int main(void) { return 0; }\n" * 5000)[:130_000])
    return path


def _run(monkeypatch, tmp_path, tag, *flags):
    """main() in-process, results into tmp_path/tag; returns (the run's JSON, the results dir)."""
    import json
    import sys

    import scripts.run_sparsela as sra
    res = tmp_path / tag
    res.mkdir()
    monkeypatch.setattr(sra, "RESULTS", res)
    monkeypatch.setattr(sra, "_CMASK", {})
    monkeypatch.setattr(sys, "argv", ["run_sparsela.py", "--corpus-file", str(_corpus(tmp_path)), *SMALL, *flags])
    sra.main()
    return json.loads((res / "tiny.json").read_text()), res


def _ckpt(res, suffix=""):
    return torch.load(res / "ckpt" / f"tiny_mingru_sparse_post{suffix}.pt", map_location="cpu", weights_only=False)


def test_yardstick_flags_leave_training_untouched(monkeypatch, tmp_path):
    """The required runner test: the same final weights and the same legacy JSON with and without the new flags."""
    legacy, ra = _run(monkeypatch, tmp_path, "a")
    rng_a = torch.get_rng_state()
    flagged, rb = _run(monkeypatch, tmp_path, "b", *FIXED, "--eval-every", "2", "--ckpt-every", "2")
    assert torch.equal(rng_a, torch.get_rng_state())                                        # the global generator ends where it ended without the flags
    sa, sb = _ckpt(ra)["state"], _ckpt(rb)["state"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)            # identical state_dict, bit for bit
    for k, v in legacy.items():
        if k != "arms":
            assert flagged[k] == v, k                                                       # every legacy top-level field
    assert len(legacy["arms"]) == len(flagged["arms"]) == 1
    for k, v in legacy["arms"][0].items():
        if k != "seconds":
            assert flagged["arms"][0][k] == v, k                                            # every legacy per-arm field but the clock


def test_with_every_flag_absent_the_json_and_checkpoint_are_the_legacy_ones(monkeypatch, tmp_path):
    from scripts.run_sparsela import YARDSTICK_FLAGS
    legacy, res = _run(monkeypatch, tmp_path, "a")
    assert not set(YARDSTICK_FLAGS) & set(legacy)                                           # no new key among the recorded args
    assert list(legacy["arms"][0]) == GOLDEN_ARM_KEYS                                       # same keys, same order, nothing added
    assert not NEW_ARM_KEYS & set(legacy["arms"][0])
    assert not set(YARDSTICK_FLAGS) & set(_ckpt(res)["args"])
    assert sorted(p.name for p in (res / "ckpt").iterdir()) == ["tiny_mingru_sparse_post.pt"]


def test_flagged_run_writes_the_new_fields_and_checkpoints(monkeypatch, tmp_path):
    out, res = _run(monkeypatch, tmp_path, "b", *FIXED, "--eval-every", "2", "--ckpt-every", "2")
    arm = out["arms"][0]
    assert out["val_fixed"] == 4 and out["val_offset"] == 110000 and out["eval_every"] == 2 and out["ckpt_every"] == 2
    assert [r["step"] for r in arm["curve_fixed"]] == [2, 4, 6]
    assert all(set(r) == {"step", "train_seconds", "bpb", "bpb_warm"} for r in arm["curve_fixed"])
    ts = [r["train_seconds"] for r in arm["curve_fixed"]]
    assert ts == sorted(ts) and 0 < ts[-1] == arm["train_seconds"] <= arm["seconds"]        # the last row closes the training clock
    assert arm["val_fixed"] == {"offset": 110000, "n_windows": 4, "bpb": arm["curve_fixed"][-1]["bpb"], "bpb_warm": arm["curve_fixed"][-1]["bpb_warm"]}
    assert arm["eval_seconds"] > 0 and arm["train_seconds"] + arm["eval_seconds"] <= arm["seconds"] + 1e-3
    assert arm["bytes_seen"] == 6 * 2 * 32 and arm["peak_vram_bytes"] == 0                   # CPU
    assert not {"joules", "gpu_uuid"} & set(arm)                                             # no meter was asked for
    assert sorted(p.name for p in (res / "ckpt").iterdir()) == [
        "tiny_mingru_sparse_post.pt", "tiny_mingru_sparse_post.step2.pt", "tiny_mingru_sparse_post.step4.pt", "tiny_mingru_sparse_post.step6.pt"]
    ck = _ckpt(res, ".step4")
    assert set(ck) == {"arm", "args", "step", "state", "opt", "sched"} and ck["step"] == 4 and ck["arm"] == "mingru_sparse_post"
    assert ck["sched"]["last_epoch"] == 4 and ck["args"]["ckpt_every"] == 2


@pytest.mark.parametrize("steps, every, expected", [
    (6, 2, [2, 4, 6]),                  # the last periodic eval is the final one: scored once, not twice
    (4, 100, [4]),                      # --eval-every larger than --steps: only the final eval
    (4, 4, [4]),
    (12, 0, [2, 4, 6, 8, 10, 12]),      # 0 keeps steps // 6
    (1, 0, [1]),
])
def test_fixed_eval_cadence(monkeypatch, tmp_path, steps, every, expected):
    flags = [*FIXED, "--steps", str(steps)] + (["--eval-every", str(every)] if every else [])
    out, _ = _run(monkeypatch, tmp_path, "c", *flags)
    arm = out["arms"][0]
    assert [r["step"] for r in arm["curve_fixed"]] == expected
    assert arm["val_fixed"]["bpb"] == arm["curve_fixed"][-1]["bpb"]
    assert [r["step"] for r in arm["curve"]] == [s for s in range(max(1, steps // 6), steps + 1, max(1, steps // 6))]  # legacy cadence kept


def test_a_fixed_region_the_corpus_cannot_supply_fails_before_any_training(monkeypatch, tmp_path):
    import scripts.run_sparsela as sra
    monkeypatch.setattr(sra, "build", lambda *a, **k: pytest.fail("a model was built"))
    near_the_end = str(_corpus(tmp_path).stat().st_size - 100)
    with pytest.raises(ValueError, match="need 129 bytes for 4 windows of 32"):
        _run(monkeypatch, tmp_path, "d", "--val-fixed", "4", "--val-offset", near_the_end)
    assert list((tmp_path / "d").iterdir()) == []                                            # nothing written


def test_evals_and_saves_never_count_as_training_time(monkeypatch, tmp_path):
    """Fake clock: a training step costs 1 s, a legacy eval 10 s, a fixed eval 100 s, a checkpoint save 1000 s."""
    import scripts.run_sparsela as sra
    from flybrain.meter import SplitClock
    t = [0.0]

    def costing(fn, dt):
        def wrapped(*a, **k):
            t[0] += dt
            return fn(*a, **k)
        return wrapped

    monkeypatch.setattr(sra, "SplitClock", lambda sync: SplitClock(sync=sync, now=lambda: t[0]))
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", costing(torch.nn.utils.clip_grad_norm_, 1.0))  # once per training step
    monkeypatch.setattr(sra, "eval_wiki", costing(sra.eval_wiki, 10.0))     # 6 periodic + 1 final
    monkeypatch.setattr(sra, "eval_fixed", costing(sra.eval_fixed, 100.0))  # steps 2, 4 and the final 6
    monkeypatch.setattr(sra, "save_ckpt", costing(sra.save_ckpt, 1000.0))   # steps 2, 4, 6
    arm = _run(monkeypatch, tmp_path, "t", *FIXED, "--eval-every", "2", "--ckpt-every", "2")[0]["arms"][0]
    assert arm["train_seconds"] == 6.0
    assert arm["eval_seconds"] == 7 * 10.0 + 3 * 100.0 + 3 * 1000.0
    assert [r["train_seconds"] for r in arm["curve_fixed"]] == [2.0, 4.0, 6.0]


def test_cpu_meter_is_a_no_op_that_still_gets_the_timing_fields(monkeypatch, tmp_path, capsys):
    out, _ = _run(monkeypatch, tmp_path, "e", "--meter")
    arm = out["arms"][0]
    assert {"train_seconds", "eval_seconds", "bytes_seen", "peak_vram_bytes"} <= set(arm)
    assert not {"joules", "mean_power_w", "contended", "telemetry_lost", "gpu_uuid", "curve_fixed", "val_fixed"} & set(arm)
    assert "meter: skipped, device is cpu" in capsys.readouterr().out


def test_meter_summary_reaches_the_json_and_the_thread_is_stopped(monkeypatch, tmp_path):
    import os

    import scripts.run_sparsela as sra
    from flybrain.meter import GpuMeter, Sample
    ok, made = Sample(100.0, 150.0, 60.0, 1000.0, ()), []

    def fake(args, device):
        m = GpuMeter("GPU-test", os.getpid(), interval=0.01, sampler=lambda: ok)
        m.ingest(ok, 0.0)                    # one sample is certain, whatever the thread has done by then
        made.append(m)
        return m

    monkeypatch.setattr(sra, "make_meter", fake)
    arm = _run(monkeypatch, tmp_path, "f", "--meter")[0]["arms"][0]
    assert arm["gpu_uuid"] == "GPU-test" and arm["contended"] is False and arm["telemetry_lost"] is False
    assert arm["power_limit_w"] == 150.0 and arm["max_temp_c"] == 60.0 and arm["samples"] >= 1
    assert made[0]._thread is not None and not made[0]._thread.is_alive()


def test_an_overheated_card_aborts_the_run_and_stops_the_thread(monkeypatch, tmp_path):
    import os

    import scripts.run_sparsela as sra
    from flybrain.meter import GpuMeter, Sample
    hot, made = Sample(100.0, 150.0, 90.0, 1000.0, ()), []

    def fake(args, device):
        m = GpuMeter("GPU-test", os.getpid(), interval=0.01, sampler=lambda: hot)
        m.ingest(hot, 0.0)
        made.append(m)
        return m

    monkeypatch.setattr(sra, "make_meter", fake)
    monkeypatch.setattr(sra, "CHECK_EVERY", 2)
    with pytest.raises(RuntimeError, match="reached 90 C"):
        _run(monkeypatch, tmp_path, "g", "--meter")
    assert not (tmp_path / "g" / "tiny.json").exists()                                       # an aborted run leaves no result to skip
    assert made[0]._thread is not None and not made[0]._thread.is_alive()


@pytest.mark.parametrize("argv", [["--val-offset", "5"], ["--eval-every", "5"], ["--val-fixed", "4", "--task", "copy"],
                                  ["--val-fixed", "-1"], ["--ckpt-every", "-2"]])
def test_flags_that_cannot_work_are_refused_at_parse_time(argv):
    from scripts.run_sparsela import parse_args
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_the_parser_defaults_are_the_legacy_behaviour():
    from scripts.run_sparsela import parse_args, yardstick_on
    args = parse_args([])
    assert (args.val_fixed, args.val_offset, args.eval_every, args.meter, args.ckpt_every) == (0, 0, 0, False, 0)
    assert yardstick_on(args) is False and yardstick_on(parse_args(["--meter"])) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_yardstick.py -q -x`
Expected: FAIL with "SystemExit: 2"

- [ ] **Step 3: Write minimal implementation**

Five replacements in `scripts/run_sparsela.py`, in this order. First the imports:

Replace in `scripts/run_sparsela.py`:
```python
import math
import sys
import time
from pathlib import Path
```

with:
```python
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
```

Replace in `scripts/run_sparsela.py`:
```python
from flybrain.data.text import batch_windows, load_bytes, ngram_bpb
from flybrain.model.hybrid import GRUHybrid
```

with:
```python
from flybrain.data.text import batch_windows, fixed_windows, load_bytes, ngram_bpb
from flybrain.meter import GpuMeter, SplitClock, device_uuid
from flybrain.model.hybrid import GRUHybrid
```

The helpers, and the argparse block becomes `parse_args`:

Replace in `scripts/run_sparsela.py`:
```python
_CMASK: dict = {}


def main():
    p = argparse.ArgumentParser()
```

with:
```python
_CMASK: dict = {}

CHECK_EVERY = 50  # steps between GpuMeter.check() calls
YARDSTICK_FLAGS = ("val_fixed", "val_offset", "eval_every", "meter", "ckpt_every")


def yardstick_on(args) -> bool:
    """True when any yardstick flag is set. Only then do the new fields reach the JSON, so a legacy command line writes a legacy file."""
    return bool(args.val_fixed or args.val_offset or args.eval_every or args.meter or args.ckpt_every)


def recorded_args(args) -> dict:
    """vars(args) as a run records it (JSON and checkpoints): the yardstick flags appear only when one of them is set."""
    return vars(args) if yardstick_on(args) else {k: v for k, v in vars(args).items() if k not in YARDSTICK_FLAGS}


def make_meter(args, device) -> GpuMeter | None:
    """A GpuMeter for the physical card behind device when --meter is set; None on CPU, where --meter does nothing."""
    if not args.meter:
        return None
    if device.type != "cuda":
        print("meter: skipped, device is cpu", flush=True)
        return None
    return GpuMeter(device_uuid(device), os.getpid())


def fixed_row(arm, model, xf, yf, args, device, step: int, train_seconds: float) -> dict:
    row = {"step": step, "train_seconds": train_seconds, **eval_fixed(model, xf, yf, args, device)}
    print(f"fixed {arm} {row}", flush=True)  # no leading spaces: run_*.sh drop lines that start with two
    return row


def save_ckpt(path: Path, arm: str, args, step: int, model, opt, sched) -> None:
    """Model, optimizer and scheduler at a step. Save only: there is no resume path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": arm, "args": recorded_args(args), "step": step, "state": model.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict()}, path)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
```

The new flags, their validation, and the head of `main` (the fixed windows are loaded first so a bad region fails before the slow n-gram baselines):

Replace in `scripts/run_sparsela.py`:
```python
    p.add_argument("--run-name", default="")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    name = args.run_name or f"sla_{args.task}_s{args.seed}"
    extra = {}
    if args.task != "copy":
        src = Path(args.corpus_file) if args.corpus_file else WIKI
        train = load_bytes(src, max_bytes=args.train_bytes)
```

with:
```python
    p.add_argument("--run-name", default="")
    p.add_argument("--val-fixed", type=int, default=0, help=">0: also score this many fixed non-overlapping windows from --val-offset every --eval-every steps and at the end (cold and warm bits/byte)")
    p.add_argument("--val-offset", type=int, default=0, help="byte offset of the fixed-window region (needs --val-fixed)")
    p.add_argument("--eval-every", type=int, default=0, help="steps between fixed-window evals, 0 = steps // 6 (needs --val-fixed); the legacy curve keeps its steps // 6 cadence")
    p.add_argument("--meter", action="store_true", help="sample the card once a second while training: joules, power, temperature, contention; aborts at 85 C (a no-op on CPU)")
    p.add_argument("--ckpt-every", type=int, default=0, help=">0: save model, optimizer and scheduler every this many steps (save only, no resume)")
    args = p.parse_args(argv)
    if min(args.val_fixed, args.val_offset, args.eval_every, args.ckpt_every) < 0:
        p.error("--val-fixed, --val-offset, --eval-every and --ckpt-every must be >= 0")
    if (args.val_offset or args.eval_every) and not args.val_fixed:
        p.error("--val-offset and --eval-every belong to the fixed-window eval and need --val-fixed N")
    if args.val_fixed and args.task == "copy":
        p.error("--val-fixed scores corpus windows; use a text task, not copy")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    name = args.run_name or f"sla_{args.task}_s{args.seed}"
    extra = {}
    xf = yf = None
    if args.task != "copy":
        src = Path(args.corpus_file) if args.corpus_file else WIKI
        if args.val_fixed:  # first: a short region or a bad offset fails now, not after the slow baselines below
            xf, yf = fixed_windows(load_bytes(src, max_bytes=args.val_fixed * args.seq_len + 1, offset=args.val_offset), args.seq_len, args.val_fixed)
        train = load_bytes(src, max_bytes=args.train_bytes)
```

The arm loop, in full as it reads after the change:

Replace in `scripts/run_sparsela.py`:
```python
    rows = []
    for arm in args.arms.split(","):
        model = build(arm, args).to(device)
        if args.state_bits and hasattr(model, "all_heads"):
            for h in model.all_heads(): h.state_bits = int(args.state_bits)
        opt = make_optimizer(model, args)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(args.steps, args.warmup, args.cosine))
        torch.manual_seed(args.seed)
        t0 = time.perf_counter(); curve = []
        for step in range(args.steps):
            if args.task == "copy":
                x, y, _ = _copy(args, args.seed * 100_000 + step)
            else:
                x, y = batch_windows(train, seq_len=args.seq_len, batch=args.batch, seed=args.seed * 100_000 + step)
            x, y = x.to(device), y.to(device)
            with _amp(args, device):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if (step + 1) % max(1, args.steps // 6) == 0:
                row = {"step": step + 1, "train_bpb": float(loss) / math.log(2)}
                row.update(eval_copy(model, args, 2, 10_000 + step, device) if args.task == "copy"
                           else eval_wiki(model, val, args, 4, 10_000 + step, device))
                curve.append(row); print(f"  {arm} {row}", flush=True)
        rep = eval_copy(model, args, 8, 20_000, device) if args.task == "copy" else eval_wiki(model, val, args, 16, 20_000, device)
        if args.save:
            (RESULTS / "ckpt").mkdir(exist_ok=True)
            torch.save({"arm": arm, "args": vars(args), "state": model.state_dict()}, RESULTS / "ckpt" / f"{name}_{arm}.pt")
        rep.update(arm=arm, seconds=time.perf_counter() - t0, trainable_scalars=numel(model), curve=curve,
                   flops_per_token_recurrent=model.flops_per_token_recurrent() if hasattr(model, "flops_per_token_recurrent") else model.flops_per_step(1))
        metric = {k: round(v, 3) for k, v in rep.items() if k.startswith(("copy", "val"))}
        print(f"{arm:10} {metric} scalars={rep['trainable_scalars']} rflops={rep['flops_per_token_recurrent']} sec={rep['seconds']:.0f}", flush=True)
        rows.append(rep)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps({"run_name": name, **vars(args), **extra, "arms": rows}, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
```

with:
```python
    rows = []
    for arm in args.arms.split(","):
        model = build(arm, args).to(device)
        if args.state_bits and hasattr(model, "all_heads"):
            for h in model.all_heads(): h.state_bits = int(args.state_bits)
        opt = make_optimizer(model, args)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(args.steps, args.warmup, args.cosine))
        torch.manual_seed(args.seed)
        timed = yardstick_on(args)
        clock = SplitClock(sync=(lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None))
        boundary = clock.boundary if timed else nullcontext  # every eval and save runs inside one; a no-op unless a yardstick flag is set
        meter = make_meter(args, device)
        curve_fixed = []
        fixed_every = args.eval_every or max(1, args.steps // 6)
        if timed and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter(); curve = []
        if meter:
            meter.start()
        if timed:
            clock.start()
        try:
            for step in range(args.steps):
                if args.task == "copy":
                    x, y, _ = _copy(args, args.seed * 100_000 + step)
                else:
                    x, y = batch_windows(train, seq_len=args.seq_len, batch=args.batch, seed=args.seed * 100_000 + step)
                x, y = x.to(device), y.to(device)
                with _amp(args, device):
                    logits, _ = model(x)
                    loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
                if meter and (step + 1) % CHECK_EVERY == 0:
                    meter.check()
                if (step + 1) % max(1, args.steps // 6) == 0:
                    with boundary():
                        row = {"step": step + 1, "train_bpb": float(loss) / math.log(2)}
                        row.update(eval_copy(model, args, 2, 10_000 + step, device) if args.task == "copy"
                                   else eval_wiki(model, val, args, 4, 10_000 + step, device))
                        curve.append(row); print(f"  {arm} {row}", flush=True)
                if args.val_fixed and (step + 1) % fixed_every == 0 and step + 1 < args.steps:  # the last step is scored once, below
                    with boundary():
                        curve_fixed.append(fixed_row(arm, model, xf, yf, args, device, step + 1, clock.train_seconds))
                if args.ckpt_every and (step + 1) % args.ckpt_every == 0:
                    with boundary():
                        save_ckpt(RESULTS / "ckpt" / f"{name}_{arm}.step{step + 1}.pt", arm, args, step + 1, model, opt, sched)
                        print(f"saved {name}_{arm}.step{step + 1}.pt", flush=True)
            with boundary():  # closes train_seconds; everything below is eval_seconds
                rep = eval_copy(model, args, 8, 20_000, device) if args.task == "copy" else eval_wiki(model, val, args, 16, 20_000, device)
                if args.val_fixed:
                    curve_fixed.append(fixed_row(arm, model, xf, yf, args, device, args.steps, clock.train_seconds))
                if args.save:
                    (RESULTS / "ckpt").mkdir(exist_ok=True)
                    torch.save({"arm": arm, "args": recorded_args(args), "state": model.state_dict()}, RESULTS / "ckpt" / f"{name}_{arm}.pt")
        finally:
            if meter:
                meter.stop()
        rep.update(arm=arm, seconds=time.perf_counter() - t0, trainable_scalars=numel(model), curve=curve,
                   flops_per_token_recurrent=model.flops_per_token_recurrent() if hasattr(model, "flops_per_token_recurrent") else model.flops_per_step(1))
        metric = {k: round(v, 3) for k, v in rep.items() if k.startswith(("copy", "val"))}
        print(f"{arm:10} {metric} scalars={rep['trainable_scalars']} rflops={rep['flops_per_token_recurrent']} sec={rep['seconds']:.0f}", flush=True)
        if timed:
            rep.update(train_seconds=clock.train_seconds, eval_seconds=clock.eval_seconds, bytes_seen=args.steps * args.batch * args.seq_len,
                       peak_vram_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0)
            if args.val_fixed:
                rep.update(curve_fixed=curve_fixed, val_fixed={"offset": args.val_offset, "n_windows": args.val_fixed,
                                                               "bpb": curve_fixed[-1]["bpb"], "bpb_warm": curve_fixed[-1]["bpb_warm"]})
            if meter:
                rep.update(meter.summary(), gpu_uuid=meter.gpu_uuid)
            print(f"{arm:10} yardstick train_s={rep['train_seconds']:.1f} eval_s={rep['eval_seconds']:.1f} peak_vram_gib={rep['peak_vram_bytes'] / 2**30:.2f}"
                  + (f" joules={rep['joules']:.0f} contended={rep['contended']}" if meter else ""), flush=True)
        rows.append(rep)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps({"run_name": name, **recorded_args(args), **extra, "arms": rows}, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_yardstick.py -q`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add scripts/run_sparsela.py tests/test_yardstick.py
git commit -m "Runner: yardstick flags (--val-fixed, --val-offset, --eval-every, --meter, --ckpt-every), all off by default"
```

---

### Task 7: cross_eval --fixed

**Files:**
- Modify: `scripts/cross_eval.py` (import, `evaluate_fixed`, `--fixed`, the length loop)
- Test: `tests/test_cross_eval_fixed.py`

**Interfaces:**
- Consumes: `fixed_windows` (Task 1).
- Produces: `evaluate_fixed(model, data, T, batch, n_windows, device) -> dict` (the row `evaluate` returns plus `bpb_warm` and `n_windows`; `bpb` is cold) and `--fixed N` (default 0). With `--fixed N > 0`, N non-overlapping windows from `--offset` are scored at every length in `--lens`; `--bytes` and `--n-batches` are ignored; the region needed is `N * max(lens) + 1` bytes and a shorter region raises before any checkpoint loads. With `--fixed` absent the rows, the printed line and the JSON `args` are what they were (`fixed` is left out of `args`). Always pass `--offset` with `--fixed`: its default, 9,100,000, is inside the kernel training bytes.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cross_eval_fixed.py`:
```python
"""cross_eval.py --fixed: the yardstick's scorer for saved checkpoints. CPU only; the checkpoint and corpus are written here."""

from __future__ import annotations

import json
import math
import sys

import pytest
import torch
from torch.nn import functional as F

from flybrain.data.text import fixed_windows, load_bytes
from flybrain.model.hybrid import GRUHybrid

CK_ARGS = dict(d_model=16, gru_hidden=16, n_layers=1, d_ff=32, learned_key_dim=16, key_conv=2, seed=1, seq_len=64)
LEGACY_ROW_KEYS = ["T", "bpb", "bpb_first_eighth", "bpb_last_eighth", "bpb_copyable", "bpb_noncopyable", "ckpt", "arm"]


def _files(tmp_path):
    torch.manual_seed(1)
    m = GRUHybrid(d_model=16, hidden=16, backbone="mingru", where="none", n_layers=1, d_ff=32, max_len=1024)
    ck, corpus = tmp_path / "tiny_mingru.pt", tmp_path / "c.bytes"
    torch.save({"arm": "mingru", "args": CK_ARGS, "state": m.state_dict()}, ck)
    corpus.write_bytes((b"int main(void) { return 0; }\n" * 400)[:6000])
    return ck, corpus


def _main(monkeypatch, tmp_path, *flags):
    import scripts.cross_eval as ce
    ck, corpus = _files(tmp_path)
    monkeypatch.setattr(ce, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["cross_eval.py", "--ckpts", str(ck), "--corpus-file", str(corpus), "--device", "cpu", "--run-name", "out", *flags])
    ce.main()
    return json.loads((tmp_path / "out.json").read_text())


def test_fixed_scores_the_windows_at_the_offset_cold_and_warm(monkeypatch, tmp_path):
    from scripts.cross_eval import rebuild
    out = _main(monkeypatch, tmp_path, "--offset", "100", "--fixed", "5", "--lens", "16,32")
    assert [r["T"] for r in out["rows"]] == [16, 32] and all(r["n_windows"] == 5 for r in out["rows"])
    m = rebuild(torch.load(tmp_path / "tiny_mingru.pt", weights_only=False)).eval()
    data = load_bytes(tmp_path / "c.bytes", max_bytes=5 * 32 + 1, offset=100)
    for r in out["rows"]:
        x, y = fixed_windows(data, r["T"], 5)                             # independent reference over the same bytes
        with torch.no_grad():
            ce = F.cross_entropy(m(x)[0].reshape(-1, 256), y.reshape(-1), reduction="none").view(5, r["T"])
        assert r["bpb"] == pytest.approx(float(ce.mean()) / math.log(2), abs=1e-5)
        assert r["bpb_warm"] == pytest.approx(float(ce[:, r["T"] // 2:].mean()) / math.log(2), abs=1e-5)
    assert out["args"]["fixed"] == 5 and set(out["rows"][0]) == set(LEGACY_ROW_KEYS) | {"bpb_warm", "n_windows"}


def test_fixed_is_deterministic_and_ignores_the_random_window_flags(monkeypatch, tmp_path):
    a = _main(monkeypatch, tmp_path, "--offset", "100", "--fixed", "5", "--lens", "32")["rows"]
    state = torch.get_rng_state()
    b = _main(monkeypatch, tmp_path, "--offset", "100", "--fixed", "5", "--lens", "32", "--n-batches", "9", "--bytes", "999")["rows"]
    assert a == b and torch.equal(state, torch.get_rng_state())


def test_without_fixed_the_rows_and_the_recorded_args_are_the_legacy_ones(monkeypatch, tmp_path, capsys):
    out = _main(monkeypatch, tmp_path, "--offset", "100", "--lens", "32", "--n-batches", "2")
    assert list(out["rows"][0]) == LEGACY_ROW_KEYS and "fixed" not in out["args"]
    assert " warm=" not in capsys.readouterr().out


def test_fixed_prints_the_warm_number(monkeypatch, tmp_path, capsys):
    _main(monkeypatch, tmp_path, "--offset", "100", "--fixed", "3", "--lens", "32")
    assert " warm=" in capsys.readouterr().out


def test_a_region_too_short_for_the_windows_fails_before_any_checkpoint_loads(monkeypatch, tmp_path):
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("a checkpoint was loaded"))
    with pytest.raises(ValueError, match="need 321 bytes for 10 windows of 32"):
        _main(monkeypatch, tmp_path, "--offset", "5900", "--fixed", "10", "--lens", "16,32")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cross_eval_fixed.py -q -x`
Expected: FAIL with "SystemExit: 2"

- [ ] **Step 3: Write minimal implementation**

Replace in `scripts/cross_eval.py`:
```python
from flybrain.data.text import batch_windows, load_bytes
```

with:
```python
from flybrain.data.text import batch_windows, fixed_windows, load_bytes
```

Replace in `scripts/cross_eval.py`:
```python
def main():
```

with:
```python
@torch.no_grad()
def evaluate_fixed(model, data, T, batch, n_windows, device):
    """Score n_windows non-overlapping windows from the start of data, no RNG: the yardstick's scorer for saved checkpoints.
    The row evaluate() returns plus bpb_warm (mean over the window's second half) and n_windows; bpb is the cold mean."""
    model.eval()
    xs, ys = fixed_windows(data, T, n_windows)
    tot = torch.zeros(T, device=device, dtype=torch.float64); cp = [0.0, 0]; ncp = [0.0, 0]
    for i in range(0, n_windows, batch):
        x, y = xs[i:i + batch], ys[i:i + batch]
        cm = copyable_mask(x, y, n=8).to(device)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="none").view(-1, T)
        tot += ce.double().sum(0)
        cp[0] += float(ce[cm].sum()); cp[1] += int(cm.sum()); ncp[0] += float(ce[~cm].sum()); ncp[1] += int((~cm).sum())
    per = tot / n_windows / math.log(2)
    q = T // 8
    return {"T": T, "bpb": float(per.mean()), "bpb_first_eighth": float(per[:q].mean()), "bpb_last_eighth": float(per[-q:].mean()),
            "bpb_copyable": cp[0] / max(cp[1], 1) / math.log(2), "bpb_noncopyable": ncp[0] / max(ncp[1], 1) / math.log(2),
            "bpb_warm": float(per[T // 2:].mean()), "n_windows": n_windows}


def main():
```

Replace in `scripts/cross_eval.py`:
```python
    p.add_argument("--bf16", action="store_true", help="cast the whole model to bfloat16 for scoring")
    args = p.parse_args()
    device = torch.device(args.device)
    data = load_bytes(args.corpus_file, max_bytes=args.bytes, offset=args.offset)
```

with:
```python
    p.add_argument("--bf16", action="store_true", help="cast the whole model to bfloat16 for scoring")
    p.add_argument("--fixed", type=int, default=0, help=">0: score this many fixed non-overlapping windows from --offset at each length (cold bpb, plus bpb_warm over the window's second half; no RNG) instead of seeded random windows; --bytes and --n-batches are then ignored")
    args = p.parse_args()
    device = torch.device(args.device)
    lens = [int(t) for t in args.lens.split(",")]
    if args.fixed:
        data = load_bytes(args.corpus_file, max_bytes=args.fixed * max(lens) + 1, offset=args.offset)
        fixed_windows(data, max(lens), args.fixed)  # a region too short for the windows fails now, before any checkpoint loads
    else:
        data = load_bytes(args.corpus_file, max_bytes=args.bytes, offset=args.offset)
```

Replace in `scripts/cross_eval.py`:
```python
        for T in [int(t) for t in args.lens.split(",")]:
            batch = max(1, 8192 // T)
            if args.ntk_train_len and getattr(getattr(m, "core", None), "rope", False):
                m.core.set_rope_scale(max(1.0, T / args.ntk_train_len))
            r = evaluate(m, data, T, batch, args.n_batches or max(2, 32 // batch), device)
            r.update(ckpt=Path(c).name, arm=ck["arm"])
            print(f"{ck['arm']:8} T={T:5d} bpb={r['bpb']:.3f} first8={r['bpb_first_eighth']:.3f} last8={r['bpb_last_eighth']:.3f} copyable={r['bpb_copyable']:.3f} non={r['bpb_noncopyable']:.3f}", flush=True)
            rows.append(r)
    (RESULTS / f"{args.run_name}.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2) + "\n")
```

with:
```python
        for T in lens:
            batch = max(1, 8192 // T)
            if args.ntk_train_len and getattr(getattr(m, "core", None), "rope", False):
                m.core.set_rope_scale(max(1.0, T / args.ntk_train_len))
            r = (evaluate_fixed(m, data, T, batch, args.fixed, device) if args.fixed
                 else evaluate(m, data, T, batch, args.n_batches or max(2, 32 // batch), device))
            r.update(ckpt=Path(c).name, arm=ck["arm"])
            print(f"{ck['arm']:8} T={T:5d} bpb={r['bpb']:.3f} first8={r['bpb_first_eighth']:.3f} last8={r['bpb_last_eighth']:.3f} copyable={r['bpb_copyable']:.3f} non={r['bpb_noncopyable']:.3f}"
                  + (f" warm={r['bpb_warm']:.3f}" if args.fixed else ""), flush=True)
            rows.append(r)
    recorded = vars(args) if args.fixed else {k: v for k, v in vars(args).items() if k != "fixed"}  # a legacy command line writes the legacy file
    (RESULTS / f"{args.run_name}.json").write_text(json.dumps({"args": recorded, "rows": rows}, indent=2) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cross_eval_fixed.py -q`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add scripts/cross_eval.py tests/test_cross_eval_fixed.py
git commit -m "cross_eval: --fixed scores fixed windows, cold and warm, for final scoring of saved checkpoints"
```

---

### Task 8: scripts/run_spine.sh

**Files:**
- Create: `scripts/run_spine.sh` (executable)
- Test: `tests/test_spine_script.py`

**Interfaces:**
- Consumes: the runner flags of Task 6.
- Produces: `scripts/run_spine.sh <device> <hybrid|tf|zeroheads> <seed> [steps] [tag]`. Run names `spine16_hybrid_s$S`, `spine16_tf_s$S`, `spine16_zeroheads_s$S`; with a tag, `spine16_<arm>_<tag>_s$S` (an equal-seconds rerun and a rerun after a contended run need a new name, and the runner skips a name whose JSON exists). `CORPORA` resolves as in `run_50m.sh`. `DRY_RUN=1` prints `<run name> <flags>` and exits before the skip check.

Flags checked against `scripts/run_20m.sh`, `scripts/run_capacity.sh`, the argparse block of `scripts/run_sparsela.py` and the args recorded in `results/m20_linux_depth4local_s0.json` (hybrid) and `results/m20_linux_s1.json` (transformer): every common flag of decision 5 equals `run_20m.sh`'s `COMMON` (same names, same values), the hybrid arm equals `run_capacity.sh`'s `depth4local`, and the transformer arm equals `run_20m.sh`'s `--arms mingru,transformer --rope` with the transformer run alone. **No flag differs in value.** Differences in form only: (1) `--amp --meter --val-fixed 128 --val-offset 202117357 --eval-every 1000` are new; (2) the transformer used to run in one process after the minGRU, now it runs alone, which changes nothing because `build()` and the loop each reseed with `--seed`; (3) `--n-key 4096` is passed to all three arms as in `run_20m.sh` although the transformer and the zero-head cell have no memory head; (4) unlike `run_50m.sh`, the script sets `pipefail` (a failed arm fails the script, so a chain of arms stops), filters the log with `grep --line-buffered` (so the log can be read live) and takes an optional tag; (5) `--ckpt-every` is not used, so the only checkpoints are the final `--save` ones, `results/ckpt/spine16_hybrid_s0_mingru_sparse_post.pt`, `spine16_tf_s0_transformer.pt` and `spine16_zeroheads_s0_mingru.pt`. Parameter counts to expect: hybrid 16,066,824, transformer 17,085,696, zero-head cell 13,930,752.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spine_script.py`:
```python
"""scripts/run_spine.sh: the three spine commands, parsed by the runner's own argparse, so a typo or a drifted value fails here
and not hours into a run. No GPU, no corpus (an empty stand-in file), no uv (a stub on PATH)."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_spine.sh"

COMMON = dict(task="code", train_bytes=200_000_000, val_bytes=2_000_000, normalize=True, key_conv=2, top_k=4, n_key=4096, lr=0.0006,
              decay=1.0, d_model=512, d_ff=1024, n_layers=8, batch=8, seq_len=2048, steps=16000, save=True, seed=0, device="cuda:1",
              warmup=1000, cosine=True, wd=0.1, chunk=256, amp=True, meter=True, val_fixed=128, val_offset=202_117_357,
              eval_every=1000, ckpt_every=0)
ARMS = {"hybrid": dict(arms="mingru_sparse_post", gate=True, mem_every=2, local_attn=512, key_convs="2,4", rope=False),
        "tf": dict(arms="transformer", gate=False, mem_every=0, local_attn=0, key_convs="", rope=True),
        "zeroheads": dict(arms="mingru", gate=False, mem_every=0, local_attn=512, key_convs="", rope=False)}


def _dry(tmp_path, *argv):
    corpora = tmp_path / "corpora"
    corpora.mkdir(exist_ok=True)
    (corpora / "linux_full.bytes").write_bytes(b"x")
    return subprocess.run(["bash", str(SCRIPT), *argv], capture_output=True, text=True,
                          env={**os.environ, "DRY_RUN": "1", "CORPORA": str(corpora)})


def _parsed(tmp_path, *argv):
    from scripts.run_sparsela import parse_args
    r = _dry(tmp_path, *argv)
    assert r.returncode == 0, r.stderr
    name, flags = r.stdout.strip().split(" ", 1)
    return name, parse_args(shlex.split(flags))


def test_the_script_is_executable_and_valid_bash():
    assert os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


@pytest.mark.parametrize("arm", ["hybrid", "tf", "zeroheads"])
def test_each_arm_parses_to_the_decided_flags(tmp_path, arm):
    name, args = _parsed(tmp_path, "cuda:1", arm, "0")
    assert name == f"spine16_{arm}_s0"
    for k, v in {**COMMON, **ARMS[arm]}.items():
        assert getattr(args, k) == v, (arm, k)
    assert args.corpus_file == str(tmp_path / "corpora" / "linux_full.bytes")


def test_steps_seed_device_and_tag_reach_the_command(tmp_path):
    name, args = _parsed(tmp_path, "cuda:0", "tf", "1", "18286", "eqsec")
    assert name == "spine16_tf_eqsec_s1" and (args.device, args.seed, args.steps, args.val_fixed, args.eval_every) == ("cuda:0", 1, 18286, 128, 1000)


@pytest.mark.parametrize("arm, baseline", [("hybrid", "m20_linux_depth4local_s0.json"), ("tf", "m20_linux_s1.json")])
def test_each_arm_is_the_recorded_16m_recipe_plus_the_yardstick_flags(tmp_path, arm, baseline):
    """Every training flag the committed 16M run recorded is the value the spine uses (seed, device, corpus path aside)."""
    args = vars(_parsed(tmp_path, "cuda:1", arm, "0")[1])
    recorded = json.loads((ROOT / "results" / baseline).read_text())
    skip = {"run_name", "device", "save", "arms", "seed", "corpus_file", "unigram_bpb", "bigram_bpb"}
    checked = [k for k in recorded if k not in skip and k in args]
    assert len(checked) >= 24
    for k in checked:
        assert args[k] == recorded[k], (arm, k)


def test_an_unknown_arm_and_a_missing_corpus_are_refused(tmp_path):
    assert _dry(tmp_path, "cuda:1", "nope", "0").returncode == 2
    r = subprocess.run(["bash", str(SCRIPT), "cuda:1", "tf", "0"], capture_output=True, text=True,
                       env={**os.environ, "CORPORA": str(tmp_path / "empty")})
    assert r.returncode == 2 and "corpus not found" in r.stdout


def _sandbox(tmp_path, uv_body: str) -> tuple[Path, dict]:
    """A copy of the script in a scratch repo root, a stand-in corpus, and a stub uv on PATH."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "run_spine.sh")
    (tmp_path / "results").mkdir()
    (tmp_path / "corpora").mkdir()
    (tmp_path / "corpora" / "linux_full.bytes").write_bytes(b"x")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "uv").write_text("#!/usr/bin/env bash\n" + uv_body)
    (tmp_path / "bin" / "uv").chmod(0o755)
    return tmp_path / "scripts" / "run_spine.sh", {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "CORPORA": str(tmp_path / "corpora")}


def test_a_finished_run_name_is_skipped_without_starting_anything(tmp_path):
    script, env = _sandbox(tmp_path, "echo STARTED; exit 0\n")
    (tmp_path / "results" / "spine16_hybrid_s0.json").write_text("{}")
    r = subprocess.run(["bash", str(script), "cuda:1", "hybrid", "0"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "skip spine16_hybrid_s0" in r.stdout and "STARTED" not in r.stdout


def test_the_runner_is_started_with_the_run_name_and_its_output_is_filtered_and_unbuffered(tmp_path):
    script, env = _sandbox(tmp_path, 'echo "uv $*"; echo "  curve line"; echo "kept line"; exit 0\n')
    r = subprocess.run(["bash", str(script), "cuda:1", "zeroheads", "1"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "=== spine16_zeroheads_s1" in r.stdout
    assert "uv run python scripts/run_sparsela.py --run-name spine16_zeroheads_s1" in r.stdout
    assert "kept line" in r.stdout and "curve line" not in r.stdout           # two-space lines are dropped, like run_50m.sh
    assert "--line-buffered" in SCRIPT.read_text()                            # the log is read while the run is going


def test_a_failing_runner_fails_the_script_so_a_chain_of_arms_stops(tmp_path):
    script, env = _sandbox(tmp_path, 'echo "Traceback: RuntimeError: GPU reached 90 C"; exit 3\n')
    r = subprocess.run(["bash", str(script), "cuda:1", "tf", "0"], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "Traceback" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_spine_script.py -q -x`
Expected: FAIL with "assert False"

- [ ] **Step 3: Write minimal implementation**

Create `scripts/run_spine.sh` (then `chmod +x scripts/run_spine.sh`):
```bash
#!/usr/bin/env bash
# Spine runs (group-learning spec 1.6): 16M, 16k steps, the run_20m.sh recipe under bf16 autocast, metered, with the dev fixed-window eval.
#   hybrid    : 4 memory heads (--mem-every 2) + one 512-window attention layer + key widths 2,4   (run_capacity.sh depth4local)
#   tf        : RoPE transformer, same d / layers / d_ff                                           (run_20m.sh transformer arm)
#   zeroheads : the same recurrence + local attention with no memory head (the Samba cell)
# usage: scripts/run_spine.sh <device> <hybrid|tf|zeroheads> <seed> [steps] [tag]
#   tag goes into the run name (spine16_<arm>_<tag>_s<seed>): an equal-seconds rerun with other steps is a new name, not a skip.
# DRY_RUN=1 prints "<run name> <flags>" and exits; tests/test_spine_script.py parses it with the runner's own argparse.
set -u -o pipefail
DEV=${1:-cuda:1}; ARM=${2:-hybrid}; S=${3:-0}; STEPS=${4:-16000}; TAG=${5:-}
C=${CORPORA:-$(cd "$(dirname "$0")/.." && pwd)/corpora}
cd "$(dirname "$0")/.."
FILE=$C/linux_full.bytes
[ -f "$FILE" ] || { echo "corpus not found: $FILE"; echo "build it with scripts/build_corpus.py, or set CORPORA=<dir>"; exit 2; }
COMMON="--task code --corpus-file $FILE --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --steps $STEPS --save --seed $S --device $DEV --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 128 --val-offset 202117357 --eval-every 1000"
case $ARM in
  hybrid)    EXTRA="--arms mingru_sparse_post --gate --mem-every 2 --local-attn 512 --key-convs 2,4" ;;
  tf)        EXTRA="--arms transformer --rope" ;;
  zeroheads) EXTRA="--arms mingru --local-attn 512" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
NAME=spine16_${ARM}${TAG:+_$TAG}_s$S
[ -n "${DRY_RUN:-}" ] && { echo "$NAME $COMMON $EXTRA"; exit 0; }
[ -f results/$NAME.json ] && { echo "skip $NAME"; exit 0; }
echo "=== $NAME"
uv run python scripts/run_sparsela.py --run-name "$NAME" $COMMON $EXTRA 2>&1 | grep --line-buffered -v "^  "
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_spine_script.py -q`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
chmod +x scripts/run_spine.sh
git add scripts/run_spine.sh tests/test_spine_script.py
git commit -m "Spine: run_spine.sh launches the hybrid, transformer and zero-head arms on one recipe"
```

---

### Task 9: spine_report.py

**Files:**
- Create: `scripts/spine_report.py`
- Test: `tests/test_spine_report.py`

**Interfaces:**
- Consumes: `results/spine16_<arm>[_<tag>]_s<seed>.json` (Task 6 fields) and `results/spine16_score_<split>_*.json` (Task 7 rows at length 2048; splits `dev`, `test`, `legacy`, `dev128`).
- Produces: `load_runs(results)`, `load_scores(results, runs)`, `decide(rows, steps) -> {"seed1": [arm], "eqsec": [{"arm", "steps", "target_seconds"}]}`, `commands(decision, device="cuda:1") -> list[str]`, `table(runs, scores) -> str`, `report(results, device) -> str`, and the CLI `uv run python scripts/spine_report.py [--results DIR] [--device DEV]`.

The two rules, applied to the seed-0 runs on dev cold bits/byte (fp32, `cross_eval.py --fixed`) and `train_seconds`. Seconds "close" means within 5 % of the faster arm's. Rule 1: a pair closer than 0.01 bits/byte, or close in seconds, gets seed 1 for both arms. Rule 2: if the better arm of a pair is also the slower one and the seconds are not close, the faster arm is rerun with its steps scaled by `slow_seconds / fast_seconds` (its own cosine schedule), as `spine16_<arm>_eqsec_s0`; an arm that is beaten by several slower arms is scaled to the slowest of them, so there is one rerun per arm. A pair whose seconds are within 5 % is a seed question, not an equal-seconds one.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spine_report.py`:
```python
"""scripts/spine_report.py: the two decision rules and the Run 53 table, on synthetic result files. CPU only."""

from __future__ import annotations

import json

import pytest

from scripts.spine_report import ARMS, commands, decide, report

CKPT_ARM = {"hybrid": "mingru_sparse_post", "tf": "transformer", "zeroheads": "mingru"}


def rows(h, t, z):
    """(dev cold bpb, train_seconds) for hybrid, tf, zeroheads."""
    return [dict(arm=a, dev_bpb=b, train_seconds=s) for a, (b, s) in zip(ARMS, (h, t, z))]


def test_a_pair_within_5_percent_in_seconds_gets_seed_one_and_nothing_else_is_rerun():
    d = decide(rows((0.70, 7000), (0.75, 7500), (0.90, 6900)), 16000)
    assert d["seed1"] == ["hybrid", "zeroheads"]       # 7000 vs 6900 is 1.4 % apart; 7000 vs 7500 is 7.1 %
    assert d["eqsec"] == [{"arm": "zeroheads", "steps": 17391, "target_seconds": 7500}]   # tf beats it, and is 8.7 % slower


def test_nothing_is_rerun_when_the_better_arm_is_also_the_faster_and_every_pair_is_apart():
    assert decide(rows((0.70, 6000), (0.75, 7000), (0.90, 8000)), 16000) == {"seed1": [], "eqsec": []}


def test_a_better_arm_that_is_slower_by_less_than_5_percent_is_a_seed_question_not_an_equal_seconds_one():
    d = decide(rows((0.70, 7100), (0.80, 7000), (0.95, 9000)), 16000)    # hybrid is better and 1.4 % slower than tf
    assert d == {"seed1": ["hybrid", "tf"], "eqsec": []}


def test_the_faster_worse_arms_are_rerun_at_the_slowest_better_arms_seconds():
    d = decide(rows((0.70, 8000), (0.75, 6000), (0.90, 5000)), 16000)
    assert d["seed1"] == []
    assert d["eqsec"] == [{"arm": "tf", "steps": 21333, "target_seconds": 8000}, {"arm": "zeroheads", "steps": 25600, "target_seconds": 8000}]


def test_the_thresholds_are_strict_at_0_01_bits_and_5_percent():
    assert decide(rows((0.70, 6000), (0.7099, 7200), (0.9, 9000)), 16000)["seed1"] == ["hybrid", "tf"]      # 0.0099 apart
    assert decide(rows((0.70, 6000), (0.7101, 7200), (0.9, 9000)), 16000)["seed1"] == []                    # 0.0101 apart
    assert decide(rows((0.70, 6000), (0.80, 6294), (0.9, 9000)), 16000)["seed1"] == ["hybrid", "tf"]        # 4.9 % apart
    assert decide(rows((0.70, 6000), (0.80, 6306), (0.9, 9000)), 16000)["seed1"] == []                      # 5.1 % apart


def test_one_pair_can_trigger_both_rules():
    d = decide(rows((0.700, 7800), (0.705, 6000), (0.95, 9000)), 16000)    # 0.005 apart, and the better arm is 30 % slower
    assert d["seed1"] == ["hybrid", "tf"] and d["eqsec"] == [{"arm": "tf", "steps": 20800, "target_seconds": 7800}]


def test_commands_name_the_runs_the_decision_asks_for():
    d = {"seed1": ["hybrid", "tf"], "eqsec": [{"arm": "zeroheads", "steps": 17391, "target_seconds": 7500}]}
    assert commands(d) == ["scripts/run_spine.sh cuda:1 hybrid 1", "scripts/run_spine.sh cuda:1 tf 1",
                           "scripts/run_spine.sh cuda:1 zeroheads 0 17391 eqsec"]
    assert commands({"seed1": [], "eqsec": []}, "cuda:0") == []


def _write(results, spec, scores=True, d128_shift=0.0):
    """spec[arm] = (name, dev, seconds); writes the run JSONs and the four score JSONs the report reads."""
    for arm, (name, dev, seconds) in spec.items():
        (results / f"{name}.json").write_text(json.dumps({"run_name": name, "steps": 16000, "arms": [{
            "arm": CKPT_ARM[arm], "trainable_scalars": 16_000_000, "train_seconds": seconds, "eval_seconds": 60.0, "peak_vram_bytes": 9 * 2**30,
            "joules": 1.0e6, "mean_power_w": 148.0, "contended": False, "telemetry_lost": False, "val_bpb": 0.76,
            "val_fixed": {"offset": 202117357, "n_windows": 128, "bpb": dev + 0.001, "bpb_warm": dev - 0.05}}]}))
    if not scores:
        return
    for split, shift in (("dev", 0.0), ("test", 0.02), ("legacy", -0.03), ("dev128", d128_shift)):
        rows_ = [{"T": 2048, "bpb": spec[a][1] + shift, "bpb_warm": spec[a][1] + shift - 0.05, "ckpt": f"{spec[a][0]}_{CKPT_ARM[a]}.pt", "arm": CKPT_ARM[a]} for a in spec]
        rows_.append({"T": 512, "bpb": 9.9, "bpb_warm": 9.9, "ckpt": f"{spec['tf'][0]}_transformer.pt", "arm": "transformer"})   # other lengths are ignored
        (results / f"spine16_score_{split}_s0.json").write_text(json.dumps({"args": {}, "rows": rows_}))


def test_report_prints_the_table_the_gaps_and_the_follow_up_commands(tmp_path):
    _write(tmp_path, {"hybrid": ("spine16_hybrid_s0", 0.70, 7000.0), "tf": ("spine16_tf_s0", 0.75, 7500.0), "zeroheads": ("spine16_zeroheads_s0", 0.90, 6900.0)})
    out = report(tmp_path)
    assert "| run | params | dev cold / warm | test cold / warm | legacy bytes cold | train_seconds | peak VRAM GiB | joules | contended |" in out
    assert "| `spine16_hybrid_s0` | 16.00M | 0.700 / 0.650 | 0.720 / 0.670 | 0.670 | 7,000 | 9.0 | 1,000,000 | no |" in out
    assert out.index("spine16_hybrid_s0") < out.index("spine16_tf_s0") < out.index("spine16_zeroheads_s0")
    assert "hybrid - tf: cold bpb dev -0.050, test -0.050, legacy -0.050; train_seconds 7,000 vs 7,500 (-6.7%)" in out
    assert "scripts/run_spine.sh cuda:1 hybrid 1" in out and "scripts/run_spine.sh cuda:1 zeroheads 0 17391 eqsec" in out
    assert "scripts/run_spine.sh cuda:1 tf 1" not in out
    assert "consistency spine16_hybrid_s0: in-run bf16 0.7010 vs fp32 re-score 0.7000, delta +0.0010" in out and "WARNING" not in out
    assert "runner val_bpb (128 random windows, the number Runs 37-49 report): hybrid 0.7600, tf 0.7600, zeroheads 0.7600" in out


def test_report_flags_an_inrun_score_that_disagrees_with_the_fp32_rescore(tmp_path):
    _write(tmp_path, {"hybrid": ("spine16_hybrid_s0", 0.70, 7000.0), "tf": ("spine16_tf_s0", 0.75, 7500.0), "zeroheads": ("spine16_zeroheads_s0", 0.90, 6900.0)}, d128_shift=0.02)
    assert "delta -0.0190  WARNING: above 0.01" in report(tmp_path)


def test_report_warns_about_a_contended_run_and_says_what_is_missing(tmp_path):
    _write(tmp_path, {"hybrid": ("spine16_hybrid_s0", 0.70, 7000.0), "tf": ("spine16_tf_s0", 0.75, 7500.0)}, scores=False)
    doc = json.loads((tmp_path / "spine16_tf_s0.json").read_text())
    doc["arms"][0].update(contended=True, telemetry_lost=True)
    (tmp_path / "spine16_tf_s0.json").write_text(json.dumps(doc))
    out = report(tmp_path)
    assert "WARNING spine16_tf_s0: contended=True telemetry_lost=True; its train_seconds is not clean" in out
    assert "| yes, telemetry lost |" in out and "n/a" in out
    assert "incomplete: need the three seed-0 runs and their dev scores" in out


def test_follow_up_runs_join_the_table_but_not_the_rules(tmp_path):
    spec = {"hybrid": ("spine16_hybrid_s0", 0.70, 7000.0), "tf": ("spine16_tf_s0", 0.75, 7500.0), "zeroheads": ("spine16_zeroheads_s0", 0.90, 6900.0)}
    _write(tmp_path, spec)
    _write(tmp_path, {"hybrid": ("spine16_hybrid_s1", 0.701, 7010.0)}, scores=False)
    _write(tmp_path, {"zeroheads": ("spine16_zeroheads_eqsec_s0", 0.88, 7500.0)}, scores=False)
    out = report(tmp_path)
    assert out.index("spine16_hybrid_s0") < out.index("spine16_zeroheads_s0") < out.index("spine16_zeroheads_eqsec_s0") < out.index("spine16_hybrid_s1")
    assert "| `spine16_hybrid_s1` | 16.00M | n/a | n/a | n/a | 7,010 |" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_spine_report.py -q -x`
Expected: FAIL with "No module named 'scripts.spine_report'"

- [ ] **Step 3: Write minimal implementation**

Create `scripts/spine_report.py`:
```python
#!/usr/bin/env python3
"""Spine write-up helper (group-learning spec 1.6): the Run 53 table, the pair gaps and the two decision rules, read from results/.

Runs: results/spine16_<arm>[_<tag>]_s<seed>.json (run_spine.sh). Scores: results/spine16_score_<split>_*.json, one cross_eval.py --fixed
call per split (dev, test, legacy, dev128) over the saved checkpoints, length 2048. Prints markdown; changes nothing."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEED_GAP_BPB = 0.01      # a pair closer than this in dev cold bits/byte gets seed 1
SEED_GAP_SECONDS = 0.05  # ... or closer than this in train_seconds, as a fraction of the faster arm's
CONSISTENCY_BPB = 0.01   # in-run bf16 score vs fp32 re-score of the same 128 dev windows
ARMS = ("hybrid", "tf", "zeroheads")
SPLITS = ("dev", "test", "legacy", "dev128")
RUN = re.compile(r"^spine16_(?P<arm>hybrid|tf|zeroheads)(?:_(?P<tag>[a-z0-9]+))?_s(?P<seed>\d+)$")
SCORE = re.compile(r"^spine16_score_(?P<split>dev128|dev|test|legacy)(?:_.*)?$")


def load_runs(results: Path) -> dict[str, dict]:
    """Every spine run JSON by run name, reduced to the fields the table and the rules read."""
    runs = {}
    for path in sorted(results.glob("spine16_*.json")):
        m = RUN.match(path.stem)
        if not m:
            continue
        doc = json.loads(path.read_text())
        a = doc["arms"][0]
        runs[path.stem] = {"name": path.stem, "arm": m["arm"], "tag": m["tag"] or "", "seed": int(m["seed"]), "steps": doc["steps"],
                           "params": a["trainable_scalars"], "train_seconds": a["train_seconds"], "peak_vram_bytes": a["peak_vram_bytes"],
                           "joules": a.get("joules"), "contended": a.get("contended"), "telemetry_lost": a.get("telemetry_lost"),
                           "runner_val_bpb": a["val_bpb"], "inrun_dev_bpb": a["val_fixed"]["bpb"]}
    return runs


def load_scores(results: Path, runs: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """scores[split][run name] = the length-2048 row of cross_eval.py --fixed for that run's checkpoint."""
    scores: dict[str, dict[str, dict]] = {s: {} for s in SPLITS}
    for path in sorted(results.glob("spine16_score_*.json")):
        m = SCORE.match(path.stem)
        if not m:
            continue
        for row in json.loads(path.read_text())["rows"]:
            for name in runs:
                if row["T"] == 2048 and row["ckpt"].startswith(name + "_"):
                    scores[m["split"]][name] = row
    return scores


def decide(rows: list[dict], steps: int) -> dict:
    """The two rules. rows: {"arm", "dev_bpb", "train_seconds"} for each seed-0 arm.
    Rule 1: a pair closer than 0.01 bits/byte, or within 5 % in train_seconds (of the faster arm's), gets seed 1 for both arms.
    Rule 2: where the better arm is also the slower one (seconds differ by 5 % or more), the faster arm is rerun with its steps
    scaled to the slower arm's train_seconds, its own cosine schedule; one rerun per arm, scaled to the slowest arm that beats it."""
    seed1: set[str] = set()
    eqsec: dict[str, dict] = {}
    for a, b in itertools.combinations(rows, 2):
        fast, slow = sorted((a, b), key=lambda r: r["train_seconds"])
        gap_s = (slow["train_seconds"] - fast["train_seconds"]) / fast["train_seconds"]
        if abs(a["dev_bpb"] - b["dev_bpb"]) < SEED_GAP_BPB or gap_s < SEED_GAP_SECONDS:
            seed1 |= {a["arm"], b["arm"]}
        if gap_s >= SEED_GAP_SECONDS and slow["dev_bpb"] < fast["dev_bpb"]:
            target = max(slow["train_seconds"], eqsec.get(fast["arm"], {}).get("target_seconds", 0.0))
            eqsec[fast["arm"]] = {"arm": fast["arm"], "steps": round(steps * target / fast["train_seconds"]), "target_seconds": target}
    return {"seed1": [arm for arm in ARMS if arm in seed1], "eqsec": [eqsec[arm] for arm in ARMS if arm in eqsec]}


def commands(decision: dict, device: str = "cuda:1") -> list[str]:
    cmds = [f"scripts/run_spine.sh {device} {arm} 1" for arm in decision["seed1"]]
    return cmds + [f"scripts/run_spine.sh {device} {e['arm']} 0 {e['steps']} eqsec" for e in decision["eqsec"]]


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _cw(row):
    return "n/a" if row is None else f"{_f(row['bpb'])} / {_f(row['bpb_warm'])}"


def table(runs: dict[str, dict], scores: dict[str, dict[str, dict]]) -> str:
    head = ("| run | params | dev cold / warm | test cold / warm | legacy bytes cold | train_seconds | peak VRAM GiB | joules | contended |\n"
            "|---|---:|---|---|---:|---:|---:|---:|---|")
    lines = [head]
    for r in sorted(runs.values(), key=lambda r: (r["seed"], r["tag"], ARMS.index(r["arm"]))):
        g = lambda split: scores[split].get(r["name"])
        legacy = g("legacy")
        joules = "n/a" if r["joules"] is None else f"{r['joules']:,.0f}"
        contended = "n/a" if r["contended"] is None else ("yes" if r["contended"] else "no") + (", telemetry lost" if r["telemetry_lost"] else "")
        lines.append(f"| `{r['name']}` | {r['params'] / 1e6:.2f}M | {_cw(g('dev'))} | {_cw(g('test'))} | {_f(legacy and legacy['bpb'])} | "
                     f"{r['train_seconds']:,.0f} | {r['peak_vram_bytes'] / 2**30:.1f} | {joules} | {contended} |")
    return "\n".join(lines)


def report(results: Path = RESULTS, device: str = "cuda:1") -> str:
    runs = load_runs(results)
    scores = load_scores(results, runs)
    out = [table(runs, scores), ""]
    base = {r["arm"]: r for r in runs.values() if r["seed"] == 0 and not r["tag"]}
    for name, r in runs.items():  # in-run bf16 vs fp32 re-score of the same 128 dev windows
        d128 = scores["dev128"].get(name)
        if d128:
            delta = r["inrun_dev_bpb"] - d128["bpb"]
            out.append(f"consistency {name}: in-run bf16 {r['inrun_dev_bpb']:.4f} vs fp32 re-score {d128['bpb']:.4f}, delta {delta:+.4f}"
                       + ("  WARNING: above 0.01, find out why before reporting" if abs(delta) > CONSISTENCY_BPB else ""))
    if base:
        out.append("runner val_bpb (128 random windows, the number Runs 37-49 report): " + ", ".join(f"{a} {base[a]['runner_val_bpb']:.4f}" for a in ARMS if a in base))
    for r in runs.values():
        if r["contended"] or r["telemetry_lost"]:
            out.append(f"WARNING {r['name']}: contended={r['contended']} telemetry_lost={r['telemetry_lost']}; its train_seconds is not clean")
    if all(arm in base and base[arm]["name"] in scores["dev"] for arm in ARMS):
        for a, b in itertools.combinations(ARMS, 2):
            na, nb = base[a]["name"], base[b]["name"]
            gaps = ", ".join(f"{s} {scores[s][na]['bpb'] - scores[s][nb]['bpb']:+.3f}" for s in ("dev", "test", "legacy") if na in scores[s] and nb in scores[s])
            out.append(f"{a} - {b}: cold bpb {gaps}; train_seconds {base[a]['train_seconds']:,.0f} vs {base[b]['train_seconds']:,.0f} "
                       f"({base[a]['train_seconds'] / base[b]['train_seconds'] - 1:+.1%})")
        rows = [{"arm": arm, "dev_bpb": scores["dev"][base[arm]["name"]]["bpb"], "train_seconds": base[arm]["train_seconds"]} for arm in ARMS]
        cmds = commands(decide(rows, base["hybrid"]["steps"]), device)
        out += ["", "follow-up runs the two rules call for (one unit, in this order):"] + (cmds or ["none: every pair is apart by 0.01 bits/byte and 5 % in seconds, and the better arm is not the slower"])
    else:
        out.append("incomplete: need the three seed-0 runs and their dev scores before the rules can be applied")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--device", default="cuda:1", help="device printed in the follow-up commands")
    args = p.parse_args()
    print(report(Path(args.results), args.device))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_spine_report.py -q`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add scripts/spine_report.py tests/test_spine_report.py
git commit -m "Spine report: the Run 53 table, the pair gaps and the two decision rules"
```

---

### Task 10: GPU regression and meter smoke test, before any spine run

**Files:**
- Create: `results/regress_fixed_m20_depth4local_s0.json` (written by check B)
- No source changes.

**Interfaces:**
- Consumes: Tasks 1 to 7 (`fixed_windows`, `GpuMeter`, `device_uuid`, `eval_fixed`, `cross_eval.py --fixed`); `results/ckpt/m20_linux_depth4local_s0_mingru_sparse_post.pt`; `corpora/linux_full.bytes`.
- Produces: evidence that the new scorer reproduces the runner's own number, that `device_uuid` names the card nvidia-smi calls GPU 1, that the meter reads the real nvidia-smi, and the new-yardstick baseline of the 16M winner on the first 128 fixed windows of the legacy bytes.

Why check A feeds the runner's own windows. The spec's check, "re-scored through the new fixed-window path on the legacy validation bytes reads 0.764 within 0.002", can only hold on the same windows. The runner's final eval draws 128 random windows (16 batches of 8, seeds 20,000 to 20,015) from the 2 MB at byte 200,100,000. A different 128 windows from the same bytes read differently by sampling alone: Run 40 in the notebook already records a drift of about 0.06 between window samples, the committed `results/extrap_m20_linux_depth4local.json` reads this very checkpoint at 0.7065 on 32 windows from the first 600 kB, and the first 128 fixed windows read 0.8313 cold and 0.7695 warm. Check A therefore sends the runner's own windows through `eval_fixed`; it fails on any bug in the new scoring arithmetic, which a different window set could not show. While writing this plan the same code, run on CPU in fp32 against the real checkpoint and corpus, read **0.763838** on the runner's windows (the committed `val_bpb` is 0.763837731530657) and **0.8313 / 0.7695** (cold / warm) on the first 128 fixed windows, through both `eval_fixed` and `cross_eval.py --fixed`. Check B runs the real CLI path on those fixed windows and must match.

- [ ] **Step 1: Free GPU 1 and read its state**

```bash
cd /games/ailab/lab/flybrain-lab
nvidia-smi --query-compute-apps=pid,used_memory --format=csv     # before: GPU 1 lists the model server (pid and about 19.5 GB); GPU 0 may list another job, which is not ours
systemctl --user stop qwen-server.service
systemctl --user is-active qwen-server.service fleet.service     # expect "inactive" twice; fleet.service stays stopped
nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv   # expect the header line only
nvidia-smi --query-gpu=index,uuid,power.limit --format=csv       # expect GPU 1 at 150.00 W; do not change it
```
If `-i 1` still lists a process after 15 s, stop here: something else owns the card.

- [ ] **Step 2: Check that cuda:1 is the card nvidia-smi calls GPU 1**

```bash
uv run python -c "import torch; from flybrain.meter import device_uuid; print(device_uuid(torch.device('cuda:1')))"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
```
Expected: the first line equals the uuid on the `1,` line (on 2026-10-03: `GPU-51164b62-fd30-253a-5738-55844d46af9c`). If it equals the `0,` line, `cuda:1` is the wrong card: stop, and put `CUDA_DEVICE_ORDER=PCI_BUS_ID` in front of every command of Tasks 10 to 13 and in the unit's `--setenv` in Task 11. A uuid that is not `GPU-` plus 8-4-4-4-12 hex means `format_uuid` raised: read the message and fix it before going on.

- [ ] **Step 3: Check A, the runner's own windows through the new scorer**

```bash
uv run python - <<'EOF'
import argparse, torch
from flybrain.data.text import batch_windows, load_bytes
from scripts.cross_eval import rebuild
from scripts.run_sparsela import eval_fixed
ck = torch.load("results/ckpt/m20_linux_depth4local_s0_mingru_sparse_post.pt", map_location="cpu", weights_only=False)
val = load_bytes("corpora/linux_full.bytes", max_bytes=2_000_000, offset=200_100_000)   # the runner's val bytes: train_bytes + 100_000
xs, ys = zip(*[batch_windows(val, 2048, 8, seed=20_000 + i) for i in range(16)])        # the runner's final eval: 16 batches of 8
dev = torch.device("cuda:1")
out = eval_fixed(rebuild(ck).to(dev), torch.cat(xs), torch.cat(ys), argparse.Namespace(batch=8, amp=False), dev)
print(out)
assert abs(out["bpb"] - 0.7638) < 0.002, out
print("check A ok: the new scorer reads the runner's windows at", round(out["bpb"], 4), "(results/m20_linux_depth4local_s0.json val_bpb 0.7638)")
EOF
```
Expected: `{'bpb': 0.76383..., 'bpb_warm': 0.6875...}` and the `check A ok` line. The agreement should be to about four decimals (the CPU run agreed to 1e-6); the assertion allows the spec's 0.002.

- [ ] **Step 4: Check B, the CLI path on fixed windows**

```bash
uv run python scripts/cross_eval.py --ckpts results/ckpt/m20_linux_depth4local_s0_mingru_sparse_post.pt --corpus-file corpora/linux_full.bytes --offset 200100000 --fixed 128 --lens 2048 --device cuda:1 --run-name regress_fixed_m20_depth4local_s0
```
Expected: `mingru_sparse_post T= 2048 bpb=0.831 first8=1.167 last8=0.750 copyable=0.166 non=1.149 warm=0.769`, each number within 0.002 of those (the CPU values). A cold value near 0.764 here would be the coincidence, not the target. If `bpb` is off by more than 0.002 while check A passed, the bug is in `evaluate_fixed` or the CLI wiring, not in the scoring arithmetic.

- [ ] **Step 5: Meter smoke test on the real card**

```bash
uv run python - <<'EOF'
import os, time, torch
from flybrain.meter import GpuMeter, device_uuid
dev = torch.device("cuda:1")
uid = device_uuid(dev)
torch.zeros(1, device=dev)                      # opens a CUDA context, so this process is a compute app on the card
m = GpuMeter(uid, os.getpid())
m.start(); time.sleep(6); m.stop()
print(uid, m.summary())
EOF
```
Expected: `samples` between 3 and 8 (each sample is two nvidia-smi calls plus a 1 s wait), `contended` False (the process's own PID is not counted), `power_limit_w` 150.0, `max_temp_c` below 85, `telemetry_lost` False, `joules` above 0. `contended` True means a process other than this one is on GPU 1: find it with `nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv` before going on. `samples` 0 with `telemetry_lost` still False (under 60 s) means every sample failed: run the two queries of Task 2 by hand and compare with `parse`.

- [ ] **Step 6: Commit**
```bash
git add results/regress_fixed_m20_depth4local_s0.json
git commit -m "Regression: the new scorer reproduces the 16M winner's 0.7638 on the runner's windows; fixed-window baseline 0.831 / 0.769"
```
Leave the model server stopped if Task 11 follows within the hour. Otherwise `systemctl --user start qwen-server.service` (the queue is empty).

---

### Task 11: Launch the seed-0 spine runs

**Files:**
- Modify: `.gitignore` (add `.worktrees/`)
- Create: `results/spine16_hybrid_s0.json`, `results/spine16_tf_s0.json`, `results/spine16_zeroheads_s0.json`, `results/spine16_s0.log` (copied from the worktree); `results/ckpt/spine16_*` (copied, git-ignored)
- No source changes.

**Interfaces:**
- Consumes: Tasks 1 to 10, committed.
- Produces: the three seed-0 result JSONs and checkpoints that Task 12 scores.

- [ ] **Step 1: Check the branch and the tests**
```bash
cd /games/ailab/lab/flybrain-lab
git branch --show-current                  # feat/group-learning
git diff --quiet && git diff --cached --quiet && echo clean
uv run pytest -q                           # all pass: 216 with this plan alone (144 before, 72 added); more once the other plans are merged
```

- [ ] **Step 2: Ignore the worktree directory and commit**
```bash
grep -qxF '.worktrees/' .gitignore || { printf '.worktrees/\n' >> .gitignore; git add .gitignore; git commit -m "Ignore .worktrees/, where pinned run checkouts live"; }
tail -3 .gitignore                          # results/ckpt/, .venv/, .worktrees/ (the probes plan adds the same line; whichever runs first commits it)
```

- [ ] **Step 3: Pin a worktree at this commit**
```bash
COMMIT=$(git rev-parse HEAD)
git worktree add .worktrees/runs "$COMMIT"
git -C .worktrees/runs log -1 --format='%h %s'      # the commit just made
ls -l .worktrees/runs/scripts/run_spine.sh           # executable
cd .worktrees/runs
uv sync --extra dev --frozen --python 3.11
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run pytest -q
cd ../..
```
Expected: the worktree prints the commit; `uv sync` builds `.worktrees/runs/.venv` with Python 3.11 (the uv cache holds only about 108 MB, so it downloads torch and the CUDA wheels, several GB; `/games` has 513 GB free); torch prints `2.14.0+cu130 True`; the tests pass. If the download is impossible, `export UV_PROJECT_ENVIRONMENT=/games/ailab/lab/flybrain-lab/.venv` for the worktree commands and add `--setenv=UV_PROJECT_ENVIRONMENT=/games/ailab/lab/flybrain-lab/.venv` to the unit below; the lock file is the same. The corpora are git-ignored, so the worktree has none: `CORPORA` points at the main checkout's, nothing is copied.

- [ ] **Step 4: Pre-flight on the card**
```bash
export CORPORA=/games/ailab/lab/flybrain-lab/corpora
stat -c %s "$CORPORA/linux_full.bytes"                       # 220000000
DRY_RUN=1 .worktrees/runs/scripts/run_spine.sh cuda:1 hybrid 0   # prints the run name and the flags
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
systemctl --user stop qwen-server.service
systemctl --user is-active qwen-server.service fleet.service     # inactive, inactive
nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv   # header only: no compute process on GPU 1
nvidia-smi --query-gpu=index,power.limit --format=csv            # GPU 1 reads 150.00 W; leave it alone
```
Do not go on if GPU 1 lists a process or its limit is not 150 W. `systemctl --user show-environment` lists `~/.local/bin` first in `PATH`, so the unit finds `uv`.

- [ ] **Step 5: Launch the three arms, one after another, in one detached unit**
```bash
systemd-run --user --unit flybrain-spine-s0 \
  --working-directory=/games/ailab/lab/flybrain-lab/.worktrees/runs \
  --setenv=CORPORA=/games/ailab/lab/flybrain-lab/corpora \
  bash -c 'set -o pipefail; { scripts/run_spine.sh cuda:1 hybrid 0 && scripts/run_spine.sh cuda:1 tf 0 && scripts/run_spine.sh cuda:1 zeroheads 0; } 2>&1 | tee results/spine16_s0.log'
```
Expected: `Running as unit: flybrain-spine-s0.service`. The `&&` chain with `pipefail` means a failed arm (including an over-temperature abort, which leaves no JSON) stops the chain; a finished arm is skipped when the same command is launched again.

**Expected duration, an estimate and not a measurement:** the bf16 hybrid took 4,919 s on GPU 1 at the earlier power cap (Run 47), and the 150 W cap is slower, so roughly 2 to 2.5 hours per arm and 6 to 7.5 hours for the three. The transformer has never run under bf16, so its time is the least certain. GPU 1 is unavailable to the model server for all of it.

- [ ] **Step 6: Watch it**
```bash
systemctl --user status flybrain-spine-s0.service --no-pager      # Active: active (running); the CGroup lists every PID of the unit
journalctl --user -u flybrain-spine-s0.service -n 30 --no-pager
tail -f .worktrees/runs/results/spine16_s0.log                    # Ctrl-C leaves the run alone
nvidia-smi -i 1 --query-gpu=power.draw,power.limit,temperature.gpu,memory.used --format=csv,noheader
```
Expect, in order: `=== spine16_hybrid_s0`; then, after several minutes in which the runner loads 200 MB and computes its n-gram baselines in pure Python, `code train=200000000 val=2000000 unigram=5.452 bigram=4.016` (the values in `results/m20_linux_depth4local_s0.json`); then about 8 minutes of training per 1,000 steps (estimate), after which one line per 1,000 steps, `fixed mingru_sparse_post {'step': 1000, 'train_seconds': ..., 'bpb': ..., 'bpb_warm': ...}`, every number finite and the bpb falling over the first few rows. The legacy `curve` lines start with two spaces and are filtered out of the log, as in `run_50m.sh`; they are in the JSON. nvidia-smi shows a power draw at or just under the 150 W limit, a temperature well under 85 C and no other process on the card. At the end of an arm: `mingru_sparse_post {...} scalars=16066824 rflops=38182912 sec=...`, then `mingru_sparse_post yardstick train_s=... eval_s=... peak_vram_gib=... joules=... contended=False`, then `wrote results/spine16_hybrid_s0.json`. The transformer arm prints `transformer` and 17085696 scalars, the zero-head arm `mingru` and 13930752. A NaN, or a bpb that rises for three rows in a row after step 3000, means stop the unit (Step 7) and investigate; an `RuntimeError: GPU ... reached 85 C` ends the chain by design: let the card cool, then run Step 5 again.

- [ ] **Step 7: Stop it by explicit PID, if it must be stopped**
```bash
systemctl --user status flybrain-spine-s0.service --no-pager      # read the PIDs in the CGroup tree
nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv  # the python PID that holds the card
kill -TERM <that PID>
```
Wait up to 30 s; if the process is still there, `kill -KILL <that PID>`. Never `pkill -f`. Because of `&&` and `pipefail`, the next arm does not start. Then `systemctl --user is-active flybrain-spine-s0.service` prints `failed` (or `inactive` once it has been cleaned up); clear a failed unit with `systemctl --user reset-failed flybrain-spine-s0.service`. Between two arms there is no python process: kill the unit's `bash -c` PID from the same CGroup listing instead. An aborted or killed arm writes no JSON.

- [ ] **Step 8: When the unit has exited, check the three results**
```bash
systemctl --user is-active flybrain-spine-s0.service     # inactive (a finished unit is removed) or failed
cd /games/ailab/lab/flybrain-lab/.worktrees/runs
ls results/spine16_*.json results/ckpt/spine16_*          # three JSONs, three checkpoints
uv run python - <<'EOF'
import json
for n in ("hybrid", "tf", "zeroheads"):
    a = json.load(open(f"results/spine16_{n}_s0.json"))["arms"][0]
    keys = ("trainable_scalars", "val_bpb", "train_seconds", "eval_seconds", "seconds", "peak_vram_bytes", "joules", "mean_power_w", "power_limit_w", "max_temp_c", "contended", "telemetry_lost", "samples", "gpu_uuid")
    print(n, {k: a[k] for k in keys}, a["val_fixed"])
EOF
cd ../..
```
Expected for every arm: `contended` False, `telemetry_lost` False, `power_limit_w` 150.0, `max_temp_c` below 85, `gpu_uuid` `GPU-51164b62-fd30-253a-5738-55844d46af9c`, `train_seconds + eval_seconds` within a few seconds of `seconds`, `mean_power_w` at most about 150, `samples` roughly 0.7 to 1 per second of arm time. A `contended` True or `telemetry_lost` True arm is kept and flagged; Task 12 explains what to do with it. Keep the model server stopped: Task 12 needs the card.

- [ ] **Step 9: Copy the results into the main checkout and commit**
```bash
cd /games/ailab/lab/flybrain-lab
cp .worktrees/runs/results/spine16_hybrid_s0.json .worktrees/runs/results/spine16_tf_s0.json .worktrees/runs/results/spine16_zeroheads_s0.json .worktrees/runs/results/spine16_s0.log results/
mkdir -p results/ckpt
cp .worktrees/runs/results/ckpt/spine16_* results/ckpt/
git status --short                                           # the three JSONs and the log are new; results/ckpt is ignored
git add results/spine16_hybrid_s0.json results/spine16_tf_s0.json results/spine16_zeroheads_s0.json results/spine16_s0.log
git commit -m "Spine runs, seed 0: hybrid, transformer and zero-head cell at 16M, one card, bf16, metered"
```
The checkpoints stay out of git. Leave `.worktrees/runs` in place: Task 13 uses the same pinned commit.

---

### Task 12: Score the checkpoints and apply the two rules

**Files:**
- Create: `results/spine16_score_dev_s0.json`, `results/spine16_score_test_s0.json`, `results/spine16_score_legacy_s0.json`, `results/spine16_score_dev128_s0.json`
- No source changes.

**Interfaces:**
- Consumes: Tasks 7 and 9; the three seed-0 checkpoints and JSONs from Task 11.
- Produces: the four score files and the output of `spine_report.py`: the table, the pair gaps and the follow-up commands.

- [ ] **Step 1: Confirm the card is free**
```bash
cd /games/ailab/lab/flybrain-lab
nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv   # header only
nvidia-smi --query-gpu=index,power.limit --format=csv                # GPU 1 at 150.00 W
```

- [ ] **Step 2: Score each checkpoint on dev, test, the legacy bytes, and the 128 in-run dev windows**
```bash
CK="results/ckpt/spine16_hybrid_s0_mingru_sparse_post.pt results/ckpt/spine16_tf_s0_transformer.pt results/ckpt/spine16_zeroheads_s0_mingru.pt"
for spec in dev:202117357:1024 test:211576468:1024 legacy:200100000:976 dev128:202117357:128; do
  IFS=: read split off n <<< "$spec"
  uv run python scripts/cross_eval.py --ckpts $CK --corpus-file corpora/linux_full.bytes --offset $off --fixed $n --lens 2048 --device cuda:1 --run-name spine16_score_${split}_s0
done
```
Expected: four blocks of three lines, `<arm> T= 2048 bpb=... warm=...`, a few minutes in all (an estimate). dev is the first 1,024 windows from byte 202,117,357 (inside the dev split, which ends at 211,576,468), test the first 1,024 windows from byte 211,576,468, legacy all 976 windows of the 2 MB from byte 200,100,000, dev128 the 128 windows the runs scored in-run. This is the only scoring of test: do not rerun it to look again, and tune nothing on it. The scoring is fp32 with no autocast, so the arms are compared without a precision confound; the in-run numbers (bf16) are in each run's `curve_fixed`.

- [ ] **Step 3: Print the report**
```bash
uv run python scripts/spine_report.py
```
Expected: the markdown table (one row per run); a `consistency` line per run whose delta (in-run bf16 against the fp32 re-score of the same 128 windows) is within 0.01, and a `WARNING` otherwise, in which case find the cause before reporting anything; the `runner val_bpb` line (the Run 37 to 49 number); one `a - b:` line per pair with the dev, test and legacy cold gaps and the seconds ratio; and the follow-up commands, or `none`. A `WARNING <run>: contended=...` line means that run's `train_seconds` is not clean: the rules below it are provisional for every pair that includes it. Rerun that arm under a new name (`scripts/run_spine.sh cuda:1 <arm> 0 16000 rerun` makes `spine16_<arm>_rerun_s0`, launched as in Task 11 Step 5 from the same worktree), score it like Task 13 Step 3, and say in the notebook which of the two runs the seconds come from; the report's rules keep reading the original names.

- [ ] **Step 4: Commit the scores**
```bash
git add results/spine16_score_dev_s0.json results/spine16_score_test_s0.json results/spine16_score_legacy_s0.json results/spine16_score_dev128_s0.json
git commit -m "Spine scores, seed 0: dev, test and the legacy bytes on fixed windows, cold and warm"
```

- [ ] **Step 5: Branch on what the rules asked for**

If the report printed `none: ...`, the queue is empty: `systemctl --user start qwen-server.service`, then go to Task 14. If it printed commands, keep the server stopped and go to Task 13.

---

### Task 13: Follow-up runs the rules asked for (skip when Task 12 printed `none`)

**Files:**
- Create: `results/spine16_followups.log`, `results/spine16_<arm>_s1.json` and `results/spine16_<arm>_eqsec_s0.json` for the runs asked for, `results/spine16_score_<split>_followups.json`
- No source changes.

**Interfaces:**
- Consumes: the command lines printed by `spine_report.py` (Task 12 Step 3), the worktree of Task 11.
- Produces: seed-1 runs (rule 1) and equal-seconds reruns `spine16_<arm>_eqsec_s0` (rule 2), scored like the seed-0 runs and shown in the table. The rules are not applied again to them.

- [ ] **Step 1: Pre-flight, as Task 11 Step 4**

Confirm that `nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv` prints the header only and GPU 1 reads 150.00 W. The model server stays stopped.

- [ ] **Step 2: Launch the printed commands in one unit**

Join the printed lines with ` && ` inside the braces. For example, if the report printed `scripts/run_spine.sh cuda:1 hybrid 1`, `scripts/run_spine.sh cuda:1 tf 1` and `scripts/run_spine.sh cuda:1 zeroheads 0 17391 eqsec`:
```bash
systemd-run --user --unit flybrain-spine-followups \
  --working-directory=/games/ailab/lab/flybrain-lab/.worktrees/runs \
  --setenv=CORPORA=/games/ailab/lab/flybrain-lab/corpora \
  bash -c 'set -o pipefail; { scripts/run_spine.sh cuda:1 hybrid 1 && scripts/run_spine.sh cuda:1 tf 1 && scripts/run_spine.sh cuda:1 zeroheads 0 17391 eqsec; } 2>&1 | tee results/spine16_followups.log'
```
Use the commands the report printed, not these. Watch, stop and read the results exactly as in Task 11 Steps 6 to 8 (unit `flybrain-spine-followups`). Each seed-1 arm takes about as long as its seed-0 run; an equal-seconds rerun takes about the slower arm's `train_seconds` (an estimate, since its steps were scaled by the measured seconds).

- [ ] **Step 3: Copy, score and commit**
```bash
cd /games/ailab/lab/flybrain-lab
cp .worktrees/runs/results/spine16_*.json .worktrees/runs/results/spine16_followups.log results/    # the seed-0 files are identical copies
cp -n .worktrees/runs/results/ckpt/spine16_* results/ckpt/
CK=$(ls results/ckpt/spine16_*_s1_*.pt results/ckpt/spine16_*_eqsec_s0_*.pt 2>/dev/null)
echo $CK
for spec in dev:202117357:1024 test:211576468:1024 legacy:200100000:976 dev128:202117357:128; do
  IFS=: read split off n <<< "$spec"
  uv run python scripts/cross_eval.py --ckpts $CK --corpus-file corpora/linux_full.bytes --offset $off --fixed $n --lens 2048 --device cuda:1 --run-name spine16_score_${split}_followups
done
uv run python scripts/spine_report.py
git status --short                       # the new run JSONs, the log and the four followups score files; results/ckpt is ignored
git add results/spine16_*.json results/spine16_followups.log
git commit -m "Spine follow-ups: seed 1 and equal-seconds reruns the two rules asked for, scored on the same windows"
systemctl --user start qwen-server.service      # the queue is empty
```
Expected: `echo $CK` lists exactly the new checkpoints; the table now has the new rows after the seed-0 rows (`_eqsec_s0` before `_s1`).

---

### Task 14: The Run 53 notebook entry

**Files:**
- Modify: `notebook.md` (insert before `## Path to V3 (ladder plumbing)`, after any Run 50 to 52 entries the other plans added)
- Do not edit `README.md` or `notes/RESULTS.md`; any wording change they would need goes in the Gate 1 report for Kyle.

**Interfaces:**
- Consumes: the output of `uv run python scripts/spine_report.py` after Tasks 12 and 13, `git rev-parse` of the worktree commit, the unit start and end times from `journalctl --user -u flybrain-spine-s0.service` (and `flybrain-spine-followups`).
- Produces: one notebook entry in the repo's Run format: heading, a setup paragraph, one table, then `**Claim:** ... **Not a claim:** ...`.

- [ ] **Step 1: Collect the slots**

| slot | source |
|---|---|
| `<<start>>`, `<<end>>` | first start and last finish of the units, `YYYY-MM-DD HH:MM`, from `journalctl --user -u flybrain-spine-s0.service --no-pager` (and the follow-up unit) |
| `<<commit>>` | `git -C .worktrees/runs rev-parse --short HEAD` |
| `<<table>>` | the table printed by `spine_report.py`, pasted as is |
| `<<pair lines>>` | the `a - b:` lines and the `runner val_bpb` line, one bullet each |
| `<<power>>` | the lowest and highest `mean_power_w` among the runs, from the JSONs |
| `<<claim>>` | the one row of the claim table below that the report meets, with the report's numbers copied in |

- [ ] **Step 2: Choose the claim**

Every number comes from the report; compute nothing new. "Gap" is hybrid minus transformer in cold bits/byte, so negative favours the hybrid; "seconds ratio" is hybrid `train_seconds` over transformer `train_seconds`.

| the report shows | the claim to write |
|---|---|
| gap at most -0.01 on dev and on test, seconds ratio at most 1.05 | At 16M and 16,000 steps on one capped card the hybrid reads <<dev gap>> on dev and <<test gap>> on test below the transformer at <<ratio>> of its training seconds. |
| gap at most -0.01 on dev and test, seconds ratio above 1.05 | The hybrid reads <<dev gap>> / <<test gap>> below the transformer but trains <<ratio>> times as long; at equal seconds (`_eqsec` row: <<its dev and test cold>>) the gap is <<equal-seconds gap>>. |
| the gaps are within 0.01 on dev or on test | At 16M the two are within noise on fixed windows (dev <<dev gap>>, test <<test gap>>); the seconds ratio is <<ratio>>; seed 1 (`_s1` rows) reads <<seed-1 gaps>>. |
| the transformer is lower by more than 0.01 | The transformer reads <<dev gap>> / <<test gap>> below the hybrid at <<ratio>> of the hybrid's training seconds; the earlier lead (0.017 to 0.025 at 20M) did not reproduce on this yardstick. |

If dev and test disagree (one gap at most -0.01, the other above +0.01), use the third row's wording and say that the splits disagree. Say the same about the zero-head cell in one more sentence, from its `a - b:` lines.

- [ ] **Step 3: Insert the entry**

Replace in `notebook.md`:
```markdown
---

## Path to V3 (ladder plumbing)
```

with:
```markdown
---

## Run 53 — the spine: hybrid vs RoPE transformer vs zero-head local-attention cell at 16M, one card, one recipe (`scripts/run_spine.sh`, `scripts/spine_report.py`, `results/spine16_*.json`, `results/spine16_score_*.json`) — <<start>> → <<end>>, GPU 1 at 150 W

16M class (d 512, 8 layers, d_ff 1024; kernel corpus; 16,000 steps of 8 × 2,048 bytes; lr 6e-4, warmup 1000, cosine, wd 0.1): the Run 37 / Run 43 recipe, all three arms under bf16 autocast (`--amp`; the transformer had never run under it), with the yardstick of the group-learning spec. One arm after another on GPU 1 (RTX 3090, power cap 150 W, model server stopped, nothing else on the card), from a worktree pinned at `<<commit>>`. `train_seconds` excludes evals and saves; peak VRAM is `torch.cuda.max_memory_allocated`; joules, `contended` and the temperature come from one nvidia-smi sample a second. Scored by `cross_eval.py --fixed` in fp32 at length 2048: dev = the first 1,024 windows from byte 202,117,357, test = the first 1,024 windows from byte 211,576,468 (scored once), legacy bytes = all 976 windows from byte 200,100,000. Cold = mean over the window, warm = mean over its second half.

<<table>>

<<pair lines>>
- Joules are logged, not a headline: at a fixed 150 W cap they are close to 150 W × seconds (mean power <<power>> W here), so they restate the seconds column.

**Claim:** <<claim>> **Not a claim:** joules as a result; more than one seed unless the table has `_s1` rows; anything beyond 16M, 16,000 steps and this kernel corpus; that the order holds at equal seconds unless the table has an `_eqsec` row; a comparison of the fixed-window numbers with the random-window `val_bpb` of Runs 37 to 49 (different windows, different numbers; the `runner val_bpb` line above is the comparable one).

---

## Path to V3 (ladder plumbing)
```

- [ ] **Step 4: Check the entry and commit**
```bash
grep -c "<<" notebook.md                 # 0: no slot left unfilled
git diff --stat                          # notebook.md only
git add notebook.md
git commit -m "Notebook: Run 53, the spine at 16M on one capped card, with the yardstick"
```

---

## Self-review

| Spec requirement | Task |
|---|---|
| 1.5 flags `--val-fixed`, `--val-offset`, `--eval-every`, `--meter`, `--ckpt-every`, all off by default | 6 (with `fixed_windows` in 1 and `eval_fixed` in 5) |
| 1.5 cold is the mean over the window, warm the mean over its second half | 5 (runner), 7 (`cross_eval.py`) |
| 1.5 per-arm fields `train_seconds`, `eval_seconds`, `bytes_seen`, `peak_vram_bytes`, `curve_fixed`, written only when a flag is on | 4 (the clock), 6 (the loop, the JSON) |
| 1.5 per-arm meter fields `joules`, `mean_power_w`, `power_limit_w`, `max_temp_c`, `contended` (plus `samples`, `telemetry_lost`, `gpu_uuid`) | 3 (accounting), 6 (merged into the arm) |
| 1.5 `train_seconds` excludes evals and saves; `seconds` keeps its meaning | 4, 6 (`test_evals_and_saves_never_count_as_training_time`) |
| 1.5 `cross_eval.py --fixed N`, off by default | 7 |
| 1.5 `flybrain/meter.py`: nvidia-smi thread, the physical card, `contended`, abort at 85 C, thermal_guard's fields | 2, 3 |
| 1.5 joules logged, not headlined | 14 (the entry says so twice) |
| 1.6 16M, 16,000 steps, the `run_20m.sh` flags plus `--amp --meter --val-fixed 128 --val-offset 202117357 --eval-every 1000 --save`; the three run names and arm flags | 8 (flag-by-flag comparison, `test_each_arm_parses_to_the_decided_flags`) |
| 1.6 all three on GPU 1, one after another, nothing else on the card | 11 (one unit, per-GPU pre-flight, `contended` in the result) |
| 1.6 final scoring on dev, test and the legacy bytes, cold and warm | 12 |
| 1.6 seed 1 for a pair closer than 0.01 bits/byte or 5 % in `train_seconds` | 9 (`decide`), 12 (report), 13 (run) |
| 1.6 equal-seconds rerun of the faster arm with scaled steps and its own cosine schedule, as its own run name | 8 (the tag argument), 9 (`decide`), 13 (run) |
| Run rules: stop the server before, start it when the queue is empty | 10, 11, 12, 13 |
| Run rules: GPU 0 only 22:00 to 05:00; the 150 W cap stays | Global Constraints; every GPU command here targets GPU 1 and only reads the cap (10, 11) |
| Run rules: nvidia-smi before each launch, refuse a busy card, kill by explicit PID | 10, 11 (Steps 4 and 7), 12, 13 |
| Run rules: detached `systemd-run --user` units from a pinned worktree, logs in `results/`, new run names only | 11, 13; skip rule in 8 (`test_a_finished_run_name_is_skipped_without_starting_anything`) |
| Testing: the runner with and without the flags gives the same `state_dict` and legacy JSON fields | 6 (`test_yardstick_flags_leave_training_untouched`) |
| Testing: the meter's joules within 5 % of mean power times seconds; `contended` trips on a second process | 3 |
| Testing: the 16M winner re-scored through the fixed-window path reads 0.764 within 0.002 | 10 (check A, on the runner's own windows; check B on fixed windows) |
| The notebook write-up of the spine | 14 |
