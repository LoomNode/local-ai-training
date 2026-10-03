"""Yardstick instruments: GPU energy, temperature and contention, and a train/eval wall clock.

The nvidia-smi queries and the 85 C stop follow local-ai-training/scripts/thermal_guard.py; nothing here imports it."""

from __future__ import annotations

import csv
import math
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

GPU_QUERY = ["nvidia-smi", "--query-gpu=uuid,power.draw,power.limit,temperature.gpu,memory.used", "--format=csv,noheader,nounits"]
APPS_QUERY = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]
TELEMETRY_TIMEOUT_S = 60.0


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
