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
