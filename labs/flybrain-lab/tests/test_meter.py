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
