"""Yardstick instruments: nvidia-smi parsing, GPU meter accounting, train/eval clock. No GPU, no nvidia-smi, no network."""

from __future__ import annotations

import subprocess
import time
import uuid

import pytest
import torch

from flybrain import meter
from flybrain.meter import GpuMeter, Sample, SplitClock, format_uuid, parse

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
    calls, kwargs = [], []

    def fake_run(argv, **kw):
        calls.append(argv)
        kwargs.append(kw)
        return subprocess.CompletedProcess(argv, 0, stdout=GPU_CSV if "--query-gpu" in argv[1] else APPS_CSV)

    monkeypatch.setattr(meter.subprocess, "run", fake_run)
    got = meter.nvidia_smi_sampler(GPU1, own_pid=4242)()
    assert calls == [["nvidia-smi", "--query-gpu=uuid,power.draw,power.limit,temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
                     ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]]
    assert got.other_pids == (1179204,)
    assert len(kwargs) == 2
    for kw in kwargs:  # the meter's thread counts a missing sample only because a failing nvidia-smi raises and a hung one times out
        assert kw.get("check") is True and kw.get("timeout") == 5.0


def test_sampler_raises_when_nvidia_smi_fails_instead_of_returning_a_sample(monkeypatch):
    def failing_run(argv, **kw):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(meter.subprocess, "run", failing_run)
    with pytest.raises(subprocess.CalledProcessError):
        meter.nvidia_smi_sampler(GPU1, own_pid=4242)()


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
    m.start(); time.sleep(0.1)
    alive = m._thread.is_alive()            # a thread killed by its first failure leaves exactly the same zeros behind
    m.stop()
    assert alive
    assert m.summary() == {"joules": 0.0, "mean_power_w": 0.0, "power_limit_w": 0.0, "max_temp_c": 0.0,
                           "contended": False, "samples": 0, "telemetry_lost": False}
    m.check()


def test_the_thread_keeps_sampling_after_every_kind_of_failure():
    errors = [FileNotFoundError("nvidia-smi"), subprocess.CalledProcessError(9, "nvidia-smi"),
              subprocess.TimeoutExpired("nvidia-smi", 5.0), ValueError("nvidia-smi gave '[N/A]' for power.draw")]
    calls = {"n": 0}

    def sampler() -> Sample:
        calls["n"] += 1
        if calls["n"] <= len(errors):
            raise errors[calls["n"] - 1]
        return s()

    m = GpuMeter(GPU1, own_pid=1, interval=0.01, sampler=sampler)
    m.start()
    deadline = time.monotonic() + 5.0
    while m.summary()["samples"] < 1 and m._thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)                    # a dead thread ends the wait at once instead of running out the deadline
    alive = m._thread.is_alive()
    m.stop()
    assert alive and calls["n"] > len(errors) and m.summary()["samples"] >= 1   # survived all four, then ingested the good sample
    assert not m._thread.is_alive()


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
