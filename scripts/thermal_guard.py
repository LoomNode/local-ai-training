"""Per-GPU, fail-closed thermal guard for integrity training runs.

This module is standalone so existing jobs importing ``gpu_guard`` retain their
current behavior.  It never changes GPU power limits and never moves a job to a
different physical GPU.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO

ADMISSION_SAMPLES = 3
ADMISSION_INTERVAL_SECONDS = 5.0
STATUS_INTERVAL_SECONDS = 10.0
PROCESS_POLL_SECONDS = 1.0
TELEMETRY_TIMEOUT_SECONDS = 60.0
TERMINATION_GRACE_SECONDS = 30.0
WARNING_TEMPERATURE_C = 80
CRITICAL_TEMPERATURE_C = 85
EXPECTED_POWER_LIMIT_W = {0: 300.0, 1: 250.0}
LOG_TAIL_BYTES = 4_096


def guard_dir() -> Path:
    """Directory for locks, per-GPU status files, and thermal-events.jsonl."""
    directory = Path(os.environ.get("LAT_GUARD_DIR", "runs/guard")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def thermal_events_path() -> Path:
    return guard_dir() / "thermal-events.jsonl"


class GPUUnavailableError(RuntimeError):
    """GPU telemetry is unavailable or malformed, so admission is denied."""


class TrainingLockBusyError(RuntimeError):
    """Another managed process owns the selected physical GPU lock."""


def lock_path(gpu: int) -> Path:
    return guard_dir() / f"training-gpu-{gpu}.lock"


def status_path(gpu: int) -> Path:
    return guard_dir() / f"status-gpu-{gpu}.json"


def required_free_mib(cmd: Sequence[object]) -> int:
    command_text = " ".join(os.fspath(part) for part in cmd).casefold()
    return 22_000 if "historical-1b" in command_text else 8_000


def parse_snapshot(gpu_csv: str, process_csv: str) -> dict[int, dict[str, object]]:
    """Parse memory, utilization, temperature, and power telemetry."""
    try:
        result: dict[int, dict[str, object]] = {}
        uuid_to_index: dict[str, int] = {}
        for row in csv.reader(gpu_csv.splitlines(), skipinitialspace=True):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) != 10:
                raise ValueError(f"expected 10 GPU fields, got {len(row)}")
            index = int(row[0])
            uuid = row[1].strip()
            result[index] = {
                "index": index,
                "uuid": uuid,
                "name": row[2].strip(),
                "totalMiB": int(row[3]),
                "usedMiB": int(row[4]),
                "freeMiB": int(row[5]),
                "utilizationPercent": int(row[6]),
                "temperatureC": int(row[7]),
                "powerDrawW": float(row[8]),
                "powerLimitW": float(row[9]),
                "computeProcesses": [],
            }
            uuid_to_index[uuid] = index
        if not result:
            raise ValueError("nvidia-smi returned no GPUs")

        for row in csv.reader(process_csv.splitlines(), skipinitialspace=True):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) != 4:
                raise ValueError(f"expected 4 process fields, got {len(row)}")
            gpu_uuid = row[0].strip()
            if gpu_uuid not in uuid_to_index:
                raise ValueError(f"process references unknown GPU {gpu_uuid}")
            processes = result[uuid_to_index[gpu_uuid]]["computeProcesses"]
            assert isinstance(processes, list)
            processes.append(
                {
                    "pid": int(row[1]),
                    "processName": row[2].strip(),
                    "usedMemoryMiB": int(row[3]),
                }
            )
        return result
    except (AssertionError, TypeError, ValueError) as error:
        raise GPUUnavailableError(f"malformed nvidia-smi snapshot: {error}") from error


def choose_gpu(
    samples: Sequence[Mapping[int, Mapping[str, object]]],
    preferred_gpu: int,
    required_mib: int,
) -> int | None:
    """Admit only the assigned physical GPU after three qualifying samples."""
    expected_cap = EXPECTED_POWER_LIMIT_W.get(preferred_gpu)
    if expected_cap is None or len(samples) < ADMISSION_SAMPLES:
        return None
    recent = samples[-ADMISSION_SAMPLES:]
    try:
        admitted = all(
            preferred_gpu in sample
            and int(sample[preferred_gpu]["freeMiB"]) >= required_mib
            and int(sample[preferred_gpu]["temperatureC"]) < WARNING_TEMPERATURE_C
            and float(sample[preferred_gpu]["powerLimitW"]) <= expected_cap
            for sample in recent
        )
    except (KeyError, TypeError, ValueError):
        return None
    return preferred_gpu if admitted else None


def thermal_decision(
    gpu_snapshot: Mapping[str, object] | None, telemetry_missing_seconds: float
) -> str:
    """Return the action state for one owned-child monitoring sample."""
    if gpu_snapshot is None:
        if telemetry_missing_seconds >= TELEMETRY_TIMEOUT_SECONDS:
            return "telemetry_missing_timeout"
        return "telemetry_missing"
    try:
        temperature = int(gpu_snapshot["temperatureC"])
    except (KeyError, TypeError, ValueError):
        if telemetry_missing_seconds >= TELEMETRY_TIMEOUT_SECONDS:
            return "telemetry_missing_timeout"
        return "telemetry_missing"
    if temperature >= CRITICAL_TEMPERATURE_C:
        return "temperature_critical"
    if temperature >= WARNING_TEMPERATURE_C:
        return "temperature_warning"
    return "ok"


def parse_proc_stat(stat_text: str) -> tuple[str, int]:
    closing_paren = stat_text.rfind(")")
    if closing_paren < 0:
        raise ValueError("missing process-name terminator")
    fields_after_name = stat_text[closing_paren + 1 :].split()
    if len(fields_after_name) < 20:
        raise ValueError("truncated process stat")
    return fields_after_name[0], int(fields_after_name[19])


def same_process_alive(proc_stat: tuple[str, int], expected_start_ticks: int) -> bool:
    state, start_ticks = proc_stat
    return state not in {"X", "Z"} and start_ticks == expected_start_ticks


def _read_proc_stat(pid: int) -> tuple[str, int] | None:
    try:
        return parse_proc_stat(Path(f"/proc/{pid}/stat").read_text(encoding="utf-8"))
    except (FileNotFoundError, ProcessLookupError):
        return None


def _query_snapshot() -> dict[int, dict[str, object]]:
    try:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,"
                "utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        process_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise GPUUnavailableError(f"nvidia-smi unavailable: {error}") from error
    return parse_snapshot(gpu_result.stdout, process_result.stdout)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log_progress(
    log_file: IO[object] | None = None, log_path: Path | None = None
) -> dict[str, object] | None:
    try:
        if log_path is not None:
            stat = log_path.stat()
            path = log_path
            with path.open("rb") as handle:
                handle.seek(max(0, stat.st_size - LOG_TAIL_BYTES))
                tail = handle.read(LOG_TAIL_BYTES)
        elif log_file is not None:
            name = getattr(log_file, "name", None)
            path = Path(name) if isinstance(name, (str, os.PathLike)) else None
            if path is not None:
                stat = path.stat()
                with path.open("rb") as handle:
                    handle.seek(max(0, stat.st_size - LOG_TAIL_BYTES))
                    tail = handle.read(LOG_TAIL_BYTES)
            else:
                descriptor = log_file.fileno()
                stat = os.fstat(descriptor)
                tail = os.pread(
                    descriptor,
                    min(LOG_TAIL_BYTES, stat.st_size),
                    max(0, stat.st_size - LOG_TAIL_BYTES),
                )
        else:
            return None
    except (OSError, TypeError, ValueError):
        return None
    lines = tail.decode("utf-8", errors="replace").splitlines()
    return {
        "path": os.fspath(path) if path is not None else None,
        "bytes": stat.st_size,
        "modifiedUTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
        "lastLine": lines[-1] if lines else None,
    }


def _atomic_write_status(gpu: int, payload: Mapping[str, object]) -> None:
    path = status_path(gpu)
    temporary = guard_dir() / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_status(
    gpu: int,
    snapshot: Mapping[int, Mapping[str, object]] | None,
    *,
    pid: int | None,
    state: str,
    log_progress: Mapping[str, object] | None,
    extra: Mapping[str, object] | None = None,
) -> None:
    occupancy = dict(snapshot[gpu]) if snapshot is not None and gpu in snapshot else None
    processes = occupancy.get("computeProcesses", []) if occupancy else []
    competing = [
        process
        for process in processes
        if not isinstance(process, Mapping) or process.get("pid") != pid
    ]
    payload: dict[str, object] = {
        "timestampUTC": _utc_now(),
        "state": state,
        "preferredGPU": gpu,
        "actualGPU": gpu if pid is not None else None,
        "gpuOccupancy": occupancy,
        "ownChildPID": pid,
        "lastLogProgress": log_progress,
        "competingProcesses": competing if occupancy is not None else None,
    }
    if extra:
        payload.update(extra)
    _atomic_write_status(gpu, payload)


def _append_thermal_event(gpu: int, pid: int, reason: str, **details: object) -> None:
    payload = {"timestampUTC": _utc_now(), "gpu": gpu, "pid": pid, "reason": reason}
    payload.update(details)
    with thermal_events_path().open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _locked_training_slot(gpu: int) -> IO[str]:
    guard_dir().mkdir(parents=True, exist_ok=True)
    lock_file = lock_path(gpu).open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_file.close()
        raise TrainingLockBusyError(
            f"another managed process owns physical GPU {gpu}"
        ) from error
    return lock_file


def _release_lock(lock_file: IO[str]) -> None:
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def _terminate_owned_child(pid: int, expected_start_ticks: int) -> str:
    """Stop only the exact PID instance recorded when the child was launched."""
    current = _read_proc_stat(pid)
    if current is None or not same_process_alive(current, expected_start_ticks):
        return "pid_dead_or_reused"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_exited"
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        current = _read_proc_stat(pid)
        if current is None or not same_process_alive(current, expected_start_ticks):
            return "terminated"
        time.sleep(PROCESS_POLL_SECONDS)
    current = _read_proc_stat(pid)
    if current is not None and same_process_alive(current, expected_start_ticks):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return "terminated"


def _child_start_ticks(pid: int) -> int:
    for _ in range(50):
        stat = _read_proc_stat(pid)
        if stat is not None and stat[0] not in {"X", "Z"}:
            return stat[1]
        time.sleep(0.01)
    raise ProcessLookupError(f"could not establish identity for child PID {pid}")


def run_training(
    cmd: Sequence[object],
    log: IO[object],
    env: Mapping[str, str],
    gpu: int,
) -> subprocess.CompletedProcess[object]:
    """Admit and monitor one child on its assigned physical GPU."""
    if gpu not in EXPECTED_POWER_LIMIT_W:
        raise GPUUnavailableError(f"physical GPU {gpu} has no configured power cap")
    lock_file = _locked_training_slot(gpu)
    try:
        required_mib = required_free_mib(cmd)
        samples: list[dict[int, dict[str, object]]] = []
        sample_number = 0
        while True:
            try:
                snapshot = _query_snapshot()
                if gpu not in snapshot:
                    raise GPUUnavailableError(f"physical GPU {gpu} is unavailable")
                samples.append(snapshot)
                samples = samples[-ADMISSION_SAMPLES:]
                sample_number += 1
                admitted = choose_gpu(samples, gpu, required_mib) == gpu
                _write_status(
                    gpu,
                    snapshot,
                    pid=None,
                    state="admitted" if admitted else "waiting_for_capacity",
                    log_progress=_log_progress(log_file=log),
                    extra={
                        "admissionSample": sample_number,
                        "requiredFreeMiB": required_mib,
                        "expectedPowerLimitW": EXPECTED_POWER_LIMIT_W[gpu],
                        "consecutiveSamplesRequired": ADMISSION_SAMPLES,
                    },
                )
                if admitted:
                    break
            except GPUUnavailableError as error:
                samples.clear()
                _write_status(
                    gpu,
                    None,
                    pid=None,
                    state="admission_telemetry_unavailable",
                    log_progress=_log_progress(log_file=log),
                    extra={"error": str(error)},
                )
            time.sleep(ADMISSION_INTERVAL_SECONDS)

        child_env = dict(env)
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        child = subprocess.Popen(
            [os.fspath(part) for part in cmd],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_env,
        )
        start_ticks = _child_start_ticks(child.pid)
        missing_since: float | None = None
        warning_active = False
        last_status = 0.0
        stop_reason: str | None = None
        while child.poll() is None:
            now = time.monotonic()
            snapshot = None
            error_text = None
            try:
                queried = _query_snapshot()
                if gpu not in queried:
                    raise GPUUnavailableError(f"physical GPU {gpu} is unavailable")
                snapshot = queried
                missing_since = None
            except GPUUnavailableError as error:
                error_text = str(error)
                if missing_since is None:
                    missing_since = now

            missing_seconds = 0.0 if missing_since is None else now - missing_since
            decision = thermal_decision(
                snapshot[gpu] if snapshot is not None else None, missing_seconds
            )
            temperature = (
                int(snapshot[gpu]["temperatureC"]) if snapshot is not None else None
            )
            if decision == "temperature_warning" and not warning_active:
                _append_thermal_event(
                    gpu, child.pid, decision, temperatureC=temperature
                )
                warning_active = True
            elif decision == "ok":
                warning_active = False
            if decision in {"temperature_critical", "telemetry_missing_timeout"}:
                stop_reason = decision
                _append_thermal_event(
                    gpu,
                    child.pid,
                    decision,
                    temperatureC=temperature,
                    telemetryMissingSeconds=missing_seconds,
                )
                _write_status(
                    gpu,
                    snapshot,
                    pid=child.pid,
                    state="stopping_owned_child",
                    log_progress=_log_progress(log_file=log),
                    extra={"stopReason": decision, "telemetryError": error_text},
                )
                _terminate_owned_child(child.pid, start_ticks)
                break
            if now - last_status >= STATUS_INTERVAL_SECONDS:
                _write_status(
                    gpu,
                    snapshot,
                    pid=child.pid,
                    state=(
                        "running_telemetry_unavailable"
                        if snapshot is None
                        else "running_temperature_warning"
                        if decision == "temperature_warning"
                        else "running"
                    ),
                    log_progress=_log_progress(log_file=log),
                    extra={
                        "telemetryMissingSeconds": missing_seconds,
                        "telemetryError": error_text,
                    },
                )
                last_status = now
            time.sleep(PROCESS_POLL_SECONDS)

        return_code = child.wait()
        try:
            final_snapshot = _query_snapshot()
        except GPUUnavailableError:
            final_snapshot = None
        _write_status(
            gpu,
            final_snapshot,
            pid=child.pid,
            state="exited",
            log_progress=_log_progress(log_file=log),
            extra={"returnCode": return_code, "stopReason": stop_reason},
        )
        return subprocess.CompletedProcess(args=cmd, returncode=return_code)
    finally:
        _release_lock(lock_file)


def monitor_existing(pid: int, gpu: int, log_path: str | os.PathLike[str]) -> dict[str, object]:
    """Thermally protect an adopted process without judging training success."""
    if gpu not in EXPECTED_POWER_LIMIT_W:
        raise GPUUnavailableError(f"physical GPU {gpu} has no configured power cap")
    lock_file = _locked_training_slot(gpu)
    try:
        initial_stat = _read_proc_stat(pid)
        if initial_stat is None or initial_stat[0] in {"X", "Z"}:
            raise ProcessLookupError(f"PID {pid} is not a live process")
        expected_start_ticks = initial_stat[1]
        path = Path(log_path)
        warning_active = False
        missing_since: float | None = None
        stop_reason: str | None = None
        while True:
            current_stat = _read_proc_stat(pid)
            if current_stat is None:
                reason = "pid_gone"
                break
            if not same_process_alive(current_stat, expected_start_ticks):
                reason = "pid_dead_or_reused"
                break
            now = time.monotonic()
            snapshot = None
            error_text = None
            try:
                queried = _query_snapshot()
                if gpu not in queried:
                    raise GPUUnavailableError(f"physical GPU {gpu} is unavailable")
                snapshot = queried
                missing_since = None
            except GPUUnavailableError as error:
                error_text = str(error)
                if missing_since is None:
                    missing_since = now

            missing_seconds = 0.0 if missing_since is None else now - missing_since
            decision = thermal_decision(
                snapshot[gpu] if snapshot is not None else None, missing_seconds
            )
            temperature = (
                int(snapshot[gpu]["temperatureC"]) if snapshot is not None else None
            )
            if decision == "temperature_warning" and not warning_active:
                _append_thermal_event(gpu, pid, decision, temperatureC=temperature)
                warning_active = True
            elif decision == "ok":
                warning_active = False
            if decision in {"temperature_critical", "telemetry_missing_timeout"}:
                reason = decision
                stop_reason = decision
                _append_thermal_event(
                    gpu,
                    pid,
                    decision,
                    temperatureC=temperature,
                    telemetryMissingSeconds=missing_seconds,
                )
                _write_status(
                    gpu,
                    snapshot,
                    pid=pid,
                    state="stopping_adopted_process",
                    log_progress=_log_progress(log_path=path),
                    extra={
                        "adoptedStartTimeTicks": expected_start_ticks,
                        "stopReason": decision,
                        "telemetryError": error_text,
                    },
                )
                _terminate_owned_child(pid, expected_start_ticks)
                break
            _write_status(
                gpu,
                snapshot,
                pid=pid,
                state=(
                    "monitoring_existing_telemetry_unavailable"
                    if snapshot is None
                    else "monitoring_existing_temperature_warning"
                    if decision == "temperature_warning"
                    else "monitoring_existing"
                ),
                log_progress=_log_progress(log_path=path),
                extra={
                    "adoptedStartTimeTicks": expected_start_ticks,
                    "telemetryMissingSeconds": missing_seconds,
                    "telemetryError": error_text,
                },
            )
            time.sleep(STATUS_INTERVAL_SECONDS)

        _write_status(
            gpu,
            None,
            pid=pid,
            state="existing_monitor_stopped",
            log_progress=_log_progress(log_path=path),
            extra={
                "monitoringStoppedReason": reason,
                "adoptedStartTimeTicks": expected_start_ticks,
                "stopReason": stop_reason,
            },
        )
        return {"pid": pid, "reason": reason, "startTimeTicks": expected_start_ticks}
    finally:
        _release_lock(lock_file)
