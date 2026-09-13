import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import thermal_guard


def gpu_snapshot(*, free=24_000, temperature=70, power_draw=200.0, power_limit=300.0):
    return {
        0: {
            "index": 0,
            "freeMiB": free,
            "temperatureC": temperature,
            "powerDrawW": power_draw,
            "powerLimitW": power_limit,
            "computeProcesses": [],
        }
    }


class ThermalAdmissionTests(unittest.TestCase):
    def test_exact_gpu_requires_three_cool_samples_at_or_below_cap(self):
        good = gpu_snapshot()
        too_hot = gpu_snapshot(temperature=80)
        over_cap = gpu_snapshot(power_limit=300.1)

        self.assertEqual(thermal_guard.choose_gpu([good, good, good], 0, 22_000), 0)
        self.assertIsNone(thermal_guard.choose_gpu([good, too_hot, good], 0, 22_000))
        self.assertIsNone(thermal_guard.choose_gpu([good, over_cap, good], 0, 22_000))

    def test_gpu_one_uses_250_watt_expected_cap_and_never_falls_back(self):
        gpu_zero = gpu_snapshot()
        samples = [
            {0: gpu_zero[0], 1: {**gpu_zero[0], "index": 1, "powerLimitW": 251.0}}
            for _ in range(3)
        ]

        self.assertIsNone(thermal_guard.choose_gpu(samples, 1, 8_000))

    def test_snapshot_parses_temperature_draw_and_limit(self):
        snapshot = thermal_guard.parse_snapshot(
            "0, GPU-a, RTX 3090, 24576, 1000, 23576, 20, 79, 241.50, 300.00\n",
            "",
        )

        self.assertEqual(snapshot[0]["temperatureC"], 79)
        self.assertEqual(snapshot[0]["powerDrawW"], 241.5)
        self.assertEqual(snapshot[0]["powerLimitW"], 300.0)

    def test_missing_thermal_field_fails_closed(self):
        with self.assertRaises(thermal_guard.GPUUnavailableError):
            thermal_guard.parse_snapshot(
                "0, GPU-a, RTX 3090, 24576, 1000, 23576, 20, N/A, 241.50, 300.00\n",
                "",
            )


class OwnedChildSafetyTests(unittest.TestCase):
    def test_85_degree_decision_requests_owned_child_stop(self):
        self.assertEqual(
            thermal_guard.thermal_decision(
                gpu_snapshot(temperature=85)[0], telemetry_missing_seconds=0
            ),
            "temperature_critical",
        )

    def test_continuous_telemetry_loss_reaches_fail_closed_stop_at_60_seconds(self):
        self.assertEqual(
            thermal_guard.thermal_decision(None, telemetry_missing_seconds=59.9),
            "telemetry_missing",
        )
        self.assertEqual(
            thermal_guard.thermal_decision(None, telemetry_missing_seconds=60.0),
            "telemetry_missing_timeout",
        )

    def test_termination_refuses_reused_pid_without_sending_signal(self):
        with (
            mock.patch.object(thermal_guard, "_read_proc_stat", return_value=("S", 222)),
            mock.patch.object(thermal_guard.os, "kill") as kill,
        ):
            result = thermal_guard._terminate_owned_child(4321, expected_start_ticks=111)

        self.assertEqual(result, "pid_dead_or_reused")
        kill.assert_not_called()

    def test_termination_signals_verified_owned_pid_and_rechecks_before_kill(self):
        proc_states = [("S", 111), ("S", 222)]
        clock = iter([0.0, 30.0])
        with (
            mock.patch.object(thermal_guard, "_read_proc_stat", side_effect=proc_states),
            mock.patch.object(thermal_guard.os, "kill") as kill,
            mock.patch.object(thermal_guard.time, "monotonic", side_effect=clock),
            mock.patch.object(thermal_guard.time, "sleep"),
        ):
            result = thermal_guard._terminate_owned_child(4321, expected_start_ticks=111)

        self.assertEqual(result, "terminated")
        kill.assert_called_once_with(4321, thermal_guard.signal.SIGTERM)

    def test_adopted_critical_temperature_stops_only_verified_target_pid(self):
        snapshot = gpu_snapshot(temperature=85)
        with tempfile.NamedTemporaryFile(mode="w") as log:
            with (
                mock.patch.object(thermal_guard, "_locked_training_slot", return_value=object()),
                mock.patch.object(thermal_guard, "_release_lock"),
                mock.patch.object(
                    thermal_guard, "_read_proc_stat", side_effect=[("S", 111), ("S", 111)]
                ),
                mock.patch.object(thermal_guard, "_query_snapshot", return_value=snapshot),
                mock.patch.object(thermal_guard, "_append_thermal_event"),
                mock.patch.object(thermal_guard, "_write_status"),
                mock.patch.object(thermal_guard, "_terminate_owned_child") as terminate,
            ):
                result = thermal_guard.monitor_existing(4321, 0, log.name)

        terminate.assert_called_once_with(4321, 111)
        self.assertEqual(result["reason"], "temperature_critical")


class LogProgressTests(unittest.TestCase):
    def test_reads_tail_from_write_only_named_log(self):
        with tempfile.NamedTemporaryFile(mode="w") as log:
            log.write("first\nlast\n")
            log.flush()

            progress = thermal_guard._log_progress(log_file=log)

        self.assertEqual(progress["lastLine"], "last")


class LockAndStatusIsolationTests(unittest.TestCase):
    def test_gpu_lock_and_status_paths_are_physical_gpu_specific(self):
        self.assertEqual(thermal_guard.lock_path(1).name, "training-gpu-1.lock")
        self.assertEqual(thermal_guard.status_path(1).name, "status-gpu-1.json")



def test_guard_dir_defaults_to_runs_guard_and_honours_env(monkeypatch, tmp_path):
    from scripts import thermal_guard

    monkeypatch.delenv("LAT_GUARD_DIR", raising=False)
    assert thermal_guard.guard_dir() == Path("runs/guard").resolve()
    assert thermal_guard.lock_path(1) == Path("runs/guard").resolve() / "training-gpu-1.lock"
    monkeypatch.setenv("LAT_GUARD_DIR", str(tmp_path))
    assert thermal_guard.guard_dir() == tmp_path.resolve()
    assert thermal_guard.status_path(0) == tmp_path.resolve() / "status-gpu-0.json"


if __name__ == "__main__":
    unittest.main()
