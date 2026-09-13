import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import momentum_grid
from scripts.momentum_grid import (
    TIE_TOLERANCE,
    Arm,
    arm_command,
    best_so_far,
    grid_arms,
    run_serial,
    summarize,
    write_manifest,
)


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
    common = dict(
        config=Path("c.toml"), dataset=Path("d"), output=tmp_path, seed=1337, lat=Path("lat")
    )
    plain = arm_command(Arm("plain", "ratchet", 0, 0.0), **common)
    momentum = arm_command(Arm("leak16-beta0.99", "ratchet", 16, 0.99), **common)
    qat = arm_command(Arm("qat", "qat", 0, 0.0), **common)
    for cmd in (plain, momentum, qat):
        assert cmd[:2] == ["lat", "train"] and "--codes" in cmd
        assert cmd[cmd.index("--codes") + 1] == "15"
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


def _write_full_metrics(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "validation_loss",
        "saturated_percent",
        "cumulative_code_moves",
        "ratchet_state_bytes",
        "support_parameter_bytes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_best_so_far_tracks_minimum_and_final(tmp_path):
    _write_metrics(tmp_path / "metrics.csv", [(0, 3.0), (200, 1.5), (400, 1.6), (600, 1.55)])
    result = best_so_far(tmp_path / "metrics.csv")
    assert result["best"] == 1.5 and result["best_step"] == 200 and result["final"] == 1.55
    assert result["trace"] == [(0, 3.0), (200, 1.5), (400, 1.6), (600, 1.55)]


def test_best_so_far_extra_phase2_columns_default_to_none_when_absent(tmp_path):
    _write_metrics(tmp_path / "metrics.csv", [(0, 3.0), (200, 1.5)])
    result = best_so_far(tmp_path / "metrics.csv")
    # saturated_percent is present in this fixture's schema (always 10.0); the other
    # three Phase 2 columns are not, and must come back as None rather than KeyError.
    assert result["saturated_percent"] == 10.0
    assert result["cumulative_code_moves"] is None
    assert result["ratchet_state_bytes"] is None
    assert result["support_parameter_bytes"] is None


def test_best_so_far_carries_phase2_metrics_from_final_row_when_present(tmp_path):
    _write_full_metrics(
        tmp_path / "metrics.csv",
        [
            {
                "step": 0,
                "validation_loss": 3.0,
                "saturated_percent": 1.0,
                "cumulative_code_moves": 10,
                "ratchet_state_bytes": 1_000_000,
                "support_parameter_bytes": 500_000,
            },
            {
                "step": 200,
                "validation_loss": 1.5,
                "saturated_percent": 5.5,
                "cumulative_code_moves": 42,
                "ratchet_state_bytes": 1_000_000,
                "support_parameter_bytes": 500_000,
            },
        ],
    )
    result = best_so_far(tmp_path / "metrics.csv")
    assert result["saturated_percent"] == 5.5
    assert result["cumulative_code_moves"] == 42.0
    assert result["ratchet_state_bytes"] == 1_000_000.0
    assert result["support_parameter_bytes"] == 500_000.0


def test_summarize_ranks_arms_and_reports_gap_fraction(tmp_path):
    _write_metrics(tmp_path / "plain" / "metrics.csv", [(0, 3.0), (5000, 1.40)])
    _write_metrics(tmp_path / "qat" / "metrics.csv", [(0, 3.0), (5000, 1.20)])
    _write_metrics(tmp_path / "leak16-beta0.99" / "metrics.csv", [(0, 3.0), (5000, 1.25)])
    result = summarize(tmp_path)
    assert result["winner"] == "leak16-beta0.99"
    assert abs(result["arms"]["leak16-beta0.99"]["gap_closed"] - 0.75) < 1e-9
    assert json.loads((tmp_path / "results.json").read_text())["winner"] == "leak16-beta0.99"


def test_summarize_applies_tie_tolerance_and_prefers_larger_leak(tmp_path):
    assert TIE_TOLERANCE == 0.005
    _write_metrics(tmp_path / "plain" / "metrics.csv", [(0, 3.0), (5000, 1.40)])
    _write_metrics(tmp_path / "qat" / "metrics.csv", [(0, 3.0), (5000, 1.20)])
    # leak8 and leak16 are 0.002 apart (within the 0.005 tolerance); leak32 is
    # 0.010 behind the minimum and must be excluded from the tie set.
    _write_metrics(tmp_path / "leak8-beta0.9" / "metrics.csv", [(0, 3.0), (5000, 1.100)])
    _write_metrics(tmp_path / "leak16-beta0.9" / "metrics.csv", [(0, 3.0), (5000, 1.102)])
    _write_metrics(tmp_path / "leak32-beta0.9" / "metrics.csv", [(0, 3.0), (5000, 1.110)])
    result = summarize(tmp_path)
    assert sorted(result["tie_set"]) == ["leak16-beta0.9", "leak8-beta0.9"]
    assert result["winner"] == "leak16-beta0.9"
    assert json.loads((tmp_path / "results.json").read_text())["tie_set"] == result["tie_set"]


def test_write_manifest_records_hashes_and_extra_keys(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("x = 1\n")
    dataset = tmp_path / "data.txt"
    dataset.write_text("hello\n")
    root = tmp_path / "root"
    root.mkdir()
    write_manifest(
        root, config=config, dataset=dataset, extra={"seed": 1337, "codes": 15, "gpu": 0}
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert manifest["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert manifest["seed"] == 1337
    assert manifest["codes"] == 15
    assert manifest["gpu"] == 0
    assert "source_commit" in manifest
    assert "started_utc" in manifest


def test_run_serial_stops_on_first_failure_and_logs_lifecycle(tmp_path, monkeypatch):
    calls = []

    def fake_run_training(cmd, log, env, gpu):
        calls.append(cmd)
        return SimpleNamespace(returncode=1 if cmd == ["bad"] else 0)

    monkeypatch.setattr(momentum_grid, "run_training", fake_run_training)
    named_commands = [("good1", ["ok"]), ("bad", ["bad"]), ("good2", ["ok2"])]
    exit_code = run_serial(named_commands, tmp_path, "study.log", gpu=0)
    assert exit_code == 1
    assert calls == [["ok"], ["bad"]]
    log_text = (tmp_path / "study.log").read_text()
    assert "START good1" in log_text
    assert "EXIT good1 0" in log_text
    assert "START bad" in log_text
    assert "EXIT bad 1" in log_text
    assert "STOPPED: bad failed" in log_text
    assert "good2" not in log_text


def test_main_completes_when_root_preexists_without_manifest(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()  # exists already, but has no manifest.json
    config = tmp_path / "config.toml"
    config.write_text("x = 1\n")
    dataset = tmp_path / "data.txt"
    dataset.write_text("d\n")

    monkeypatch.setattr(
        momentum_grid,
        "run_training",
        lambda cmd, log, env, gpu: SimpleNamespace(returncode=0),
    )
    exit_code = momentum_grid.main(
        [
            "--root", str(root),
            "--config", str(config),
            "--dataset", str(dataset),
            "--gpu", "0",
        ]
    )
    assert exit_code == 0
    assert (root / "manifest.json").exists()
