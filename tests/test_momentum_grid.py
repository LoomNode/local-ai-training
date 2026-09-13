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
    arm_status,
    assign_queues,
    best_so_far,
    build_parser,
    child_argv,
    grid_arms,
    plan_continue,
    run_serial,
    summarize,
    write_manifest,
)


def test_grid_has_twenty_unique_matched_arms():
    arms = grid_arms()
    assert len(arms) == 20
    assert len({arm.name for arm in arms}) == 20
    assert Arm("plain", "ratchet", 0, 0.0) in arms
    assert Arm("qat", "qat", 0, 0.0) in arms
    # Momentum cells: weight_mode "ratchet" with a nonzero beta, including the
    # leak-0 (pressure leak off) beta-only cells such as "leak0-beta0.9".
    momentum = [arm for arm in arms if arm.weight_mode == "ratchet" and arm.beta > 0]
    assert sorted({arm.leak for arm in momentum}) == [0, 8, 12, 16, 24, 32]
    assert sorted({arm.beta for arm in momentum}) == [0.9, 0.99, 0.999]
    assert len(momentum) == 18


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


def test_summarize_tie_break_prefers_leak_zero_over_nonzero_leak(tmp_path):
    # leak0-beta0.99 (no forgetting) and leak16-beta0.99 are within TIE_TOLERANCE
    # (0.002 apart); leak 0 is the weakest possible leak, so it must win the tie
    # even though its best is slightly worse than leak16's.
    _write_metrics(tmp_path / "plain" / "metrics.csv", [(0, 3.0), (5000, 1.30)])
    _write_metrics(tmp_path / "qat" / "metrics.csv", [(0, 3.0), (5000, 1.10)])
    _write_metrics(tmp_path / "leak0-beta0.99" / "metrics.csv", [(0, 3.0), (5000, 1.202)])
    _write_metrics(tmp_path / "leak16-beta0.99" / "metrics.csv", [(0, 3.0), (5000, 1.200)])
    result = summarize(tmp_path)
    assert sorted(result["tie_set"]) == ["leak0-beta0.99", "leak16-beta0.99"]
    assert result["winner"] == "leak0-beta0.99"


def test_summarize_ignores_interrupted_dirs_left_by_continue(tmp_path):
    # A --continue run moves a partial arm aside to "<arm>.interrupted-<UTC>" but
    # never deletes it; its metrics.csv (with a lower, mid-run best) must not be
    # scored as a candidate -- only the exact arm-name directory should count.
    _write_metrics(tmp_path / "plain" / "metrics.csv", [(0, 3.0), (5000, 1.40)])
    _write_metrics(tmp_path / "qat" / "metrics.csv", [(0, 3.0), (5000, 1.20)])
    _write_metrics(tmp_path / "leak16-beta0.99" / "metrics.csv", [(0, 3.0), (5000, 1.20)])
    _write_metrics(
        tmp_path / "leak16-beta0.99.interrupted-2026-09-13T17:00:00Z" / "metrics.csv",
        [(0, 3.0), (1000, 1.10)],
    )
    result = summarize(tmp_path)
    assert result["winner"] == "leak16-beta0.99"
    assert result["arms"]["leak16-beta0.99"]["best"] == 1.20
    assert "leak16-beta0.99.interrupted-2026-09-13T17:00:00Z" not in result["arms"]


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


def test_assign_queues_splits_round_robin_across_cards():
    arms = grid_arms()
    queues = assign_queues(arms, (0, 1))
    assert set(queues) == {0, 1}
    assert sum(len(q) for q in queues.values()) == len(arms)
    # round-robin by index: arms[0]->gpu0, arms[1]->gpu1, arms[2]->gpu0, ...
    assert queues[0][0] is arms[0]
    assert queues[1][0] is arms[1]
    assert queues[0][1] is arms[2]


def test_assign_queues_single_gpu_keeps_original_order():
    arms = grid_arms()
    queues = assign_queues(arms, (0,))
    assert queues[0] == arms


def test_arm_status_complete_partial_absent(tmp_path):
    _write_metrics(tmp_path / "done" / "metrics.csv", [(0, 3.0), (2000, 1.6), (5000, 1.0)])
    _write_metrics(tmp_path / "running" / "metrics.csv", [(0, 3.0), (2000, 1.5)])
    (tmp_path / "empty-dir").mkdir()
    assert arm_status(tmp_path, "done", 5000) == "complete"
    assert arm_status(tmp_path, "running", 5000) == "partial"
    assert arm_status(tmp_path, "empty-dir", 5000) == "partial"
    assert arm_status(tmp_path, "missing", 5000) == "absent"


def test_plan_continue_skips_complete_moves_partial_and_queues_absent(tmp_path):
    root = tmp_path / "root"
    _write_metrics(root / "plain" / "metrics.csv", [(0, 3.0), (5000, 1.0)])
    _write_metrics(root / "qat" / "metrics.csv", [(0, 3.0), (5000, 1.0)])
    _write_metrics(root / "leak8-beta0.9" / "metrics.csv", [(0, 3.0), (1800, 1.5)])
    (root / "leak8-beta0.9.log").write_text("partial log\n")
    arms = grid_arms()
    now = "20260913T000000Z"

    plan = plan_continue(root, arms, 5000, now)

    assert plan["skipped"] == ["plain", "qat"]
    assert plan["moved_aside"] == [
        {"arm": "leak8-beta0.9", "moved_to": f"leak8-beta0.9.interrupted-{now}"}
    ]
    # 20 arms - 2 complete = 18 queued (partial rerun + all absent arms)
    assert len(plan["queued"]) == 18
    # leak0-beta0.9/0.99/0.999 precede leak8-beta0.9 in grid order and are absent
    # (never written above), so they queue ahead of the partial leak8-beta0.9 arm.
    assert plan["queued"][0] == "leak0-beta0.9"
    assert "leak8-beta0.9" in plan["queued"]
    assert "plain" not in plan["queued"] and "qat" not in plan["queued"]

    moved_dir = root / f"leak8-beta0.9.interrupted-{now}"
    assert moved_dir.exists() and (moved_dir / "metrics.csv").exists()
    assert not (root / "leak8-beta0.9").exists()
    assert (root / f"leak8-beta0.9.interrupted-{now}.log").exists()
    assert not (root / "leak8-beta0.9.log").exists()


def test_plan_continue_dry_run_does_not_move_anything(tmp_path):
    root = tmp_path / "root"
    _write_metrics(root / "plain" / "metrics.csv", [(0, 3.0), (5000, 1.0)])
    _write_metrics(root / "leak8-beta0.9" / "metrics.csv", [(0, 3.0), (1800, 1.5)])
    (root / "leak8-beta0.9.log").write_text("partial log\n")
    arms = grid_arms()

    plan = plan_continue(root, arms, 5000, "20260913T000000Z", dry_run=True)

    assert plan["skipped"] == ["plain"]
    assert plan["moved_aside"] == [
        {"arm": "leak8-beta0.9", "moved_to": "leak8-beta0.9.interrupted-20260913T000000Z"}
    ]
    # 20 arms - 1 skipped (plain) = 19 queued
    assert len(plan["queued"]) == 19
    # nothing actually moved
    assert (root / "leak8-beta0.9").exists()
    assert (root / "leak8-beta0.9.log").exists()
    assert not (root / "leak8-beta0.9.interrupted-20260913T000000Z").exists()


def test_child_argv_round_trips_through_the_parser():
    parser = build_parser()
    parent_args = parser.parse_args(
        [
            "--root", "myroot",
            "--gpus", "0,1",
            "--config", "cfg.toml",
            "--dataset", "ds.bin",
            "--seed", "42",
            "--continue",
        ]
    )
    argv = child_argv(parent_args, gpu=1)
    child_args = parser.parse_args(argv[3:])
    assert child_args.root == parent_args.root
    assert child_args.gpus == parent_args.gpus
    assert child_args.config == parent_args.config
    assert child_args.dataset == parent_args.dataset
    assert child_args.seed == parent_args.seed
    assert child_args.continue_ is True
    assert child_args.queue_gpu == 1


def test_child_argv_omits_continue_flag_when_not_set():
    parser = build_parser()
    parent_args = parser.parse_args(["--gpus", "0,1"])
    argv = child_argv(parent_args, gpu=0)
    assert "--continue" not in argv
    child_args = parser.parse_args(argv[3:])
    assert child_args.continue_ is False
    assert child_args.queue_gpu == 0


def test_single_gpu_dry_run_order_and_format_unchanged(capsys):
    exit_code = momentum_grid.main(["--dry-run"])
    assert exit_code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    names = [line.split(" ", 1)[0] for line in lines]
    assert names[:3] == ["plain", "qat", "leak0-beta0.9"]
    # today's format is "<arm> <command...>" -- no leading gpu column
    assert lines[0].startswith("plain ")
    assert "train" in lines[0]


def test_deprecated_gpu_flag_sets_single_card_gpus(capsys):
    exit_code = momentum_grid.main(["--dry-run", "--gpu", "1"])
    assert exit_code == 0
    # still the unchanged single-gpu format since --gpu resolves to one card
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split(" ", 1)[0] == "plain"


def test_multi_gpu_dry_run_prints_gpu_arm_command(capsys):
    exit_code = momentum_grid.main(["--dry-run", "--gpus", "0,1"])
    assert exit_code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    first = lines[0].split(" ")
    assert first[0] == "0"
    assert first[1] == "plain"
    second = lines[1].split(" ")
    assert second[0] == "1"
    assert second[1] == "qat"


def test_continue_without_manifest_errors(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    try:
        momentum_grid.main(["--root", str(root), "--continue", "--dry-run"])
        raised = False
    except SystemExit as exc:
        raised = True
        assert "nothing to continue" in str(exc)
    assert raised


def test_non_continue_refuses_existing_manifest(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "manifest.json").write_text("{}\n")
    try:
        momentum_grid.main(["--root", str(root)])
        raised = False
    except SystemExit as exc:
        raised = True
        assert "already has a manifest" in str(exc)
    assert raised
