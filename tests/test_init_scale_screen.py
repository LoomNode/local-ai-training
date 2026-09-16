import csv
import json
from pathlib import Path

import pytest

from scripts.init_scale_screen import (
    CELLS,
    build_parser,
    cell_command,
    format_table,
    main,
    summarize,
)
from scripts.momentum_grid import assign_queues


def test_cells_have_the_six_expected_k_values():
    assert [cell.k for cell in CELLS] == [1, 2, 3, 4, 6, 8]
    assert [cell.name for cell in CELLS] == ["k1", "k2", "k3", "k4", "k6", "k8"]


def test_cell_command_appends_scale_multiplier_except_for_k1(tmp_path):
    common = dict(
        config=Path("c.toml"), dataset=Path("d"), output=tmp_path, seed=1337, lat=Path("lat")
    )
    k1_cell = next(c for c in CELLS if c.name == "k1")
    k4_cell = next(c for c in CELLS if c.name == "k4")

    k1_cmd = cell_command(k1_cell, **common)
    k4_cmd = cell_command(k4_cell, **common)

    assert "--scale-multiplier" not in k1_cmd
    assert k4_cmd[k4_cmd.index("--scale-multiplier") + 1] == "4"
    for cmd in (k1_cmd, k4_cmd):
        assert cmd[:2] == ["lat", "train"]
        assert cmd[cmd.index("--codes") + 1] == "15"
        assert cmd[cmd.index("--seed") + 1] == "1337"
        assert cmd[cmd.index("--weight-mode") + 1] == "ratchet"
        assert cmd[cmd.index("--config") + 1] == "c.toml"
        assert cmd[cmd.index("--dataset-path") + 1] == "d"
        assert cmd[cmd.index("--output") + 1] == str(tmp_path)


def test_assign_queues_splits_six_cells_across_two_gpus():
    queues = assign_queues(list(CELLS), (0, 1))
    assert [c.name for c in queues[0]] == ["k1", "k3", "k6"]
    assert [c.name for c in queues[1]] == ["k2", "k4", "k8"]


def _write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "validation_loss",
        "saturated_percent",
        "zero_percent",
        "move_percent",
        "code_histogram",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _histogram(counts: dict[int, int]) -> str:
    return json.dumps({str(k): v for k, v in counts.items()})


def _rows_for(final_loss: float, *, saturated=10.0, zero=20.0, move=5.0, counts=None) -> list:
    counts = counts if counts is not None else {-6: 5, -1: 20, 0: 50, 1: 20, 6: 5}
    return [
        {
            "step": 200,
            "validation_loss": 3.0,
            "saturated_percent": 15.0,
            "zero_percent": 25.0,
            "move_percent": 8.0,
            "code_histogram": _histogram(counts),
        },
        {
            "step": 5000,
            "validation_loss": final_loss,
            "saturated_percent": saturated,
            "zero_percent": zero,
            "move_percent": move,
            "code_histogram": _histogram(counts),
        },
    ]


def _write_reference(reference_root: Path, plain_best: float, qat_best: float) -> None:
    reference_root.mkdir(parents=True, exist_ok=True)
    (reference_root / "results.json").write_text(
        json.dumps(
            {
                "arms": {
                    "plain": {"best": plain_best, "best_step": 5000, "final": plain_best},
                    "qat": {"best": qat_best, "best_step": 5000, "final": qat_best},
                }
            }
        )
    )


def test_summarize_picks_winner_computes_advances_and_identity_check(tmp_path):
    root = tmp_path / "root"
    reference_root = tmp_path / "reference"
    _write_reference(reference_root, plain_best=1.1671, qat_best=1.1250)

    # k1 must reproduce the reference plain best exactly.
    _write_metrics(root / "k1" / "metrics.csv", _rows_for(1.1671))
    _write_metrics(root / "k2" / "metrics.csv", _rows_for(1.150))
    _write_metrics(root / "k3" / "metrics.csv", _rows_for(1.140))
    _write_metrics(root / "k4" / "metrics.csv", _rows_for(1.130))
    _write_metrics(root / "k6" / "metrics.csv", _rows_for(1.155))
    _write_metrics(root / "k8" / "metrics.csv", _rows_for(1.160))

    result = summarize(root, reference_root)

    assert result["winner"] == "k4"
    assert result["winner_best"] == 1.130
    assert result["advances"] is True  # 1.130 < 1.1671 - 0.005
    expected_gap_closed = (1.1671 - 1.130) / (1.1671 - 1.1250)
    assert abs(result["gap_closed"] - expected_gap_closed) < 1e-9
    assert result["identity_check"] == {
        "k1_best": 1.1671,
        "reference_plain_best": 1.1671,
        "identical": True,
    }
    assert result["verdict"] == "advances"

    k4_record = result["cells"]["k4"]
    assert k4_record["saturated_percent"] == 10.0
    assert k4_record["zero_percent"] == 20.0
    assert k4_record["move_percent"] == 5.0
    # counts: {-6: 5, -1: 20, 0: 50, 1: 20, 6: 5}, total 100
    assert k4_record["tail_ge5"] == pytest.approx(10.0)  # -6 and 6
    assert k4_record["tail_le1"] == pytest.approx(90.0)  # -1, 0, 1

    on_disk = json.loads((root / "results.json").read_text())
    assert on_disk["winner"] == "k4"


def test_summarize_does_not_advance_when_winner_is_close_to_plain(tmp_path):
    root = tmp_path / "root"
    reference_root = tmp_path / "reference"
    _write_reference(reference_root, plain_best=1.1671, qat_best=1.1250)
    for name in ("k1", "k2", "k3", "k4", "k6", "k8"):
        _write_metrics(root / name / "metrics.csv", _rows_for(1.1671))

    result = summarize(root, reference_root)

    assert result["advances"] is False
    assert result["verdict"] == "no-advance"


def test_summarize_incomplete_root_yields_incomplete_verdict(tmp_path):
    root = tmp_path / "root"
    reference_root = tmp_path / "reference"
    _write_reference(reference_root, plain_best=1.1671, qat_best=1.1250)
    _write_metrics(root / "k1" / "metrics.csv", _rows_for(1.1671))
    _write_metrics(root / "k2" / "metrics.csv", _rows_for(1.150))

    result = summarize(root, reference_root)

    assert result["verdict"] == "incomplete"
    assert result["winner"] == "k2"


def test_format_table_includes_expected_columns_and_rows(tmp_path):
    root = tmp_path / "root"
    reference_root = tmp_path / "reference"
    _write_reference(reference_root, plain_best=1.1671, qat_best=1.1250)
    for name in ("k1", "k2", "k3", "k4", "k6", "k8"):
        _write_metrics(root / name / "metrics.csv", _rows_for(1.1671))
    result = summarize(root, reference_root)

    table = format_table(result["cells"])

    assert "best" in table
    assert "saturated %" in table
    assert "tail>=5 %" in table
    assert "tail<=1 %" in table
    assert "move %" in table
    for name in ("k1", "k2", "k3", "k4", "k6", "k8"):
        assert name in table


def test_dry_run_prints_six_commands_and_creates_nothing(tmp_path, capsys):
    config = tmp_path / "c.toml"
    config.write_text("x = 1\n")
    dataset = tmp_path / "d.txt"
    dataset.write_text("d\n")

    exit_code = main(
        [
            "--root", str(tmp_path / "root"),
            "--config", str(config),
            "--dataset", str(dataset),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 6
    names = [line.split(" ", 1)[0] for line in lines]
    assert names == ["k1", "k2", "k3", "k4", "k6", "k8"]
    assert not (tmp_path / "root").exists()


def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.root == Path("/games/ailab/local-ai-training/runs/init-scale-screen-2026-09-15")
    assert args.dataset == Path("/games/ailab/local-ai-training/data/text8/text8")
    assert args.reference_root == Path(
        "/games/ailab/local-ai-training/runs/momentum-grid-2026-09-13"
    )
    assert args.gpus == "0,1"
    assert str(args.config).endswith("configs/scaleup_text8_25m_5k.toml")
    assert args.config.is_absolute()


def test_main_summarize_only_does_not_run_commands(tmp_path, monkeypatch):
    import scripts.init_scale_screen as init_scale_screen

    def boom(*args, **kwargs):
        raise AssertionError("run_serial must not be called with --summarize-only")

    monkeypatch.setattr(init_scale_screen, "run_serial", boom)

    root = tmp_path / "root"
    reference_root = tmp_path / "reference"
    _write_reference(reference_root, plain_best=1.1671, qat_best=1.1250)
    _write_metrics(root / "k1" / "metrics.csv", _rows_for(1.1671))

    exit_code = main(
        [
            "--root", str(root),
            "--reference-root", str(reference_root),
            "--summarize-only",
        ]
    )

    assert exit_code == 0
    assert (root / "results.json").exists()


def test_queue_gpu_child_mode_runs_only_its_slice(tmp_path, monkeypatch):
    import scripts.init_scale_screen as init_scale_screen

    captured = {}

    def _fake_run_serial(named_commands, root, log_name, gpu):
        captured["names"] = [name for name, _ in named_commands]
        captured["log_name"] = log_name
        captured["gpu"] = gpu
        return 0

    monkeypatch.setattr(init_scale_screen, "run_serial", _fake_run_serial)

    config = tmp_path / "c.toml"
    config.write_text("x = 1\n")
    dataset = tmp_path / "d.txt"
    dataset.write_text("d\n")
    root = tmp_path / "root"

    exit_code = main(
        [
            "--root", str(root),
            "--config", str(config),
            "--dataset", str(dataset),
            "--gpus", "0,1",
            "--queue-gpu", "0",
        ]
    )

    assert exit_code == 0
    assert captured["names"] == ["k1", "k3", "k6"]
    assert captured["log_name"] == "queue-gpu0.log"
    assert captured["gpu"] == 0


def test_parent_mode_spawns_one_child_per_gpu_writes_manifest_and_summarizes(
    tmp_path, monkeypatch
):
    import scripts.init_scale_screen as init_scale_screen

    root = tmp_path / "root"
    reference_root = tmp_path / "reference"
    _write_reference(reference_root, plain_best=1.1671, qat_best=1.1250)
    for name in ("k1", "k2", "k3", "k4", "k6", "k8"):
        _write_metrics(root / name / "metrics.csv", _rows_for(1.1671))

    config = tmp_path / "c.toml"
    config.write_text("x = 1\n")
    dataset = tmp_path / "d.txt"
    dataset.write_text("d\n")

    spawned_argv = []
    real_popen = init_scale_screen.subprocess.Popen

    class _FakeProcess:
        def wait(self):
            return 0

    def _fake_popen(argv, cwd=None, **kwargs):
        # write_manifest also shells out (git rev-parse/status) through this same
        # subprocess module; only intercept the child-driver spawns, not those.
        if isinstance(argv, list) and "scripts.init_scale_screen" in argv:
            spawned_argv.append(argv)
            return _FakeProcess()
        return real_popen(argv, cwd=cwd, **kwargs)

    monkeypatch.setattr(init_scale_screen.subprocess, "Popen", _fake_popen)

    exit_code = main(
        [
            "--root", str(root),
            "--config", str(config),
            "--dataset", str(dataset),
            "--reference-root", str(reference_root),
            "--gpus", "0,1",
        ]
    )

    assert exit_code == 0
    assert len(spawned_argv) == 2
    for argv in spawned_argv:
        assert "--queue-gpu" in argv
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["cells"] == [{"name": c.name, "k": c.k} for c in CELLS]
    assert manifest["reference_root"] == str(reference_root)
    assert "spec_path" in manifest
    assert (root / "results.json").exists()


def test_child_argv_round_trips_through_the_parser():
    parser = build_parser()
    parent_args = parser.parse_args(
        [
            "--root", "myroot",
            "--config", "c.toml",
            "--dataset", "d.txt",
            "--reference-root", "refroot",
            "--gpus", "0,1",
        ]
    )
    from scripts.init_scale_screen import child_argv

    argv = child_argv(parent_args, gpu=1)
    child_args = parser.parse_args(argv[3:])
    assert child_args.root == parent_args.root
    assert child_args.config == parent_args.config
    assert child_args.dataset == parent_args.dataset
    assert child_args.reference_root == parent_args.reference_root
    assert child_args.gpus == parent_args.gpus
    assert child_args.queue_gpu == 1
