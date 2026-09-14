import csv
import json
from pathlib import Path

from scripts.update_rule_screen import (
    CELLS,
    Cell,
    build_parser,
    cell_command,
    main,
    summarize,
)


def test_cells_have_the_five_expected_lever_combinations():
    assert CELLS == (
        Cell("stoch", True, 0.0),
        Cell("pw0.5", False, 0.5),
        Cell("pw1.0", False, 1.0),
        Cell("stoch-pw0.5", True, 0.5),
        Cell("stoch-pw1.0", True, 1.0),
    )


def test_cell_command_carries_the_right_flags(tmp_path):
    common = dict(
        config=Path("c.toml"), dataset=Path("d"), output=tmp_path, seed=1337, lat=Path("lat")
    )
    stoch = cell_command(Cell("stoch", True, 0.0), **common)
    pw = cell_command(Cell("pw0.5", False, 0.5), **common)
    both = cell_command(Cell("stoch-pw0.5", True, 0.5), **common)

    for cmd in (stoch, pw, both):
        assert cmd[:2] == ["lat", "train"]
        assert cmd[cmd.index("--codes") + 1] == "15"
        assert cmd[cmd.index("--seed") + 1] == "1337"
        assert cmd[cmd.index("--weight-mode") + 1] == "ratchet"
        assert cmd[cmd.index("--config") + 1] == "c.toml"
        assert cmd[cmd.index("--dataset-path") + 1] == "d"
        assert cmd[cmd.index("--output") + 1] == str(tmp_path)

    assert "--stochastic-bucket" in stoch
    assert "--pressure-weight" not in stoch

    assert "--stochastic-bucket" not in pw
    assert pw[pw.index("--pressure-weight") + 1] == "0.5"

    assert "--stochastic-bucket" in both
    assert both[both.index("--pressure-weight") + 1] == "0.5"


def _write_metrics(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "validation_loss"])
        writer.writeheader()
        for step, loss in rows:
            writer.writerow({"step": step, "validation_loss": loss})


def test_summarize_picks_winner_and_computes_advances_and_gap_closed(tmp_path):
    root = tmp_path / "root"
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    (reference_root / "results.json").write_text(
        json.dumps({"arms": {"plain": {"best": 1.1671}, "qat": {"best": 1.1250}}})
    )

    _write_metrics(root / "stoch" / "metrics.csv", [(0, 3.0), (5000, 1.150)])
    _write_metrics(root / "pw0.5" / "metrics.csv", [(0, 3.0), (5000, 1.140)])
    _write_metrics(root / "pw1.0" / "metrics.csv", [(0, 3.0), (5000, 1.160)])
    _write_metrics(root / "stoch-pw0.5" / "metrics.csv", [(0, 3.0), (5000, 1.155)])
    _write_metrics(root / "stoch-pw1.0" / "metrics.csv", [(0, 3.0), (5000, 1.170)])

    result = summarize(root, reference_root)

    assert result["winner"] == "pw0.5"
    assert result["references"] == {"plain": {"best": 1.1671}, "qat": {"best": 1.1250}}
    assert result["advances"] is True  # 1.140 < 1.1671 - 0.005
    expected_gap_closed = (1.1671 - 1.140) / (1.1671 - 1.1250)
    assert abs(result["gap_closed"] - expected_gap_closed) < 1e-9
    assert json.loads((root / "results.json").read_text())["winner"] == "pw0.5"


def test_summarize_does_not_advance_when_winner_is_close_to_plain(tmp_path):
    root = tmp_path / "root"
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    (reference_root / "results.json").write_text(
        json.dumps({"arms": {"plain": {"best": 1.1671}, "qat": {"best": 1.1250}}})
    )
    _write_metrics(root / "stoch" / "metrics.csv", [(0, 3.0), (5000, 1.165)])

    result = summarize(root, reference_root)

    assert result["winner"] == "stoch"
    assert result["advances"] is False  # 1.165 is not below plain - 0.005


def test_dry_run_prints_five_commands(tmp_path, capsys):
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
    assert len(lines) == 5
    names = [line.split(" ", 1)[0] for line in lines]
    assert names == ["stoch", "pw0.5", "pw1.0", "stoch-pw0.5", "stoch-pw1.0"]
    assert not (tmp_path / "root").exists()  # dry-run must not create the root


def test_build_parser_defaults():
    args = build_parser().parse_args([])
    assert args.root == Path("/games/ailab/local-ai-training/runs/update-rule-screen-2026-09-14")
    assert args.dataset == Path("/games/ailab/local-ai-training/data/text8/text8")
    assert args.reference_root == Path(
        "/games/ailab/local-ai-training/runs/momentum-grid-2026-09-13"
    )
    assert str(args.config).endswith("configs/scaleup_text8_25m_5k.toml")
    assert args.config.is_absolute()


def test_main_summarize_only_uses_existing_root_without_running_commands(tmp_path, monkeypatch):
    import scripts.update_rule_screen as update_rule_screen

    def boom(*args, **kwargs):
        raise AssertionError("run_serial must not be called with --summarize-only")

    monkeypatch.setattr(update_rule_screen, "run_serial", boom)

    root = tmp_path / "root"
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    (reference_root / "results.json").write_text(
        json.dumps({"arms": {"plain": {"best": 1.1671}, "qat": {"best": 1.1250}}})
    )
    _write_metrics(root / "stoch" / "metrics.csv", [(0, 3.0), (5000, 1.150)])

    exit_code = main(
        [
            "--root", str(root),
            "--reference-root", str(reference_root),
            "--summarize-only",
        ]
    )

    assert exit_code == 0
    assert (root / "results.json").exists()
