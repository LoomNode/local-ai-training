import csv
import json
from pathlib import Path

from scripts.momentum_grid import Arm, arm_command, best_so_far, grid_arms, summarize


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


def test_best_so_far_tracks_minimum_and_final(tmp_path):
    _write_metrics(tmp_path / "metrics.csv", [(0, 3.0), (200, 1.5), (400, 1.6), (600, 1.55)])
    result = best_so_far(tmp_path / "metrics.csv")
    assert result["best"] == 1.5 and result["best_step"] == 200 and result["final"] == 1.55
    assert result["trace"] == [(0, 3.0), (200, 1.5), (400, 1.6), (600, 1.55)]


def test_summarize_ranks_arms_and_reports_gap_fraction(tmp_path):
    _write_metrics(tmp_path / "plain" / "metrics.csv", [(0, 3.0), (5000, 1.40)])
    _write_metrics(tmp_path / "qat" / "metrics.csv", [(0, 3.0), (5000, 1.20)])
    _write_metrics(tmp_path / "leak16-beta0.99" / "metrics.csv", [(0, 3.0), (5000, 1.25)])
    result = summarize(tmp_path)
    assert result["winner"] == "leak16-beta0.99"
    assert abs(result["arms"]["leak16-beta0.99"]["gap_closed"] - 0.75) < 1e-9
    assert json.loads((tmp_path / "results.json").read_text())["winner"] == "leak16-beta0.99"
