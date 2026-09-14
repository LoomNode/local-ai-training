import csv
import json
from pathlib import Path

from scripts import gap_ladder
from scripts.gap_ladder import (
    ARMS,
    RUNGS,
    SEEDS,
    assign_queues,
    build_parser,
    child_argv,
    ladder_items,
    summarize,
)


def test_ladder_items_order_and_shape():
    items = ladder_items(SEEDS)
    assert len(items) == 18
    # Rung-major (99m first), arm-major, seed-minor.
    expected_rungs = ["99m"] * 9 + ["7m"] * 9
    assert [rung for rung, _, _ in items] == expected_rungs
    expected_arms = (["plain"] * 3 + ["qat"] * 3 + ["fp32"] * 3) * 2
    assert [arm.name for _, arm, _ in items] == expected_arms
    expected_seeds = list(SEEDS) * 6
    assert [seed for _, _, seed in items] == expected_seeds

    root = Path("root")
    dataset = Path("dataset")
    for rung, arm, seed in items:
        cmd = gap_ladder._command(rung, arm, seed, root, dataset)
        if arm.name == "fp32":
            assert "--codes" not in cmd
        else:
            assert cmd[cmd.index("--codes") + 1] == "15"


def test_assign_queues_gives_both_cards_a_99m_plain_run():
    items = ladder_items(SEEDS)
    queues = assign_queues(items, (0, 1))
    for gpu in (0, 1):
        names = {(rung, arm.name) for rung, arm, _ in queues[gpu]}
        assert ("99m", "plain") in names


def _write_metrics(path: Path, losses, resolved_steps: int | None = 30000, complete: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "validation_loss",
        "resolved_steps",
        "saturated_percent",
        "cumulative_code_moves",
        "ratchet_state_bytes",
        "support_parameter_bytes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        last_index = len(losses) - 1
        for index, loss in enumerate(losses):
            step = index * 200
            if complete and resolved_steps is not None and index == last_index:
                step = resolved_steps
            writer.writerow(
                {
                    "step": step,
                    "validation_loss": loss,
                    "resolved_steps": resolved_steps if resolved_steps is not None else "",
                    "saturated_percent": 1.0,
                    "cumulative_code_moves": 10.0,
                    "ratchet_state_bytes": 100.0,
                    "support_parameter_bytes": 1000.0,
                }
            )


def _write_rung(root: Path, tag: str, per_arm_losses: dict[str, dict[int, list[float]]]):
    for arm_name, per_seed in per_arm_losses.items():
        for seed, losses in per_seed.items():
            name = f"{tag}-{arm_name}-seed{seed}"
            _write_metrics(root / name / "metrics.csv", losses)


def _write_baseline(root: Path, per_arm_losses: dict[str, dict[int, list[float]]]):
    per_seed: dict[str, dict[str, dict]] = {}
    for arm_name, by_seed in per_arm_losses.items():
        for seed, losses in by_seed.items():
            best = min(losses)
            per_seed.setdefault(str(seed), {})[arm_name] = {
                "best": best,
                "best_step": losses.index(best) * 200,
                "final": losses[-1],
                "last_four_mean": sum(losses[-4:]) / len(losses[-4:]),
                "saturated_percent": 0.0,
                "cumulative_code_moves": 0.0,
                "ratchet_state_bytes": 0.0,
                "support_parameter_bytes": 100.0,
                "complete": True,
            }
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.json").write_text(json.dumps({"per_seed": per_seed}))


def _full_losses(plain, qat, fp32):
    """Nine complete runs (3 arms x 3 seeds) with the given best-loss per arm."""
    return {
        "plain": {seed: [3.0, plain + 0.1, plain] for seed in SEEDS},
        "qat": {seed: [3.0, qat + 0.1, qat] for seed in SEEDS},
        "fp32": {seed: [3.0, fp32 + 0.1, fp32] for seed in SEEDS},
    }


def test_summarize_computes_gaps_and_shrinking_verdict(tmp_path):
    root = tmp_path / "root"
    baseline_root = tmp_path / "baseline"
    root.mkdir()

    # plain_minus_qat means: 7m=0.10, 25m=0.06, 99m=0.02 -- strictly decreasing,
    # 7m - 99m = 0.08 >= 0.02 -> shrinking.
    _write_rung(root, "7m", _full_losses(plain=1.20, qat=1.10, fp32=1.00))
    _write_rung(root, "99m", _full_losses(plain=1.02, qat=1.00, fp32=0.90))
    _write_baseline(baseline_root, _full_losses(plain=1.06, qat=1.00, fp32=0.95))

    result = summarize(root, baseline_root)
    assert result["order"] == ["7m", "25m", "99m"]
    assert result["verdict"] == "shrinking"
    for tag in ("7m", "25m", "99m"):
        assert result["rungs"][tag]["complete"] is True
    gaps_7m = result["rungs"]["7m"]["gaps"]["plain_minus_qat"]
    assert abs(gaps_7m["mean"] - 0.10) < 1e-9
    assert gaps_7m["sd"] == 0.0
    assert result["rungs"]["99m"]["ratchet_weights"] == gap_ladder.RATCHET_WEIGHTS["99m"]
    assert result["rungs"]["25m"]["ratchet_weights"] == gap_ladder.RATCHET_WEIGHTS["25m"]
    # per_seed records are trimmed to the baseline field set (no "trace"/"complete").
    entry = result["rungs"]["7m"]["per_seed"][str(SEEDS[0])]["plain"]
    assert set(entry.keys()) == set(gap_ladder._BASELINE_FIELDS)

    with (root / "results.json").open() as handle:
        on_disk = json.load(handle)
    assert on_disk["verdict"] == "shrinking"


def test_summarize_growing_verdict(tmp_path):
    root = tmp_path / "root"
    baseline_root = tmp_path / "baseline"
    root.mkdir()

    # plain_minus_qat means: 7m=0.02, 25m=0.06, 99m=0.10 -- strictly increasing,
    # 99m - 7m = 0.08 >= 0.02 -> growing.
    _write_rung(root, "7m", _full_losses(plain=1.02, qat=1.00, fp32=0.90))
    _write_rung(root, "99m", _full_losses(plain=1.20, qat=1.10, fp32=1.00))
    _write_baseline(baseline_root, _full_losses(plain=1.06, qat=1.00, fp32=0.95))

    result = summarize(root, baseline_root)
    assert result["verdict"] == "growing"


def test_summarize_flat_verdict_when_gap_shrinks_below_threshold(tmp_path):
    root = tmp_path / "root"
    baseline_root = tmp_path / "baseline"
    root.mkdir()

    # plain_minus_qat means: 7m=0.10, 25m=0.09, 99m=0.09 -- decreasing but not strictly
    # (25m == 99m), and 7m - 99m = 0.01 < 0.02 -> flat either way.
    _write_rung(root, "7m", _full_losses(plain=1.20, qat=1.10, fp32=1.00))
    _write_rung(root, "99m", _full_losses(plain=1.09, qat=1.00, fp32=0.90))
    _write_baseline(baseline_root, _full_losses(plain=1.09, qat=1.00, fp32=0.95))

    result = summarize(root, baseline_root)
    assert result["verdict"] == "flat"


def test_summarize_incomplete_rung_yields_incomplete_verdict(tmp_path):
    root = tmp_path / "root"
    baseline_root = tmp_path / "baseline"
    root.mkdir()

    _write_rung(root, "99m", _full_losses(plain=1.02, qat=1.00, fp32=0.90))
    _write_baseline(baseline_root, _full_losses(plain=1.06, qat=1.00, fp32=0.95))

    # 7m rung: only 8 of 9 runs present (missing fp32 for one seed).
    seven_m = _full_losses(plain=1.20, qat=1.10, fp32=1.00)
    del seven_m["fp32"][SEEDS[-1]]
    _write_rung(root, "7m", seven_m)

    result = summarize(root, baseline_root)
    assert result["rungs"]["7m"]["complete"] is False
    assert result["verdict"] == "incomplete"


def test_summarize_incomplete_when_a_run_stops_short(tmp_path):
    root = tmp_path / "root"
    baseline_root = tmp_path / "baseline"
    root.mkdir()

    _write_rung(root, "99m", _full_losses(plain=1.02, qat=1.00, fp32=0.90))
    _write_baseline(baseline_root, _full_losses(plain=1.06, qat=1.00, fp32=0.95))
    _write_rung(root, "7m", _full_losses(plain=1.20, qat=1.10, fp32=1.00))
    # Overwrite one run's metrics.csv so its last step never reaches resolved_steps.
    _write_metrics(
        root / f"7m-plain-seed{SEEDS[0]}" / "metrics.csv",
        [3.0, 1.3, 1.2],
        resolved_steps=30000,
        complete=False,
    )
    with (root / f"7m-plain-seed{SEEDS[0]}" / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert int(rows[-1]["step"]) != 30000

    result = summarize(root, baseline_root)
    assert result["rungs"]["7m"]["complete"] is False
    assert result["verdict"] == "incomplete"


def test_child_argv_round_trips_through_the_parser():
    parser = build_parser()
    parent_args = parser.parse_args(
        [
            "--root",
            "myroot",
            "--gpus",
            "0,1",
            "--dataset",
            "ds.bin",
            "--baseline-root",
            "baseroot",
            "--seeds",
            "1337,1338,1339",
        ]
    )
    argv = child_argv(parent_args, gpu=1)
    child_args = parser.parse_args(argv[3:])
    assert child_args.root == parent_args.root
    assert child_args.gpus == parent_args.gpus
    assert child_args.dataset == parent_args.dataset
    assert child_args.baseline_root == parent_args.baseline_root
    assert child_args.seeds == parent_args.seeds
    assert child_args.queue_gpu == 1


def test_rungs_and_arms_constants():
    assert list(RUNGS.keys()) == ["99m", "7m"]
    assert [arm.name for arm in ARMS] == ["plain", "qat", "fp32"]
    fp32 = next(arm for arm in ARMS if arm.name == "fp32")
    assert fp32.weight_mode == "fp32"
