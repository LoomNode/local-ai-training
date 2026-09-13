import csv
from pathlib import Path

from scripts import momentum_30k
from scripts.momentum_30k import assign_queues, child_argv, confirmation_arms, summarize


def test_confirmation_arms_are_four_per_seed_and_fp32_has_no_codes_knobs():
    pairs = confirmation_arms(16, 0.99, (1337, 1338, 1339))
    assert len(pairs) == 12
    names = sorted({arm.name for arm, _ in pairs})
    assert names == ["fp32", "momentum", "plain", "qat"]
    momentum = next(arm for arm, _ in pairs if arm.name == "momentum")
    assert (momentum.leak, momentum.beta, momentum.weight_mode) == (16, 0.99, "ratchet")
    fp32 = next(arm for arm, _ in pairs if arm.name == "fp32")
    assert fp32.weight_mode == "fp32" and fp32.leak == 0 and fp32.beta == 0.0


def test_assign_queues_gives_every_card_every_arm_type():
    pairs = confirmation_arms(16, 0.99, (1337, 1338, 1339))
    queues = assign_queues(pairs, (0, 1))
    assert sorted(len(q) for q in queues.values()) == [6, 6]
    for queue in queues.values():
        assert {arm.name for arm, _ in queue} == {"fp32", "momentum", "plain", "qat"}


def _write(path: Path, losses):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "validation_loss"])
        writer.writeheader()
        for index, loss in enumerate(losses):
            writer.writerow({"step": index * 200, "validation_loss": loss})


def _write_resolved(path: Path, losses, resolved_steps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "validation_loss", "resolved_steps"])
        writer.writeheader()
        for index, loss in enumerate(losses):
            writer.writerow(
                {"step": index * 200, "validation_loss": loss, "resolved_steps": resolved_steps}
            )


def test_summarize_reports_gaps_and_verdict(tmp_path):
    # Every arm's trailing four evaluations are deliberately unequal (not a flat
    # plateau) so this test can only pass under a genuine 4-wide last_four_mean
    # window -- a 3-wide (or 1-wide) window would read a different tail and
    # produce a different mean. `best` is the minimum over the whole series,
    # chosen so the gap arithmetic below stays simple:
    #   momentum best=1.03, qat best=1.01, plain best=1.15, fp32 best=0.94
    #   momentum_minus_qat   = 1.03 - 1.01 = 0.02
    #   momentum_minus_plain = 1.03 - 1.15 = -0.12
    #   fraction_recovered   = (1.15 - 1.03) / (1.15 - 1.01) = 0.12 / 0.14
    # momentum's last four evaluations are [1.06, 1.03, 1.04, 1.03], mean 1.04
    # (this is not its best/minimum, which is 1.03 -- best and last_four_mean
    # are deliberately different values here). None of these fixtures carry a
    # resolved_steps column, so completeness is "unknown" for every arm and
    # every seed is treated as complete (unknown never excludes a seed).
    for seed in (1337, 1338, 1339):
        _write(tmp_path / f"fp32-seed{seed}" / "metrics.csv", [3.0, 1.00, 0.96, 0.94, 0.95, 0.94])
        _write(tmp_path / f"qat-seed{seed}" / "metrics.csv", [3.0, 1.12, 1.04, 1.01, 1.02, 1.01])
        _write(tmp_path / f"plain-seed{seed}" / "metrics.csv", [3.0, 1.28, 1.19, 1.15, 1.17, 1.15])
        _write(
            tmp_path / f"momentum-seed{seed}" / "metrics.csv",
            [3.0, 1.15, 1.06, 1.03, 1.04, 1.03],
        )
    result = summarize(tmp_path)
    gaps = result["gaps"]
    assert abs(gaps["momentum_minus_qat"]["mean"] - 0.02) < 1e-9
    assert abs(gaps["momentum_minus_plain"]["mean"] + 0.12) < 1e-9
    assert abs(gaps["fraction_recovered"]["mean"] - (0.12 / 0.14)) < 1e-9
    assert result["verdict"] == "flipped"
    assert result["per_seed"]["1337"]["momentum"]["last_four_mean"] == 1.04
    assert result["per_seed"]["1337"]["momentum"]["complete"] is True
    assert "excluded_seeds" not in result


def test_summarize_excludes_a_seed_whose_run_stopped_short(tmp_path):
    full_losses = [3.0, 1.00, 0.96, 0.94, 0.95, 0.94]  # last step = 5 * 200 = 1000
    for seed in (1337, 1338):
        for name in ("fp32", "qat", "plain", "momentum"):
            _write_resolved(
                tmp_path / f"{name}-seed{seed}" / "metrics.csv", full_losses, resolved_steps=1000
            )
    # seed 1339: every arm completes except momentum, which stops short at step 600
    for name in ("fp32", "qat", "plain"):
        _write_resolved(
            tmp_path / f"{name}-seed1339" / "metrics.csv", full_losses, resolved_steps=1000
        )
    _write_resolved(
        tmp_path / "momentum-seed1339" / "metrics.csv", full_losses[:4], resolved_steps=1000
    )

    result = summarize(tmp_path)
    assert result["seeds"] == ["1337", "1338"]
    assert result["excluded_seeds"] == ["1339"]
    assert result["verdict"] is None
    assert result["per_seed"]["1339"]["momentum"]["complete"] is False
    assert result["per_seed"]["1339"]["plain"]["complete"] is True
    assert result["per_seed"]["1337"]["momentum"]["complete"] is True
    # gaps are computed over the remaining (complete) seeds only
    assert abs(result["gaps"]["momentum_minus_qat"]["mean"] - 0.0) < 1e-9


def test_fraction_recovered_guards_zero_denominator(tmp_path):
    def write(name, seed, val):
        _write(tmp_path / f"{name}-seed{seed}" / "metrics.csv", [val, val])

    # seed 1337: qat best == plain best -> zero denominator, guarded out of the stats
    write("fp32", 1337, 0.5)
    write("qat", 1337, 1.0)
    write("plain", 1337, 1.0)
    write("momentum", 1337, 0.95)

    # seed 1338: a normal gap
    write("fp32", 1338, 0.5)
    write("qat", 1338, 0.8)
    write("plain", 1338, 1.0)
    write("momentum", 1338, 0.9)

    result = summarize(tmp_path)
    fraction_recovered = result["gaps"]["fraction_recovered"]
    assert fraction_recovered is not None
    # only seed 1338 contributes: (1.0 - 0.9) / (1.0 - 0.8) = 0.5
    assert abs(fraction_recovered["mean"] - 0.5) < 1e-9
    assert fraction_recovered["sd"] == 0.0


def test_child_argv_round_trips_through_the_parser():
    parser = momentum_30k.build_parser()
    parent_args = parser.parse_args(
        [
            "--leak", "16",
            "--beta", "0.99",
            "--gpus", "0,1",
            "--config", "cfg.toml",
            "--dataset", "ds.bin",
            "--root", "myroot",
        ]
    )
    argv = child_argv(parent_args, gpu=1)
    child_args = parser.parse_args(argv[3:])
    assert child_args.config == parent_args.config
    assert child_args.dataset == parent_args.dataset
    assert child_args.leak == parent_args.leak
    assert child_args.beta == parent_args.beta
    assert child_args.gpus == parent_args.gpus
    assert child_args.root == parent_args.root
    assert child_args.queue_gpu == 1
