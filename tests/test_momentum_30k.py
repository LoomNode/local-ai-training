import csv
from pathlib import Path

from scripts.momentum_30k import assign_queues, confirmation_arms, summarize


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
    # are deliberately different values here).
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
    assert result["per_seed"][1337]["momentum"]["last_four_mean"] == 1.04
