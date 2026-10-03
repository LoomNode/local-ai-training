"""scripts/spine_report.py: the two decision rules and the Run 53 table, on synthetic result files. CPU only."""

from __future__ import annotations

import json

import pytest

from scripts.spine_report import ARMS, commands, decide, report

CKPT_ARM = {"hybrid": "mingru_sparse_post", "tf": "transformer", "zeroheads": "mingru"}


def rows(h, t, z):
    """(dev cold bpb, train_seconds) for hybrid, tf, zeroheads."""
    return [dict(arm=a, dev_bpb=b, train_seconds=s) for a, (b, s) in zip(ARMS, (h, t, z))]


def test_a_pair_within_5_percent_in_seconds_gets_seed_one_and_nothing_else_is_rerun():
    d = decide(rows((0.70, 7000), (0.75, 7500), (0.90, 6900)), 16000)
    assert d["seed1"] == ["hybrid", "zeroheads"]       # 7000 vs 6900 is 1.4 % apart; 7000 vs 7500 is 7.1 %
    assert d["eqsec"] == [{"arm": "zeroheads", "steps": 17391, "target_seconds": 7500}]   # tf beats it, and is 8.7 % slower


def test_nothing_is_rerun_when_the_better_arm_is_also_the_faster_and_every_pair_is_apart():
    assert decide(rows((0.70, 6000), (0.75, 7000), (0.90, 8000)), 16000) == {"seed1": [], "eqsec": []}


def test_a_better_arm_that_is_slower_by_less_than_5_percent_is_a_seed_question_not_an_equal_seconds_one():
    d = decide(rows((0.70, 7100), (0.80, 7000), (0.95, 9000)), 16000)    # hybrid is better and 1.4 % slower than tf
    assert d == {"seed1": ["hybrid", "tf"], "eqsec": []}


def test_the_faster_worse_arms_are_rerun_at_the_slowest_better_arms_seconds():
    d = decide(rows((0.70, 8000), (0.75, 6000), (0.90, 5000)), 16000)
    assert d["seed1"] == []
    assert d["eqsec"] == [{"arm": "tf", "steps": 21333, "target_seconds": 8000}, {"arm": "zeroheads", "steps": 25600, "target_seconds": 8000}]


def test_the_thresholds_are_strict_at_0_01_bits_and_5_percent():
    assert decide(rows((0.70, 6000), (0.7099, 7200), (0.9, 9000)), 16000)["seed1"] == ["hybrid", "tf"]      # 0.0099 apart
    assert decide(rows((0.70, 6000), (0.7101, 7200), (0.9, 9000)), 16000)["seed1"] == []                    # 0.0101 apart
    assert decide(rows((0.70, 6000), (0.80, 6294), (0.9, 9000)), 16000)["seed1"] == ["hybrid", "tf"]        # 4.9 % apart
    assert decide(rows((0.70, 6000), (0.80, 6306), (0.9, 9000)), 16000)["seed1"] == []                      # 5.1 % apart


def test_one_pair_can_trigger_both_rules():
    d = decide(rows((0.700, 7800), (0.705, 6000), (0.95, 9000)), 16000)    # 0.005 apart, and the better arm is 30 % slower
    assert d["seed1"] == ["hybrid", "tf"] and d["eqsec"] == [{"arm": "tf", "steps": 20800, "target_seconds": 7800}]


def test_commands_name_the_runs_the_decision_asks_for():
    d = {"seed1": ["hybrid", "tf"], "eqsec": [{"arm": "zeroheads", "steps": 17391, "target_seconds": 7500}]}
    assert commands(d) == ["scripts/run_spine.sh cuda:1 hybrid 1", "scripts/run_spine.sh cuda:1 tf 1",
                           "scripts/run_spine.sh cuda:1 zeroheads 0 17391 eqsec"]
    assert commands({"seed1": [], "eqsec": []}, "cuda:0") == []


def _write(results, spec, scores=True, d128_shift=0.0):
    """spec[arm] = (name, dev, seconds); writes the run JSONs and the four score JSONs the report reads."""
    for arm, (name, dev, seconds) in spec.items():
        (results / f"{name}.json").write_text(json.dumps({"run_name": name, "steps": 16000, "arms": [{
            "arm": CKPT_ARM[arm], "trainable_scalars": 16_000_000, "train_seconds": seconds, "eval_seconds": 60.0, "peak_vram_bytes": 9 * 2**30,
            "joules": 1.0e6, "mean_power_w": 148.0, "contended": False, "telemetry_lost": False, "val_bpb": 0.76,
            "val_fixed": {"offset": 202117357, "n_windows": 128, "bpb": dev + 0.001, "bpb_warm": dev - 0.05}}]}))
    if not scores:
        return
    for split, shift in (("dev", 0.0), ("test", 0.02), ("legacy", -0.03), ("dev128", d128_shift)):
        rows_ = [{"T": 2048, "bpb": spec[a][1] + shift, "bpb_warm": spec[a][1] + shift - 0.05, "ckpt": f"{spec[a][0]}_{CKPT_ARM[a]}.pt", "arm": CKPT_ARM[a]} for a in spec]
        rows_.append({"T": 512, "bpb": 9.9, "bpb_warm": 9.9, "ckpt": f"{spec['tf'][0]}_transformer.pt", "arm": "transformer"})   # other lengths are ignored
        (results / f"spine16_score_{split}_s0.json").write_text(json.dumps({"args": {}, "rows": rows_}))


def test_report_prints_the_table_the_gaps_and_the_follow_up_commands(tmp_path):
    _write(tmp_path, {"hybrid": ("spine16_hybrid_s0", 0.70, 7000.0), "tf": ("spine16_tf_s0", 0.75, 7500.0), "zeroheads": ("spine16_zeroheads_s0", 0.90, 6900.0)})
    out = report(tmp_path)
    assert "| run | params | dev cold / warm | test cold / warm | legacy bytes cold | train_seconds | peak VRAM GiB | joules | contended |" in out
    assert "| `spine16_hybrid_s0` | 16.00M | 0.700 / 0.650 | 0.720 / 0.670 | 0.670 | 7,000 | 9.0 | 1,000,000 | no |" in out
    assert out.index("spine16_hybrid_s0") < out.index("spine16_tf_s0") < out.index("spine16_zeroheads_s0")
    assert "hybrid - tf: cold bpb dev -0.050, test -0.050, legacy -0.050; train_seconds 7,000 vs 7,500 (-6.7%)" in out
    assert "scripts/run_spine.sh cuda:1 hybrid 1" in out and "scripts/run_spine.sh cuda:1 zeroheads 0 17391 eqsec" in out
    assert "scripts/run_spine.sh cuda:1 tf 1" not in out
    assert "consistency spine16_hybrid_s0: in-run bf16 0.7010 vs fp32 re-score 0.7000, delta +0.0010" in out and "WARNING" not in out
    assert "runner val_bpb (128 random windows, the number Runs 37-49 report): hybrid 0.7600, tf 0.7600, zeroheads 0.7600" in out


def test_report_flags_an_inrun_score_that_disagrees_with_the_fp32_rescore(tmp_path):
    _write(tmp_path, {"hybrid": ("spine16_hybrid_s0", 0.70, 7000.0), "tf": ("spine16_tf_s0", 0.75, 7500.0), "zeroheads": ("spine16_zeroheads_s0", 0.90, 6900.0)}, d128_shift=0.02)
    assert "delta -0.0190  WARNING: above 0.01" in report(tmp_path)


def test_report_warns_about_a_contended_run_and_says_what_is_missing(tmp_path):
    _write(tmp_path, {"hybrid": ("spine16_hybrid_s0", 0.70, 7000.0), "tf": ("spine16_tf_s0", 0.75, 7500.0)}, scores=False)
    doc = json.loads((tmp_path / "spine16_tf_s0.json").read_text())
    doc["arms"][0].update(contended=True, telemetry_lost=True)
    (tmp_path / "spine16_tf_s0.json").write_text(json.dumps(doc))
    out = report(tmp_path)
    assert "WARNING spine16_tf_s0: contended=True telemetry_lost=True; its train_seconds is not clean" in out
    assert "| yes, telemetry lost |" in out and "n/a" in out
    assert "incomplete: need the three seed-0 runs and their dev scores" in out


def test_follow_up_runs_join_the_table_but_not_the_rules(tmp_path):
    spec = {"hybrid": ("spine16_hybrid_s0", 0.70, 7000.0), "tf": ("spine16_tf_s0", 0.75, 7500.0), "zeroheads": ("spine16_zeroheads_s0", 0.90, 6900.0)}
    _write(tmp_path, spec)
    _write(tmp_path, {"hybrid": ("spine16_hybrid_s1", 0.701, 7010.0)}, scores=False)
    _write(tmp_path, {"zeroheads": ("spine16_zeroheads_eqsec_s0", 0.88, 7500.0)}, scores=False)
    out = report(tmp_path)
    assert out.index("spine16_hybrid_s0") < out.index("spine16_zeroheads_s0") < out.index("spine16_zeroheads_eqsec_s0") < out.index("spine16_hybrid_s1")
    assert "| `spine16_hybrid_s1` | 16.00M | n/a | n/a | n/a | 7,010 |" in out
