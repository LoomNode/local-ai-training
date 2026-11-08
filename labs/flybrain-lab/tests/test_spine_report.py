"""scripts/spine_report.py: the two decision rules and the Run 53 table, on synthetic result files. CPU only."""

from __future__ import annotations

import itertools
import json

import pytest

from scripts.spine_report import ARMS, commands, decide, main, report

CKPT_ARM = {"hybrid": "mingru_sparse_post", "tf": "transformer", "zeroheads": "mingru"}


def rows(h, t, z):
    """(dev cold bpb, train_seconds) for hybrid, tf, zeroheads."""
    return [dict(arm=a, dev_bpb=b, train_seconds=s) for a, (b, s) in zip(ARMS, (h, t, z))]


def test_a_pair_within_5_percent_in_seconds_gets_seed_one_and_only_the_pair_that_is_apart_gets_an_equal_seconds_rerun():
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


# --- arithmetic pins: how a step count is rounded, and which seconds a rerun is scaled to ------------------------------------------------------

def test_a_scaled_step_count_is_rounded_to_the_nearest_whole_step_and_printed_without_a_fraction():
    d = decide(rows((0.70, 8000), (0.75, 7000), (0.90, 9000)), 16000)    # tf is beaten by the slower hybrid: 16000 * 8000 / 7000 = 18285.71, so 18286 (round, not int)
    assert d == {"seed1": [], "eqsec": [{"arm": "tf", "steps": 18286, "target_seconds": 8000}]}   # the .30 and .33 fractions in the tests above round down
    assert isinstance(d["eqsec"][0]["steps"], int)
    assert commands(d) == ["scripts/run_spine.sh cuda:1 tf 0 18286 eqsec"]


def test_the_rerun_target_is_the_slowest_better_arm_whichever_order_the_arms_come_in():
    expected = {"seed1": [], "eqsec": [{"arm": "tf", "steps": 21333, "target_seconds": 8000}, {"arm": "zeroheads", "steps": 25600, "target_seconds": 8000}]}
    for order in itertools.permutations(rows((0.70, 8000), (0.75, 6000), (0.90, 5000))):    # zeroheads is beaten by hybrid (8000 s) and tf (6000 s); either pair can come first
        assert decide(list(order), 16000) == expected, [r["arm"] for r in order]


# --- wiring pins: which column, which runs and which device feed the rules --------------------------------------------------------------------
# _write shifts every arm by the same amount in every column, so a rule that read the wrong column, split or run would still decide the same way.
# These cases give each arm its own numbers in each column and add runs the rules must not read, so a wrong read changes the printed commands.

SECONDS = (27000.0, 10000.0, 13500.0)   # train_seconds of hybrid, tf, zeroheads in the ratio of the first spine runs (2.7 x and 2 x): only the bits/byte can decide
DECOY = (0.10, 99999.0)                 # (dev cold bpb, train_seconds) of a run the rules must never read: the best and by far the slowest
COLUMNS = {                             # cold bpb of hybrid, tf, zeroheads, and what the rules would call for if they read that column instead
    "dev":      (0.70, 0.75, 0.90),     # the column they read: tf and zeroheads are beaten by the slower hybrid, so both are rerun at its seconds
    "dev warm": (0.60, 0.70, 0.695),    # the same two reruns, and seed 1 for tf and zeroheads (0.005 apart)
    "test":     (0.90, 0.75, 0.70),     # tf alone, rerun at zeroheads' seconds
    "legacy":   (0.80, 0.805, 0.60),    # seed 1 for hybrid and tf (0.005 apart), and tf rerun at the hybrid's seconds
    "dev128":   (0.75, 0.70, 0.80),     # zeroheads alone
}
DEV_COLD = ["scripts/run_spine.sh cuda:1 tf 0 43200 eqsec", "scripts/run_spine.sh cuda:1 zeroheads 0 32000 eqsec"]    # 16000 * 27000 / 10000, 16000 * 27000 / 13500
MIXED = {"hybrid": ("spine16_hybrid_s0", 0.70, 7000.0), "tf": ("spine16_tf_s0", 0.75, 7500.0), "zeroheads": ("spine16_zeroheads_s0", 0.90, 6900.0)}   # seed 1 for hybrid and zeroheads, and zeroheads rerun at tf's 7500 s


def _scores(results, entries, suffix="s0"):
    """entries[split] = [(run name, arm, cold, warm), ...]: spine16_score_<split>_<suffix>.json with a length-2048 row for each. The numbers are given outright."""
    for split, items in entries.items():
        rows_ = [{"T": 2048, "bpb": cold, "bpb_warm": warm, "ckpt": f"{name}_{CKPT_ARM[arm]}.pt", "arm": CKPT_ARM[arm]} for name, arm, cold, warm in items]
        (results / f"spine16_score_{split}_{suffix}.json").write_text(json.dumps({"args": {}, "rows": rows_}))


def _write_columns(results):
    """The three seed-0 runs, and in each split the per-arm numbers of COLUMNS (warm is cold - 0.05 except in dev, where it is the 'dev warm' row)."""
    _write(results, {arm: (f"spine16_{arm}_s0", COLUMNS["dev"][i], SECONDS[i]) for i, arm in enumerate(ARMS)}, scores=False)
    cold = {split: COLUMNS[split] for split in ("dev", "test", "legacy", "dev128")}
    warm = {split: tuple(c - 0.05 for c in cold[split]) for split in cold} | {"dev": COLUMNS["dev warm"]}
    _scores(results, {split: [(f"spine16_{arm}_s0", arm, cold[split][i], warm[split][i]) for i, arm in enumerate(ARMS)] for split in cold})


def _write_decoys(results, arms=ARMS):
    """Runs the rules must never read: another seed (s1), a tag that sorts before s0 (eqsec) and one that sorts after it (tuned), so a dict that keeps
    the first or the last run of an arm cannot hide a leak. DECOY in every column. Returns their names."""
    decoys = [(f"spine16_{arm}_{part}", arm) for part in ("s1", "eqsec_s0", "tuned_s0") for arm in arms]
    for name, arm in decoys:
        _write(results, {arm: (name, *DECOY)}, scores=False)
    _scores(results, {split: [(name, arm, DECOY[0], DECOY[0] - 0.05) for name, arm in decoys] for split in ("dev", "test", "legacy", "dev128")}, suffix="followups")
    return [name for name, _ in decoys]


def _follow_ups(out):
    """What the report prints under its follow-up header: the commands, or the one none: line."""
    lines = out.splitlines()
    return lines[lines.index("follow-up runs the two rules call for (one unit, in this order):") + 1:]


def _table_runs(out):
    """The run names in the report's table, in order."""
    return [row.split("|")[1].strip(" `") for row in out.split("\n\n")[0].splitlines()[2:]]


def test_the_rules_read_the_dev_cold_column_and_no_other(tmp_path):
    _write_columns(tmp_path)
    assert _follow_ups(report(tmp_path)) == DEV_COLD


def test_each_other_column_and_each_decoy_would_have_changed_the_decision():
    """The wiring cases have teeth: reading any other column, or letting a decoy replace any one seed-0 arm, calls for different runs."""
    by_column = {name: commands(decide(rows(*zip(cold, SECONDS)), 16000)) for name, cold in COLUMNS.items()}
    assert by_column["dev"] == DEV_COLD
    assert len({tuple(cmds) for cmds in by_column.values()}) == len(COLUMNS)       # no two columns call for the same runs
    for i, arm in enumerate(ARMS):
        with_decoy = list(zip(COLUMNS["dev"], SECONDS))
        with_decoy[i] = DECOY
        assert commands(decide(rows(*with_decoy), 16000)) != DEV_COLD, arm


def test_other_seeds_and_tagged_runs_never_enter_the_rules_but_still_fill_the_table(tmp_path):
    _write(tmp_path, {arm: (f"spine16_{arm}_s0", COLUMNS["dev"][i], SECONDS[i]) for i, arm in enumerate(ARMS)})
    decoys = _write_decoys(tmp_path)
    out = report(tmp_path)
    assert _follow_ups(out) == DEV_COLD
    assert set(decoys) <= set(_table_runs(out))
    assert all(f"| `{name}` | 16.00M | 0.100 / 0.050 |" in out for name in decoys)      # and their scores were matched to them


def test_a_follow_up_run_never_stands_in_for_a_missing_seed_zero_run(tmp_path):
    _write(tmp_path, {arm: (f"spine16_{arm}_s0", COLUMNS["dev"][i], SECONDS[i]) for i, arm in enumerate(ARMS) if arm != "zeroheads"})
    decoys = _write_decoys(tmp_path, arms=("zeroheads",))
    out = report(tmp_path)
    assert "incomplete: need the three seed-0 runs and their dev scores" in out and "scripts/run_spine.sh" not in out
    assert set(decoys) <= set(_table_runs(out))


def test_the_device_names_the_card_in_every_follow_up_command(tmp_path):
    _write(tmp_path, MIXED)
    out = report(tmp_path, "cuda:0")
    assert _follow_ups(out) == ["scripts/run_spine.sh cuda:0 hybrid 1", "scripts/run_spine.sh cuda:0 zeroheads 1", "scripts/run_spine.sh cuda:0 zeroheads 0 17391 eqsec"]
    assert "cuda:1" not in out


def test_the_command_line_hands_its_results_directory_and_device_to_the_report(tmp_path, monkeypatch, capsys):
    _write(tmp_path, MIXED)
    monkeypatch.setattr("sys.argv", ["spine_report.py", "--results", str(tmp_path), "--device", "cuda:0"])
    main()
    out = capsys.readouterr().out
    assert out == report(tmp_path, "cuda:0") + "\n"
    assert "scripts/run_spine.sh cuda:0 hybrid 1" in out and "cuda:1" not in out


def test_report_says_none_when_no_follow_up_run_is_called_for(tmp_path):
    _write(tmp_path, {"hybrid": ("spine16_hybrid_s0", 0.70, 6000.0), "tf": ("spine16_tf_s0", 0.75, 7000.0), "zeroheads": ("spine16_zeroheads_s0", 0.90, 8000.0)})
    out = report(tmp_path)
    assert _follow_ups(out) == ["none: every pair is apart by 0.01 bits/byte and 5 % in seconds, and the better arm is not the slower"]
    assert "scripts/run_spine.sh" not in out


# --- the joules note, and the equal-seconds reruns: a gap line against the run each was matched to and one against its own arm's seed-0 run ----------

NOTE = "note: joules cover the whole arm, evals and saves included; train_seconds excludes them"
TODAYS_MIXED_REPORT = [                 # report(MIXED) as it was printed before the note and the rerun lines existed
    "| run | params | dev cold / warm | test cold / warm | legacy bytes cold | train_seconds | peak VRAM GiB | joules | contended |",
    "|---|---:|---|---|---:|---:|---:|---:|---|",
    "| `spine16_hybrid_s0` | 16.00M | 0.700 / 0.650 | 0.720 / 0.670 | 0.670 | 7,000 | 9.0 | 1,000,000 | no |",
    "| `spine16_tf_s0` | 16.00M | 0.750 / 0.700 | 0.770 / 0.720 | 0.720 | 7,500 | 9.0 | 1,000,000 | no |",
    "| `spine16_zeroheads_s0` | 16.00M | 0.900 / 0.850 | 0.920 / 0.870 | 0.870 | 6,900 | 9.0 | 1,000,000 | no |",
    "",
    "consistency spine16_hybrid_s0: in-run bf16 0.7010 vs fp32 re-score 0.7000, delta +0.0010",
    "consistency spine16_tf_s0: in-run bf16 0.7510 vs fp32 re-score 0.7500, delta +0.0010",
    "consistency spine16_zeroheads_s0: in-run bf16 0.9010 vs fp32 re-score 0.9000, delta +0.0010",
    "runner val_bpb (128 random windows, the number Runs 37-49 report): hybrid 0.7600, tf 0.7600, zeroheads 0.7600",
    "hybrid - tf: cold bpb dev -0.050, test -0.050, legacy -0.050; train_seconds 7,000 vs 7,500 (-6.7%)",
    "hybrid - zeroheads: cold bpb dev -0.200, test -0.200, legacy -0.200; train_seconds 7,000 vs 6,900 (+1.4%)",
    "tf - zeroheads: cold bpb dev -0.150, test -0.150, legacy -0.150; train_seconds 7,500 vs 6,900 (+8.7%)",
    "",
    "follow-up runs the two rules call for (one unit, in this order):",
    "scripts/run_spine.sh cuda:1 hybrid 1",
    "scripts/run_spine.sh cuda:1 zeroheads 1",
    "scripts/run_spine.sh cuda:1 zeroheads 0 17391 eqsec",
]
RERUN_COLD = {"dev": 0.88, "test": 0.905, "legacy": 0.8575}      # a zeroheads rerun with its own number in each split, so a gap read from the wrong split is wrong


def _write_eqsec(results, arm, seconds, cold=None):
    """The run spine16_<arm>_eqsec_s0 with this train_seconds; cold = {split: cold bpb} also writes its score rows (none when None). Returns its name.
    Its warm number is 0.08 below its cold one, where every other run's is 0.05 below, so a warm gap cannot pass for a cold gap."""
    name = f"spine16_{arm}_eqsec_s0"
    _write(results, {arm: (name, 0.5, seconds)}, scores=False)
    if cold:
        _scores(results, {split: [(name, arm, bpb, bpb - 0.08)] for split, bpb in cold.items()}, suffix=f"eqsec_{arm}")
    return name


def test_a_note_under_the_table_says_the_joules_cover_the_whole_arm(tmp_path):
    _write(tmp_path, MIXED)
    out = report(tmp_path)
    table, rest = out.split("\n\n", 1)
    assert table.splitlines()[0].startswith("| run |") and rest.splitlines()[0] == NOTE and out.count(NOTE) == 1


def test_with_no_eqsec_run_the_report_is_what_it_was_plus_the_note(tmp_path):
    _write(tmp_path, MIXED)
    assert report(tmp_path).split("\n") == TODAYS_MIXED_REPORT[:6] + [NOTE] + TODAYS_MIXED_REPORT[6:]


def test_each_eqsec_run_gets_a_gap_line_against_the_run_it_was_matched_to_and_one_against_its_own_arms_seed_zero_run(tmp_path):
    _write(tmp_path, MIXED)                                                  # hybrid 7,000 s, tf 7,500 s, zeroheads 6,900 s; dev 0.70, 0.75, 0.90; test +0.02, legacy -0.03
    _write_eqsec(tmp_path, "zeroheads", 7480.0, RERUN_COLD)
    lines = report(tmp_path).splitlines()
    last_pair = lines.index("tf - zeroheads: cold bpb dev -0.150, test -0.150, legacy -0.150; train_seconds 7,500 vs 6,900 (+8.7%)")
    assert lines[last_pair + 1:last_pair + 5] == [
        "spine16_zeroheads_eqsec_s0 - spine16_tf_s0 (matched on train_seconds): cold bpb dev +0.1300, test +0.1350, legacy +0.1375; train_seconds 7,480 vs 7,500 (ratio 99.7%)",
        "spine16_zeroheads_eqsec_s0 - spine16_zeroheads_s0 (own arm, seed 0): cold bpb dev -0.0200, test -0.0150, legacy -0.0125; train_seconds 7,480 vs 6,900 (ratio 108.4%)",
        "",
        "follow-up runs the two rules call for (one unit, in this order):"]   # right after the pair lines, before the follow-up commands
    assert lines[last_pair + 5:] == ["scripts/run_spine.sh cuda:1 hybrid 1", "scripts/run_spine.sh cuda:1 zeroheads 1", "scripts/run_spine.sh cuda:1 zeroheads 0 17391 eqsec"]


@pytest.mark.parametrize("arm, seconds, matched", [
    ("zeroheads", 7480.0, "spine16_tf_s0"),          # the nearer of the other two arms' seconds (hybrid 7,000, tf 7,500)
    ("zeroheads", 7100.0, "spine16_hybrid_s0"),      # the nearest, not the slowest
    ("zeroheads", 6930.0, "spine16_hybrid_s0"),      # zeroheads' own seed-0 run (6,900) is nearer still, but the match is another arm's
    ("hybrid", 7400.0, "spine16_tf_s0"),
    ("tf", 6000.0, "spine16_zeroheads_s0"),          # the nearest is the fastest of the others (6,900; hybrid 7,000); tf's own is 7,500
])
def test_an_eqsec_run_is_matched_to_the_other_arms_seed_zero_run_nearest_in_train_seconds(tmp_path, arm, seconds, matched):
    _write(tmp_path, MIXED)
    name = _write_eqsec(tmp_path, arm, seconds, {"dev": 0.8, "test": 0.8, "legacy": 0.8})
    mine = [line for line in report(tmp_path).splitlines() if line.startswith(name + " - ")]
    assert len(mine) == 2
    assert mine[0].startswith(f"{name} - {matched} (matched on train_seconds): ") and mine[1].startswith(f"{name} - spine16_{arm}_s0 (own arm, seed 0): ")


def test_only_eqsec_runs_get_gap_lines_in_arm_order_and_a_missing_score_prints_n_a(tmp_path):
    _write(tmp_path, MIXED)
    _write(tmp_path, {"hybrid": ("spine16_hybrid_s1", 0.701, 7010.0)}, scores=False)              # another seed and another tag: table rows, no gap lines
    _write(tmp_path, {"hybrid": ("spine16_hybrid_tuned_s0", 0.702, 7020.0)}, scores=False)
    _write_eqsec(tmp_path, "zeroheads", 7480.0, {"dev": 0.88})                                     # scored in dev only
    _write_eqsec(tmp_path, "tf", 7000.0)                                                           # not scored at all yet
    mine = [line for line in report(tmp_path).splitlines() if line.startswith("spine16_") and " - spine16_" in line]
    assert [line.split(" (")[0] for line in mine] == ["spine16_tf_eqsec_s0 - spine16_hybrid_s0", "spine16_tf_eqsec_s0 - spine16_tf_s0",
                                                      "spine16_zeroheads_eqsec_s0 - spine16_tf_s0", "spine16_zeroheads_eqsec_s0 - spine16_zeroheads_s0"]
    assert [line.split("): ")[1] for line in mine] == [
        "cold bpb dev n/a, test n/a, legacy n/a; train_seconds 7,000 vs 7,000 (ratio 100.0%)",
        "cold bpb dev n/a, test n/a, legacy n/a; train_seconds 7,000 vs 7,500 (ratio 93.3%)",
        "cold bpb dev +0.1300, test n/a, legacy n/a; train_seconds 7,480 vs 7,500 (ratio 99.7%)",
        "cold bpb dev -0.0200, test n/a, legacy n/a; train_seconds 7,480 vs 6,900 (ratio 108.4%)"]
