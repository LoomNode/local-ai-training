"""The lab board's pure parts (tools/board/server.py): the unit-log parser, the pace and ETA, the queue's claims, the leaderboard join."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("board_server", Path(__file__).resolve().parents[1] / "tools" / "board" / "server.py")
board = importlib.util.module_from_spec(spec)
spec.loader.exec_module(board)

LOG = """lock tests gpu1 start 2026-10-09 23:06:41 at 6308b29

=== l4dir_s1_r 23:06:41: 24939 steps = twice the base's calibrated cost
code train=200000000 val=2000000 unigram=5.452 bigram=4.016
fixed transformer {'step': 2000, 'train_seconds': 282.2, 'bpb': 0.9364, 'bpb_warm': 0.8677}
fixed transformer {'step': 4000, 'train_seconds': 569.4, 'bpb': 0.8531, 'bpb_warm': 0.7838}
transformer yardstick train_s=3629.0 eval_s=17.7 peak_vram_gib=1.43 joules=545971 contended=False

=== calibration session 9 00:10:00
=== cal9_base 00:10:00  load 2.70  gpu0 36 %
fixed transformer {'step': 100, 'train_seconds': 22.0, 'bpb': 2.0, 'bpb_warm': 1.9}
=== l6x4_mt_r 00:12:00: 70,688 steps
fixed transformer {'step': 2000, 'train_seconds': 400.0, 'bpb': 1.0, 'bpb_warm': 0.9}
Traceback (most recent call last):
"""


def test_the_log_parser_reads_each_run_with_its_budget_points_and_end_and_does_not_take_a_calibrations_points():
    runs = board.parse_log(LOG)
    assert [r["name"] for r in runs] == ["l4dir_s1_r", "l6x4_mt_r"]
    a, b = runs
    assert a["budget"] == 24939 and a["started"] == "23:06:41" and a["finished"] and a["train_s"] == 3629.0
    assert [p["step"] for p in a["points"]] == [2000, 4000] and a["points"][1]["bpb"] == 0.8531
    assert b["budget"] == 70688 and not b["finished"] and [p["step"] for p in b["points"]] == [2000]
    assert b["errors"] and b["errors"][0].startswith("Traceback")


def test_the_pace_is_the_last_two_points_and_the_eta_is_the_steps_left_at_that_pace_less_the_logs_age():
    points = [{"step": 2000, "train_seconds": 282.2, "bpb": 1, "bpb_warm": 1}, {"step": 4000, "train_seconds": 569.4, "bpb": 1, "bpb_warm": 1}]
    pace, eta = board.pace_eta(points, 24939, age_s=60.0)
    assert abs(pace - (569.4 - 282.2) / 2000) < 1e-9
    assert abs(eta - ((24939 - 4000) * pace - 60.0)) < 1e-6
    pace1, eta1 = board.pace_eta(points[:1], 24939, 0.0)
    assert abs(pace1 - 282.2 / 2000) < 1e-9 and eta1 > 0
    assert board.pace_eta([], 10, 0.0) == (None, None)
    assert board.pace_eta(points, 3000, 1e9)[1] == 0.0


def test_a_running_arm_is_a_claim_held_by_an_active_unit_whose_run_json_does_not_exist(tmp_path):
    (tmp_path / "claim_a_r").write_text("tables-x.service\n")
    (tmp_path / "claim_b_r").write_text("tables-x.service\n")
    (tmp_path / "claim_c_r").write_text("tables-old.service\n")
    (tmp_path / "b_r.json").write_text("{}")
    got = board.running_arms(["tables-x.service"], board.claims(tmp_path), tmp_path)
    assert got == [{"name": "a_r", "unit": "tables-x.service"}]


def test_the_leaderboard_joins_each_score_to_its_run_and_sorts_by_cold(tmp_path):
    tables = tmp_path / "tables"; tables.mkdir()
    (tables / "k_two.json").write_text(json.dumps({"name": "k_two", "ckpt": "results/ckpt/two_r_transformer.pt", "arm": "transformer", "bpb": 0.7, "bpb_warm": 0.6}))
    (tables / "k_one.json").write_text(json.dumps({"name": "k_one", "ckpt": "results/ckpt/one_r_transformer.pt", "arm": "transformer", "bpb": 0.65, "bpb_warm": 0.58}))
    (tmp_path / "one_r.json").write_text(json.dumps({"n_layers": 4, "seed": 0, "mem_orders": "8,16", "mem_feat": "/m/feat_x",
                                                     "arms": [{"trainable_scalars": 123, "bytes_seen": 10, "train_seconds": 5.0,
                                                               "curve_fixed": [{"step": 2000, "train_seconds": 1.0, "bpb": 1.0, "bpb_warm": 0.9}]}]}))
    rows = board.leaderboard(tables, [tmp_path])
    assert [r["run"] for r in rows] == ["one_r", "two_r"]
    one, two = rows
    assert one["cold"] == 0.65 and one["steps"] == 2000 and one["params"] == 123 and one["layers"] == 4 and one["orders"] == 2 and one["mem_feat"] == "feat_x"
    assert two["steps"] is None and two["params"] is None
