"""The lab board's pure parts (tools/board/server.py): the unit-log parser, the pace and ETA, the queue's claims, the leaderboard join, and the
big train (open-ended runs: logs in tables/big/ with several headers, a run adopted from the dry run's log, the scorer's summary.json)."""
import importlib.util
import json
import re
import time
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


# --- the big train: runs with no end step -------------------------------------------------------------------------------------------------
def fixed(step, secs, bpb, warm):
    return f"fixed transformer {{'step': {step}, 'train_seconds': {secs}, 'bpb': {bpb}, 'bpb_warm': {warm}}}"


def header(name, at, layers, how, date="2026-10-11"):
    """The line bigtrain.sh writes; how is 'fresh' or '--resume results/ckpt/<name>_transformer.step<N>.pt'."""
    return f"=== {name} {at}: 10000000 steps ({layers} layers, cooldown 1, {how}, code a5e9e6b, {date})"


CODE = "code train=5283527978 val=2000000 unigram=5.099 bigram=3.997"
# the dry run: 100-step checkpoints, killed after 6,100 steps; its checkpoint at step 6100 became big12's start
DRY12 = "\n".join([
    "", header("dry12", "11:16:57", 12, "fresh"), CODE, "saved dry12_transformer.step100.pt",
    fixed(2000, 1560.1, 1.2289, 1.1396), "saved dry12_transformer.step2000.pt",
    fixed(4000, 3096.5, 0.9507, 0.8583), "saved dry12_transformer.step4000.pt",
    fixed(6000, 4272.7, 0.8975, 0.8038), "saved dry12_transformer.step6000.pt", "saved dry12_transformer.step6100.pt",
    "Terminated", "=== dry12 exited 2026-10-11 14:39:01 status 143", ""])
RESUMED12 = "resumed big12_transformer at step 6100 of 10000000 from results/ckpt/big12_transformer.step6100.pt (train_seconds so far 4300.0)"
BIG12_START = "\n".join(["", header("big12", "14:41:21", 12, "--resume results/ckpt/big12_transformer.step6100.pt"), CODE, RESUMED12])
BIG12 = BIG12_START + "\n" + "\n".join([fixed(8000, 5150.0, 0.8613, 0.7703), "saved big12_transformer.step8000.pt",
                                       fixed(10000, 6000.0, 0.8400, 0.7500), "saved big12_transformer.step10000.pt"]) + "\n"


def big_dirs(tmp_path, big12=BIG12, dry12=DRY12):
    """A results/tables/big/ with the 12-layer run's logs and a run worktree holding its claim."""
    tables, run = tmp_path / "tables", tmp_path / "run"
    (tables / "big").mkdir(parents=True); run.mkdir()
    (tables / "big" / "big12.log").write_text(big12)
    if dry12 is not None:
        (tables / "big" / "dry12.log").write_text(dry12)
    (run / "claim_big12").write_text("big12-gpu1.service\n")
    return tables, run


def big_of(tables, run, units, now=None):
    return board.big_state(tables, board.claims(run), units, board.parsed_logs(tables), now or time.time())


def test_the_log_parser_reads_the_resume_step_the_last_save_the_layers_and_every_header_of_a_big_log():
    runs = board.parse_log(DRY12 + BIG12)
    assert [r["name"] for r in runs] == ["dry12", "big12"]
    dry, big = runs
    assert dry["budget"] == 10_000_000 and dry["resumed"] == 0 and dry["layers"] == 12 and dry["saved"] == 6100
    assert [p["step"] for p in dry["points"]] == [2000, 4000, 6000]
    assert big["resumed"] == 6100 and big["saved"] == 10000 and [p["step"] for p in big["points"]] == [8000, 10000]
    # the resume step is known from the header alone, before the "resumed" line is printed (a few seconds into the run)
    assert board.parse_log(BIG12_START.rsplit("\n", 1)[0])[0]["resumed"] == 6100


BIG8_FIRST = "\n".join(["", header("big8", "16:00:00", 8, "fresh"), CODE,
                        fixed(2000, 700.0, 1.20, 1.10), "saved big8_transformer.step2000.pt",
                        fixed(4000, 1400.0, 0.95, 0.85), "saved big8_transformer.step4000.pt",
                        fixed(6000, 2100.0, 0.90, 0.80), "Terminated", "=== big8 exited 2026-10-11 17:20:00 status 143", ""])
BIG8_SECOND = "\n".join(["", header("big8", "20:10:00", 8, "--resume results/ckpt/big8_transformer.step4000.pt"), CODE,
                         "resumed big8_transformer at step 4000 of 10000000 from results/ckpt/big8_transformer.step4000.pt (train_seconds so far 1400.0)"])


def test_a_run_restarted_after_a_game_has_one_curve_deduplicated_by_step_with_the_last_point_kept():
    # resumed from the checkpoint at 4000 after being cut at 6000: step 6000 is trained again and its new point replaces the old one
    log = BIG8_FIRST + BIG8_SECOND + "\n" + fixed(6000, 2150.0, 0.91, 0.81) + "\nsaved big8_transformer.step6000.pt\n" + fixed(8000, 2850.0, 0.88, 0.78) + "\n"
    segments = board.parse_log(log)
    assert len(segments) == 2
    curve = board.merged_points(segments)
    assert [p["step"] for p in curve] == [2000, 4000, 6000, 8000]
    assert curve[2]["bpb"] == 0.91 and curve[2]["train_seconds"] == 2150.0
    # the same step logged twice with nothing dropped (resumed at 6000, which evaluates again at once): the later line wins
    again = "\n".join(["", header("big8", "20:10:00", 8, "--resume results/ckpt/big8_transformer.step6000.pt"), CODE,
                       "resumed big8_transformer at step 6000 of 10000000 from results/ckpt/big8_transformer.step6000.pt (train_seconds so far 2100.0)",
                       fixed(6000, 2100.5, 0.9005, 0.8005), fixed(8000, 2850.0, 0.88, 0.78), ""])
    curve = board.merged_points(board.parse_log(BIG8_FIRST + again))
    assert [p["step"] for p in curve] == [2000, 4000, 6000, 8000] and curve[2]["bpb"] == 0.9005


def test_points_beyond_the_step_a_run_resumed_at_are_dropped_until_it_reaches_them_again():
    # killed after the eval at step 6000, resumed from the checkpoint at 4000: the old 6000 is not the run's state any more
    curve = board.merged_points(board.parse_log(BIG8_FIRST + BIG8_SECOND))
    assert [p["step"] for p in curve] == [2000, 4000]
    # a restart from scratch (a "fresh" header) supersedes everything before it
    fresh = board.parse_log(BIG8_FIRST + "\n" + header("big8", "21:00:00", 8, "fresh") + "\n" + fixed(2000, 650.0, 1.19, 1.09) + "\n")
    assert [(p["step"], p["bpb"]) for p in board.merged_points(fresh)] == [(2000, 1.19)]
    # a header whose resume step is not known yet (no --resume path, no "fresh", no "resumed" line) drops nothing
    unknown = board.parse_log(BIG8_FIRST + "\n=== big8 22:00:00: 10000000 steps (8 layers)\n")
    assert [p["step"] for p in board.merged_points(unknown)] == [2000, 4000, 6000]


def test_the_log_search_includes_the_big_subdirectory(tmp_path):
    tables, run = big_dirs(tmp_path)
    (tables / "queue.log").write_text("=== a_r 10:00:00: 100 steps\n")
    logs = board.parsed_logs(tables)
    assert {Path(p).name for p in logs} == {"queue.log", "big12.log", "dry12.log"}
    assert any(p.endswith("big/big12.log") for p in logs)


def test_big12_gets_the_dry_runs_curve_merged_in_and_a_step_pace_and_passes_from_the_log(tmp_path):
    tables, run = big_dirs(tmp_path)
    now = time.time()
    big = big_of(tables, run, ["big12-gpu1.service"], now)
    assert list(big) == ["big12", "big8"]
    b = big["big12"]
    assert [row[0] for row in b["curve"]] == [2000, 4000, 6000, 8000, 10000]  # 2,000-6,000 come from dry12.log
    assert b["curve"][0] == [2000, 1.2289, 1.1396] and b["curve"][-1] == [10000, 0.84, 0.75]
    assert b["step"] == 10000 and b["layers"] == 12 and b["log"] == "big12.log" and b["restarts"] == 0
    assert abs(b["pace"] - (6000.0 - 5150.0) / 2000) < 1e-9 and abs(b["train_s"] - 6000.0) < 1e-9
    assert abs(b["passes"] - 10000 * 8 * 2048 / 5283527978) < 1e-3
    assert b["state"] == "running" and b["running"] and b["unit"] == "big12-gpu1.service"
    assert b["budget"] == 10_000_000 and b["open_ended"] is True
    assert b["log_age_s"] is not None and 0 <= b["log_age_s"] < 60
    json.dumps(big, allow_nan=False)


def test_a_run_that_has_only_just_resumed_shows_the_step_it_resumed_at_and_the_history_pace(tmp_path):
    tables, run = big_dirs(tmp_path, big12=BIG12_START + "\n")
    b = big_of(tables, run, ["big12-gpu1.service"])["big12"]
    assert b["step"] == 6100  # not the last point's 6000: the run has already done 100 steps of its own
    assert [row[0] for row in b["curve"]] == [2000, 4000, 6000]
    assert abs(b["pace"] - (4272.7 - 3096.5) / 2000) < 1e-9


def test_a_budget_of_a_million_steps_or_more_is_open_ended_so_it_has_a_pace_but_no_eta():
    assert board.is_open_ended(10_000_000) and board.is_open_ended(1_000_000)
    assert not board.is_open_ended(999_999) and not board.is_open_ended(24939) and not board.is_open_ended(None)
    points = [{"step": 2000, "train_seconds": 800.0, "bpb": 1, "bpb_warm": 1}, {"step": 4000, "train_seconds": 1600.0, "bpb": 1, "bpb_warm": 1}]
    pace, eta = board.pace_eta(points, 10_000_000, 30.0)
    assert abs(pace - 0.4) < 1e-9 and eta is None
    assert board.pace_eta(points, 24939, 30.0)[1] is not None  # a finite budget still gets its ETA


def test_the_state_marks_an_open_ended_queue_arm_and_carries_the_big_runs(tmp_path, monkeypatch):
    tables, run = big_dirs(tmp_path)
    (tables / "queue.log").write_text("\n".join(["=== qarm 10:00:00: 10000000 steps", fixed(2000, 400.0, 1.0, 0.9), fixed(4000, 790.0, 0.95, 0.85), ""]))
    (run / "claim_qarm").write_text("tables-x.service\n")
    monkeypatch.setattr(board, "RESULTS", tmp_path)
    monkeypatch.setattr(board, "RUN_RESULTS", run)
    monkeypatch.setattr(board, "DATA", tmp_path / "data")
    monkeypatch.setattr(board, "gpus", lambda: [])
    monkeypatch.setattr(board, "active_units", lambda *a, **k: ["tables-x.service", "big12-gpu1.service", "big8-super.service"])
    s = board.build_state()
    assert s["units"] == ["tables-x.service"]  # the queue's units only: a big run's unit is not a queue arm
    assert [r["name"] for r in s["runs"]] == ["qarm"]  # claim_big12 is held by an active unit but big12 is not a queue arm either
    arm = s["runs"][0]
    assert arm["open_ended"] is True and arm["eta_s"] is None and arm["pace"] > 0
    assert s["big"]["big12"]["running"] and s["big"]["big12"]["step"] == 10000 and s["big_error"] is None
    assert s["big"]["big8"]["state"] == "not started"
    json.dumps(s["big"], allow_nan=False)


def test_a_run_is_running_while_its_claims_unit_is_active_and_paused_or_not_started_while_only_its_supervisor_is(tmp_path):
    tables, run = big_dirs(tmp_path)
    up = big_of(tables, run, ["big12-gpu1.service", "big8-super.service"])
    assert up["big12"]["state"] == "running" and up["big12"]["supervisor"] is None
    big8 = up["big8"]  # no log yet: the supervisor waits for the card to be idle
    assert big8["state"] == "not started" and big8["waiting"] and big8["supervisor"] == "big8-super.service"
    assert not big8["running"] and big8["step"] is None and big8["passes"] is None and big8["pace"] is None
    assert big8["curve"] == [] and big8["clean"] == {"dev": [], "oldk": [], "dev30": [], "oldk30": []} and big8["open_ended"] is True
    # the owner's game: big8 has run and stopped, the supervisor is waiting to restart it
    (tables / "big" / "big8.log").write_text(BIG8_FIRST)
    paused = big_of(tables, run, ["big12-gpu1.service", "big8-super.service"])["big8"]
    assert paused["state"] == "paused" and paused["waiting"] and not paused["running"] and paused["step"] == 6000 and paused["layers"] == 8
    # stopped by the game before its first checkpoint: a header and no points is still a run that was started
    (tables / "big" / "big8.log").write_text(header("big8", "16:00:00", 8, "fresh") + "\n" + CODE + "\n")
    early = big_of(tables, run, ["big8-super.service"])["big8"]
    assert early["state"] == "paused" and early["step"] == 0 and early["curve"] == [] and early["passes"] == 0.0
    (tables / "big" / "big8.log").write_text(BIG8_FIRST)
    # back on its card after the game: a second header and an active unit (claim_big8 is written by bigtrain.sh)
    (tables / "big" / "big8.log").write_text(BIG8_FIRST + BIG8_SECOND + "\n")
    (run / "claim_big8").write_text("big8-gpu0.service\n")
    again = big_of(tables, run, ["big12-gpu1.service", "big8-gpu0.service", "big8-super.service"])["big8"]
    assert again["state"] == "running" and again["unit"] == "big8-gpu0.service" and again["restarts"] == 1 and again["step"] == 4000
    assert [r[0] for r in again["curve"]] == [2000, 4000]
    # a claim left behind by a unit that is gone does not make a run running; with no supervisor either it is stopped
    gone = big_of(tables, run, [])
    assert gone["big12"]["state"] == "stopped" and not gone["big12"]["running"] and gone["big8"]["state"] == "stopped"


SUMMARY = {
    "big12": [
        {"step": 2000, "dev": {"cold": 1.2726, "warm": 1.163, "n": 613}},
        {"step": 4000, "dev": {"cold": 0.9797, "warm": 0.8729, "n": 613}},
        {"step": 10000, "dev": {"cold": 0.8, "warm": 0.7, "n": 613}, "oldk": {"cold": 0.61, "warm": 0.55, "n": 890},
         "dev30": {"cold": 0.79, "warm": 0.69, "n": 613}},
    ],
    "big8": [],
}


def test_the_summary_json_becomes_clean_scores_per_tag_and_a_missing_tag_is_just_an_empty_series(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(SUMMARY))
    got = board.read_summary(p)
    clean = got["big12"]["clean"]
    assert clean["dev"] == [[2000, 1.2726, 1.163], [4000, 0.9797, 0.8729], [10000, 0.8, 0.7]]
    assert clean["oldk"] == [[10000, 0.61, 0.55]] and clean["dev30"] == [[10000, 0.79, 0.69]] and clean["oldk30"] == []
    assert got["big12"]["windows"] == {"dev": 613, "oldk": 890, "dev30": 613}
    assert got["big8"]["clean"] == {"dev": [], "oldk": [], "dev30": [], "oldk30": []}
    # rows out of order and rows without a usable score do not break it
    p.write_text(json.dumps({"big12": [{"step": 4000, "dev": {"cold": 0.98, "warm": 0.87}}, {"step": 2000, "dev": {"cold": 1.27}},
                                       {"step": 6000, "dev": {"cold": None}}, {"step": "x"}, "junk"]}))
    assert board.read_summary(p)["big12"]["clean"]["dev"] == [[2000, 1.27, None], [4000, 0.98, 0.87]]


def test_the_summary_json_may_be_missing_or_half_written_and_the_last_good_read_is_kept(tmp_path):
    p = tmp_path / "summary.json"
    assert board.read_summary(p) == {}
    p.write_text("")  # the scorer truncates the file before it writes it
    assert board.read_summary(p) == {}
    p.write_text(json.dumps(SUMMARY))
    assert board.read_summary(p)["big12"]["clean"]["dev"][0][0] == 2000
    p.write_text('{"big12": [{"step": 2000, "dev": {"co')
    assert board.read_summary(p)["big12"]["clean"]["dev"][0][0] == 2000


def test_the_state_of_a_big_run_carries_its_clean_scores_from_the_summary(tmp_path):
    tables, run = big_dirs(tmp_path)
    assert big_of(tables, run, ["big12-gpu1.service"])["big12"]["clean"]["dev"] == []  # the scorer has not written anything yet
    (tables / "big" / "summary.json").write_text(json.dumps(SUMMARY))
    b = big_of(tables, run, ["big12-gpu1.service"])["big12"]
    assert [row[:2] for row in b["clean"]["dev"]] == [[2000, 1.2726], [4000, 0.9797], [10000, 0.8]]
    assert b["clean"]["oldk"] == [[10000, 0.61, 0.55]] and b["windows"]["dev"] == 613
    assert big_of(tables, run, [])["big8"]["clean"]["dev"] == []


def test_a_long_series_is_thinned_evenly_keeping_the_first_and_the_last_point():
    rows = [[i * 2000, 1.0 / (i + 1)] for i in range(5000)]
    out = board.thin(rows, 1500)
    assert len(out) <= 1501 and out[0] == rows[0] and out[-1] == rows[-1]
    assert [r[0] for r in out] == sorted(r[0] for r in out)
    short = rows[:10]
    assert board.thin(short, 1500) == short


def test_a_fault_in_the_big_train_data_leaves_the_rest_of_the_state_intact(tmp_path, monkeypatch):
    tables, run = big_dirs(tmp_path)
    monkeypatch.setattr(board, "RESULTS", tmp_path)
    monkeypatch.setattr(board, "RUN_RESULTS", run)
    monkeypatch.setattr(board, "DATA", tmp_path / "data")
    monkeypatch.setattr(board, "gpus", lambda: [])
    monkeypatch.setattr(board, "active_units", lambda *a, **k: [])

    def boom(*a, **k):
        raise ValueError("a log nobody expected")
    monkeypatch.setattr(board, "big_state", boom)
    s = board.build_state()
    assert s["big"] == {} and s["big_error"] == "ValueError: a log nobody expected"
    assert {"runs", "last", "leaderboard", "gpus", "windows"} <= set(s)


def test_every_element_the_page_script_looks_up_by_id_exists_in_the_page():
    html = (Path(__file__).resolve().parents[1] / "tools" / "board" / "index.html").read_text()
    used = set(re.findall(r'\$\("#([\w-]+)"\)', html[html.index("<script>"):]))
    present = set(re.findall(r'id="([\w-]+)"', html))  # the markup, and the ids the script writes into the chart it draws
    assert {"bigcard", "bigrows", "bigchart", "biglegend", "bigtablebody", "bigx", "runcard", "tip"} <= used
    assert used <= present, used - present
