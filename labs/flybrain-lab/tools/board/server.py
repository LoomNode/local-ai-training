#!/usr/bin/env python3
"""The lab board (Kyle, 2026-10-09: "a visualization of the model training or running would be really neat. it could be pretty simple").
One page, two views. Training: the big train (open-ended runs from results/tables/big/: step, passes, pace, the in-run curve and the
scorer's clean scores), the two cards, the queue (step N of M, pace, ETA from the unit's log), the running arm's curve against a
reference run, the leaderboard from the dev-window scores. Reading: dev windows dumped by checks/dump_window.py, every byte coloured by
the memory that carried it. Standard library only; read-only; the GPUs are touched only through nvidia-smi.
Run:   python3 tools/board/server.py        (port 8800; BOARD_PORT, BOARD_RESULTS, BOARD_RUN, BOARD_DATA override the defaults)
Open:  http://goblin:8800"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("BOARD_RESULTS", HERE.parents[1] / "results"))  # the speed worktree: run JSONs, tables/*.log, tables/k_*.json
RUN_RESULTS = Path(os.environ.get("BOARD_RUN", "/games/ailab/lab/flybrain-lab/.worktrees/speed-run4/results"))  # the run worktree: claims, JSONs as runs end
DATA = Path(os.environ.get("BOARD_DATA", HERE / "data"))
PORT = int(os.environ.get("BOARD_PORT", "8800"))
CACHE_S = 5.0
PLAIN_RUN = os.environ.get("BOARD_PLAIN", "speed_tf_r_3600_s0")  # the plain 8x512 transformer's hour (--rope-table, no memory), the bar
PLAIN_COLD = 0.806  # its dev-window score (notes/SPEED.md, "State of the round"); it was scored before k_*.json existed

RUN_HEADER = re.compile(r"^=== (\S+) (\d\d:\d\d:\d\d): ([\d,]+) steps(.*)$")  # group 4: the rest (the big train writes layers, resume, code there)
HEADER_RESUME = re.compile(r"--resume \S*?\.step(\d+)\.pt")  # "... cooldown 1, --resume results/ckpt/big12_transformer.step24700.pt, code a5e9e6b ..."
HEADER_FRESH = re.compile(r"\bfresh\b")
HEADER_LAYERS = re.compile(r"\((\d+) layers\b")
RESUMED_LINE = re.compile(r"^resumed \S+ at step (\d+) of \d+")  # "resumed big12_transformer at step 24700 of 10000000 from ... (train_seconds so far 11976.9)"
SAVED_LINE = re.compile(r"^saved \S+?\.step(\d+)\.pt")  # "saved big12_transformer.step26000.pt"
CURVE_LINE = re.compile(r"^fixed \w+ (\{.*\})\s*$")
YARDSTICK = re.compile(r"^\w+ yardstick train_s=([\d.]+)")
ERROR_LINE = re.compile(r"Traceback|Error|failed|not built|pre-flight failed")


# --- the unit logs: what a check script tees to results/tables/<queue>.log ------------------------------------------------------------
def parse_log(text: str) -> list[dict]:
    """The runs in a queue's log, in order: name, start time, step budget, the fixed-window curve points, whether the yardstick line
    (the end of training) was reached, and error lines. A "=== " line without a step budget (a calibration, a scoring step) closes the
    current run, so its own curve points are not taken for the run before it. A log may hold several headers of one run (the big train
    appends one each time it is resumed): "resumed" is the step a segment started from (0 for "fresh", None while unknown), "saved" its
    last checkpoint step, "layers" the layer count in the header."""
    runs: list[dict] = []
    cur = None
    for line in text.splitlines():
        h = RUN_HEADER.match(line)
        if h:
            rest = h.group(4)
            rm, lm = HEADER_RESUME.search(rest), HEADER_LAYERS.search(rest)
            cur = {"name": h.group(1), "started": h.group(2), "budget": int(h.group(3).replace(",", "")), "points": [], "finished": False,
                   "train_s": None, "errors": [], "resumed": int(rm.group(1)) if rm else (0 if HEADER_FRESH.search(rest) else None),
                   "saved": None, "layers": int(lm.group(1)) if lm else None}
            runs.append(cur)
            continue
        if line.startswith("=== "):
            cur = None
            continue
        if cur is None:
            continue
        r = RESUMED_LINE.match(line)
        if r:
            cur["resumed"] = int(r.group(1))
            continue
        sv = SAVED_LINE.match(line)
        if sv:
            cur["saved"] = int(sv.group(1))
            continue
        c = CURVE_LINE.match(line)
        if c:
            try:
                d = ast.literal_eval(c.group(1))
                cur["points"].append({"step": int(d["step"]), "train_seconds": float(d["train_seconds"]), "bpb": float(d["bpb"]),
                                      "bpb_warm": float(d.get("bpb_warm", float("nan")))})
            except (ValueError, SyntaxError, KeyError):
                pass
            continue
        y = YARDSTICK.match(line)
        if y:
            cur["finished"] = True; cur["train_s"] = float(y.group(1))
            continue
        if ERROR_LINE.search(line):
            cur["errors"].append(line.strip()[:200])
    return runs


OPEN_ENDED_FROM = 1_000_000  # a step budget this large means "until told to stop" (the big train asks for 10,000,000): no ETA, no progress bar


def is_open_ended(budget: int | None) -> bool:
    return budget is not None and budget >= OPEN_ENDED_FROM


def step_pace(points: list[dict]) -> float | None:
    """Seconds per step over the last two curve points (the first point alone, from its train_seconds, if there is one)."""
    if not points:
        return None
    if len(points) >= 2:
        a, b = points[-2], points[-1]
        return (b["train_seconds"] - a["train_seconds"]) / max(1, b["step"] - a["step"])
    return points[-1]["train_seconds"] / max(1, points[-1]["step"])


def pace_eta(points: list[dict], budget: int, age_s: float) -> tuple[float | None, float | None]:
    """The pace (step_pace) and the seconds left to the budget, less the time since the log was last written. An open-ended budget has
    a pace and no ETA."""
    pace = step_pace(points)
    if pace is None:
        return None, None
    if is_open_ended(budget):
        return pace, None
    return pace, max(0.0, (budget - points[-1]["step"]) * pace - age_s)


_log_cache: dict[str, tuple[tuple[int, int], float, list[dict]]] = {}  # path -> ((mtime_ns, size), mtime, runs)


def parsed_logs(tables_dir: Path) -> dict[str, tuple[float, list[dict]]]:
    """{log path: (mtime, runs)} for every queue log and every log of the big train (tables/big/), re-parsed only when the file changed
    (its mtime or its size: two writes in one timestamp tick still differ in size)."""
    out = {}
    for p in sorted([*tables_dir.glob("*.log"), *(tables_dir / "big").glob("*.log")]):
        try:
            st = p.stat()
        except OSError:
            continue
        key = (st.st_mtime_ns, st.st_size)
        hit = _log_cache.get(str(p))
        if hit is None or hit[0] != key:
            try:
                hit = (key, st.st_mtime, parse_log(p.read_text(errors="replace")))
            except OSError:
                continue
            _log_cache[str(p)] = hit
        out[str(p)] = (hit[1], hit[2])
    return out


# --- the queue: systemd units and the claim files the check scripts write -------------------------------------------------------------
def active_units(patterns: tuple[str, ...] = ("tables-*",)) -> list[str]:
    try:
        out = subprocess.run(["systemctl", "--user", "list-units", "--type=service", "--state=active", "--plain", "--no-legend", *patterns],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln.split()[0] for ln in out.splitlines() if ln.strip()]


def claims(run_dir: Path) -> dict[str, str]:
    """{arm name: unit} from results/claim_<name> (a check script writes its unit's name there when it starts an arm)."""
    out = {}
    for p in run_dir.glob("claim_*"):
        try:
            out[p.name[len("claim_"):]] = p.read_text().strip()
        except OSError:
            pass
    return out


def running_arms(units: list[str], claimed: dict[str, str], run_dir: Path) -> list[dict]:
    """The arms whose claim is held by an active unit and whose run JSON does not exist yet."""
    return [{"name": n, "unit": u} for n, u in sorted(claimed.items()) if u in units and not (run_dir / f"{n}.json").exists()]


def run_from_logs(name: str, logs: dict[str, tuple[float, list[dict]]]) -> tuple[dict | None, str | None, float]:
    """The newest log entry for an arm: (run, log path, log mtime)."""
    best = (None, None, 0.0)
    for path, (mtime, runs) in logs.items():
        for r in runs:
            if r["name"] == name and mtime >= best[2]:
                best = (r, path, mtime)
    return best


# --- the cards ---------------------------------------------------------------------------------------------------------------------
def gpus() -> list[dict]:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,power.draw,power.limit,temperature.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    holders: dict[str, list[dict]] = {}
    for ln in apps.splitlines():
        parts = [s.strip() for s in ln.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid, mem = parts[0], parts[1], parts[2]
        holders.setdefault(uuid, []).append({"pid": int(pid) if pid.isdigit() else pid, "mem_mib": _num(mem), "who": _who(pid)})
    out = []
    for ln in q.splitlines():
        parts = [s.strip() for s in ln.split(",")]
        if len(parts) < 9:
            continue
        out.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2], "util": _num(parts[3]), "power_w": _num(parts[4]),
                    "power_limit_w": _num(parts[5]), "temp_c": _num(parts[6]), "mem_used_mib": _num(parts[7]), "mem_total_mib": _num(parts[8]),
                    "holders": holders.get(parts[1], [])})
    return out


def _num(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def _who(pid: str) -> str:
    """The user unit a process runs under (from its cgroup), else its command name."""
    try:
        cg = Path(f"/proc/{pid}/cgroup").read_text()
        m = re.search(r"/([\w@.:-]+\.service)\s*$", cg, re.M)
        if m and not m.group(1).startswith("user@"):
            return m.group(1)
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return "?"


# --- the leaderboard and the curves: the dev-window scores and the run JSONs ----------------------------------------------------------
def run_json(name: str, dirs: list[Path]) -> dict | None:
    for d in dirs:
        p = d / f"{name}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (OSError, ValueError):
                return None
    return None


def first_arm(d: dict) -> dict | None:
    arms = d.get("arms")
    if isinstance(arms, list) and arms:
        return arms[0]
    if isinstance(arms, dict) and arms:
        return next(iter(arms.values()))
    return None


# Hand-written notes for the arms whose point is not in their run JSON (the speed round's state table, notes/SPEED.md).
NOTES = {
    "base_tf_mt_r": "the base arm", "l4b_mt_r": "4 layers at one hour, the tier arms' base", "l4plainm_r": "the control: no corpus table",
    "l4dist_r": "no corpus table, taught by the table model", "l4drop5_r": "table withdrawn at half the positions while training",
    "l4ramp_r": "weaned off the table over the run", "l4dir_r": "the best at one hour", "l4fly_r": "the fly's sparse key",
    "l4rep_r": "table admits only repeated contexts", "l4n_mt_r": "learned slots on top of the table", "l4c_mt_r": "a 2^29 table",
    "hyb_mt_r": "the fly hybrid backbone", "gla_mt_r": "GLA backbone", "w4k_tf_mt_r": "window 4096", "s12_tf_mt_r": "12 layers of 384",
    "s6_tf_mt_r": "6 layers of 768", "c30_mt_r": "skips 30 % of the stack", "c50l8_mt_r": "skips 50 % of the stack", "e_mt_r": "PAQ bit tables",
    "paq_mt_r": "PAQ keys", "top2_mt_r": "top-two slots", "counters_tf_mt_r": "counter inputs", "keys_tf_mt_r": "template, word and line keys",
    "orders_tf_mt_r": "more n-gram orders", "all_tf_mt_r": "18 keys through a fused channel", "allf_tf_mt_r": "18 keys, fused, equal seconds",
    "alleqf_tf_mt_r": "18 keys, equal seconds", "all28f_tf_mt_r": "the 2^28 tables' first run", "l0b_mt_r": "no network: the table's answers mixed",
    "l12x4_mt_r": "12 layers at four hours: the best so far, 0.007 under 8", "s6x4_tf_mt_r": "6 x 768 at four hours: width loses, +0.008 to 8 x 512",
    "s12x4_tf_mt_r": "12 x 384 at four hours: ties 12 x 512 with half the parameters", "l12x4s1_mt_r": "12 layers at four hours, second seed: the gain over 8 was the seed", "l16x4_mt_r": "16 layers at four hours: ties 8 and 12, still falling at the end", "l4pk_mt_r": "4 layers at the hour on the package corpus (5.3 GB, its own 2^28 table)", "l4pk29_mt_r": "the same with the 2^29 table", "l4pk30_mt_r": "the same with the 2^30 table: 0.724 on the package dev windows", "l4pk28x2_mt_r": "two hours on the package corpus with the 2^28 table: 0.763 on the clean dev windows, 0.742 read through 2^30", "l4pk30x2_mt_r": "two hours with the 2^30 table (the recipe test's other half)",
    "wsd_r": "the restart test: 4 layers at the hour under warmup-stable-decay", "wsdk_r": "the same, killed and resumed from a checkpoint", "l8x4_mt_r": "8 layers at four hours", "l6x4_mt_r": "6 layers at four hours: ties 8", "l8x2s1_mt_r": "8 layers at two hours, second seed",
    "l6x2dir_mt_r": "the project tier at two hours with 6 layers: no gain", "l4b_s1_r": "4 layers at one hour, second seed",
    "l4x2dir_mt_r": "the project tier at two hours with 4 layers",
}


def describe(name: str, d: dict | None) -> str:
    """An arm in words, from its run JSON and its name: layers, memory, keys, tiers, seed, budget, and the hand-written note."""
    if not d:
        return name
    a = first_arm(d) or {}
    bits = []
    arm = (a.get("arm") or d.get("arms_list") or "transformer") if isinstance(a, dict) else "transformer"
    if arm != "transformer":
        bits.append(arm)
    if d.get("n_layers") is not None:
        bits.append(f"{d['n_layers']} layer{'s' if d['n_layers'] != 1 else ''}")
    feat = Path(d["mem_feat"]).name if d.get("mem_feat") else ""
    orders = [int(o) for o in str(d.get("mem_orders") or "").split(",") if o.strip().isdigit()]
    if feat:
        size = "2^29" if "29" in feat else "2^28" if "28" in feat else ""
        n_keys = len([o for o in orders if o < 2000])
        bits.append(f"{size + ' ' if size else ''}corpus table, {n_keys} key{'s' if n_keys != 1 else ''}")
    if any(2000 <= o < 3000 for o in orders):
        bits.append("project tier")
    if any(3000 <= o < 4000 for o in orders):
        bits.append("fly key")
    if d.get("match_feat") and not feat:
        bits.append("in-window lookup only")
    if a.get("distill") or d.get("distill_from"):
        bits.append("distilled")
    if d.get("mem_dropout"):
        bits.append("table dropout")
    if "x4" in name:
        bits.append("four hours")
    elif "x2" in name:
        bits.append("two hours")
    if d.get("seed"):
        bits.append(f"seed {d['seed']}")
    note = NOTES.get(name)
    label = " · ".join(bits) if bits else name
    return f"{label} — {note}" if note else label


def run_summary(name: str, dirs: list[Path]) -> dict | None:
    """Steps, parameters, layers, bytes seen and logged training seconds of a finished run, with its fixed-window curve."""
    d = run_json(name, dirs)
    a = first_arm(d) if d else None
    if not d or not a:
        return None
    curve = a.get("curve_fixed") or []
    return {"name": name, "label": describe(name, d), "steps": curve[-1]["step"] if curve else None, "params": a.get("trainable_scalars"),
            "layers": d.get("n_layers"), "bytes_seen": a.get("bytes_seen"), "train_s": a.get("train_seconds"), "seed": d.get("seed"),
            "orders": d.get("mem_orders"), "mem_feat": Path(d["mem_feat"]).name if d.get("mem_feat") else None,
            "plain": not d.get("mem_feat") and not d.get("match_feat"), "distill": bool(a.get("distill")), "points": curve}


def leaderboard(tables_dir: Path, dirs: list[Path]) -> list[dict]:
    """Every k_*.json (one scored arm each) joined to its run by the checkpoint's name, sorted by cold bits/byte."""
    rows = []
    for p in sorted(tables_dir.glob("k_*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if "bpb" not in d:
            continue
        ckpt = Path(d.get("ckpt", "")).name
        run = ckpt[: -len(f"_{d.get('arm', 'transformer')}.pt")] if ckpt.endswith(".pt") else ""
        s = run_summary(run, dirs) if run else None
        rows.append({"score": d.get("name", p.stem), "run": run, "label": s["label"] if s else run or p.stem, "plain": bool(s and s["plain"]),
                     "cold": d["bpb"], "warm": d.get("bpb_warm"), "arm": d.get("arm"),
                     "steps": s and s["steps"], "params": s and s["params"], "layers": s and s["layers"], "seed": s and s["seed"],
                     "mem_feat": s and s["mem_feat"], "orders": s and len(s["orders"].split(",")) if s and s.get("orders") else None,
                     "train_s": s and s["train_s"], "scored": p.stat().st_mtime})
    rows.sort(key=lambda r: r["cold"])
    return rows


def windows_index(data_dir: Path) -> list[dict]:
    out = []
    for p in sorted(data_dir.glob("*_w*.json")):
        m = re.match(r"^(.+)_w(\d+)\.json$", p.name)
        if not m:
            continue
        head = ""
        try:
            d = json.loads(p.read_text())
            t = d.get("text", "")
            h = re.search(r"// ==== (\S+) ====", t)
            head = h.group(1) if h else t[:48].replace("\n", " ")
        except (OSError, ValueError):
            pass
        out.append({"file": p.name, "model": m.group(1), "window": int(m.group(2)), "head": head})
    return out


# --- the big train: runs with no end step (logs and scores in results/tables/big/) ---------------------------------------------------
BIG_RUNS = {"big12": ["dry12"], "big8": []}  # run -> the earlier names of the same run: big12 began as the dry run dry12 (its curve is big12's start)
BIG_TAGS = ("dev", "oldk", "dev30", "oldk30")  # the scorer's tags: clean dev windows (every checkpoint: the bar) and clean old-kernel windows, each also through 2^30
BIG_BATCH, BIG_WINDOW, BIG_CORPUS = 8, 2048, 5_283_527_978  # bytes per step = batch x window; one pass = the package corpus once
BIG_MAX_POINTS = 1500  # the longest series sent to the page: an open-ended run grows without bound, so old points are thinned evenly


def passes(step: int) -> float:
    return step * BIG_BATCH * BIG_WINDOW / BIG_CORPUS


def merged_points(segments: list[dict]) -> list[dict]:
    """The curve of a run that was stopped and resumed (the owner's game takes its card, a restart): every point of every segment in
    order, de-duplicated by step with the last one kept, sorted by step. A segment that resumes at step N supersedes the earlier points
    beyond N (those steps are trained again) and a fresh start supersedes them all; a segment whose start is not known yet drops nothing."""
    by_step: dict[int, dict] = {}
    for seg in segments:
        start = seg.get("resumed")
        if start is not None:
            by_step = {s: p for s, p in by_step.items() if s <= start}
        for p in seg["points"]:
            by_step[p["step"]] = p
    return [by_step[s] for s in sorted(by_step)]


def thin(rows: list, cap: int) -> list:
    """At most about cap rows, evenly spaced, the first and the last kept."""
    n = len(rows)
    if n <= cap:
        return rows
    idx = list(range(0, n, -(-n // cap)))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [rows[i] for i in idx]


def _finite(x, digits: int = 4) -> float | None:
    return round(float(x), digits) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


_summary_cache: dict[str, tuple[tuple[int, int], dict]] = {}


def read_summary(path: Path) -> dict[str, dict]:
    """The scorer's summary.json ({run: [{"step": N, "dev": {"cold", "warm", "n"}, "oldk": {...}, "dev30": {...}, "oldk30": {...}}, ...]},
    rewritten every minute, a tag only where that checkpoint was scored on it) as {run: {"clean": {tag: [[step, cold, warm], ...]},
    "windows": {tag: n}}}: every tag of BIG_TAGS is a series, empty where nothing was scored. {} while the file does not exist; a file
    caught half written (the scorer truncates it before it writes) leaves the last good read in place."""
    try:
        st = path.stat()
    except OSError:
        return {}
    key = (st.st_mtime_ns, st.st_size)
    hit = _summary_cache.get(str(path))
    if hit and hit[0] == key:
        return hit[1]
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return hit[1] if hit else {}
    out: dict[str, dict] = {}
    for run, rows in (raw.items() if isinstance(raw, dict) else []):
        clean: dict[str, list] = {t: [] for t in BIG_TAGS}
        windows: dict[str, int] = {}
        good = [r for r in rows if isinstance(r, dict) and isinstance(r.get("step"), int)] if isinstance(rows, list) else []
        for row in sorted(good, key=lambda r: r["step"]):
            for tag, v in row.items():
                cold = _finite(v.get("cold")) if isinstance(v, dict) else None
                if tag == "step" or cold is None:
                    continue
                clean.setdefault(tag, []).append([row["step"], cold, _finite(v.get("warm"))])
                if isinstance(v.get("n"), int):
                    windows[tag] = v["n"]
        out[run] = {"clean": {t: thin(series, BIG_MAX_POINTS) for t, series in clean.items()}, "windows": windows}
    _summary_cache[str(path)] = (key, out)
    return out


def big_state(tables_dir: Path, claimed: dict[str, str], units: list[str], logs: dict[str, tuple[float, list[dict]]], now: float) -> dict[str, dict]:
    """Every big run (BIG_RUNS) as the page draws it: whether it runs, its step, passes and pace (no ETA: it has no end), the in-run curve
    with the earlier names' logs merged in, and the scorer's clean scores. units are the active user units, claimed is {run: unit} from the
    claim files, logs is parsed_logs()."""
    scores = read_summary(tables_dir / "big" / "summary.json")
    out = {}
    for name, earlier in BIG_RUNS.items():
        r = _big_run(name, earlier, claimed, units, logs, scores.get(name), now)
        # the cooled-down snapshots (checks/bigsnap.sh): their scores are the summary's "<run>snap" series, the stop rule's evidence
        r["snap"] = (scores.get(f"{name}snap") or {}).get("clean") or {t: [] for t in BIG_TAGS}
        r["snap_running"] = snap_running(name, claimed, units, logs)
        out[name] = r
    return out


def snap_running(name: str, claimed: dict[str, str], units: list[str], logs: dict[str, tuple[float, list[dict]]]) -> dict | None:
    """The snapshot of `name` being cooled down now, if any: a claim <name>_snap<m> held by an active unit, with its step and budget from
    its log (resumed at m, the cooldown's checkpoints and curve points after it)."""
    for run, unit in sorted(claimed.items()):
        if not run.startswith(f"{name}_snap") or unit not in units:
            continue
        segs = [r for _, (_, rs) in sorted(logs.items(), key=lambda kv: kv[1][0]) for r in rs if r["name"] == run]
        if not segs:
            return {"name": run, "step": None, "budget": None, "from": None}
        seg = segs[-1]
        steps = [p["step"] for p in seg["points"]] + [v for v in (seg.get("resumed"), seg.get("saved")) if v is not None]
        return {"name": run, "step": max(steps) if steps else None, "budget": seg.get("budget"), "from": seg.get("resumed")}
    return None


def _big_run(name: str, earlier: list[str], claimed: dict[str, str], units: list[str], logs: dict[str, tuple[float, list[dict]]],
             scores: dict | None, now: float) -> dict:
    names = [*earlier, name]
    found = sorted(((names.index(r["name"]), mtime, i, r, path) for path, (mtime, runs) in logs.items() for i, r in enumerate(runs)
                    if r["name"] in names), key=lambda t: t[:3])  # the earlier names first, then by log time and position in the log
    own = [t for t in found if t[3]["name"] == name]
    curve = merged_points([t[3] for t in found])
    step = None
    if found:  # the newest segment may be ahead of the last curve point: it resumed at N, or saved a checkpoint
        ahead = [v for v in (own[-1][3].get("resumed"), own[-1][3].get("saved")) if v is not None] if own else []
        step = max([curve[-1]["step"] if curve else 0, *ahead])
    tail = found[-1] if found else None
    budget = tail[3]["budget"] if tail else None
    unit = claimed.get(name) or next((u for u in units if u.startswith(f"{name}-gpu") and not u.endswith("-guard.service")), None)
    running = unit is not None and unit in units
    supervisor = next((u for u in units if u.startswith(f"{name}-super") and u.endswith(".service")), None)
    started = bool(own) or bool(step)  # a header of its own in a log, or progress inherited from an earlier name
    if running:
        status = "running"
    elif supervisor and started:
        status = "paused"  # the owner's game has the card; the supervisor restarts the run when the card has been idle for three minutes
    elif started:
        status = "stopped"
    else:
        status = "not started"
    pace = step_pace(curve)
    if pace is not None and not (math.isfinite(pace) and pace > 0):
        pace = None
    scores = scores or {"clean": {t: [] for t in BIG_TAGS}, "windows": {}}
    return {"name": name, "state": status, "running": running, "waiting": bool(supervisor) and not running, "unit": unit, "supervisor": supervisor,
            "budget": budget, "open_ended": budget is None or is_open_ended(budget), "layers": tail[3].get("layers") if tail else None,
            "step": step, "passes": round(passes(step), 4) if step is not None else None, "passes_per_step": passes(1),
            "pace": round(pace, 5) if pace else None, "passes_per_day": round(86400 / pace * passes(1), 3) if pace else None,
            "train_s": round(curve[-1]["train_seconds"], 1) if curve else None, "restarts": max(0, len(own) - 1),
            "log": Path(tail[4]).name if tail else None, "log_age_s": round(now - tail[1], 1) if tail else None,
            "clean": scores["clean"], "windows": scores["windows"],  # the long series last, so that the top of the JSON reads at a glance
            "curve": thin([[p["step"], _finite(p["bpb"]), _finite(p["bpb_warm"])] for p in curve if _finite(p["bpb"]) is not None], BIG_MAX_POINTS)}


# --- the state the page polls ------------------------------------------------------------------------------------------------------
_state_lock = threading.Lock()
_state: tuple[float, dict] | None = None


def build_state() -> dict:
    now = time.time()
    tables = RESULTS / "tables"
    dirs = [RESULTS, RUN_RESULTS]
    logs = parsed_logs(tables)
    all_units = active_units(("tables-*", "big*"))
    units = [u for u in all_units if u.startswith("tables-")]  # the queue's; the big train's units are read by big_state
    claimed = claims(RUN_RESULTS)
    arms = running_arms(units, claimed, RUN_RESULTS)
    runs = []
    for a in arms:
        r, path, mtime = run_from_logs(a["name"], logs)
        if r is None:
            runs.append({**a, "budget": None, "open_ended": False, "points": [], "pace": None, "eta_s": None, "started": None, "log": None, "errors": []})
            continue
        pace, eta = pace_eta(r["points"], r["budget"], now - mtime)
        runs.append({**a, "budget": r["budget"], "open_ended": is_open_ended(r["budget"]), "points": r["points"], "pace": pace, "eta_s": eta, "started": r["started"],
                     "log": Path(path).name, "errors": r["errors"][-3:], "log_age_s": now - mtime})
    last = None
    if not runs:  # the newest finished run, from the logs (its JSON may still be in the run worktree only)
        cands = [(mt, r) for path, (mt, rs) in logs.items() for r in rs if r["finished"] and r["points"]]
        if cands:
            mt, r = max(cands, key=lambda c: (c[0], c[1]["points"][-1]["train_seconds"]))
            last = {"name": r["name"], "budget": r["budget"], "points": r["points"], "train_s": r["train_s"], "started": r["started"]}
    try:
        big, big_error = big_state(tables, claimed, all_units, logs, now), None
    except Exception as e:  # a fault in the big train's panel must not take the older views down with it
        big, big_error = {}, f"{type(e).__name__}: {e}"
    board = leaderboard(tables, dirs)
    labels = {r["run"]: r["label"] for r in board if r["run"]}
    for r in runs + ([last] if last else []):
        r["label"] = labels.get(r["name"]) or describe(r["name"], run_json(r["name"], dirs)) or r["name"]
        if r["label"] == r["name"]:  # not finished yet and not scored: describe it from the arm it repeats, by name
            stem = re.sub(r"_s1(_r)$", r"\1", r["name"]).replace("x4_", "x2_")
            lab = ""
            for base in (stem, stem.replace("_r", "_mt_r"), stem.replace("dir_mt_r", "_mt_r").replace("dir_r", "b_mt_r")):
                lab = labels.get(base) or describe(base, run_json(base, dirs)) or ""
                if lab and lab != base: break
            lab = (lab or r["name"]).split(" — ")[0]
            if "_s1_" in r["name"]: lab += " · seed 1"
            if "x4_" in r["name"]: lab = lab.replace("two hours", "four hours")
            if "dir" in r["name"] and "project tier" not in lab: lab += " · project tier"
            note = NOTES.get(r["name"])
            r["label"] = lab + (" — " + note if note else "")
    plain_row = next((r for r in board if r["plain"]), None)
    plain = plain_row["run"] if plain_row else (PLAIN_RUN if run_json(PLAIN_RUN, dirs) else None)
    best = board[0]["run"] if board else None
    return {"time": now, "big": big, "big_error": big_error, "gpus": gpus(), "units": units, "runs": runs, "last": last, "plain_run": plain, "best_run": best,
            "plain_cold": plain_row["cold"] if plain_row else PLAIN_COLD, "plain_label": "plain transformer, no memory",
            "runs_available": [{"name": n, "label": labels.get(n, n)} for n in sorted({r["run"] for r in board if r["run"]} | {r["name"] for r in runs})],
            "leaderboard": board, "windows": windows_index(DATA)}


def state() -> dict:
    global _state
    with _state_lock:
        if _state is None or time.time() - _state[0] > CACHE_S:
            _state = (time.time(), build_state())
        return _state[1]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            except OSError:
                self._send(404, b"no index.html", "text/plain")
        elif path == "/api/state":
            self._json(state())
        elif path.startswith("/api/curve/"):
            name = Path(path[len("/api/curve/"):]).name
            s = run_summary(name, [RESULTS, RUN_RESULTS])
            self._json(s if s else {"error": f"no run {name}"}, 200 if s else 404)
        elif path == "/api/windows":
            self._json(windows_index(DATA))
        elif path.startswith("/api/window/"):
            name = Path(path[len("/api/window/"):]).name
            p = DATA / name
            if name.endswith(".json") and p.exists():
                self._send(200, p.read_bytes(), "application/json")
            else:
                self._json({"error": f"no window {name}"}, 404)
        else:
            self._send(404, b"not found", "text/plain")


if __name__ == "__main__":
    print(f"lab board on http://0.0.0.0:{PORT}  results {RESULTS}  run {RUN_RESULTS}  data {DATA}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
