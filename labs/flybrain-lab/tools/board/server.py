#!/usr/bin/env python3
"""The lab board (Kyle, 2026-10-09: "a visualization of the model training or running would be really neat. it could be pretty simple").
One page, two views. Training: the two cards, the queue (step N of M, pace, ETA from the unit's log), the running arm's curve against a
reference run, the leaderboard from the dev-window scores. Reading: dev windows dumped by checks/dump_window.py, every byte coloured by
the memory that carried it. Standard library only; read-only; the GPUs are touched only through nvidia-smi.
Run:   python3 tools/board/server.py        (port 8800; BOARD_PORT, BOARD_RESULTS, BOARD_RUN, BOARD_DATA override the defaults)
Open:  http://goblin:8800"""
from __future__ import annotations

import ast
import json
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

RUN_HEADER = re.compile(r"^=== (\S+) (\d\d:\d\d:\d\d): ([\d,]+) steps")
CURVE_LINE = re.compile(r"^fixed \w+ (\{.*\})\s*$")
YARDSTICK = re.compile(r"^\w+ yardstick train_s=([\d.]+)")
ERROR_LINE = re.compile(r"Traceback|Error|failed|not built|pre-flight failed")


# --- the unit logs: what a check script tees to results/tables/<queue>.log ------------------------------------------------------------
def parse_log(text: str) -> list[dict]:
    """The runs in a queue's log, in order: name, start time, step budget, the fixed-window curve points, whether the yardstick line
    (the end of training) was reached, and error lines. A "=== " line without a step budget (a calibration, a scoring step) closes the
    current run, so its own curve points are not taken for the run before it."""
    runs: list[dict] = []
    cur = None
    for line in text.splitlines():
        h = RUN_HEADER.match(line)
        if h:
            cur = {"name": h.group(1), "started": h.group(2), "budget": int(h.group(3).replace(",", "")), "points": [], "finished": False,
                   "train_s": None, "errors": []}
            runs.append(cur)
            continue
        if line.startswith("=== "):
            cur = None
            continue
        if cur is None:
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


def pace_eta(points: list[dict], budget: int, age_s: float) -> tuple[float | None, float | None]:
    """Seconds per step over the last two curve points (the first point alone if there is one), and the seconds left to the budget,
    less the time since the log was last written."""
    if not points:
        return None, None
    if len(points) >= 2:
        a, b = points[-2], points[-1]
        pace = (b["train_seconds"] - a["train_seconds"]) / max(1, b["step"] - a["step"])
    else:
        pace = points[-1]["train_seconds"] / max(1, points[-1]["step"])
    eta = max(0.0, (budget - points[-1]["step"]) * pace - age_s)
    return pace, eta


_log_cache: dict[str, tuple[float, list[dict]]] = {}


def parsed_logs(tables_dir: Path) -> dict[str, tuple[float, list[dict]]]:
    """{log path: (mtime, runs)} for every queue log, re-parsed only when the file changed."""
    out = {}
    for p in sorted(tables_dir.glob("*.log")):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        hit = _log_cache.get(str(p))
        if hit is None or hit[0] != mtime:
            try:
                hit = (mtime, parse_log(p.read_text(errors="replace")))
            except OSError:
                continue
            _log_cache[str(p)] = hit
        out[str(p)] = hit
    return out


# --- the queue: systemd units and the claim files the check scripts write -------------------------------------------------------------
def active_units() -> list[str]:
    try:
        out = subprocess.run(["systemctl", "--user", "list-units", "--type=service", "--state=active", "--plain", "--no-legend", "tables-*"],
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
    "s12x4_tf_mt_r": "12 x 384 at four hours: the narrow arm", "wsd_r": "the restart test: 4 layers at the hour under warmup-stable-decay", "wsdk_r": "the same, killed and resumed from a checkpoint", "l8x4_mt_r": "8 layers at four hours", "l6x4_mt_r": "6 layers at four hours: ties 8", "l8x2s1_mt_r": "8 layers at two hours, second seed",
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


# --- the state the page polls ------------------------------------------------------------------------------------------------------
_state_lock = threading.Lock()
_state: tuple[float, dict] | None = None


def build_state() -> dict:
    now = time.time()
    tables = RESULTS / "tables"
    dirs = [RESULTS, RUN_RESULTS]
    logs = parsed_logs(tables)
    units = active_units()
    arms = running_arms(units, claims(RUN_RESULTS), RUN_RESULTS)
    runs = []
    for a in arms:
        r, path, mtime = run_from_logs(a["name"], logs)
        if r is None:
            runs.append({**a, "budget": None, "points": [], "pace": None, "eta_s": None, "started": None, "log": None, "errors": []})
            continue
        pace, eta = pace_eta(r["points"], r["budget"], now - mtime)
        runs.append({**a, "budget": r["budget"], "points": r["points"], "pace": pace, "eta_s": eta, "started": r["started"],
                     "log": Path(path).name, "errors": r["errors"][-3:], "log_age_s": now - mtime})
    last = None
    if not runs:  # the newest finished run, from the logs (its JSON may still be in the run worktree only)
        cands = [(mt, r) for path, (mt, rs) in logs.items() for r in rs if r["finished"] and r["points"]]
        if cands:
            mt, r = max(cands, key=lambda c: (c[0], c[1]["points"][-1]["train_seconds"]))
            last = {"name": r["name"], "budget": r["budget"], "points": r["points"], "train_s": r["train_s"], "started": r["started"]}
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
    return {"time": now, "gpus": gpus(), "units": units, "runs": runs, "last": last, "plain_run": plain, "best_run": best,
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
