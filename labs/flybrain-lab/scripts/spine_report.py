#!/usr/bin/env python3
"""Spine write-up helper (group-learning spec 1.6): the Run 53 table, the pair gaps and the two decision rules, read from results/.

Runs: results/spine16_<arm>[_<tag>]_s<seed>.json (run_spine.sh). Scores: results/spine16_score_<split>_*.json, one cross_eval.py --fixed
call per split (dev, test, legacy, dev128) over the saved checkpoints, length 2048. Prints markdown; changes nothing."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEED_GAP_BPB = 0.01      # a pair closer than this in dev cold bits/byte gets seed 1
SEED_GAP_SECONDS = 0.05  # ... or closer than this in train_seconds, as a fraction of the faster arm's
CONSISTENCY_BPB = 0.01   # in-run bf16 score vs fp32 re-score of the same 128 dev windows
ARMS = ("hybrid", "tf", "zeroheads")
SPLITS = ("dev", "test", "legacy", "dev128")
RUN = re.compile(r"^spine16_(?P<arm>hybrid|tf|zeroheads)(?:_(?P<tag>[a-z0-9]+))?_s(?P<seed>\d+)$")
SCORE = re.compile(r"^spine16_score_(?P<split>dev128|dev|test|legacy)(?:_.*)?$")
JOULES_NOTE = "note: joules cover the whole arm, evals and saves included; train_seconds excludes them"


def load_runs(results: Path) -> dict[str, dict]:
    """Every spine run JSON by run name, reduced to the fields the table and the rules read."""
    runs = {}
    for path in sorted(results.glob("spine16_*.json")):
        m = RUN.match(path.stem)
        if not m:
            continue
        doc = json.loads(path.read_text())
        a = doc["arms"][0]
        runs[path.stem] = {"name": path.stem, "arm": m["arm"], "tag": m["tag"] or "", "seed": int(m["seed"]), "steps": doc["steps"],
                           "params": a["trainable_scalars"], "train_seconds": a["train_seconds"], "peak_vram_bytes": a["peak_vram_bytes"],
                           "joules": a.get("joules"), "contended": a.get("contended"), "telemetry_lost": a.get("telemetry_lost"),
                           "runner_val_bpb": a["val_bpb"], "inrun_dev_bpb": a["val_fixed"]["bpb"]}
    return runs


def load_scores(results: Path, runs: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """scores[split][run name] = the length-2048 row of cross_eval.py --fixed for that run's checkpoint."""
    scores: dict[str, dict[str, dict]] = {s: {} for s in SPLITS}
    for path in sorted(results.glob("spine16_score_*.json")):
        m = SCORE.match(path.stem)
        if not m:
            continue
        for row in json.loads(path.read_text())["rows"]:
            for name in runs:
                if row["T"] == 2048 and row["ckpt"].startswith(name + "_"):
                    scores[m["split"]][name] = row
    return scores


def decide(rows: list[dict], steps: int) -> dict:
    """The two rules. rows: {"arm", "dev_bpb", "train_seconds"} for each seed-0 arm.
    Rule 1: a pair closer than 0.01 bits/byte, or within 5 % in train_seconds (of the faster arm's), gets seed 1 for both arms.
    Rule 2: where the better arm is also the slower one (seconds differ by 5 % or more), the faster arm is rerun with its steps
    scaled to the slower arm's train_seconds, its own cosine schedule; one rerun per arm, scaled to the slowest arm that beats it."""
    seed1: set[str] = set()
    eqsec: dict[str, dict] = {}
    for a, b in itertools.combinations(rows, 2):
        fast, slow = sorted((a, b), key=lambda r: r["train_seconds"])
        gap_s = (slow["train_seconds"] - fast["train_seconds"]) / fast["train_seconds"]
        if abs(a["dev_bpb"] - b["dev_bpb"]) < SEED_GAP_BPB or gap_s < SEED_GAP_SECONDS:
            seed1 |= {a["arm"], b["arm"]}
        if gap_s >= SEED_GAP_SECONDS and slow["dev_bpb"] < fast["dev_bpb"]:
            target = max(slow["train_seconds"], eqsec.get(fast["arm"], {}).get("target_seconds", 0.0))
            eqsec[fast["arm"]] = {"arm": fast["arm"], "steps": round(steps * target / fast["train_seconds"]), "target_seconds": target}
    return {"seed1": [arm for arm in ARMS if arm in seed1], "eqsec": [eqsec[arm] for arm in ARMS if arm in eqsec]}


def commands(decision: dict, device: str = "cuda:1") -> list[str]:
    cmds = [f"scripts/run_spine.sh {device} {arm} 1" for arm in decision["seed1"]]
    return cmds + [f"scripts/run_spine.sh {device} {e['arm']} 0 {e['steps']} eqsec" for e in decision["eqsec"]]


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _cw(row):
    return "n/a" if row is None else f"{_f(row['bpb'])} / {_f(row['bpb_warm'])}"


def table(runs: dict[str, dict], scores: dict[str, dict[str, dict]]) -> str:
    head = ("| run | params | dev cold / warm | test cold / warm | legacy bytes cold | train_seconds | peak VRAM GiB | joules | contended |\n"
            "|---|---:|---|---|---:|---:|---:|---:|---|")
    lines = [head]
    for r in sorted(runs.values(), key=lambda r: (r["seed"], r["tag"], ARMS.index(r["arm"]))):
        g = lambda split: scores[split].get(r["name"])
        legacy = g("legacy")
        joules = "n/a" if r["joules"] is None else f"{r['joules']:,.0f}"
        contended = "n/a" if r["contended"] is None else ("yes" if r["contended"] else "no") + (", telemetry lost" if r["telemetry_lost"] else "")
        lines.append(f"| `{r['name']}` | {r['params'] / 1e6:.2f}M | {_cw(g('dev'))} | {_cw(g('test'))} | {_f(legacy and legacy['bpb'])} | "
                     f"{r['train_seconds']:,.0f} | {r['peak_vram_bytes'] / 2**30:.1f} | {joules} | {contended} |")
    return "\n".join(lines)


def _gap_line(r: dict, other: dict, how: str, scores: dict[str, dict[str, dict]]) -> str:
    """Run r minus run other: the cold bits/byte gap in dev, test and legacy to 4 decimals (n/a where either has no score) and the train_seconds ratio."""
    gaps = ", ".join(f"{s} " + (f"{scores[s][r['name']]['bpb'] - scores[s][other['name']]['bpb']:+.4f}" if r["name"] in scores[s] and other["name"] in scores[s] else "n/a")
                     for s in ("dev", "test", "legacy"))
    return (f"{r['name']} - {other['name']} ({how}): cold bpb {gaps}; train_seconds {r['train_seconds']:,.0f} vs {other['train_seconds']:,.0f} "
            f"(ratio {r['train_seconds'] / other['train_seconds']:.1%})")


def report(results: Path = RESULTS, device: str = "cuda:1") -> str:
    runs = load_runs(results)
    scores = load_scores(results, runs)
    out = [table(runs, scores), "", JOULES_NOTE]
    base = {r["arm"]: r for r in runs.values() if r["seed"] == 0 and not r["tag"]}
    for name, r in runs.items():  # in-run bf16 vs fp32 re-score of the same 128 dev windows
        d128 = scores["dev128"].get(name)
        if d128:
            delta = r["inrun_dev_bpb"] - d128["bpb"]
            out.append(f"consistency {name}: in-run bf16 {r['inrun_dev_bpb']:.4f} vs fp32 re-score {d128['bpb']:.4f}, delta {delta:+.4f}"
                       + ("  WARNING: above 0.01, find out why before reporting" if abs(delta) > CONSISTENCY_BPB else ""))
    if base:
        out.append("runner val_bpb (128 random windows, the number Runs 37-49 report): " + ", ".join(f"{a} {base[a]['runner_val_bpb']:.4f}" for a in ARMS if a in base))
    for r in runs.values():
        if r["contended"] or r["telemetry_lost"]:
            out.append(f"WARNING {r['name']}: contended={r['contended']} telemetry_lost={r['telemetry_lost']}; its train_seconds is not clean")
    if all(arm in base and base[arm]["name"] in scores["dev"] for arm in ARMS):
        for a, b in itertools.combinations(ARMS, 2):
            na, nb = base[a]["name"], base[b]["name"]
            gaps = ", ".join(f"{s} {scores[s][na]['bpb'] - scores[s][nb]['bpb']:+.3f}" for s in ("dev", "test", "legacy") if na in scores[s] and nb in scores[s])
            out.append(f"{a} - {b}: cold bpb {gaps}; train_seconds {base[a]['train_seconds']:,.0f} vs {base[b]['train_seconds']:,.0f} "
                       f"({base[a]['train_seconds'] / base[b]['train_seconds'] - 1:+.1%})")
        for r in sorted((r for r in runs.values() if r["tag"] == "eqsec"), key=lambda r: (r["seed"], ARMS.index(r["arm"]))):   # each equal-seconds rerun that is present
            matched = min((base[a] for a in ARMS if a != r["arm"]), key=lambda b: abs(b["train_seconds"] - r["train_seconds"]))   # the other arm's seed-0 run nearest in seconds
            out += [_gap_line(r, matched, "matched on train_seconds", scores), _gap_line(r, base[r["arm"]], "own arm, seed 0", scores)]
        rows = [{"arm": arm, "dev_bpb": scores["dev"][base[arm]["name"]]["bpb"], "train_seconds": base[arm]["train_seconds"]} for arm in ARMS]
        cmds = commands(decide(rows, base["hybrid"]["steps"]), device)
        out += ["", "follow-up runs the two rules call for (one unit, in this order):"] + (cmds or ["none: every pair is apart by 0.01 bits/byte and 5 % in seconds, and the better arm is not the slower"])
    else:
        out.append("incomplete: need the three seed-0 runs and their dev scores before the rules can be applied")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--device", default="cuda:1", help="device printed in the follow-up commands")
    args = p.parse_args()
    print(report(Path(args.results), args.device))


if __name__ == "__main__":
    main()
