"""Phase 1 of the momentum confirmation: grid the two update-rule knobs at the 5k screen.

Spec: docs/superpowers/specs/2026-09-13-momentum-confirmation-design.md. Runs 17 matched arms
serially on one GPU under the thermal guard, records a manifest, and summarizes results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.thermal_guard import run_training

REPO = Path(__file__).resolve().parent.parent
LEAKS = (8, 12, 16, 24, 32)
BETAS = (0.9, 0.99, 0.999)
CODES = 15
# Phase 1 selection rule: cells within this many nats of the minimum best@5k
# are considered tied; the tie is broken by preferring the larger leak period.
TIE_TOLERANCE = 0.005
_LEAK_FROM_NAME = re.compile(r"^leak(\d+)-beta")


@dataclass(frozen=True)
class Arm:
    name: str
    weight_mode: str
    leak: int
    beta: float


def grid_arms() -> list[Arm]:
    arms = [Arm("plain", "ratchet", 0, 0.0), Arm("qat", "qat", 0, 0.0)]
    arms += [Arm(f"leak{k}-beta{b}", "ratchet", k, b) for k in LEAKS for b in BETAS]
    return arms


def arm_command(
    arm: Arm, *, config: Path, dataset: Path, output: Path, seed: int, lat: Path
) -> list[str]:
    cmd = [
        str(lat), "train", "--config", str(config), "--dataset-path", str(dataset),
        "--seed", str(seed), "--codes", str(CODES), "--weight-mode", arm.weight_mode,
        "--output", str(output),
    ]
    if arm.leak:
        cmd += ["--pressure-leak-period", str(arm.leak)]
    if arm.beta:
        cmd += ["--rms-ema-beta", str(arm.beta)]
    return cmd


# Phase 2 metrics read from the final row when the column exists; None otherwise.
_OPTIONAL_FINAL_ROW_COLUMNS = (
    "saturated_percent",
    "cumulative_code_moves",
    "ratchet_state_bytes",
    "support_parameter_bytes",
)


def best_so_far(metrics_csv: Path) -> dict:
    trace = []
    rows = []
    with metrics_csv.open() as handle:
        for row in csv.DictReader(handle):
            trace.append((int(row["step"]), float(row["validation_loss"])))
            rows.append(row)
    best_step, best = min(trace, key=lambda item: item[1])
    last_row = rows[-1]
    result = {"best": best, "best_step": best_step, "final": trace[-1][1], "trace": trace}
    for column in _OPTIONAL_FINAL_ROW_COLUMNS:
        value = last_row.get(column)
        result[column] = float(value) if value not in (None, "") else None
    return result


def _leak_from_cell_name(name: str) -> int:
    match = _LEAK_FROM_NAME.match(name)
    return int(match.group(1)) if match else -1


def summarize(root: Path) -> dict:
    arms = {}
    for directory in sorted(p for p in root.iterdir() if (p / "metrics.csv").exists()):
        arms[directory.name] = best_so_far(directory / "metrics.csv")
    plain, qat = arms.get("plain"), arms.get("qat")
    for name, record in arms.items():
        if plain and qat and name not in ("plain", "qat"):
            gap = plain["best"] - qat["best"]
            # gap is plain-minus-qat; a negative gap (QAT worse than plain) makes
            # gap_closed sign-inverted and meaningless, so it's reported as None
            # the same way a zero gap (division by zero) is.
            record["gap_closed"] = (plain["best"] - record["best"]) / gap if gap else None
    candidates = [n for n in arms if n not in ("plain", "qat")]
    tie_set: list[str] = []
    winner = None
    if candidates:
        min_best = min(arms[n]["best"] for n in candidates)
        tie_set = sorted(
            n for n in candidates if arms[n]["best"] - min_best <= TIE_TOLERANCE
        )
        # Selection rule: among cells within TIE_TOLERANCE nats of the minimum,
        # prefer the largest leak period (weaker forgetting); ties on leak are
        # broken by the lowest best.
        winner = min(tie_set, key=lambda n: (-_leak_from_cell_name(n), arms[n]["best"]))
    result = {"winner": winner, "tie_set": tie_set, "arms": arms}
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(root: Path, *, config: Path, dataset: Path, extra: dict) -> None:
    """Write manifest.json with provenance shared by both Phase 1 and Phase 2 drivers.

    Records source commit, working-tree status, config path + sha256, dataset
    sha256, and a start timestamp; `extra` (each driver's own fields, e.g. seed
    and arms for the grid, or leak/beta/seeds/queues for the 30k confirmation)
    is merged in on top.
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    payload = {
        "source_commit": commit, "git_status": status, "config": str(config),
        "config_sha256": _sha256(config), "dataset_sha256": _sha256(dataset),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload.update(extra)
    (root / "manifest.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


def run_serial(
    named_commands: list[tuple[str, list[str]]], root: Path, log_name: str, gpu: int
) -> int:
    """Run commands one at a time under the thermal guard, stopping on first failure.

    Shared by the Phase 1 grid (serial on one card) and each Phase 2 per-card
    queue. Writes START/EXIT lines (and STOPPED on failure) to `root/log_name`
    and each command's own `root/{name}.log`.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with (root / log_name).open("a") as study:
        for name, cmd in named_commands:
            print(f"START {name}", file=study, flush=True)
            with (root / f"{name}.log").open("w") as log:
                result = run_training(cmd, log=log, env=env, gpu=gpu)
            print(f"EXIT {name} {result.returncode}", file=study, flush=True)
            if result.returncode != 0:
                print(f"STOPPED: {name} failed", file=study, flush=True)
                return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "runs/momentum-grid-2026-09-13")
    parser.add_argument("--gpu", type=int, default=0, help="physical GPU index")
    parser.add_argument("--config", type=Path, default=REPO / "configs/scaleup_text8_25m_5k.toml")
    parser.add_argument("--dataset", type=Path, default=REPO / "data/text8/text8")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    arms = grid_arms()
    if args.summarize_only:
        print(json.dumps(summarize(args.root)["winner"]))
        return 0
    lat = REPO / ".venv/bin/lat"
    named_commands = [
        (
            arm.name,
            arm_command(
                arm, config=args.config, dataset=args.dataset, output=args.root / arm.name,
                seed=args.seed, lat=lat,
            ),
        )
        for arm in arms
    ]
    if args.dry_run:
        for name, cmd in named_commands:
            print(name, " ".join(cmd))
        return 0
    if (args.root / "manifest.json").exists():
        raise SystemExit(f"{args.root} already has a manifest; refusing to rerun into it")
    args.root.mkdir(parents=True, exist_ok=True)
    write_manifest(
        args.root,
        config=args.config,
        dataset=args.dataset,
        extra={
            "seed": args.seed, "codes": CODES, "gpu": args.gpu,
            "arms": [asdict(a) for a in arms],
        },
    )
    # thermal_guard.run_training pins CUDA_VISIBLE_DEVICES on the child itself from `gpu`;
    # do not set it here too, or a conflicting value could race its own assignment.
    exit_code = run_serial(named_commands, args.root, "study.log", args.gpu)
    if exit_code != 0:
        return exit_code
    summary = summarize(args.root)
    print("winner:", summary["winner"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
