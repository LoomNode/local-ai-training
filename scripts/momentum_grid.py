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


def best_so_far(metrics_csv: Path) -> dict:
    trace = []
    with metrics_csv.open() as handle:
        for row in csv.DictReader(handle):
            trace.append((int(row["step"]), float(row["validation_loss"])))
    best_step, best = min(trace, key=lambda item: item[1])
    return {"best": best, "best_step": best_step, "final": trace[-1][1], "trace": trace}


def summarize(root: Path) -> dict:
    arms = {}
    for directory in sorted(p for p in root.iterdir() if (p / "metrics.csv").exists()):
        arms[directory.name] = best_so_far(directory / "metrics.csv")
    plain, qat = arms.get("plain"), arms.get("qat")
    for name, record in arms.items():
        if plain and qat and name not in ("plain", "qat"):
            gap = plain["best"] - qat["best"]
            record["gap_closed"] = (plain["best"] - record["best"]) / gap if gap else None
    candidates = [n for n in arms if n not in ("plain", "qat")]
    winner = min(candidates, key=lambda n: arms[n]["best"]) if candidates else None
    result = {"winner": winner, "arms": arms}
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path, args: argparse.Namespace, arms: list[Arm]) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    payload = {
        "source_commit": commit, "git_status": status, "config": str(args.config),
        "config_sha256": _sha256(args.config), "dataset_sha256": _sha256(args.dataset),
        "seed": args.seed, "codes": CODES, "gpu": args.gpu, "arms": [asdict(a) for a in arms],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


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
    commands = {
        arm.name: arm_command(
            arm, config=args.config, dataset=args.dataset, output=args.root / arm.name,
            seed=args.seed, lat=lat,
        )
        for arm in arms
    }
    if args.dry_run:
        for name, cmd in commands.items():
            print(name, " ".join(cmd))
        return 0
    if (args.root / "manifest.json").exists():
        raise SystemExit(f"{args.root} already has a manifest; refusing to rerun into it")
    args.root.mkdir(parents=True)
    _manifest(args.root, args, arms)
    # thermal_guard.run_training pins CUDA_VISIBLE_DEVICES on the child itself from `gpu`;
    # do not set it here too, or a conflicting value could race its own assignment.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with (args.root / "study.log").open("a") as study:
        for name, cmd in commands.items():
            print(f"START {name}", file=study, flush=True)
            with (args.root / f"{name}.log").open("w") as log:
                result = run_training(cmd, log=log, env=env, gpu=args.gpu)
            print(f"EXIT {name} {result.returncode}", file=study, flush=True)
            if result.returncode != 0:
                print(f"STOPPED: {name} failed", file=study, flush=True)
                return 1
    summary = summarize(args.root)
    print("winner:", summary["winner"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
