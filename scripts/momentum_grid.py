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

from local_ai_training.config import ExperimentConfig
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
    # Restrict to directories whose name is exactly an arm name -- a --continue run
    # moves partial arms aside to "<arm>.interrupted-<UTC>" (never deleting them),
    # and those keep their own metrics.csv, so a plain glob would wrongly score them
    # as extra candidates (the leak/beta regex even matches their name prefix).
    arm_names = {arm.name for arm in grid_arms()}
    arms = {}
    for directory in sorted(
        p for p in root.iterdir() if p.name in arm_names and (p / "metrics.csv").exists()
    ):
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


def assign_queues(items, gpus: tuple[int, ...]) -> dict[int, list]:
    """Split `items` round-robin across `gpus`, in index order.

    Shared by the Phase 1 grid (arms) and the Phase 2 confirmation (arm, seed
    pairs) drivers. With a single gpu this returns the items in their
    original order under that one key -- today's serial behavior.
    """
    queues: dict[int, list] = {gpu: [] for gpu in gpus}
    for index, item in enumerate(items):
        queues[gpus[index % len(gpus)]].append(item)
    return queues


def arm_status(root: Path, arm: str, expected_steps: int) -> str:
    """Classify one arm directory under `root` for --continue.

    "complete" when metrics.csv exists and its last row's step equals
    expected_steps; "partial" when the arm directory exists but is not
    complete (missing/short metrics.csv); "absent" when the directory
    doesn't exist at all.
    """
    directory = root / arm
    metrics_csv = directory / "metrics.csv"
    if metrics_csv.exists():
        with metrics_csv.open() as handle:
            rows = list(csv.DictReader(handle))
        if rows and int(rows[-1]["step"]) == expected_steps:
            return "complete"
    if directory.exists():
        return "partial"
    return "absent"


def plan_continue(
    root: Path, arms: list[Arm], expected_steps: int, now: str, *, dry_run: bool = False
) -> dict:
    """Classify every arm for --continue and move partial ones aside.

    Returns {"skipped": [names], "moved_aside": [{"arm", "moved_to"}],
    "queued": [names]} in `arms` order. Complete arms are skipped; partial
    arms are moved aside to "<arm>.interrupted-<now>" (directory and its
    "<arm>.log", never deleted) and requeued; absent arms are queued as-is.
    With dry_run, nothing is moved but the same plan is returned.
    """
    skipped: list[str] = []
    moved_aside: list[dict] = []
    queued: list[str] = []
    for arm in arms:
        status = arm_status(root, arm.name, expected_steps)
        if status == "complete":
            skipped.append(arm.name)
            continue
        if status == "partial":
            moved_to = f"{arm.name}.interrupted-{now}"
            moved_aside.append({"arm": arm.name, "moved_to": moved_to})
            if not dry_run:
                (root / arm.name).rename(root / moved_to)
                arm_log = root / f"{arm.name}.log"
                if arm_log.exists():
                    arm_log.rename(root / f"{moved_to}.log")
        queued.append(arm.name)
    return {"skipped": skipped, "moved_aside": moved_aside, "queued": queued}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "runs/momentum-grid-2026-09-13")
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="deprecated: alias for --gpus with a single physical GPU index",
    )
    parser.add_argument(
        "--gpus", default=None, help="comma-separated physical GPU indices, e.g. 0,1"
    )
    parser.add_argument("--queue-gpu", type=int, help="internal: run this card's queue only")
    parser.add_argument("--config", type=Path, default=REPO / "configs/scaleup_text8_25m_5k.toml")
    parser.add_argument("--dataset", type=Path, default=REPO / "data/text8/text8")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--continue", dest="continue_", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def resolve_gpus(args: argparse.Namespace) -> tuple[int, ...]:
    """The gpus tuple for this invocation: --gpu (deprecated) beats --gpus beats default."""
    if args.gpu is not None:
        return (args.gpu,)
    if args.gpus is not None:
        return tuple(int(g) for g in args.gpus.split(","))
    return (0,)


def child_argv(args: argparse.Namespace, gpu: int) -> list[str]:
    """The argv for one per-card queue subprocess, re-parseable by build_parser()."""
    argv = [
        sys.executable,
        "-m",
        "scripts.momentum_grid",
        "--root",
        str(args.root),
        "--gpus",
        ",".join(str(g) for g in resolve_gpus(args)),
        "--config",
        str(args.config),
        "--dataset",
        str(args.dataset),
        "--seed",
        str(args.seed),
        "--queue-gpu",
        str(gpu),
    ]
    if getattr(args, "continue_", False):
        argv.append("--continue")
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.summarize_only:
        print(json.dumps(summarize(args.root)["winner"]))
        return 0

    arms = grid_arms()
    gpus = resolve_gpus(args)
    lat = REPO / ".venv/bin/lat"

    def command_for(arm: Arm) -> list[str]:
        return arm_command(
            arm, config=args.config, dataset=args.dataset, output=args.root / arm.name,
            seed=args.seed, lat=lat,
        )

    manifest_path = args.root / "manifest.json"
    multi = len(gpus) > 1 or args.continue_

    # A per-card child spawned by the multi-gpu/--continue parent below: recompute
    # the same queue assignment the parent used and run just this card's slice.
    if args.queue_gpu is not None:
        if args.continue_:
            manifest = json.loads(manifest_path.read_text())
            queued_names = manifest["continued"][-1]["queued"]
            by_name = {arm.name: arm for arm in arms}
            queue_arms = [by_name[name] for name in queued_names]
        else:
            queue_arms = arms
        queues = assign_queues(queue_arms, gpus)
        named_commands = [
            (arm.name, command_for(arm)) for arm in queues.get(args.queue_gpu, [])
        ]
        log_name = f"queue-gpu{args.queue_gpu}.log"
        return run_serial(named_commands, args.root, log_name, args.queue_gpu)

    if not multi:
        # Today's exact single-GPU serial behavior: no child processes, no gpu
        # column in dry-run output, log name "study.log".
        named_commands = [(arm.name, command_for(arm)) for arm in arms]
        if args.dry_run:
            for name, cmd in named_commands:
                print(name, " ".join(cmd))
            return 0
        if manifest_path.exists():
            raise SystemExit(f"{args.root} already has a manifest; refusing to rerun into it")
        args.root.mkdir(parents=True, exist_ok=True)
        write_manifest(
            args.root,
            config=args.config,
            dataset=args.dataset,
            extra={
                "seed": args.seed, "codes": CODES, "gpus": list(gpus),
                "arms": [asdict(a) for a in arms],
            },
        )
        # thermal_guard.run_training pins CUDA_VISIBLE_DEVICES on the child itself from
        # `gpu`; do not set it here too, or a conflicting value could race its own assignment.
        exit_code = run_serial(named_commands, args.root, "study.log", gpus[0])
        if exit_code != 0:
            return exit_code
        summary = summarize(args.root)
        print("winner:", summary["winner"])
        return 0

    # Multi-gpu and/or --continue: classify (and possibly move partial arms aside)
    # before any child starts, so every child derives an identical queue.
    if args.continue_:
        if not manifest_path.exists():
            raise SystemExit("nothing to continue")
        expected_steps = ExperimentConfig.from_toml(args.config).steps
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        plan = plan_continue(args.root, arms, expected_steps, now, dry_run=args.dry_run)
        by_name = {arm.name: arm for arm in arms}
        queued_arms = [by_name[name] for name in plan["queued"]]
        if args.dry_run:
            print("skipped:", plan["skipped"])
            print("moved_aside:", plan["moved_aside"])
            for index, arm in enumerate(queued_arms):
                gpu = gpus[index % len(gpus)]
                print(gpu, arm.name, " ".join(command_for(arm)))
            return 0
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("continued", []).append(
            {
                "at": now, "gpus": list(gpus), "skipped": plan["skipped"],
                "moved_aside": plan["moved_aside"], "queued": plan["queued"],
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    else:
        if args.dry_run:
            for index, arm in enumerate(arms):
                gpu = gpus[index % len(gpus)]
                print(gpu, arm.name, " ".join(command_for(arm)))
            return 0
        if manifest_path.exists():
            raise SystemExit(f"{args.root} already has a manifest; refusing to rerun into it")
        args.root.mkdir(parents=True, exist_ok=True)
        write_manifest(
            args.root,
            config=args.config,
            dataset=args.dataset,
            extra={
                "seed": args.seed, "codes": CODES, "gpus": list(gpus),
                "arms": [asdict(a) for a in arms],
            },
        )

    children = [subprocess.Popen(child_argv(args, gpu), cwd=REPO) for gpu in gpus]
    codes = [child.wait() for child in children]
    if any(code != 0 for code in codes):
        return 1
    summary = summarize(args.root)
    print("winner:", summary["winner"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
