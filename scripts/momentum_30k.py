"""Phase 2 of the momentum confirmation: converged 30k, three seeds, four matched arms.

Spec: docs/superpowers/specs/2026-09-13-momentum-confirmation-design.md. Splits 12 runs across
two per-card queues (each a subprocess of this script with --queue-gpu) under the thermal guard.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from scripts.momentum_grid import (
    CODES,
    Arm,
    arm_command,
    assign_queues,
    best_so_far,
    run_serial,
    write_manifest,
)

REPO = Path(__file__).resolve().parent.parent
SEEDS = (1337, 1338, 1339)
FLIP_THRESHOLD = 0.03


def confirmation_arms(leak: int, beta: float, seeds: tuple[int, ...]) -> list[tuple[Arm, int]]:
    arms = [
        Arm("momentum", "ratchet", leak, beta),
        Arm("plain", "ratchet", 0, 0.0),
        Arm("qat", "qat", 0, 0.0),
        Arm("fp32", "fp32", 0, 0.0),
    ]
    # Arm-major (not seed-major): each arm's 3-seed block has odd length, so the
    # index-parity round robin in assign_queues rotates which card an arm lands
    # on block-to-block, giving every card every arm type. A seed-major order
    # has even-length (4-arm) blocks, which locks each arm to one fixed card.
    return [(arm, seed) for arm in arms for seed in seeds]


def _run_name(arm: Arm, seed: int) -> str:
    return f"{arm.name}-seed{seed}"


def _command(arm: Arm, seed: int, root: Path, config: Path, dataset: Path) -> list[str]:
    cmd = arm_command(
        arm,
        config=config,
        dataset=dataset,
        output=root / _run_name(arm, seed),
        seed=seed,
        lat=REPO / ".venv/bin/lat",
    )
    if arm.weight_mode == "fp32" and "--codes" in cmd:  # FP32 control takes no code count
        index = cmd.index("--codes")
        del cmd[index : index + 2]
    return cmd


def _stats(values: list[float]) -> dict | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return {
        "mean": statistics.fmean(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else 0.0,
    }


def _last_row(metrics_csv: Path) -> dict[str, str]:
    with metrics_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1]


def _is_complete(last_row: dict[str, str]) -> bool:
    """A run is complete when its last row's step reached resolved_steps.

    resolved_steps may be absent from synthetic test fixtures (and possibly
    older metrics.csv files); treat that as "unknown" and assume complete
    rather than penalizing fixtures/runs that don't carry the column.
    """
    resolved = last_row.get("resolved_steps")
    if resolved in (None, ""):
        return True
    return int(last_row["step"]) == int(resolved)


def _fraction_recovered(momentum: float, plain: float, qat: float) -> float | None:
    gap = plain - qat
    return (plain - momentum) / gap if gap else None


def summarize(root: Path) -> dict:
    per_seed: dict[str, dict[str, dict]] = {}
    for directory in sorted(p for p in root.iterdir() if (p / "metrics.csv").exists()):
        name, seed = directory.name.rsplit("-seed", 1)
        metrics_csv = directory / "metrics.csv"
        record = best_so_far(metrics_csv)
        losses = [loss for _, loss in record["trace"]]
        record["last_four_mean"] = statistics.fmean(losses[-4:])
        record["complete"] = _is_complete(_last_row(metrics_csv))
        per_seed.setdefault(seed, {})[name] = record
    required = {"momentum", "plain", "qat", "fp32"}
    all_seeds = sorted(
        (s for s, arms in per_seed.items() if required <= arms.keys()), key=int
    )
    seeds = [s for s in all_seeds if all(per_seed[s][arm]["complete"] for arm in required)]
    excluded_seeds = [s for s in all_seeds if s not in seeds]

    def best(arm: str) -> list[float]:
        return [per_seed[s][arm]["best"] for s in seeds]

    momentum, plain, qat, fp32 = best("momentum"), best("plain"), best("qat"), best("fp32")
    gaps = (
        {
            "momentum_minus_qat": _stats(
                [m - q for m, q in zip(momentum, qat, strict=True)]
            ),
            "momentum_minus_plain": _stats(
                [m - p for m, p in zip(momentum, plain, strict=True)]
            ),
            "qat_minus_fp32": _stats(
                [q - f for q, f in zip(qat, fp32, strict=True)]
            ),
            "fraction_recovered": _stats(
                [
                    _fraction_recovered(m, p, q)
                    for m, p, q in zip(momentum, plain, qat, strict=True)
                ]
            ),
        }
        if seeds
        else {}
    )
    verdict = None
    if seeds:
        beats_plain = all(m < p for m, p in zip(momentum, plain, strict=True))
        if beats_plain and gaps["momentum_minus_qat"]["mean"] <= FLIP_THRESHOLD:
            verdict = "flipped"
        elif beats_plain:
            verdict = "partial"
        else:
            verdict = "not_confirmed"
    if excluded_seeds:
        verdict = None
    result = {"seeds": seeds, "per_seed": per_seed, "gaps": gaps, "verdict": verdict}
    if excluded_seeds:
        result["excluded_seeds"] = excluded_seeds
    (root / "results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def _run_queue(gpu: int, root: Path, pairs, config: Path, dataset: Path) -> int:
    named_commands = [
        (_run_name(arm, seed), _command(arm, seed, root, config, dataset)) for arm, seed in pairs
    ]
    return run_serial(named_commands, root, f"queue-gpu{gpu}.log", gpu)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "runs/momentum-30k-2026-09-13")
    parser.add_argument("--leak", type=int, required=False)
    parser.add_argument("--beta", type=float, required=False)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--queue-gpu", type=int, help="internal: run this card's queue only")
    parser.add_argument("--config", type=Path, default=REPO / "configs/scaleup_text8_25m_30k.toml")
    parser.add_argument("--dataset", type=Path, default=REPO / "data/text8/text8")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def child_argv(args: argparse.Namespace, gpu: int) -> list[str]:
    """The argv for one per-card queue subprocess, re-parseable by build_parser()."""
    return [
        sys.executable,
        "-m",
        "scripts.momentum_30k",
        "--root",
        str(args.root),
        "--leak",
        str(args.leak),
        "--beta",
        str(args.beta),
        "--gpus",
        args.gpus,
        "--queue-gpu",
        str(gpu),
        "--config",
        str(args.config),
        "--dataset",
        str(args.dataset),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.summarize_only:
        print(json.dumps(summarize(args.root)["verdict"]))
        return 0
    if args.leak is None or args.beta is None:
        parser.error("--leak and --beta (the Phase 1 winner) are required")
    gpus = tuple(int(g) for g in args.gpus.split(","))
    pairs = confirmation_arms(args.leak, args.beta, SEEDS)
    queues = assign_queues(pairs, gpus)
    if args.queue_gpu is not None:
        return _run_queue(
            args.queue_gpu, args.root, queues[args.queue_gpu], args.config, args.dataset
        )
    if args.dry_run:
        # Printed in pair order (round-robin by pair order, matching assign_queues)
        # rather than grouped by queue, so the GPU column alternates 0/1/0/1/...
        for index, (arm, seed) in enumerate(pairs):
            gpu = gpus[index % len(gpus)]
            print(gpu, " ".join(_command(arm, seed, args.root, args.config, args.dataset)))
        return 0
    if (args.root / "manifest.json").exists():
        raise SystemExit(f"{args.root} already has a manifest; refusing to rerun into it")
    args.root.mkdir(parents=True, exist_ok=True)
    write_manifest(
        args.root,
        config=args.config,
        dataset=args.dataset,
        extra={
            "leak": args.leak,
            "beta": args.beta,
            "codes": CODES,
            "seeds": SEEDS,
            "queues": {
                gpu: [[asdict(arm), seed] for arm, seed in pairs] for gpu, pairs in queues.items()
            },
        },
    )
    children = [
        subprocess.Popen(child_argv(args, gpu), cwd=REPO)
        for gpu in gpus
    ]
    codes = [child.wait() for child in children]
    summary = summarize(args.root)
    print("verdict:", summary["verdict"])
    return 0 if all(code == 0 for code in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
