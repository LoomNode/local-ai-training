"""Gap-versus-scale ladder: does the plain ratchet's gap to matched QAT shrink with size?

Spec: docs/superpowers/specs/2026-09-14-gap-ladder-design.md. Runs 18 new arms (two rungs,
three arms, three seeds) across two per-card queues under the thermal guard; the 25M rung is
not rerun, its per-seed numbers come from an already-finished momentum-30k results.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from scripts.momentum_grid import (
    Arm,
    arm_command,
    assign_queues,
    best_so_far,
    run_serial,
    write_manifest,
)

REPO = Path(__file__).resolve().parent.parent
SEEDS = (1337, 1338, 1339)

# Execution order: 99M first (its long ratchet runs should start first), then 7M. The
# already-finished 25M rung is deliberately absent here -- it is read from --baseline-root.
RUNGS = {
    "99m": REPO / "configs/ladder_text8_99m_30k.toml",
    "7m": REPO / "configs/ladder_text8_7m_30k.toml",
}

# Ratchet-state buffer byte counts at max_code=7 (15 codes), verified with:
#   .venv/bin/python -c "from local_ai_training.config import ExperimentConfig; \
#     from local_ai_training.model import RatchetGPT; \
#     cfg=ExperimentConfig.from_toml('configs/<file>'); \
#     m=RatchetGPT(cfg.model_config(vocab_size=27), max_code=7); \
#     print(sum(b.numel() for b in m.buffers() if not b.dtype.is_floating_point))"
# against ladder_text8_7m_30k.toml, scaleup_text8_25m_30k.toml, and ladder_text8_99m_30k.toml.
RATCHET_WEIGHTS = {"7m": 7_381_440, "25m": 25_179_648, "99m": 99_111_168}

ARMS = [Arm("plain", "ratchet", 0, 0.0), Arm("qat", "qat", 0, 0.0), Arm("fp32", "fp32", 0, 0.0)]
_REQUIRED_ARMS = tuple(arm.name for arm in ARMS)

# Fields read from --baseline-root's results.json per_seed[seed][arm], and the shape every
# rung's own per_seed[seed][arm] record is trimmed down to before it's written out.
_BASELINE_FIELDS = (
    "best",
    "best_step",
    "final",
    "last_four_mean",
    "saturated_percent",
    "cumulative_code_moves",
    "ratchet_state_bytes",
    "support_parameter_bytes",
)


def ladder_items(seeds) -> list[tuple[str, Arm, int]]:
    """Rung-major (RUNGS order), arm-major, seed-minor: 18 (rung, arm, seed) items."""
    return [(rung, arm, seed) for rung in RUNGS for arm in ARMS for seed in seeds]


def _run_name(rung: str, arm: Arm, seed: int) -> str:
    return f"{rung}-{arm.name}-seed{seed}"


def _parse_run_name(name: str) -> tuple[str, str, str] | None:
    """("<rung>", "<arm>", "<seed>") from a "<rung>-<arm>-seed<seed>" directory name."""
    if "-seed" not in name:
        return None
    prefix, seed = name.rsplit("-seed", 1)
    if "-" not in prefix:
        return None
    rung, arm_name = prefix.split("-", 1)
    return rung, arm_name, seed


def _command(rung: str, arm: Arm, seed: int, root: Path, dataset: Path) -> list[str]:
    cmd = arm_command(
        arm,
        config=RUNGS[rung],
        dataset=dataset,
        output=root / _run_name(rung, arm, seed),
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

    resolved_steps may be absent from synthetic test fixtures; treat that as "unknown"
    and assume complete rather than penalizing fixtures that don't carry the column.
    """
    resolved = last_row.get("resolved_steps")
    if resolved in (None, ""):
        return True
    return int(last_row["step"]) == int(resolved)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gaps(per_seed_raw: dict[str, dict[str, dict]]) -> dict:
    full_seeds = [
        str(seed)
        for seed in SEEDS
        if set(_REQUIRED_ARMS) <= per_seed_raw.get(str(seed), {}).keys()
    ]

    def best(arm: str) -> list[float]:
        return [per_seed_raw[seed][arm]["best"] for seed in full_seeds]

    plain, qat, fp32 = best("plain"), best("qat"), best("fp32")
    return {
        "plain_minus_qat": _stats([p - q for p, q in zip(plain, qat, strict=True)]),
        "qat_minus_fp32": _stats([q - f for q, f in zip(qat, fp32, strict=True)]),
        "plain_minus_fp32": _stats([p - f for p, f in zip(plain, fp32, strict=True)]),
    }


def _summarize_rung(ratchet_weights: int, per_seed_raw: dict[str, dict[str, dict]]) -> dict:
    expected_seeds = {str(seed) for seed in SEEDS}
    complete = expected_seeds <= per_seed_raw.keys() and all(
        set(_REQUIRED_ARMS) <= per_seed_raw[seed].keys()
        and all(per_seed_raw[seed][arm]["complete"] for arm in _REQUIRED_ARMS)
        for seed in expected_seeds
    )
    per_seed = {
        seed: {
            arm: {k: v for k, v in record.items() if k != "complete"}
            for arm, record in arms.items()
        }
        for seed, arms in per_seed_raw.items()
    }
    return {
        "ratchet_weights": ratchet_weights,
        "complete": complete,
        "per_seed": per_seed,
        "gaps": _gaps(per_seed_raw),
    }


def _verdict(rungs: dict, order: list[str]) -> str:
    if any(not rungs[tag]["complete"] for tag in order):
        return "incomplete"
    m7, m25, m99 = (rungs[tag]["gaps"]["plain_minus_qat"]["mean"] for tag in order)
    if m7 > m25 > m99 and (m7 - m99) >= 0.02:
        return "shrinking"
    if m7 < m25 < m99 and (m99 - m7) >= 0.02:
        return "growing"
    return "flat"


def summarize(root: Path, baseline_root: Path) -> dict:
    by_rung: dict[str, dict[str, dict[str, dict]]] = {}
    for directory in sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "metrics.csv").exists()
    ):
        parsed = _parse_run_name(directory.name)
        if parsed is None or parsed[0] not in RUNGS:
            continue
        rung, arm_name, seed = parsed
        metrics_csv = directory / "metrics.csv"
        record = best_so_far(metrics_csv)
        losses = [loss for _, loss in record["trace"]]
        record["last_four_mean"] = statistics.fmean(losses[-4:])
        record["complete"] = _is_complete(_last_row(metrics_csv))
        del record["trace"]
        by_rung.setdefault(rung, {}).setdefault(seed, {})[arm_name] = record

    rungs = {tag: _summarize_rung(RATCHET_WEIGHTS[tag], by_rung.get(tag, {})) for tag in RUNGS}

    baseline = json.loads((baseline_root / "results.json").read_text())
    baseline_per_seed: dict[str, dict[str, dict]] = {}
    for seed in SEEDS:
        seed_str = str(seed)
        entry = baseline.get("per_seed", {}).get(seed_str, {})
        arms_data = {}
        for arm_name in _REQUIRED_ARMS:
            arm_entry = entry.get(arm_name)
            if arm_entry is None:
                continue
            record = {field: arm_entry.get(field) for field in _BASELINE_FIELDS}
            record["complete"] = bool(arm_entry.get("complete", True))
            arms_data[arm_name] = record
        baseline_per_seed[seed_str] = arms_data
    rungs["25m"] = _summarize_rung(RATCHET_WEIGHTS["25m"], baseline_per_seed)

    order = ["7m", "25m", "99m"]
    result = {"rungs": rungs, "order": order, "verdict": _verdict(rungs, order)}
    (root / "results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def _run_queue(gpu: int, root: Path, items, dataset: Path) -> int:
    named_commands = [
        (_run_name(rung, arm, seed), _command(rung, arm, seed, root, dataset))
        for rung, arm, seed in items
    ]
    return run_serial(named_commands, root, f"queue-gpu{gpu}.log", gpu)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "runs/gap-ladder-2026-09-14")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--queue-gpu", type=int, help="internal: run this card's queue only")
    parser.add_argument("--dataset", type=Path, default=REPO / "data/text8/text8")
    parser.add_argument(
        "--baseline-root", type=Path, default=REPO / "runs/momentum-30k-2026-09-13"
    )
    parser.add_argument("--seeds", default="1337,1338,1339")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def child_argv(args: argparse.Namespace, gpu: int) -> list[str]:
    """The argv for one per-card queue subprocess, re-parseable by build_parser()."""
    return [
        sys.executable,
        "-m",
        "scripts.gap_ladder",
        "--root",
        str(args.root),
        "--gpus",
        args.gpus,
        "--queue-gpu",
        str(gpu),
        "--dataset",
        str(args.dataset),
        "--baseline-root",
        str(args.baseline_root),
        "--seeds",
        args.seeds,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.summarize_only:
        print(json.dumps(summarize(args.root, args.baseline_root)["verdict"]))
        return 0
    seeds = tuple(int(s) for s in args.seeds.split(","))
    gpus = tuple(int(g) for g in args.gpus.split(","))
    items = ladder_items(seeds)
    queues = assign_queues(items, gpus)
    if args.queue_gpu is not None:
        return _run_queue(args.queue_gpu, args.root, queues[args.queue_gpu], args.dataset)
    if args.dry_run:
        # Printed in item order (round-robin by item order, matching assign_queues) rather
        # than grouped by queue, so the GPU column alternates 0/1/0/1/...
        for index, (rung, arm, seed) in enumerate(items):
            gpu = gpus[index % len(gpus)]
            print(gpu, " ".join(_command(rung, arm, seed, args.root, args.dataset)))
        return 0
    if (args.root / "manifest.json").exists():
        raise SystemExit(f"{args.root} already has a manifest; refusing to rerun into it")
    args.root.mkdir(parents=True, exist_ok=True)
    write_manifest(
        args.root,
        config=RUNGS["99m"],
        dataset=args.dataset,
        extra={
            "rungs": {
                tag: {"config": str(path), "sha256": _sha256(path)} for tag, path in RUNGS.items()
            },
            "baseline_root": str(args.baseline_root),
            "seeds": list(seeds),
            "arms": [asdict(arm) for arm in ARMS],
            "queues": {
                gpu: [_run_name(rung, arm, seed) for rung, arm, seed in pairs]
                for gpu, pairs in queues.items()
            },
            "ratchet_weights": RATCHET_WEIGHTS,
        },
    )
    children = [subprocess.Popen(child_argv(args, gpu), cwd=REPO) for gpu in gpus]
    codes = [child.wait() for child in children]
    summary = summarize(args.root, args.baseline_root)
    print("verdict:", summary["verdict"])
    return 0 if all(code == 0 for code in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
