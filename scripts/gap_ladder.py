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
import time
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
_ARMS_BY_NAME = {arm.name: arm for arm in ARMS}

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


def ladder_items(
    seeds, rung_seeds: dict[str, tuple[int, ...]] | None = None
) -> list[tuple[str, Arm, int]]:
    """Rung-major (RUNGS order), arm-major, seed-minor.

    `rung_seeds` overrides `seeds` for just the named rung(s); item order is unchanged
    (a narrowed rung simply contributes fewer seed-minor items in its own slot).
    """
    rung_seeds = rung_seeds or {}
    return [
        (rung, arm, seed)
        for rung in RUNGS
        for arm in ARMS
        for seed in rung_seeds.get(rung, seeds)
    ]


def _parse_rung_seeds(pairs: list[str] | None) -> dict[str, tuple[int, ...]]:
    """Parse repeated `--rung-seeds RUNG=S1,S2` into {rung: (seed, ...)}."""
    result: dict[str, tuple[int, ...]] = {}
    for pair in pairs or []:
        rung, _, seeds_str = pair.partition("=")
        result[rung] = tuple(int(s) for s in seeds_str.split(","))
    return result


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


def _run_status(root: Path, name: str) -> str:
    """Classify one run directory under `root` for --continue (arm_status-equivalent).

    "complete" when metrics.csv exists and its last row's step reached resolved_steps
    (via `_is_complete`); "partial" when the run directory exists but is not complete
    (missing or short metrics.csv); "absent" when the directory doesn't exist at all.
    """
    directory = root / name
    metrics_csv = directory / "metrics.csv"
    if metrics_csv.exists():
        with metrics_csv.open() as handle:
            rows = list(csv.DictReader(handle))
        if rows and _is_complete(rows[-1]):
            return "complete"
    if directory.exists():
        return "partial"
    return "absent"


def plan_continue(
    root: Path,
    items: list[tuple[str, Arm, int]],
    excluded: set[str],
    now: str,
    *,
    dry_run: bool = False,
) -> dict:
    """Classify every ladder item for --continue; move partial run dirs aside.

    Returns {"skipped": [names], "moved_aside": [{"name", "moved_to"}],
    "items": [remaining (rung, arm, seed) items]} in `items` order. Excluded items are
    dropped from the plan untouched regardless of status (a run still in flight on
    another card); complete items are skipped; partial items are moved aside to
    "root/.interrupted-<name>-<now>" (the run directory and its "<name>.log", never
    deleted) and requeued; absent items are queued as-is. With dry_run nothing is
    moved but the same plan is returned.
    """
    skipped: list[str] = []
    moved_aside: list[dict] = []
    remaining: list[tuple[str, Arm, int]] = []
    for rung, arm, seed in items:
        name = _run_name(rung, arm, seed)
        if name in excluded:
            continue
        status = _run_status(root, name)
        if status == "complete":
            skipped.append(name)
            continue
        if status == "partial":
            moved_to = f".interrupted-{name}-{now}"
            moved_aside.append({"name": name, "moved_to": moved_to})
            if not dry_run:
                (root / name).rename(root / moved_to)
                run_log = root / f"{name}.log"
                if run_log.exists():
                    run_log.rename(root / f"{moved_to}.log")
        remaining.append((rung, arm, seed))
    return {"skipped": skipped, "moved_aside": moved_aside, "items": remaining}


def _queue_for_gpu(queues: dict, gpu: int) -> list[str]:
    """Look up this gpu's pinned queue; a JSON round-trip turns int keys to strings."""
    if str(gpu) in queues:
        return queues[str(gpu)]
    return queues.get(gpu, [])


def _items_from_names(names: list[str]) -> list[tuple[str, Arm, int]]:
    """Map pinned run names back to (rung, arm, seed) via `_parse_run_name`."""
    items = []
    for name in names:
        parsed = _parse_run_name(name)
        if parsed is None:
            raise SystemExit(f"cannot parse pinned run name: {name}")
        rung, arm_name, seed_str = parsed
        items.append((rung, _ARMS_BY_NAME[arm_name], int(seed_str)))
    return items


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gaps(per_seed_raw: dict[str, dict[str, dict]], seeds: tuple[int, ...]) -> dict:
    full_seeds = [
        str(seed)
        for seed in seeds
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


def _summarize_rung(
    ratchet_weights: int, per_seed_raw: dict[str, dict[str, dict]], seeds: tuple[int, ...]
) -> dict:
    expected_seeds = {str(seed) for seed in seeds}
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
        "seeds": list(seeds),
        "complete": complete,
        "per_seed": per_seed,
        "gaps": _gaps(per_seed_raw, seeds),
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


def _effective_seeds_by_rung(root: Path) -> dict[str, tuple[int, ...]]:
    """Per-rung seeds for `summarize`.

    Reads `rung_seeds` from the manifest's last `continued` entry when present, else
    falls back to the manifest's flat `seeds` for every rung (7m, 99m, 25m); with no
    manifest at all (e.g. synthetic test fixtures), falls back to the module SEEDS.
    """
    tags = (*RUNGS, "25m")
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {tag: SEEDS for tag in tags}
    manifest = json.loads(manifest_path.read_text())
    default_seeds = tuple(manifest.get("seeds", SEEDS))
    continued = manifest.get("continued")
    rung_seeds = continued[-1].get("rung_seeds", {}) if continued else {}
    return {tag: tuple(rung_seeds[tag]) if tag in rung_seeds else default_seeds for tag in tags}


def summarize(root: Path, baseline_root: Path) -> dict:
    effective_seeds = _effective_seeds_by_rung(root)
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

    rungs = {
        tag: _summarize_rung(RATCHET_WEIGHTS[tag], by_rung.get(tag, {}), effective_seeds[tag])
        for tag in RUNGS
    }

    baseline = json.loads((baseline_root / "results.json").read_text())
    baseline_per_seed: dict[str, dict[str, dict]] = {}
    for seed in effective_seeds["25m"]:
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
    rungs["25m"] = _summarize_rung(
        RATCHET_WEIGHTS["25m"], baseline_per_seed, effective_seeds["25m"]
    )

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
    parser.add_argument(
        "--rung-seeds",
        action="append",
        default=None,
        metavar="RUNG=S1,S2",
        help="override --seeds for one rung only (repeatable)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="NAME",
        help="drop this run from the plan without touching its directory (repeatable)",
    )
    parser.add_argument("--continue", dest="continue_", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="with --continue, write the pinned plan and exit without spawning children",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def child_argv(args: argparse.Namespace, gpu: int) -> list[str]:
    """The argv for one per-card queue subprocess, re-parseable by build_parser()."""
    argv = [
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
    for pair in getattr(args, "rung_seeds", None) or []:
        argv += ["--rung-seeds", pair]
    for name in getattr(args, "exclude", None) or []:
        argv += ["--exclude", name]
    if getattr(args, "continue_", False):
        argv.append("--continue")
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.summarize_only:
        print(json.dumps(summarize(args.root, args.baseline_root)["verdict"]))
        return 0

    seeds = tuple(int(s) for s in args.seeds.split(","))
    rung_seeds = _parse_rung_seeds(args.rung_seeds)
    excluded = set(args.exclude or [])
    gpus = tuple(int(g) for g in args.gpus.split(","))
    manifest_path = args.root / "manifest.json"

    # Child mode, pinned plan: do NOT recompute -- read the last `continued` entry.
    if args.queue_gpu is not None and args.continue_:
        if not manifest_path.exists():
            print(f"{manifest_path} does not exist; nothing to continue", file=sys.stderr)
            return 2
        manifest = json.loads(manifest_path.read_text())
        continued = manifest.get("continued")
        if not continued:
            print(
                f"{manifest_path} has no 'continued' entry; run --continue first",
                file=sys.stderr,
            )
            return 2
        names = _queue_for_gpu(continued[-1]["queues"], args.queue_gpu)
        items = _items_from_names(names)
        return _run_queue(args.queue_gpu, args.root, items, args.dataset)

    items = ladder_items(seeds, rung_seeds)

    # Ordinary child mode: recompute the same assignment the parent used.
    if args.queue_gpu is not None:
        queues = assign_queues(items, gpus)
        return _run_queue(args.queue_gpu, args.root, queues[args.queue_gpu], args.dataset)

    if args.continue_:
        if not manifest_path.exists():
            raise SystemExit(f"{args.root} has no manifest; nothing to continue")
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        plan = plan_continue(args.root, items, excluded, now, dry_run=args.dry_run)
        queues = assign_queues(plan["items"], gpus)
        queues_by_name = {
            gpu: [_run_name(rung, arm, seed) for rung, arm, seed in pairs]
            for gpu, pairs in queues.items()
        }
        if args.dry_run:
            # Planned queues after skip/exclude; nothing written, nothing moved.
            print(json.dumps(queues_by_name, indent=2))
            return 0
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("continued", []).append(
            {
                "at": now,
                "rung_seeds": {rung: list(rung_seeds.get(rung, seeds)) for rung in RUNGS},
                "excluded": sorted(excluded),
                "skipped": plan["skipped"],
                "moved_aside": plan["moved_aside"],
                "queues": queues_by_name,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        if args.plan_only:
            print(json.dumps(queues_by_name, indent=2))
            return 0
        children = [subprocess.Popen(child_argv(args, gpu), cwd=REPO) for gpu in gpus]
        codes = [child.wait() for child in children]
        summary = summarize(args.root, args.baseline_root)
        print("verdict:", summary["verdict"])
        return 0 if all(code == 0 for code in codes) else 1

    if args.dry_run:
        # Printed in item order (round-robin by item order, matching assign_queues) rather
        # than grouped by queue, so the GPU column alternates 0/1/0/1/...
        for index, (rung, arm, seed) in enumerate(items):
            gpu = gpus[index % len(gpus)]
            print(gpu, " ".join(_command(rung, arm, seed, args.root, args.dataset)))
        return 0
    if manifest_path.exists():
        raise SystemExit(f"{args.root} already has a manifest; refusing to rerun into it")
    args.root.mkdir(parents=True, exist_ok=True)
    queues = assign_queues(items, gpus)
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
