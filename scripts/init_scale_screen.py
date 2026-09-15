"""Init-scale multiplier screen: does starting every row's scale at a multiple k of the
current rule (row_max_abs / max_code) lower best-so-far validation loss versus k = 1?

References (the momentum grid's `plain` and `qat` cells) are copied from
`--reference-root/results.json` rather than rerun. The k = 1 cell must reproduce the
reference plain cell's best-so-far exactly (the 25M/256-token model is bit-repeatable);
`identity_check` in the summary reports this.

Spec: docs/superpowers/specs/2026-09-15-init-scale-screen-design.md. Reuses
`run_serial`, `write_manifest`, `assign_queues`, and `best_so_far` from
`scripts.momentum_grid` -- `momentum_grid.REPO` is the checkout that contains this
script (this worktree), so the driver runs this worktree's own `.venv/bin/lat`. The
`--queue-gpu N` child mode mirrors scripts/gap_ladder.py's: the parent writes the
manifest once, then spawns one child per GPU that recomputes the same
`assign_queues` split and runs its own slice serially under the thermal guard.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.momentum_grid import assign_queues, best_so_far, run_serial, write_manifest

REPO = Path(__file__).resolve().parent.parent
CODES = 15
SEED = 1337
# Selection rule: the best cell must beat plain by more than this many nats to advance
# (same margin as the update-rule lever screen).
ADVANCE_MARGIN = 0.005
SPEC_PATH = REPO / "docs/superpowers/specs/2026-09-15-init-scale-screen-design.md"


@dataclass(frozen=True)
class Cell:
    name: str
    k: int


CELLS: tuple[Cell, ...] = tuple(Cell(f"k{k}", k) for k in (1, 2, 3, 4, 6, 8))


def cell_command(
    cell: Cell, *, config: Path, dataset: Path, output: Path, seed: int, lat: Path
) -> list[str]:
    cmd = [
        str(lat), "train", "--config", str(config), "--dataset-path", str(dataset),
        "--seed", str(seed), "--codes", str(CODES), "--weight-mode", "ratchet",
        "--output", str(output),
    ]
    if cell.k != 1:
        cmd += ["--scale-multiplier", str(cell.k)]
    return cmd


def _last_row(metrics_csv: Path) -> dict[str, str]:
    with metrics_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1]


def _tail_shares(code_histogram_json: str) -> dict[str, float]:
    """Shares (0-100) of code mass at |code| >= 5 and |code| <= 1 from a code_histogram
    column (a JSON dict of code -> count, as written by metrics.collect_ratchet_metrics)."""
    counts = json.loads(code_histogram_json) if code_histogram_json else {}
    total = sum(counts.values())
    if not total:
        return {"tail_ge5": 0.0, "tail_le1": 0.0}
    ge5 = sum(count for code, count in counts.items() if abs(int(code)) >= 5)
    le1 = sum(count for code, count in counts.items() if abs(int(code)) <= 1)
    return {"tail_ge5": 100.0 * ge5 / total, "tail_le1": 100.0 * le1 / total}


def _cell_metrics(metrics_csv: Path) -> dict:
    record = best_so_far(metrics_csv)
    last = _last_row(metrics_csv)
    record["saturated_percent"] = float(last.get("saturated_percent") or 0.0)
    record["zero_percent"] = float(last.get("zero_percent") or 0.0)
    record["move_percent"] = float(last.get("move_percent") or 0.0)
    record.update(_tail_shares(last.get("code_histogram", "{}")))
    return record


def summarize(root: Path, reference_root: Path) -> dict:
    cell_names = {cell.name for cell in CELLS}
    cells = {}
    for directory in sorted(
        p for p in root.iterdir() if p.name in cell_names and (p / "metrics.csv").exists()
    ):
        cells[directory.name] = _cell_metrics(directory / "metrics.csv")

    reference_arms = json.loads((reference_root / "results.json").read_text())["arms"]
    references = {"plain": reference_arms["plain"], "qat": reference_arms["qat"]}
    plain_best = references["plain"]["best"]
    qat_best = references["qat"]["best"]

    winner = min(cells, key=lambda name: cells[name]["best"]) if cells else None
    winner_best = cells[winner]["best"] if winner is not None else None
    advances = winner_best < plain_best - ADVANCE_MARGIN if winner_best is not None else None
    gap = plain_best - qat_best
    gap_closed = (
        (plain_best - winner_best) / gap if winner_best is not None and gap else None
    )

    k1_best = cells["k1"]["best"] if "k1" in cells else None
    identity_check = {
        "k1_best": k1_best,
        "reference_plain_best": plain_best,
        "identical": abs(k1_best - plain_best) < 1e-6 if k1_best is not None else False,
    }

    complete = cell_names <= cells.keys()
    verdict = "incomplete"
    if complete:
        verdict = "advances" if advances else "no-advance"

    result = {
        "cells": cells,
        "references": references,
        "winner": winner,
        "winner_best": winner_best,
        "advances": advances,
        "gap_closed": gap_closed,
        "identity_check": identity_check,
        "verdict": verdict,
    }
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


_TABLE_COLUMNS = (
    ("best", "best"),
    ("best_step", "best_step"),
    ("final", "final"),
    ("saturated_percent", "saturated %"),
    ("tail_ge5", "tail>=5 %"),
    ("tail_le1", "tail<=1 %"),
    ("move_percent", "move %"),
)


def format_table(cells: dict) -> str:
    """A Markdown table (cell, best, best_step, final, saturated %, tail>=5 %, tail<=1 %,
    move %) for the cells summarize() computed."""
    headers = ["cell", *(header for _, header in _TABLE_COLUMNS)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for name in sorted(cells):
        record = cells[name]
        row = [name]
        for key, _ in _TABLE_COLUMNS:
            value = record.get(key)
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("/games/ailab/local-ai-training/runs/init-scale-screen-2026-09-15"),
    )
    parser.add_argument("--config", type=Path, default=REPO / "configs/scaleup_text8_25m_5k.toml")
    parser.add_argument(
        "--dataset", type=Path, default=Path("/games/ailab/local-ai-training/data/text8/text8")
    )
    parser.add_argument(
        "--reference-root", type=Path,
        default=Path("/games/ailab/local-ai-training/runs/momentum-grid-2026-09-13"),
    )
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--queue-gpu", type=int, help="internal: run this card's queue only")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def child_argv(args: argparse.Namespace, gpu: int) -> list[str]:
    """The argv for one per-card queue subprocess, re-parseable by build_parser()."""
    return [
        sys.executable, "-m", "scripts.init_scale_screen",
        "--root", str(args.root),
        "--config", str(args.config),
        "--dataset", str(args.dataset),
        "--reference-root", str(args.reference_root),
        "--gpus", args.gpus,
        "--queue-gpu", str(gpu),
    ]


def _run_queue(gpu: int, root: Path, cells: list[Cell], config: Path, dataset: Path) -> int:
    lat = REPO / ".venv/bin/lat"
    named_commands = [
        (
            cell.name,
            cell_command(
                cell, config=config, dataset=dataset, output=root / cell.name,
                seed=SEED, lat=lat,
            ),
        )
        for cell in cells
    ]
    return run_serial(named_commands, root, f"queue-gpu{gpu}.log", gpu)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.summarize_only:
        summary = summarize(args.root, args.reference_root)
        print(format_table(summary["cells"]))
        print("winner:", summary["winner"])
        print("verdict:", summary["verdict"])
        return 0

    gpus = tuple(int(g) for g in args.gpus.split(","))
    lat = REPO / ".venv/bin/lat"

    if args.dry_run:
        for cell in CELLS:
            cmd = cell_command(
                cell, config=args.config, dataset=args.dataset,
                output=args.root / cell.name, seed=SEED, lat=lat,
            )
            print(cell.name, " ".join(cmd))
        return 0

    # Child mode: recompute the same queue assignment the parent used and run just
    # this card's slice, exactly like scripts/gap_ladder.py's --queue-gpu.
    if args.queue_gpu is not None:
        queues = assign_queues(list(CELLS), gpus)
        return _run_queue(
            args.queue_gpu, args.root, queues[args.queue_gpu], args.config, args.dataset
        )

    manifest_path = args.root / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"{args.root} already has a manifest; refusing to rerun into it")
    args.root.mkdir(parents=True, exist_ok=True)
    write_manifest(
        args.root, config=args.config, dataset=args.dataset,
        extra={
            "seed": SEED, "codes": CODES, "gpus": list(gpus),
            "cells": [asdict(cell) for cell in CELLS],
            "reference_root": str(args.reference_root),
            "spec_path": str(SPEC_PATH),
        },
    )
    children = [subprocess.Popen(child_argv(args, gpu), cwd=REPO) for gpu in gpus]
    codes = [child.wait() for child in children]
    summary = summarize(args.root, args.reference_root)
    print(format_table(summary["cells"]))
    print("winner:", summary["winner"])
    print("verdict:", summary["verdict"])
    if summary["verdict"] != "incomplete" and not summary["identity_check"]["identical"]:
        print(
            "ERROR: k1 does not reproduce the reference plain best-so-far exactly: "
            f"{summary['identity_check']}",
            file=sys.stderr,
        )
        return 1
    return 0 if all(code == 0 for code in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
