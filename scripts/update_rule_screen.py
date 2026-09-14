"""Phase 1 screen for the two zero-state update-rule levers (stochastic bucketing,
pressure-weighted effective weight): 5 cells at the 5k config, run serially on one GPU.

References (the momentum grid's `plain` and `qat` cells) are copied from
`--reference-root/results.json` rather than rerun.

Spec: docs/superpowers/specs/2026-09-14-update-rule-levers-design.md. Reuses
`run_serial`, `write_manifest`, and `best_so_far` from `scripts.momentum_grid` --
`momentum_grid.REPO` is the checkout that contains this script (this worktree),
so the driver runs this worktree's own `.venv/bin/lat`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.momentum_grid import best_so_far, run_serial, write_manifest

REPO = Path(__file__).resolve().parent.parent
CODES = 15
SEED = 1337
# Selection rule: a cell must beat plain by more than this many nats to advance.
ADVANCE_MARGIN = 0.005


@dataclass(frozen=True)
class Cell:
    name: str
    stochastic: bool
    pressure_weight: float


CELLS: tuple[Cell, ...] = (
    Cell("stoch", True, 0.0),
    Cell("pw0.5", False, 0.5),
    Cell("pw1.0", False, 1.0),
    Cell("stoch-pw0.5", True, 0.5),
    Cell("stoch-pw1.0", True, 1.0),
)


def cell_command(
    cell: Cell, *, config: Path, dataset: Path, output: Path, seed: int, lat: Path
) -> list[str]:
    cmd = [
        str(lat), "train", "--config", str(config), "--dataset-path", str(dataset),
        "--seed", str(seed), "--codes", str(CODES), "--weight-mode", "ratchet",
        "--output", str(output),
    ]
    if cell.stochastic:
        cmd += ["--stochastic-bucket"]
    if cell.pressure_weight:
        cmd += ["--pressure-weight", str(cell.pressure_weight)]
    return cmd


def summarize(root: Path, reference_root: Path) -> dict:
    cell_names = {cell.name for cell in CELLS}
    cells = {}
    for directory in sorted(
        p for p in root.iterdir() if p.name in cell_names and (p / "metrics.csv").exists()
    ):
        cells[directory.name] = best_so_far(directory / "metrics.csv")

    reference_arms = json.loads((reference_root / "results.json").read_text())["arms"]
    references = {"plain": reference_arms["plain"], "qat": reference_arms["qat"]}

    winner = min(cells, key=lambda name: cells[name]["best"]) if cells else None
    advances = None
    gap_closed = None
    if winner is not None:
        plain_best = references["plain"]["best"]
        qat_best = references["qat"]["best"]
        winner_best = cells[winner]["best"]
        advances = winner_best < plain_best - ADVANCE_MARGIN
        gap = plain_best - qat_best
        gap_closed = (plain_best - winner_best) / gap if gap else None

    result = {
        "cells": cells,
        "references": references,
        "winner": winner,
        "advances": advances,
        "gap_closed": gap_closed,
    }
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("/games/ailab/local-ai-training/runs/update-rule-screen-2026-09-14"),
    )
    parser.add_argument("--config", type=Path, default=REPO / "configs/scaleup_text8_25m_5k.toml")
    parser.add_argument(
        "--dataset", type=Path, default=Path("/games/ailab/local-ai-training/data/text8/text8")
    )
    parser.add_argument(
        "--reference-root", type=Path,
        default=Path("/games/ailab/local-ai-training/runs/momentum-grid-2026-09-13"),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.summarize_only:
        summary = summarize(args.root, args.reference_root)
        print("winner:", summary["winner"])
        return 0

    lat = REPO / ".venv/bin/lat"
    named_commands = [
        (
            cell.name,
            cell_command(
                cell, config=args.config, dataset=args.dataset,
                output=args.root / cell.name, seed=SEED, lat=lat,
            ),
        )
        for cell in CELLS
    ]

    if args.dry_run:
        for name, cmd in named_commands:
            print(name, " ".join(cmd))
        return 0

    manifest_path = args.root / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"{args.root} already has a manifest; refusing to rerun into it")
    args.root.mkdir(parents=True, exist_ok=True)
    write_manifest(
        args.root, config=args.config, dataset=args.dataset,
        extra={
            "seed": SEED, "codes": CODES, "gpu": args.gpu,
            "cells": [asdict(cell) for cell in CELLS],
            "reference_root": str(args.reference_root),
        },
    )
    exit_code = run_serial(named_commands, args.root, "study.log", args.gpu)
    if exit_code != 0:
        return exit_code
    summary = summarize(args.root, args.reference_root)
    print("winner:", summary["winner"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
