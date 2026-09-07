#!/usr/bin/env python3
"""Capacity-bound suite: full char-level WikiText-2, multiple seeds, 4 arms.

The synthetic runs were data-bound (the ``structured`` corpus entropy rate is
2.1405 nats/token and every arm finished within 0.05 of it), so nothing could
separate. This drives the same arms on a corpus a 1.4M model cannot exhaust.

One process per GPU, then aggregate:

    python scripts/run_capacity.py --seeds 0 1 --device cuda:0 &
    python scripts/run_capacity.py --seeds 2   --device cuda:1 &
    wait
    python scripts/run_capacity.py --aggregate-only --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrt.train import ABLATION_ARMS, aggregate_seeds, run_capacity_comparison


def main() -> None:
    p = argparse.ArgumentParser(description="Capacity-bound multi-seed suite")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--run-name", default="capacity")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arms", nargs="+", default=list(ABLATION_ARMS))
    p.add_argument("--results-dir", default=None)
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training; just summarise existing per-seed JSON.",
    )
    args = p.parse_args()

    results_dir = args.results_dir or "results"

    if not args.aggregate_only:
        print(
            f"device={args.device} steps={args.steps} seeds={args.seeds} "
            f"arms={args.arms} run_name={args.run_name}"
        )
        for seed in args.seeds:
            run_capacity_comparison(
                steps=args.steps,
                device=args.device,
                run_name=args.run_name,
                seed=seed,
                arms=tuple(args.arms),
                results_dir=args.results_dir,
            )

    aggregate_seeds(
        args.run_name,
        args.seeds,
        arms=args.arms,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
