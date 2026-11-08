#!/usr/bin/env python3
"""Efficiency suite: native param sizes, Pareto over (loss, size, wall).

Unlike born/ablation matched-param races, this does **not** force equal
parameter counts. Sharing arms keep their natural unique weights; baseline
keeps cfg.n_layer. Ranking is multi-metric (Pareto + eval/Mparam + wall).

Default corpus is **wikitext2** (capacity-bound). Structured synthetic pins
near its entropy floor so loss cannot separate arms — use it only for smoke.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrt.budget import BORN_ARMS
from scrt.train import aggregate_seeds, run_efficiency_comparison


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Efficiency suite (no ParamPad by default): "
            "baseline / looped / shared_residual / modulated / shared_plus_tiny"
        )
    )
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--run-name", default="efficiency")
    p.add_argument(
        "--corpus",
        default="wikitext2",
        choices=["structured", "synonym", "shakespeare", "wikitext2"],
        help="Default wikitext2 (capacity-bound). structured is smoke-only.",
    )
    p.add_argument("--results-dir", default=None)
    p.add_argument(
        "--arms",
        nargs="+",
        default=list(BORN_ARMS),
        help="Subset of arms to run",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to run sequentially (default: 0 1 2).",
    )
    p.add_argument(
        "--match-params",
        action="store_true",
        help="Force old matched-param protocol (default is native sizes).",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training; aggregate existing per-seed JSON.",
    )
    args = p.parse_args()

    if args.device == "cuda":
        args.device = (
            "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda:1"
        )

    results_dir = args.results_dir or "results"

    if args.corpus == "structured":
        print(
            "WARNING: structured corpus has ~2.14 nats/token entropy floor; "
            "arms pin near it and native-size Pareto degenerates to ranking "
            "by param count. Prefer --corpus wikitext2 for real claims."
        )

    print(
        f"device={args.device} steps={args.steps} corpus={args.corpus} "
        f"match_params={args.match_params} run_name={args.run_name} "
        f"seeds={args.seeds} arms={args.arms}"
    )

    if not args.aggregate_only:
        for seed in args.seeds:
            print(f"\n########## SEED {seed} ##########")
            run_efficiency_comparison(
                steps=args.steps,
                device=args.device,
                run_name=args.run_name,
                seed=seed,
                match_params=args.match_params,
                corpus=args.corpus,
                results_dir=results_dir,
                arms=tuple(args.arms),
            )

    aggregate_seeds(
        args.run_name,
        args.seeds,
        arms=args.arms,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
