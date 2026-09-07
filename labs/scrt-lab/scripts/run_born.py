#!/usr/bin/env python3
"""Born-consolidated experiment: share by construction (no merge machinery)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrt.train import run_born_comparison


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Born-consolidated suite: baseline / looped / shared_residual / "
            "modulated / shared_plus_tiny"
        )
    )
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--run-name", default="born")
    p.add_argument(
        "--corpus",
        default="structured",
        choices=["structured", "synonym", "shakespeare", "wikitext2"],
    )
    p.add_argument("--results-dir", default=None)
    p.add_argument("--no-match-params", action="store_true")
    args = p.parse_args()

    if args.device == "cuda":
        args.device = (
            "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda:1"
        )

    print(
        f"device={args.device} steps={args.steps} corpus={args.corpus} "
        f"match_params={not args.no_match_params} run_name={args.run_name}"
    )
    run_born_comparison(
        steps=args.steps,
        device=args.device,
        run_name=args.run_name,
        match_params=not args.no_match_params,
        corpus=args.corpus,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
