#!/usr/bin/env python3
"""Saturation experiment: long horizon, consolidate in the second half."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrt.train import run_saturation_comparison


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "SCRT saturation run: matched 4-arm budget, consolidate only after "
            "start_frac (default 0.5) through end of training"
        )
    )
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--run-name", default="saturation")
    p.add_argument(
        "--corpus",
        default="structured",
        choices=["structured", "synonym", "shakespeare", "wikitext2"],
    )
    p.add_argument("--results-dir", default=None)
    p.add_argument("--no-match-params", action="store_true")
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Disable free/regrow; stronger dropout/weight decay (follow-up experiment)",
    )
    args = p.parse_args()

    if args.device == "cuda":
        args.device = (
            "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda:1"
        )

    if args.merge_only and args.run_name == "saturation":
        args.run_name = "sat_merge"

    print(
        f"device={args.device} steps={args.steps} corpus={args.corpus} "
        f"match_params={not args.no_match_params} run_name={args.run_name} "
        f"merge_only={args.merge_only}"
    )
    run_saturation_comparison(
        steps=args.steps,
        device=args.device,
        run_name=args.run_name,
        match_params=not args.no_match_params,
        corpus=args.corpus,
        results_dir=args.results_dir,
        merge_only=args.merge_only,
    )


if __name__ == "__main__":
    main()
