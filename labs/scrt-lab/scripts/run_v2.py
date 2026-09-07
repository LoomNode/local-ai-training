#!/usr/bin/env python3
"""v2 matched-budget experiment: harder data + gentler residual consolidation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrt.train import run_v2_comparison


def main() -> None:
    p = argparse.ArgumentParser(description="SCRT v2 experiment")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument(
        "--device",
        default="cuda:1",
        help="default cuda:1 (dedicated research 3090)",
    )
    p.add_argument("--run-name", default="v2")
    p.add_argument(
        "--corpus",
        default="structured",
        choices=["structured", "synonym", "shakespeare", "wikitext2"],
        help="structured = harder synthetic (default); shakespeare/wikitext2 = char LM",
    )
    p.add_argument("--no-match-params", action="store_true")
    args = p.parse_args()

    if args.device == "cuda":
        args.device = (
            "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cuda:1"
        )

    print(
        f"device={args.device} steps={args.steps} corpus={args.corpus} "
        f"match_params={not args.no_match_params}"
    )
    run_v2_comparison(
        steps=args.steps,
        device=args.device,
        run_name=args.run_name,
        match_params=not args.no_match_params,
        corpus=args.corpus,
    )


if __name__ == "__main__":
    main()
