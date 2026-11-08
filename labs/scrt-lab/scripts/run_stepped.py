#!/usr/bin/env python3
"""Matched-budget stepped experiment on a single GPU."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrt.train import run_stepped_comparison


def main() -> None:
    p = argparse.ArgumentParser(description="SCRT stepped matched-budget run")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument(
        "--device",
        default="cuda:1",
        help="torch device (default cuda:1 — dedicated research 3090)",
    )
    p.add_argument("--run-name", default="stepped")
    p.add_argument("--corpus-length", type=int, default=500_000)
    p.add_argument(
        "--no-match-params",
        action="store_true",
        help="Disable exact param matching (debug only)",
    )
    args = p.parse_args()

    # Bare ``cuda`` → research card (GPU 1). Do not remap if CUDA_VISIBLE_DEVICES
    # already isolates a single device (then cuda:0 is that card).
    if args.device == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            args.device = "cuda:0"
        else:
            args.device = "cuda:1"

    print(f"device={args.device}  steps={args.steps}  match_params={not args.no_match_params}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    run_stepped_comparison(
        steps=args.steps,
        device=args.device,
        run_name=args.run_name,
        match_params=not args.no_match_params,
        corpus_length=args.corpus_length,
    )


if __name__ == "__main__":
    main()
