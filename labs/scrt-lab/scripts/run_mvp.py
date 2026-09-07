#!/usr/bin/env python3
"""Run the three-arm MVP comparison (baseline / looped / scrt)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as scripts/run_mvp.py without install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrt.train import run_mvp_comparison


def main() -> None:
    p = argparse.ArgumentParser(description="SCRT MVP comparison run")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument(
        "--device",
        default="cuda:1",
        help="torch device (default cuda:1 — dedicated research 3090)",
    )
    p.add_argument("--run-name", default="mvp")
    args = p.parse_args()
    if args.device == "cuda":
        args.device = "cuda:1"
    run_mvp_comparison(steps=args.steps, device=args.device, run_name=args.run_name)


if __name__ == "__main__":
    main()
