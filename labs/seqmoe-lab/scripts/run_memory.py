#!/usr/bin/env python3
"""Tiny sequential vs joint: optimizer_numel and peak_vram_mb (0 on CPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from seqmoe.config import resolve_device, tiny_config
from seqmoe.memory import compare_sequential_vs_joint


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--out", default=None, help="optional JSON path")
    args = p.parse_args()
    cfg = tiny_config(device=resolve_device(args.device))
    result = compare_sequential_vs_joint(cfg, steps=args.steps)
    print(f"{'arm':12} {'optimizer_numel':>16} {'peak_vram_mb':>14} {'resident':>10}")
    for arm in ("sequential", "joint"):
        row = result[arm]
        print(
            f"{arm:12} {row['optimizer_numel']:16d} {row['peak_vram_mb']:14.2f} "
            f"{row['resident_expert_count']:10d}"
        )
    print(f"joint/seq optimizer_numel ratio = {result['ratio']}")
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str))
        print(f"wrote {path}")
    seq_n = result["sequential"]["optimizer_numel"]
    joint_n = result["joint"]["optimizer_numel"]
    if seq_n * 2 != joint_n:
        print("WARNING: expected joint optimizer_numel == 2 * sequential", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
