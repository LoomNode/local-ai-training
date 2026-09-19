#!/usr/bin/env python3
"""Run the ablation ladder and write results JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biointel.ladder import run_ladder

RESULTS = ROOT / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action="store_true", help="plumbing run (few steps)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="")
    args = p.parse_args()
    name = args.run_name or ("ladder_tiny" if args.tiny else "ladder_cpu")
    rows = run_ladder(tiny=args.tiny, seed=args.seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    payload = {"run_name": name, "tiny": args.tiny, "seed": args.seed, "arms": rows}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    hdr = f"{'arm':28} {'params':>8} {'state':>8} {'N':>4} {'E':>5} {'train':>7} {'eval':>7}"
    print(hdr)
    for r in rows:
        print(
            f"{r['arm']:28} {r['trainable_scalars']:8d} {r['persistent_state_scalars']:8d} "
            f"{r['neuron_count']:4d} {r['synapse_count']:5d} {r['train_acc']:7.3f} {r['eval_acc']:7.3f}"
        )


if __name__ == "__main__":
    main()
