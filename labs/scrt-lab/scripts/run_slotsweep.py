#!/usr/bin/env python3
"""Does residual-slot capacity bind at all? Vary occupancy at fixed parameter count.

The capacity run left the bank looking over-provisioned: a thrashing scrt run that
collapsed to 4 active slots (1.3372) and a fixed one holding 14 (1.3365) landed
within a tenth of the seed noise of each other. If 4 slots buy as much as 16, then
slot capacity is not the binding constraint at this scale, no slot is under pressure
to duplicate another, and there is nothing for consolidation to consolidate.

Occupancy lives in a buffer, not a parameter, so every variant here has the *same*
1,452,672 parameters — only how many of them are switched on differs. Uses
``scrt_static`` so occupancy is frozen for the whole run (scrt_static never
consolidates), making this a clean capacity sweep rather than a policy test.

    python scripts/run_slotsweep.py --device cuda:1 --occupancy 4 8 16 --seeds 0 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrt.config import capacity_config, resolve_device
from scrt.train import train_arm


def main() -> None:
    p = argparse.ArgumentParser(description="Slot-occupancy sweep at fixed params")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--occupancy", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 2])
    p.add_argument("--results-dir", default="results_slots")
    p.add_argument("--run-name", default="slots")
    args = p.parse_args()

    results: dict[int, dict[int, float]] = {}
    param_counts: set[int] = set()

    for occ in args.occupancy:
        results[occ] = {}
        for seed in args.seeds:
            cfg = capacity_config(device=resolve_device(args.device), steps=args.steps)
            cfg.train.seed = seed
            cfg.train.results_dir = args.results_dir
            cfg.train.run_name = f"{args.run_name}_occ{occ}_s{seed}"
            cfg.model = replace(cfg.model, init_occupied=occ)
            print(f"\n=== occupancy={occ} seed={seed} ===")
            r = train_arm(
                cfg,
                arm="scrt_static",
                write_results=True,
                match_params=True,
                eval_batches=20,
                final_eval_batches=50,
            )
            results[occ][seed] = r.best_eval_loss
            param_counts.add(r.n_params)

    print("\n=== slot-occupancy sweep (scrt_static, params fixed) ===")
    print(f"param counts seen: {sorted(param_counts)}")
    print(f"{'occupied':>9} {'best_eval mean':>15} {'std':>8}  values")
    rows = []
    for occ in args.occupancy:
        vals = [results[occ][s] for s in args.seeds]
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append((occ, mean, sd, vals))
        print(f"{occ:>9} {mean:>15.4f} {sd:>8.4f}  {[round(v, 4) for v in vals]}")

    spread = max(r[1] for r in rows) - min(r[1] for r in rows)
    max_sd = max(r[2] for r in rows)
    if len(args.seeds) < 3:
        verdict = "UNDERPOWERED (need >=3 seeds; 2-seed stds gave false verdicts before)"
    elif spread > max_sd:
        verdict = "CAPACITY BINDS"
    else:
        verdict = "CAPACITY DOES NOT BIND"
    print(
        f"\nspread across occupancy = {spread:.4f} nats; "
        f"max within-occupancy std = {max_sd:.4f} -> {verdict}"
    )

    out = Path(args.results_dir) / f"{args.run_name}_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "param_counts": sorted(param_counts),
                "seeds": args.seeds,
                "results": {str(k): v for k, v in results.items()},
                "spread": spread,
                "max_std": max_sd,
                "capacity_binds": bool(spread > max_sd),
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
