#!/usr/bin/env python3
"""Tiny three-arm next-byte smoke: fly topology vs shuffle vs matched GRU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.accounting import count_resources
from flybrain.baselines.gru import matched_gru
from flybrain.data.connectome import degree_matched_shuffle, synthetic_flylike
from flybrain.data.text import bytes_tensor
from flybrain.model.brain import FlyLanguageModel
from flybrain.train import generate, train_steps

RESULTS = ROOT / "results"
DEFAULT_CORPUS = (
    "the cat sat on the mat. the cat sat on the hat. the cat sat on the mat.\n"
    "the cat sat on the mat. the cat sat on the hat. the cat sat on the mat.\n"
)


def _report(model, neuron_count: int, synapse_count: int) -> dict:
    persistent = model.persistent_state_scalars(batch=1)
    flops = model.flops_per_step(batch=1) if hasattr(model, "flops_per_step") else 0
    r = count_resources(
        model,
        persistent_state_scalars=persistent,
        neuron_count=neuron_count,
        synapse_count=synapse_count,
        flops_per_step=flops,
    )
    return r.as_dict()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="tiny_cpu")
    args = p.parse_args()

    g = synthetic_flylike(n=args.n, seed=args.seed)
    sh = degree_matched_shuffle(g, seed=args.seed + 1)
    fly = FlyLanguageModel(g, state_dim=4)
    rnd = FlyLanguageModel(sh, state_dim=4)
    gru = matched_gru(fly)
    data = bytes_tensor(DEFAULT_CORPUS * 4)

    arms = {
        "fly": (fly, g.n, g.n_edges),
        "shuffle": (rnd, sh.n, sh.n_edges),
        "gru": (gru, 0, 0),
    }
    rows = []
    for name, (model, n_neu, n_syn) in arms.items():
        t0 = time.perf_counter()
        stats = train_steps(
            model,
            data,
            steps=args.steps,
            seq_len=args.seq_len,
            batch=args.batch,
            lr=args.lr,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - t0
        sample = generate(model, "the cat ", n_bytes=24)
        row = {
            "arm": name,
            **stats,
            **_report(model, n_neu, n_syn),
            "seconds": elapsed,
            "sample": sample,
        }
        rows.append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.run_name}.json"
    payload = {
        "run_name": args.run_name,
        "n": args.n,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "seed": args.seed,
        "note": "plumbing smoke on a synthetic fly-like graph; not a FlyWire result",
        "arms": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    hdr = f"{'arm':10} {'params':>8} {'state':>8} {'init_bpb':>9} {'final_bpb':>9} {'sec':>7}"
    print(hdr)
    for r in rows:
        print(
            f"{r['arm']:10} {r['trainable_scalars']:8d} {r['persistent_state_scalars']:8d} "
            f"{r['initial_bpb']:9.3f} {r['final_bpb']:9.3f} {r['seconds']:7.2f}"
        )
        print(f"  sample: {r['sample']!r}")


if __name__ == "__main__":
    main()
