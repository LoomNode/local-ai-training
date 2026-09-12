#!/usr/bin/env python3
"""Delayed identity: fly vs shuffle vs GRU. Cue, wait τ, emit the cue."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.accounting import count_resources, param_breakdown
from flybrain.baselines.gru import matched_gru
from flybrain.data.connectome import degree_matched_shuffle, synthetic_flylike
from flybrain.model.brain import (
    CELL_KINDS,
    DECODE_POPS,
    MESSAGE_KINDS,
    SKIP_KINDS,
    UPDATE_KINDS,
    FlyLanguageModel,
)
from flybrain.train import train_delay

RESULTS = ROOT / "results"


def _pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


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
    out = r.as_dict()
    out["breakdown"] = param_breakdown(model)
    return out


@torch.no_grad()
def _probe(model, delay: int, n_classes: int, silence: int) -> list[dict]:
    cues = [0, 1, n_classes // 2, n_classes - 1]
    rows = []
    for cue in cues:
        x = torch.full((1, delay + 1), silence, dtype=torch.long)
        x[0, 0] = cue
        x = x.to(next(model.parameters()).device)
        logits, _ = model(x)
        pred = int(logits[0, -1].argmax())
        rows.append({"cue": cue, "pred": pred, "ok": pred == cue})
    return rows


def run_delay(
    delay: int,
    *,
    n: int,
    state_dim: int,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    device: torch.device,
    n_classes: int,
    silence: int,
    message: str,
    cell: str,
    readout: str,
    decode_pop: str,
    skip: str,
    update: str,
    n_sensory: int | None,
) -> list[dict]:
    g = synthetic_flylike(n=n, seed=seed, n_sensory=n_sensory)
    sh = degree_matched_shuffle(g, seed=seed + 1)
    fly = FlyLanguageModel(
        g,
        state_dim=state_dim,
        readout=readout,
        message=message,
        cell=cell,
        decode_pop=decode_pop,
        skip=skip,
        update=update,
    ).to(device)
    rnd = FlyLanguageModel(
        sh,
        state_dim=state_dim,
        readout=readout,
        message=message,
        cell=cell,
        decode_pop=decode_pop,
        skip=skip,
        update=update,
    ).to(device)
    gru = matched_gru(fly).to(device)
    specs = [
        ("fly", fly, g.n, g.n_edges),
        ("shuffle", rnd, sh.n, sh.n_edges),
        ("gru", gru, 0, 0),
    ]
    rows = []
    for name, model, n_neu, n_syn in specs:
        t0 = time.perf_counter()
        stats = train_delay(
            model,
            n_classes=n_classes,
            delay=delay,
            steps=steps,
            batch=batch,
            lr=lr,
            seed=seed,
            silence=silence,
            log_every=max(1, steps // 8),
            eval_batches=8,
        )
        elapsed = time.perf_counter() - t0
        probe = _probe(model, delay, n_classes, silence)
        row = {
            "arm": name,
            "delay": delay,
            "n_classes": n_classes,
            "silence": silence,
            "readout": readout,
            "state_dim": state_dim,
            "message": message,
            "cell": cell,
            "decode_pop": decode_pop,
            "skip": skip,
            "update": update,
            "n_sensory": int(g.sensory_idx.numel()),
            **stats,
            **_report(model, n_neu, n_syn),
            "seconds": elapsed,
            "probe": probe,
        }
        rows.append(row)
        print(
            f"τ={delay:<2} {readout:7} {name:8} acc {row['initial_val_acc']:.3f}->{row['val_acc']:.3f} "
            f"val_bpb {row['initial_val_bpb']:.3f}->{row['val_bpb']:.3f} "
            f"params {row['trainable_scalars']} {elapsed:.1f}s "
            f"probe {[(p['cue'], p['pred']) for p in probe]}",
            flush=True,
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--state-dim", type=int, default=4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-classes", type=int, default=16)
    p.add_argument("--silence", type=int, default=255)
    p.add_argument("--delays", default="0,1,2,4")
    p.add_argument("--device", default="cpu")
    p.add_argument("--readout", default="mean")
    p.add_argument("--message", default="scalar", choices=MESSAGE_KINDS)
    p.add_argument("--n-sensory", type=int, default=None)
    p.add_argument("--cell", default="type", choices=CELL_KINDS)
    p.add_argument("--decode-pop", default="readout", choices=DECODE_POPS)
    p.add_argument("--skip", default="none", choices=SKIP_KINDS)
    p.add_argument("--update", default="ordered", choices=UPDATE_KINDS)
    p.add_argument("--run-name", default="")
    args = p.parse_args()

    device = _pick_device(args.device)
    delays = [int(s.strip()) for s in args.delays.split(",") if s.strip()]
    name = args.run_name or f"delay_n{args.n}_{args.update}"
    chance = 1.0 / args.n_classes
    print(
        f"device={device} n={args.n} d={args.state_dim} readout={args.readout} "
        f"update={args.update} n_classes={args.n_classes} chance={chance:.4f} "
        f"delays={delays} steps={args.steps} batch={args.batch}",
        flush=True,
    )
    rows: list[dict] = []
    for delay in delays:
        rows.extend(
            run_delay(
                delay,
                n=args.n,
                state_dim=args.state_dim,
                steps=args.steps,
                batch=args.batch,
                lr=args.lr,
                seed=args.seed,
                device=device,
                n_classes=args.n_classes,
                silence=args.silence,
                message=args.message,
                cell=args.cell,
                readout=args.readout,
                decode_pop=args.decode_pop,
                skip=args.skip,
                update=args.update,
                n_sensory=args.n_sensory,
            )
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    payload = {
        "run_name": name,
        "device": str(device),
        "n": args.n,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "n_classes": args.n_classes,
        "silence": args.silence,
        "delays": delays,
        "chance_acc": chance,
        "uniform_acc": 1.0 / 256,
        "state_dim": args.state_dim,
        "message": args.message,
        "n_sensory": args.n_sensory,
        "cell": args.cell,
        "decode_pop": args.decode_pop,
        "skip": args.skip,
        "update": args.update,
        "readout": args.readout,
        "note": (
            "delayed identity (cue then silence); synthetic fly-like graph, not FlyWire; "
            "accuracy on last step; ordered hops default so τ=0 is not structurally impossible"
        ),
        "arms": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
