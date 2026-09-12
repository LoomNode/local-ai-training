#!/usr/bin/env python3
"""Held-out next-byte eval: fly vs shuffle vs GRU on a WikiText byte slice."""

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
from flybrain.data.connectome import (
    degree_matched_shuffle,
    region_preserving_shuffle,
    synthetic_flylike,
)

SHUFFLE_KINDS = ("destination", "region")
from flybrain.data.text import load_bytes, ngram_bpb, split_bytes
from flybrain.model.brain import (
    CELL_KINDS,
    DECODE_POPS,
    MESSAGE_KINDS,
    SKIP_KINDS,
    UPDATE_KINDS,
    FlyLanguageModel,
)
from flybrain.train import generate, train_steps

RESULTS = ROOT / "results"
WIKI = Path("/games/ailab/lab/scrt-lab/data/wiki.train.raw")


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


def run_readout(
    readout: str,
    train: torch.Tensor,
    val: torch.Tensor,
    *,
    n: int,
    state_dim: int,
    steps: int,
    seq_len: int,
    batch: int,
    lr: float,
    seed: int,
    device: torch.device,
    message: str = "scalar",
    n_sensory: int | None = None,
    cell: str = "type",
    decode_pop: str = "readout",
    skip: str = "none",
    update: str = "parallel",
    shuffle: str = "destination",
) -> list[dict]:
    g = synthetic_flylike(n=n, seed=seed, n_sensory=n_sensory)
    if shuffle == "region":
        sh = region_preserving_shuffle(g, seed=seed + 1)
    elif shuffle == "destination":
        sh = degree_matched_shuffle(g, seed=seed + 1)
    else:
        raise ValueError(f"unknown shuffle {shuffle}")
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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        stats = train_steps(
            model,
            train,
            steps=steps,
            seq_len=seq_len,
            batch=batch,
            lr=lr,
            seed=seed,
            val_data=val,
            log_every=max(1, steps // 8),
            eval_batches=4,
        )
        elapsed = time.perf_counter() - t0
        sample = generate(model, "The ", n_bytes=48)
        peak = 0.0
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        row = {
            "arm": name,
            "readout": readout,
            "state_dim": state_dim,
            "message": message,
            "cell": cell,
            "decode_pop": decode_pop,
            "skip": skip,
            "update": update,
            "shuffle": shuffle,
            "n_sensory": int(g.sensory_idx.numel()),
            **stats,
            **_report(model, n_neu, n_syn),
            "seconds": elapsed,
            "peak_vram_mb": peak,
            "sample": sample,
        }
        rows.append(row)
        print(
            f"{readout:8} {message:6} {cell:6} pop={decode_pop:8} skip={skip:16} "
            f"upd={update:8} shuf={shuffle:12} sens={row['n_sensory']:<3} {name:8} "
            f"train {row['initial_bpb']:.3f}->{row['final_bpb']:.3f} "
            f"val {row['initial_val_bpb']:.3f}->{row['val_bpb']:.3f} "
            f"params {row['trainable_scalars']} {elapsed:.1f}s",
            flush=True,
        )
        print(f"  sample: {sample!r}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--state-dim", type=int, default=4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-bytes", type=int, default=80_000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--corpus", type=Path, default=WIKI)
    p.add_argument("--device", default="auto")
    p.add_argument("--readout", default="flatten,mean")
    p.add_argument("--message", default="scalar", choices=MESSAGE_KINDS)
    p.add_argument("--n-sensory", type=int, default=None)
    p.add_argument("--cell", default="type", choices=CELL_KINDS)
    p.add_argument("--decode-pop", default="readout", choices=DECODE_POPS)
    p.add_argument("--skip", default="none", choices=SKIP_KINDS)
    p.add_argument("--update", default="parallel", choices=UPDATE_KINDS)
    p.add_argument("--shuffle", default="destination", choices=SHUFFLE_KINDS)
    p.add_argument("--run-name", default="")
    args = p.parse_args()

    device = _pick_device(args.device)
    raw = load_bytes(args.corpus, max_bytes=args.train_bytes)
    train, val = split_bytes(raw, val_frac=args.val_frac)
    readouts = [s.strip() for s in args.readout.split(",") if s.strip()]
    name = args.run_name or f"wiki_{device.type}_n{args.n}_d{args.state_dim}"

    uni = ngram_bpb(train, val, order=1)
    bi = ngram_bpb(train, val, order=2)
    print(
        f"device={device} n={args.n} d={args.state_dim} message={args.message} "
        f"cell={args.cell} decode_pop={args.decode_pop} skip={args.skip} "
        f"update={args.update} shuffle={args.shuffle} n_sensory={args.n_sensory} "
        f"steps={args.steps} seq={args.seq_len} "
        f"train={int(train.numel())} val={int(val.numel())} "
        f"unigram={uni:.3f} bigram={bi:.3f} corpus={args.corpus}",
        flush=True,
    )
    rows: list[dict] = []
    for readout in readouts:
        rows.extend(
            run_readout(
                readout,
                train,
                val,
                n=args.n,
                state_dim=args.state_dim,
                steps=args.steps,
                seq_len=args.seq_len,
                batch=args.batch,
                lr=args.lr,
                seed=args.seed,
                device=device,
                message=args.message,
                n_sensory=args.n_sensory,
                cell=args.cell,
                decode_pop=args.decode_pop,
                skip=args.skip,
                update=args.update,
                shuffle=args.shuffle,
            )
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    payload = {
        "run_name": name,
        "device": str(device),
        "n": args.n,
        "steps": args.steps,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "train_bytes": int(train.numel()),
        "val_bytes": int(val.numel()),
        "corpus": str(args.corpus),
        "state_dim": args.state_dim,
        "message": args.message,
        "n_sensory": args.n_sensory,
        "cell": args.cell,
        "decode_pop": args.decode_pop,
        "skip": args.skip,
        "update": args.update,
        "shuffle": args.shuffle,
        "unigram_bpb": ngram_bpb(train, val, order=1),
        "bigram_bpb": ngram_bpb(train, val, order=2),
        "note": "held-out WikiText-2 byte slice; synthetic fly-like graph, not FlyWire",
        "arms": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
