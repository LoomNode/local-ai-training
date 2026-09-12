#!/usr/bin/env python3
"""Sparse conjunction: fly vs region-shuffle vs GRU. k-hot set → class."""

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
from flybrain.data.task import make_sparse_patterns, sparse_binding_batch
from flybrain.model.brain import (
    CELL_KINDS,
    DECODE_POPS,
    INJECT_KINDS,
    MESSAGE_KINDS,
    READOUT_KINDS,
    SKIP_KINDS,
    UPDATE_KINDS,
    FlyLanguageModel,
)
from flybrain.train import _forward_task, train_last_step

RESULTS = ROOT / "results"
SHUFFLE_KINDS = ("destination", "region")


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


def _shuffle(g, kind: str, seed: int):
    if kind == "region":
        return region_preserving_shuffle(g, seed)
    if kind == "destination":
        return degree_matched_shuffle(g, seed)
    raise ValueError(f"unknown shuffle {kind}")


@torch.no_grad()
def _probe(model, patterns: torch.Tensor) -> list[dict]:
    device = next(model.parameters()).device
    rows = []
    n_classes = int(patterns.shape[0])
    for cls in (0, 1, n_classes // 2, n_classes - 1):
        x = patterns[cls : cls + 1].to(device)
        logits, _ = _forward_task(model, x)
        pred = int(logits[0, -1].argmax())
        rows.append({"class": cls, "pred": pred, "ok": pred == cls})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--state-dim", type=int, default=4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-classes", type=int, default=32)
    p.add_argument("--n-features", type=int, default=16)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n-distractors", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--readout", default="emit", choices=READOUT_KINDS)
    p.add_argument("--message", default="scalar", choices=MESSAGE_KINDS)
    p.add_argument("--n-sensory", type=int, default=16)
    p.add_argument("--n-readout", type=int, default=8)
    p.add_argument("--n-kenyon", type=int, default=96)
    p.add_argument("--claws", type=int, default=3)
    p.add_argument("--cell", default="type", choices=CELL_KINDS)
    p.add_argument("--decode-pop", default="readout", choices=DECODE_POPS)
    p.add_argument("--skip", default="none", choices=SKIP_KINDS)
    p.add_argument("--update", default="ordered", choices=UPDATE_KINDS)
    p.add_argument("--inject", default="local", choices=INJECT_KINDS)
    p.add_argument("--kenyon-top-k", type=int, default=12)
    p.add_argument("--kenyon-theta", type=float, default=1.5)
    p.add_argument("--shuffle", default="region", choices=SHUFFLE_KINDS)
    p.add_argument("--run-name", default="")
    args = p.parse_args()

    device = _pick_device(args.device)
    patterns = make_sparse_patterns(args.n_classes, args.n_features, args.k, seed=args.seed)
    chance = 1.0 / args.n_classes
    vocab = max(args.n_features, args.n_classes)
    name = args.run_name or f"bind_n{args.n}_{args.update}_{args.shuffle}_{args.readout}"
    print(
        f"device={device} n={args.n} d={args.state_dim} readout={args.readout} "
        f"update={args.update} shuffle={args.shuffle} inject={args.inject} "
        f"n_classes={args.n_classes} n_features={args.n_features} k={args.k} "
        f"distractors={args.n_distractors} kenyon={args.n_kenyon} claws={args.claws} "
        f"top_k={args.kenyon_top_k} theta={args.kenyon_theta} chance={chance:.4f} "
        f"steps={args.steps} batch={args.batch}",
        flush=True,
    )

    def batch_fn(batch: int, seed: int):
        return sparse_binding_batch(
            batch,
            patterns,
            seed,
            n_features=args.n_features,
            n_distractors=args.n_distractors,
        )

    g = synthetic_flylike(
        n=args.n,
        seed=args.seed,
        n_sensory=args.n_sensory,
        n_readout=args.n_readout,
        n_kenyon=args.n_kenyon,
        claws=args.claws,
        class_claws=patterns,
    )
    sh = _shuffle(g, args.shuffle, args.seed + 1)
    fly = FlyLanguageModel(
        g,
        state_dim=args.state_dim,
        vocab_size=vocab,
        readout=args.readout,
        message=args.message,
        cell=args.cell,
        decode_pop=args.decode_pop,
        skip=args.skip,
        update=args.update,
        inject=args.inject,
        kenyon_top_k=args.kenyon_top_k,
        kenyon_theta=args.kenyon_theta,
        n_out=args.n_classes,
    ).to(device)
    rnd = FlyLanguageModel(
        sh,
        state_dim=args.state_dim,
        vocab_size=vocab,
        readout=args.readout,
        message=args.message,
        cell=args.cell,
        decode_pop=args.decode_pop,
        skip=args.skip,
        update=args.update,
        inject=args.inject,
        kenyon_top_k=args.kenyon_top_k,
        kenyon_theta=args.kenyon_theta,
        n_out=args.n_classes,
    ).to(device)
    gru = matched_gru(fly).to(device)
    specs = [
        ("fly", fly, g.n, g.n_edges),
        ("shuffle", rnd, sh.n, sh.n_edges),
        ("gru", gru, 0, 0),
    ]
    rows = []
    for name_arm, model, n_neu, n_syn in specs:
        t0 = time.perf_counter()
        stats = train_last_step(
            model,
            batch_fn,
            steps=args.steps,
            batch=args.batch,
            lr=args.lr,
            seed=args.seed,
            log_every=max(1, args.steps // 8),
            eval_batches=32,
        )
        elapsed = time.perf_counter() - t0
        probe = _probe(model, patterns)
        row = {
            "arm": name_arm,
            "n_classes": args.n_classes,
            "n_features": args.n_features,
            "k": args.k,
            "n_distractors": args.n_distractors,
            "n_kenyon": args.n_kenyon,
            "claws": args.claws,
            "inject": args.inject,
            "kenyon_top_k": args.kenyon_top_k,
            "kenyon_theta": args.kenyon_theta,
            "decoder_in": int(model.decoder.in_features) if hasattr(model, "decoder") else 0,
            "readout": args.readout,
            "state_dim": args.state_dim,
            "message": args.message,
            "cell": args.cell,
            "decode_pop": args.decode_pop,
            "skip": args.skip,
            "update": args.update,
            "shuffle": args.shuffle,
            "n_sensory": int(g.sensory_idx.numel()),
            **stats,
            **_report(model, n_neu, n_syn),
            "seconds": elapsed,
            "probe": probe,
        }
        rows.append(row)
        print(
            f"{args.readout:7} {name_arm:8} acc {row['initial_val_acc']:.3f}->{row['val_acc']:.3f} "
            f"val_bpb {row['initial_val_bpb']:.3f}->{row['val_bpb']:.3f} "
            f"params {row['trainable_scalars']} {elapsed:.1f}s "
            f"probe {[(p['class'], p['pred']) for p in probe]}",
            flush=True,
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
        "n_features": args.n_features,
        "k": args.k,
        "n_distractors": args.n_distractors,
        "n_kenyon": args.n_kenyon,
        "n_readout": args.n_readout,
        "claws": args.claws,
        "inject": args.inject,
        "kenyon_top_k": args.kenyon_top_k,
        "kenyon_theta": args.kenyon_theta,
        "chance_acc": chance,
        "patterns": patterns.tolist(),
        "state_dim": args.state_dim,
        "message": args.message,
        "n_sensory": args.n_sensory,
        "cell": args.cell,
        "decode_pop": args.decode_pop,
        "skip": args.skip,
        "update": args.update,
        "shuffle": args.shuffle,
        "readout": args.readout,
        "note": (
            "sparse k-hot conjunction with distractors; mushroom-body expansion + small emit head; "
            "synthetic graph, not FlyWire; region shuffle is the topology control"
        ),
        "arms": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
