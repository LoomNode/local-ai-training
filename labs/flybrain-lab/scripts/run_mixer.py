#!/usr/bin/env python3
"""Coincidence mixer vs matched MLP vs GRU on sparse binding. No connectome."""

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
from flybrain.baselines.mlp import matched_mlp
from flybrain.data.task import make_sparse_patterns, sparse_binding_batch
from flybrain.model.mixer import CoincidenceMixer
from flybrain.train import train_last_step

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


def _report(model) -> dict:
    persistent = model.persistent_state_scalars(batch=1) if hasattr(model, "persistent_state_scalars") else 0
    r = count_resources(
        model,
        persistent_state_scalars=persistent,
        neuron_count=int(getattr(model, "n_expand", 0)),
        synapse_count=int(getattr(model, "n_expand", 0) * getattr(model, "claws", 0)),
        flops_per_step=0,
    )
    out = r.as_dict()
    out["breakdown"] = param_breakdown(model)
    return out


@torch.no_grad()
def _probe(model, patterns: torch.Tensor) -> list[dict]:
    device = next(model.parameters()).device
    n_classes = int(patterns.shape[0])
    rows = []
    for cls in (0, 1, n_classes // 2, n_classes - 1):
        x = patterns[cls : cls + 1].to(device)
        logits, _ = model(x)
        pred = int(logits[0, -1].argmax())
        rows.append({"class": cls, "pred": pred, "ok": pred == cls})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-classes", type=int, default=16)
    p.add_argument("--n-features", type=int, default=12)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n-distractors", type=int, default=5)
    p.add_argument("--n-expand", type=int, default=80)
    p.add_argument("--claws", type=int, default=4)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--n-read", type=int, default=16)
    p.add_argument("--theta", type=float, default=3.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-name", default="")
    args = p.parse_args()

    device = _pick_device(args.device)
    patterns = make_sparse_patterns(args.n_classes, args.n_features, args.k, seed=args.seed)
    chance = 1.0 / args.n_classes
    name = args.run_name or f"mixer_bind_s{args.seed}"
    print(
        f"device={device} mixer expand={args.n_expand} claws={args.claws} top_k={args.top_k} "
        f"read={args.n_read} theta={args.theta} n_classes={args.n_classes} "
        f"n_features={args.n_features} k={args.k} distractors={args.n_distractors} "
        f"chance={chance:.4f} steps={args.steps}",
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

    mixer = CoincidenceMixer(
        n_features=args.n_features,
        n_classes=args.n_classes,
        n_expand=args.n_expand,
        claws=args.claws,
        top_k=args.top_k,
        n_read=args.n_read,
        theta=args.theta,
        seed=args.seed,
    ).to(device)
    mlp = matched_mlp(mixer).to(device)
    gru = matched_gru(mixer, vocab_size=max(args.n_features, args.n_classes)).to(device)
    rows = []
    for arm_name, model in (("mixer", mixer), ("mlp", mlp), ("gru", gru)):
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
            "arm": arm_name,
            **stats,
            **_report(model),
            "seconds": elapsed,
            "probe": probe,
            "decoder_in": int(model.decoder.in_features) if hasattr(model, "decoder") else -1,
        }
        rows.append(row)
        print(
            f"{arm_name:6} acc {row['initial_val_acc']:.3f}->{row['val_acc']:.3f} "
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
        "n_classes": args.n_classes,
        "n_features": args.n_features,
        "k": args.k,
        "n_distractors": args.n_distractors,
        "n_expand": args.n_expand,
        "claws": args.claws,
        "top_k": args.top_k,
        "n_read": args.n_read,
        "theta": args.theta,
        "chance_acc": chance,
        "steps": args.steps,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "note": (
            "coincidence mixer (frozen sparse expand + AND + top-k + thin read) "
            "vs matched MLP vs GRU; no connectome, not FlyWire"
        ),
        "arms": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
