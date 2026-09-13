#!/usr/bin/env python3
"""WikiText next-byte: coincidence block vs matched dense FFN vs matched GRU. No connectome."""

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

from flybrain.baselines.gru import matched_gru
from flybrain.data.text import load_bytes, ngram_bpb, split_bytes
from flybrain.model.seqmixer import (
    CoincidenceLM,
    flop_matched_dense_ffn,
    flop_matched_gru,
    matched_dense_ffn,
)
from flybrain.train import generate, train_steps

RESULTS = ROOT / "results"
WIKI = Path("/games/ailab/lab/scrt-lab/data/wiki.train.raw")


def _arm(name: str, model, train, val, args) -> dict:
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    rep = train_steps(
        model,
        train,
        steps=args.steps,
        seq_len=args.seq_len,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
        val_data=val,
        log_every=max(1, args.steps // 8),
    )
    rep["seconds"] = time.perf_counter() - t0
    rep["arm"] = name
    rep["flops_per_step"] = model.flops_per_step(batch=1)
    rep["persistent_state_scalars"] = model.persistent_state_scalars(batch=1)
    try:
        rep["sample"] = generate(model, "The ", n_bytes=32)
    except Exception:  # noqa: BLE001
        rep["sample"] = ""
    print(
        f"{name:6} val={rep['val_bpb']:.3f} train={rep['final_bpb']:.3f} "
        f"scalars={rep['trainable_scalars']} flops/step={rep['flops_per_step']} "
        f"sec={rep['seconds']:.1f} sample={rep['sample']!r}",
        flush=True,
    )
    return rep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(WIKI))
    p.add_argument("--train-bytes", type=int, default=80_000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=24)
    p.add_argument("--n-expand", type=int, default=256)
    p.add_argument("--claws", type=int, default=4)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--n-read", type=int, default=24)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--arms", default="mixer,dense,gru")
    p.add_argument("--match", default="scalars", choices=("scalars", "flops"))
    p.add_argument("--run-name", default="")
    args = p.parse_args()

    raw = load_bytes(args.corpus, max_bytes=args.train_bytes)
    train, val = split_bytes(raw, val_frac=args.val_frac)
    uni, bi = ngram_bpb(train, val, order=1), ngram_bpb(train, val, order=2)
    name = args.run_name or f"seqmix_wiki_d{args.d_model}_D{args.n_expand}"
    print(
        f"d={args.d_model} D={args.n_expand} claws={args.claws} top_k={args.top_k} m={args.n_read} "
        f"steps={args.steps} train={int(train.numel())} val={int(val.numel())} "
        f"unigram={uni:.3f} bigram={bi:.3f}",
        flush=True,
    )
    torch.manual_seed(args.seed)
    mixer = CoincidenceLM(
        d_model=args.d_model,
        n_expand=args.n_expand,
        claws=args.claws,
        top_k=args.top_k,
        n_read=args.n_read,
        theta_init=args.theta,
        seed=args.seed,
    )
    torch.manual_seed(args.seed)
    dense = matched_dense_ffn(mixer) if args.match == "scalars" else flop_matched_dense_ffn(mixer)
    torch.manual_seed(args.seed)
    gru = matched_gru(mixer) if args.match == "scalars" else flop_matched_gru(mixer)
    models = {"mixer": mixer, "dense": dense, "gru": gru}
    rows = [_arm(a, models[a], train, val, args) for a in args.arms.split(",") if a in models]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    payload = {
        "run_name": name,
        **{k: getattr(args, k) for k in vars(args)},
        "train_bytes": int(train.numel()),
        "val_bytes": int(val.numel()),
        "unigram_bpb": uni,
        "bigram_bpb": bi,
        "note": "coincidence block as residual recurrent layer; no connectome; matched dense FFN and GRU",
        "arms": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
