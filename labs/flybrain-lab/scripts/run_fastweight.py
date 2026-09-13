#!/usr/bin/env python3
"""Fast weights on the coincidence block: in-context copy and WikiText.

Arms (FLOP-matched): sparse+FW, sparse no FW, dense+FW, GRU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.data.task import copy_batch
from flybrain.data.text import load_bytes, ngram_bpb, split_bytes
from flybrain.model.fastweight import FastWeightLM, flop_matched_dense_fw
from flybrain.model.seqmixer import flop_matched_gru
from flybrain.train import train_steps

RESULTS = ROOT / "results"
WIKI = Path("/games/ailab/lab/scrt-lab/data/wiki.train.raw")


def _numel(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def eval_copy(model, seq_len, span, batch, n_batches, seed) -> dict:
    model.eval()
    tot, n, hit = 0.0, 0, 0
    for i in range(n_batches):
        x, y, mask = copy_batch(batch, seq_len=seq_len, span=span, seed=seed + i)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view_as(y)
        tot += float(ce[mask].sum())
        n += int(mask.sum())
        hit += int((logits.argmax(-1)[mask] == y[mask]).sum())
    model.train()
    return {"copy_bpb": tot / n / math.log(2), "copy_acc": hit / n}


def train_copy(model, args) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    curve = []
    for step in range(args.steps):
        x, y, mask = copy_batch(args.batch, seq_len=args.seq_len, span=args.span, seed=args.seed * 100_000 + step)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, args.steps // 6) == 0 or step + 1 == args.steps:
            row = {"step": step + 1, "train_bpb": float(loss) / math.log(2)}
            row.update(eval_copy(model, args.seq_len, args.span, args.batch, 4, 10_000 + step))
            curve.append(row)
    out = eval_copy(model, args.seq_len, args.span, args.batch, 8, 20_000)
    out["curve"] = curve
    return out


def build_arms(args) -> dict:
    torch.manual_seed(args.seed)
    sfw = FastWeightLM(
        d_model=args.d_model, n_expand=args.n_expand, claws=args.claws, top_k=args.top_k,
        n_read=args.n_read, seed=args.seed, code="sparse", fast=True, decay=args.decay,
    )
    torch.manual_seed(args.seed)
    s0 = FastWeightLM(
        d_model=args.d_model, n_expand=args.n_expand, claws=args.claws, top_k=args.top_k,
        n_read=args.n_read, seed=args.seed, code="sparse", fast=False,
    )
    torch.manual_seed(args.seed)
    if args.dense_expand > 0:
        dfw = FastWeightLM(d_model=args.d_model, n_expand=args.dense_expand, n_read=args.n_read,
                           seed=args.seed, code="dense", fast=True, decay=args.decay)
    else:
        dfw = flop_matched_dense_fw(sfw)
    torch.manual_seed(args.seed)
    gru = flop_matched_gru(sfw)
    return {"sparse_fw": sfw, "sparse_nofw": s0, "dense_fw": dfw, "gru": gru}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="copy", choices=("copy", "wiki"))
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--span", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=48)
    p.add_argument("--n-expand", type=int, default=1024)
    p.add_argument("--claws", type=int, default=4)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--n-read", type=int, default=48)
    p.add_argument("--decay", type=float, default=0.95)
    p.add_argument("--arms", default="sparse_fw,sparse_nofw,dense_fw,gru")
    p.add_argument("--dense-expand", type=int, default=0, help="if >0, dense_fw uses this D instead of FLOP matching")
    p.add_argument("--corpus", default=str(WIKI))
    p.add_argument("--train-bytes", type=int, default=80_000)
    p.add_argument("--run-name", default="")
    args = p.parse_args()

    arms = build_arms(args)
    name = args.run_name or f"fw_{args.task}_s{args.seed}"
    rows = []
    extra = {}
    if args.task == "wiki":
        raw = load_bytes(args.corpus, max_bytes=args.train_bytes)
        train, val = split_bytes(raw, val_frac=0.1)
        extra = {"unigram_bpb": ngram_bpb(train, val, 1), "bigram_bpb": ngram_bpb(train, val, 2)}
    for arm in args.arms.split(","):
        model = arms[arm]
        torch.manual_seed(args.seed)
        t0 = time.perf_counter()
        if args.task == "copy":
            rep = train_copy(model, args)
            metric = f"copy_bpb={rep['copy_bpb']:.3f} acc={rep['copy_acc']:.3f}"
        else:
            rep = train_steps(model, train, steps=args.steps, seq_len=args.seq_len, batch=8, lr=args.lr,
                              seed=args.seed, val_data=val, log_every=max(1, args.steps // 6))
            metric = f"val_bpb={rep['val_bpb']:.3f}"
        rep.update(arm=arm, seconds=time.perf_counter() - t0, trainable_scalars=_numel(model),
                   flops_per_step=model.flops_per_step(1), n_expand=getattr(model, "n_expand", None),
                   persistent_state_scalars=model.persistent_state_scalars(1))
        print(f"{arm:12} {metric} scalars={rep['trainable_scalars']} flops={rep['flops_per_step']} "
              f"D={rep['n_expand']} sec={rep['seconds']:.1f}", flush=True)
        rows.append(rep)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps({"run_name": name, **vars(args), **extra, "arms": rows}, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
