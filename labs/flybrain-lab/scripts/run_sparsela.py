#!/usr/bin/env python3
"""Long-sequence tests for sparse-key linear attention vs learned keys, Gaussian keys, softmax, GRU."""

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

from flybrain.baselines.gru import ByteGRU
from flybrain.data.task import copy_batch
from flybrain.data.text import batch_windows, load_bytes, ngram_bpb
from flybrain.model.sparsela import SparseKeyLM

RESULTS = ROOT / "results"
WIKI = Path("/games/ailab/lab/scrt-lab/data/wiki.train.raw")


def numel(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(arm: str, args) -> torch.nn.Module:
    torch.manual_seed(args.seed)
    if arm == "gru":
        return ByteGRU(args.d_model, args.gru_hidden)
    n_key = args.n_key if arm in ("sparse", "gausstopk", "gauss") else args.learned_key
    return SparseKeyLM(key=arm, d_model=args.d_model, n_key=n_key, claws=args.claws, top_k=args.top_k,
                       theta=args.theta, n_layers=args.n_layers, d_ff=args.d_ff, decay=args.decay, seed=args.seed,
                       key_conv=args.key_conv)


@torch.no_grad()
def eval_copy(model, args, n_batches, seed, device):
    model.eval(); tot = n = hit = 0
    for i in range(n_batches):
        x, y, mask = copy_batch(args.batch, seq_len=args.seq_len, span=args.span, seed=seed + i)
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="none").view_as(y)
        tot += float(ce[mask].sum()); n += int(mask.sum()); hit += int((logits.argmax(-1)[mask] == y[mask]).sum())
    model.train()
    return {"copy_bpb": tot / n / math.log(2), "copy_acc": hit / n}


@torch.no_grad()
def eval_wiki(model, val, args, n_batches, seed, device):
    model.eval(); T = args.seq_len
    tot = torch.zeros(T, device=device); cnt = 0
    for i in range(n_batches):
        x, y = batch_windows(val, seq_len=T, batch=args.batch, seed=seed + i)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="none").view(-1, T)
        tot += ce.sum(0); cnt += ce.size(0)
    model.train()
    per_pos = (tot / cnt / math.log(2)).tolist()
    q = T // 4
    return {"val_bpb": sum(per_pos) / T, "val_bpb_first_q": sum(per_pos[:q]) / q,
            "val_bpb_last_q": sum(per_pos[-q:]) / q}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="copy", choices=("copy", "wiki"))
    p.add_argument("--arms", default="sparse,gausstopk,gauss,learned,softmax,gru")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--span", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-key", type=int, default=1024)
    p.add_argument("--learned-key", type=int, default=64)
    p.add_argument("--claws", type=int, default=4)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--theta", type=float, default=1.0)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--decay", type=float, default=0.98)
    p.add_argument("--gru-hidden", type=int, default=160)
    p.add_argument("--key-conv", type=int, default=1)
    p.add_argument("--train-bytes", type=int, default=2_000_000)
    p.add_argument("--val-bytes", type=int, default=200_000)
    p.add_argument("--run-name", default="")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    name = args.run_name or f"sla_{args.task}_s{args.seed}"
    extra = {}
    if args.task == "wiki":
        train = load_bytes(WIKI, max_bytes=args.train_bytes)
        val = load_bytes(WIKI, max_bytes=args.val_bytes, offset=args.train_bytes + 100_000)
        extra = {"unigram_bpb": ngram_bpb(train, val, 1), "bigram_bpb": ngram_bpb(train, val, 2)}
        print(f"wiki train={train.numel()} val={val.numel()} unigram={extra['unigram_bpb']:.3f} bigram={extra['bigram_bpb']:.3f}", flush=True)
    rows = []
    for arm in args.arms.split(","):
        model = build(arm, args).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        torch.manual_seed(args.seed)
        t0 = time.perf_counter(); curve = []
        for step in range(args.steps):
            if args.task == "copy":
                x, y, _ = copy_batch(args.batch, seq_len=args.seq_len, span=args.span, seed=args.seed * 100_000 + step)
            else:
                x, y = batch_windows(train, seq_len=args.seq_len, batch=args.batch, seed=args.seed * 100_000 + step)
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if (step + 1) % max(1, args.steps // 6) == 0:
                row = {"step": step + 1, "train_bpb": float(loss) / math.log(2)}
                row.update(eval_copy(model, args, 2, 10_000 + step, device) if args.task == "copy"
                           else eval_wiki(model, val, args, 4, 10_000 + step, device))
                curve.append(row); print(f"  {arm} {row}", flush=True)
        rep = eval_copy(model, args, 8, 20_000, device) if args.task == "copy" else eval_wiki(model, val, args, 16, 20_000, device)
        rep.update(arm=arm, seconds=time.perf_counter() - t0, trainable_scalars=numel(model), curve=curve,
                   flops_per_token_recurrent=model.flops_per_token_recurrent() if hasattr(model, "flops_per_token_recurrent") else model.flops_per_step(1))
        metric = {k: round(v, 3) for k, v in rep.items() if k.startswith(("copy", "val"))}
        print(f"{arm:10} {metric} scalars={rep['trainable_scalars']} rflops={rep['flops_per_token_recurrent']} sec={rep['seconds']:.0f}", flush=True)
        rows.append(rep)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps({"run_name": name, **vars(args), **extra, "arms": rows}, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
