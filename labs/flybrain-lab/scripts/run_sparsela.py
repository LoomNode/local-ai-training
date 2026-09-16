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
from flybrain.data.corpus import copyable_mask
from flybrain.data.task import copy_batch, multi_copy_batch
from flybrain.data.text import batch_windows, load_bytes, ngram_bpb
from flybrain.model.hybrid import GRUHybrid
from flybrain.model.sparsela import SparseKeyLM

RESULTS = ROOT / "results"
WIKI = Path("/games/ailab/lab/scrt-lab/data/wiki.train.raw")


def numel(m) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(arm: str, args) -> torch.nn.Module:
    torch.manual_seed(args.seed)
    if arm == "gru":
        return ByteGRU(args.d_model, args.gru_hidden)
    if arm in ("mingru", "gla", "transformer"):
        return GRUHybrid(d_model=args.d_model, hidden=args.gru_hidden, key="sparse", where="none", backbone=arm,
                         n_layers=args.n_layers, d_ff=args.d_ff, learned_key=args.learned_key_dim, key_conv=args.key_conv, seed=args.seed,
                         max_len=max(1024, args.seq_len), rope=args.rope)
    if arm.count("_") == 2:  # <backbone>_<key>_<pre|post>
        bb, key, where = arm.split("_")
        return GRUHybrid(d_model=args.d_model, hidden=args.gru_hidden, key=key, where=where, n_key=args.n_key, backbone=bb,
                         n_layers=args.n_layers, d_ff=args.d_ff,
                         top_k=args.top_k, claws=args.claws, theta=args.theta, learned_key=args.learned_key_dim,
                         normalize=args.normalize, key_conv=args.key_conv, decay=args.decay, seed=args.seed,
                         shuffle_keys=args.shuffle_keys, gate=args.gate, max_len=max(1024, args.seq_len), rope=args.rope, chunk=args.chunk,
                         mem_every=args.mem_every, key_convs=tuple(int(k) for k in args.key_convs.split(",")) if args.key_convs else None,
                         count_head=args.count_head, learn_decay=args.learn_decay, local_attn=args.local_attn,
                         decays=tuple(float(x) for x in args.decays.split(",")) if args.decays else None,
                         local_at=tuple(int(i) for i in args.local_at.split(",")) if args.local_at else None)
    n_key = args.n_key if arm in ("sparse", "gausstopk", "gauss", "mixed") else args.learned_key
    return SparseKeyLM(key=arm, d_model=args.d_model, n_key=n_key, claws=args.claws, top_k=args.top_k,
                       theta=args.theta, n_layers=args.n_layers, d_ff=args.d_ff, decay=args.decay, seed=args.seed,
                       key_conv=args.key_conv, freeze_conv=args.freeze_conv,
                       normalize=args.normalize, signed=args.signed, learned_key=args.learned_key_dim)


def lr_lambda(steps: int, warmup: int, cosine: bool):
    def f(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if not cosine:
            return 1.0
        p = (step - warmup) / max(1, steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


def make_optimizer(model, args):
    if not args.wd:
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)


def _copy(args, seed):
    if args.n_spans > 1:
        return multi_copy_batch(args.batch, seq_len=args.seq_len, span=args.span, n_spans=args.n_spans, seed=seed)
    return copy_batch(args.batch, seq_len=args.seq_len, span=args.span, seed=seed)


@torch.no_grad()
def eval_copy(model, args, n_batches, seed, device):
    model.eval(); tot = n = hit = 0
    for i in range(n_batches):
        x, y, mask = _copy(args, seed + i)
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
    cp_sum = cp_n = ncp_sum = ncp_n = 0.0
    for i in range(n_batches):
        x, y = batch_windows(val, seq_len=T, batch=args.batch, seed=seed + i)
        cm = _CMASK.setdefault((seed + i, T, args.batch), copyable_mask(x, y, n=8)).to(device)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="none").view(-1, T)
        tot += ce.sum(0); cnt += ce.size(0)
        cp_sum += float(ce[cm].sum()); cp_n += int(cm.sum())
        ncp_sum += float(ce[~cm].sum()); ncp_n += int((~cm).sum())
    model.train()
    per_pos = (tot / cnt / math.log(2)).tolist()
    q = T // 4
    return {"val_bpb": sum(per_pos) / T, "val_bpb_first_q": sum(per_pos[:q]) / q,
            "val_bpb_last_q": sum(per_pos[-q:]) / q,
            "val_bpb_copyable": cp_sum / max(cp_n, 1) / math.log(2), "copyable_frac": cp_n / max(cp_n + ncp_n, 1),
            "val_bpb_noncopyable": ncp_sum / max(ncp_n, 1) / math.log(2)}


_CMASK: dict = {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="copy", choices=("copy", "wiki", "code", "terminal"))
    p.add_argument("--corpus-file", default="")
    p.add_argument("--n-spans", type=int, default=1)
    p.add_argument("--save", action="store_true")
    p.add_argument("--learned-key-dim", type=int, default=64)
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
    p.add_argument("--freeze-conv", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--signed", action="store_true")
    p.add_argument("--shuffle-keys", action="store_true", help="leakage control for gru_* arms")
    p.add_argument("--gate", action="store_true", help="match-strength gate on the memory head output")
    p.add_argument("--rope", action="store_true", help="rotary positions for the transformer backbone")
    p.add_argument("--chunk", type=int, default=0, help="chunk length for the linear-cost memory head (0 = T x T parallel form)")
    p.add_argument("--mem-every", type=int, default=0, help="extra memory head after every k-th backbone layer (memory that grows with depth)")
    p.add_argument("--key-convs", default="", help="comma list of key widths cycled over heads, e.g. 2,4 (default: --key-conv for all)")
    p.add_argument("--count-head", action="store_true", help="next-byte count memory mixed into the output distribution")
    p.add_argument("--learn-decay", action="store_true", help="learned recency decay per head (init at --decay, which must be in (0,1))")
    p.add_argument("--local-attn", type=int, default=0, help=">0: one sliding-window attention layer of this window mid-stack")
    p.add_argument("--local-at", default="", help="comma list of layer counts after which a local-attention layer sits (default n_layers//2); several = several layers")
    p.add_argument("--decays", default="", help="comma list of per-head decays cycled over heads in all_heads() order (top first), e.g. 1.0,0.999")
    p.add_argument("--warmup", type=int, default=0, help="linear lr warmup steps")
    p.add_argument("--cosine", action="store_true", help="cosine decay to 10%% of lr after warmup")
    p.add_argument("--wd", type=float, default=0.0, help="AdamW weight decay (0 = plain Adam)")
    p.add_argument("--train-bytes", type=int, default=2_000_000)
    p.add_argument("--val-bytes", type=int, default=200_000)
    p.add_argument("--run-name", default="")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    name = args.run_name or f"sla_{args.task}_s{args.seed}"
    extra = {}
    if args.task != "copy":
        src = Path(args.corpus_file) if args.corpus_file else WIKI
        train = load_bytes(src, max_bytes=args.train_bytes)
        val = load_bytes(src, max_bytes=args.val_bytes, offset=args.train_bytes + 100_000)
        extra = {"unigram_bpb": ngram_bpb(train, val, 1), "bigram_bpb": ngram_bpb(train, val, 2)}
        print(f"{args.task} train={train.numel()} val={val.numel()} unigram={extra['unigram_bpb']:.3f} bigram={extra['bigram_bpb']:.3f}", flush=True)
    rows = []
    for arm in args.arms.split(","):
        model = build(arm, args).to(device)
        opt = make_optimizer(model, args)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(args.steps, args.warmup, args.cosine))
        torch.manual_seed(args.seed)
        t0 = time.perf_counter(); curve = []
        for step in range(args.steps):
            if args.task == "copy":
                x, y, _ = _copy(args, args.seed * 100_000 + step)
            else:
                x, y = batch_windows(train, seq_len=args.seq_len, batch=args.batch, seed=args.seed * 100_000 + step)
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if (step + 1) % max(1, args.steps // 6) == 0:
                row = {"step": step + 1, "train_bpb": float(loss) / math.log(2)}
                row.update(eval_copy(model, args, 2, 10_000 + step, device) if args.task == "copy"
                           else eval_wiki(model, val, args, 4, 10_000 + step, device))
                curve.append(row); print(f"  {arm} {row}", flush=True)
        rep = eval_copy(model, args, 8, 20_000, device) if args.task == "copy" else eval_wiki(model, val, args, 16, 20_000, device)
        if args.save:
            (RESULTS / "ckpt").mkdir(exist_ok=True)
            torch.save({"arm": arm, "args": vars(args), "state": model.state_dict()}, RESULTS / "ckpt" / f"{name}_{arm}.pt")
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
