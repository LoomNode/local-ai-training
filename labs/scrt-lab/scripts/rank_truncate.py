#!/usr/bin/env python3
"""Is the FFN compressible as a *basis*, even with no duplicate neurons?

Pairwise similarity found almost nothing (zero neuron pairs above 0.8 anywhere;
block 2's best partner reached 0.771). But "no two neurons are twins" does not imply
"this cannot be stored in less" — a population can be jointly low-rank while every
pair is distinct. That is the form of redundancy the origin thread actually described
("Layer A = 0.7*P1 + 0.2*P7 + 0.4*P19 + residual"), and merge-two-and-free cannot see
it at all.

Two independent axes, both on a trained checkpoint, no retraining:

1. **rank truncation** — SVD each MLP's ``fc`` (4C x C) and ``proj`` (C x 4C) to rank
   r. A rank-r factor costs ``(4C + C) * r = 960r`` params against 147,456 dense, so
   anything below r=154 is a real parameter saving. This measures the basis-style
   compressibility the thesis needs.
2. **neuron pruning** — zero the lowest-contribution-energy neurons
   (||a_i|| * ||w_i|| on a probe batch). This measures how many neurons are simply
   *useless*, which is the prune axis, not the merge axis.

The two answer different questions and can disagree. Flat loss under truncation with
steep loss under pruning would mean: every neuron matters individually, but jointly
they are compressible — i.e. share-a-basis is right and merge-a-pair is wrong.

    python scripts/rank_truncate.py --checkpoint results_redund/baseline_11000k.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scrt.budget import build_matched_models, unwrap
from scrt.config import capacity_config, resolve_device
from scrt.data import TokenBatcher, build_corpus
from scrt.model import BaselineGPT, model_forward


@torch.no_grad()
def eval_loss(model, batcher: TokenBatcher, batches: int = 30) -> float:
    model.eval()
    tot = 0.0
    for _ in range(batches):
        x, y = batcher.sample()
        fr = model_forward(model, x, y)
        tot += float(fr.loss.item())
    return tot / batches


@torch.no_grad()
def truncate_to_rank(model: BaselineGPT, r: int) -> None:
    """In-place: replace each MLP weight with its rank-r SVD reconstruction."""
    for blk in model.blocks:
        for lin in (blk.mlp.fc, blk.mlp.proj):
            W = lin.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            k = min(r, S.numel())
            lin.weight.data.copy_((U[:, :k] * S[:k]) @ Vh[:k]).to(lin.weight.dtype)


@torch.no_grad()
def neuron_energy(model: BaselineGPT, batcher: TokenBatcher, n_batches: int):
    """Per block, per neuron: ||a_i|| * ||w_i|| over probe tokens."""
    sums = {i: None for i in range(len(model.blocks))}
    handles = []

    def mk(i):
        def hook(_m, _inp, out):
            a = F.gelu(out).detach().float().reshape(-1, out.shape[-1])
            e = a.pow(2).sum(dim=0)
            sums[i] = e if sums[i] is None else sums[i] + e

        return hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.mlp.fc.register_forward_hook(mk(i)))
    model.eval()
    for _ in range(n_batches):
        x, _ = batcher.sample()
        model(x)
    for h in handles:
        h.remove()

    out = []
    for i, blk in enumerate(model.blocks):
        wn = blk.mlp.proj.weight.data.float().norm(dim=0)  # (N,)
        out.append(sums[i].sqrt().cpu() * wn.cpu())
    return out


@torch.no_grad()
def prune_neurons(model: BaselineGPT, energies, keep_frac: float) -> None:
    for blk, e in zip(model.blocks, energies):
        n = e.numel()
        k = int(n * keep_frac)
        drop = torch.argsort(e)[: n - k]
        blk.mlp.fc.weight.data[drop] = 0.0
        blk.mlp.proj.weight.data[:, drop] = 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="MLP rank truncation + neuron pruning")
    p.add_argument("--checkpoint", default="results_redund/baseline_11000k.pt")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-chars", type=int, default=11_000_000)
    p.add_argument("--eval-batches", type=int, default=30)
    p.add_argument("--out", default="results_redund/rank_truncate.json")
    args = p.parse_args()

    device = resolve_device(args.device)
    cfg = capacity_config(device=device)
    cfg.train.seed = args.seed
    cfg.train.max_chars = args.max_chars

    torch.manual_seed(cfg.train.seed)
    matched = build_matched_models(replace(cfg.model, vocab_size=256), exact=True)
    holder, used_mcfg, _ = matched["baseline"]
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    holder.load_state_dict(blob["state_dict"])
    holder = holder.to(device)
    print(f"loaded {args.checkpoint} (best_eval={blob.get('best_eval'):.4f})")

    corpus = build_corpus(
        kind=cfg.train.corpus,
        vocab_size=cfg.model.vocab_size,
        length=cfg.train.corpus_length,
        seed=cfg.train.seed,
        max_chars=cfg.train.max_chars,
        vocab_cap=cfg.train.char_vocab_cap,
    )
    split = int(corpus.data.numel() * 0.9)
    ev = TokenBatcher(
        corpus.data[split:],
        cfg.train.seq_len,
        cfg.train.batch_size,
        device=device,
        seed=cfg.train.seed + 1,
    )

    C = used_mcfg.n_embd
    N = 4 * C
    dense_per_mlp = 2 * (N * C)
    n_blocks = len(unwrap(holder).blocks)
    total = sum(q.numel() for q in holder.parameters())

    base = eval_loss(holder, ev, args.eval_batches)
    print(f"\nbaseline eval loss = {base:.4f}   "
          f"(MLP is {n_blocks*dense_per_mlp} params, "
          f"{100*n_blocks*dense_per_mlp/total:.1f}% of model)")

    print(f"\n=== rank truncation (break-even at r={dense_per_mlp//2//(N+C)}) ===")
    print(f"{'rank':>5} {'eval':>8} {'delta':>8} {'MLP params':>11} {'model saved':>12}")
    rank_rows = []
    for r in (192, 160, 128, 96, 64, 48, 32, 16, 8):
        m = copy.deepcopy(holder)
        truncate_to_rank(unwrap(m), r)
        L = eval_loss(m, ev, args.eval_batches)
        mlp_p = n_blocks * 2 * (N + C) * min(r, C)
        saved = n_blocks * dense_per_mlp - mlp_p
        rank_rows.append({"rank": r, "eval": L, "delta": L - base,
                          "mlp_params": mlp_p, "saved": saved,
                          "saved_pct": 100 * saved / total})
        print(f"{r:>5} {L:>8.4f} {L-base:>+8.4f} {mlp_p:>11d} "
              f"{100*saved/total:>11.1f}%")
        del m

    energies = neuron_energy(unwrap(holder), ev, 8)
    print("\n=== neuron pruning (lowest contribution energy first) ===")
    print(f"{'keep':>6} {'eval':>8} {'delta':>8} {'model saved':>12}")
    prune_rows = []
    for kf in (1.0, 0.9, 0.75, 0.5, 0.25, 0.1):
        m = copy.deepcopy(holder)
        prune_neurons(unwrap(m), energies, kf)
        L = eval_loss(m, ev, args.eval_batches)
        saved = int(n_blocks * dense_per_mlp * (1 - kf))
        prune_rows.append({"keep_frac": kf, "eval": L, "delta": L - base,
                           "saved": saved, "saved_pct": 100 * saved / total})
        print(f"{kf:>6.2f} {L:>8.4f} {L-base:>+8.4f} {100*saved/total:>11.1f}%")
        del m

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"checkpoint": args.checkpoint, "baseline_eval": base,
         "rank_truncation": rank_rows, "neuron_pruning": prune_rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
