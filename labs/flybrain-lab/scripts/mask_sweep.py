#!/usr/bin/env python3
"""Where does the head's gain land? Bin every validation byte by the longest earlier-in-window n-gram
match (2, 3, 4, 6, 8, or none) and score saved checkpoints per bin."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.data.corpus import copyable_mask
from flybrain.data.text import batch_windows, load_bytes
from flybrain.model.hybrid import GRUHybrid
from flybrain.model.inputs import channel_kwargs

NS = (2, 3, 4, 6, 8)


def longest_match_bin(x: Tensor, y: Tensor) -> Tensor:
    """0 = no match at n=2; otherwise the largest n in NS at which y[t] continues an earlier n-gram."""
    bins = torch.zeros_like(x)
    for n in NS:  # ascending; later (longer) overwrite
        bins[copyable_mask(x, y, n=n)] = n
    return bins


def rebuild(ck: dict):
    a = ck["args"]; arm = ck["arm"]
    if arm in ("mingru", "gla", "transformer"):
        extra = {} if arm == "transformer" else {"local_attn": a.get("local_attn", 0),
                                                 "local_at": tuple(int(i) for i in a["local_at"].split(",")) if a.get("local_at") else None}
        return GRUHybrid(d_model=a["d_model"], hidden=a["gru_hidden"], key="sparse", where="none", backbone=arm, n_layers=a["n_layers"],
                         d_ff=a["d_ff"], learned_key=a["learned_key_dim"], key_conv=a["key_conv"], seed=a["seed"], max_len=max(1024, a["seq_len"]),
                         rope=a.get("rope", False), **extra, **channel_kwargs(a))
    bb, key, where = arm.split("_")
    return GRUHybrid(d_model=a["d_model"], hidden=a["gru_hidden"], key=key, where=where, n_key=a["n_key"], backbone=bb,
                     n_layers=a["n_layers"], d_ff=a["d_ff"], top_k=a["top_k"], claws=a["claws"], theta=a["theta"],
                     learned_key=a["learned_key_dim"], normalize=a["normalize"], key_conv=a["key_conv"], decay=a["decay"], seed=a["seed"],
                     gate=a.get("gate", False), max_len=max(1024, a["seq_len"]), rope=a.get("rope", False), chunk=a.get("chunk", 0),
                     mem_every=a.get("mem_every", 0), key_convs=tuple(int(k) for k in a["key_convs"].split(",")) if a.get("key_convs") else None,
                     count_head=a.get("count_head", False), learn_decay=a.get("learn_decay", False), local_attn=a.get("local_attn", 0),
                     decays=tuple(float(x) for x in a["decays"].split(",")) if a.get("decays") else None,
                     local_at=tuple(int(i) for i in a["local_at"].split(",")) if a.get("local_at") else None, **channel_kwargs(a))


@torch.no_grad()
def score(model, val, T, batch, n_batches, device):
    model.eval(); sums = {}; counts = {}
    for i in range(n_batches):
        x, y = batch_windows(val, seq_len=T, batch=batch, seed=20_000 + i)
        bins = longest_match_bin(x, y).to(device)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="none").view_as(y)
        for b in [0] + list(NS):
            m = bins == b
            sums[b] = sums.get(b, 0.0) + float(ce[m].sum()); counts[b] = counts.get(b, 0) + int(m.sum())
    return {b: (sums[b] / max(counts[b], 1) / math.log(2), counts[b]) for b in sums}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--arms", default="mingru,mingru_sparse_post")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-batches", type=int, default=16)
    args = p.parse_args()
    device = torch.device(args.device)
    out = {}
    for arm in args.arms.split(","):
        ck = torch.load(ROOT / "results" / "ckpt" / f"{args.run_name}_{arm}.pt", map_location="cpu", weights_only=False)
        a = ck["args"]
        val = load_bytes(Path(a["corpus_file"]), max_bytes=a["val_bytes"], offset=a["train_bytes"] + 100_000)
        model = rebuild(ck).to(device); model.load_state_dict(ck["state"])
        out[arm] = score(model, val, a["seq_len"], a["batch"], args.n_batches, device)
    arms = list(out); total = sum(c for _, c in out[arms[0]].values())
    print(f"{'longest match':>14} {'frac':>6} " + " ".join(f"{a:>18}" for a in arms) + f" {'gain':>7}")
    for b in [0] + list(NS):
        row = [out[a][b][0] for a in arms]
        print(f"{('none' if b == 0 else f'{b}-gram'):>14} {out[arms[0]][b][1] / total:6.3f} " + " ".join(f"{v:18.3f}" for v in row) + f" {row[0] - row[-1]:7.3f}")
    (ROOT / "results" / f"{args.run_name}_masksweep.json").write_text(json.dumps({"run_name": args.run_name, "arms": {a: {str(k): v for k, v in d.items()} for a, d in out.items()}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
