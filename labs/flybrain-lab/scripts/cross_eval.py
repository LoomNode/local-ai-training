#!/usr/bin/env python3
"""Evaluate saved checkpoints on another corpus at several context lengths, no gradient steps.

The gain from a longer window on a *shifted* corpus is in-context adaptation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.baselines.gru import ByteGRU
from flybrain.data.corpus import copyable_mask
from flybrain.data.text import batch_windows, load_bytes
from flybrain.model.hybrid import GRUHybrid
from flybrain.model.sparsela import SparseKeyLM

RESULTS = ROOT / "results"


def rebuild(ck: dict):
    a = ck["args"]; arm = ck["arm"]
    if arm == "gru":
        m = ByteGRU(a["d_model"], a["gru_hidden"])
    elif arm.startswith("gru_"):
        _, key, where = arm.split("_")
        m = GRUHybrid(d_model=a["d_model"], hidden=a["gru_hidden"], key=key, where=where, n_key=a["n_key"], top_k=a["top_k"],
                      claws=a["claws"], theta=a["theta"], learned_key=a["learned_key_dim"], normalize=a["normalize"],
                      key_conv=a["key_conv"], decay=a["decay"], seed=a["seed"])
    else:
        n_key = a["n_key"] if arm in ("sparse", "gausstopk", "gauss", "mixed") else a["learned_key"]
        m = SparseKeyLM(key=arm, d_model=a["d_model"], n_key=n_key, claws=a["claws"], top_k=a["top_k"], theta=a["theta"],
                        n_layers=a["n_layers"], d_ff=a["d_ff"], decay=a["decay"], seed=a["seed"], key_conv=a["key_conv"],
                        freeze_conv=a["freeze_conv"], normalize=a["normalize"], signed=a["signed"], learned_key=a["learned_key_dim"])
    m.load_state_dict(ck["state"])
    return m


@torch.no_grad()
def evaluate(model, data, T, batch, n_batches, device):
    model.eval()
    tot = torch.zeros(T, device=device); cnt = 0; cp = [0.0, 0]; ncp = [0.0, 0]
    for i in range(n_batches):
        x, y = batch_windows(data, seq_len=T, batch=batch, seed=30_000 + i)
        cm = copyable_mask(x, y, n=8).to(device)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="none").view(-1, T)
        tot += ce.sum(0); cnt += ce.size(0)
        cp[0] += float(ce[cm].sum()); cp[1] += int(cm.sum()); ncp[0] += float(ce[~cm].sum()); ncp[1] += int((~cm).sum())
    per = tot / cnt / math.log(2)
    q = T // 8
    return {"T": T, "bpb": float(per.mean()), "bpb_first_eighth": float(per[:q].mean()), "bpb_last_eighth": float(per[-q:].mean()),
            "bpb_copyable": cp[0] / max(cp[1], 1) / math.log(2), "bpb_noncopyable": ncp[0] / max(ncp[1], 1) / math.log(2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--corpus-file", required=True)
    p.add_argument("--offset", type=int, default=9_100_000)
    p.add_argument("--bytes", type=int, default=600_000)
    p.add_argument("--lens", default="512,2048")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-name", default="cross_eval")
    args = p.parse_args()
    device = torch.device(args.device)
    data = load_bytes(args.corpus_file, max_bytes=args.bytes, offset=args.offset)
    rows = []
    for c in args.ckpts:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        m = rebuild(ck).to(device)
        for T in [int(t) for t in args.lens.split(",")]:
            batch = max(1, 8192 // T)
            r = evaluate(m, data, T, batch, max(2, 32 // batch), device)
            r.update(ckpt=Path(c).name, arm=ck["arm"])
            print(f"{ck['arm']:8} T={T:5d} bpb={r['bpb']:.3f} first8={r['bpb_first_eighth']:.3f} last8={r['bpb_last_eighth']:.3f} copyable={r['bpb_copyable']:.3f} non={r['bpb_noncopyable']:.3f}", flush=True)
            rows.append(r)
    (RESULTS / f"{args.run_name}.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
