#!/usr/bin/env python3
"""Swap test (2026-11-05): two checkpoints of the same recipe trained on different corpora (A, B); score A, B and the
hybrids that take one part from the other model, on both corpora, no gradient steps. If bits/byte follow a part, the
domain lives in that part. Parts: heads (every memory head's key thresholds, write/read projections, key conv, norms,
gate: the table machinery; the table contents are per-window working memory and are not in a checkpoint), local (the
local-attention layer). The minGRU layers, embedding and decoder are the core and stay with the receiving model.
Pair checkpoints of the same seed: the sparse key matrix is drawn from the seed, so same-seed heads share their keys."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.data.text import load_bytes
from scripts.cross_eval import evaluate, rebuild

RESULTS = ROOT / "results"
PARTS = {"heads": ("mem.", "mems."), "local": ("local.",)}


def take(state_to: dict, state_from: dict, parts: tuple[str, ...]) -> dict:
    prefixes = tuple(p for part in parts for p in PARTS[part])
    out = dict(state_to)
    n = 0
    for k, v in state_from.items():
        if k.startswith(prefixes):
            if out[k].shape != v.shape:
                raise ValueError(f"{k}: {tuple(out[k].shape)} vs {tuple(v.shape)}")
            out[k] = v; n += 1
    assert n, parts
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="checkpoint A")
    p.add_argument("--b", required=True, help="checkpoint B (same recipe and seed, other corpus)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-batches", type=int, default=16)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()
    device = torch.device(args.device)
    cks = {"A": torch.load(args.a, map_location="cpu", weights_only=False), "B": torch.load(args.b, map_location="cpu", weights_only=False)}
    vals = {}
    for tag, ck in cks.items():
        a = ck["args"]
        vals[tag] = load_bytes(Path(a["corpus_file"]), max_bytes=a["val_bytes"], offset=a["train_bytes"] + 100_000)
        print(f"{tag}: {Path(args.a if tag == 'A' else args.b).name}  corpus {Path(a['corpus_file']).name}  seed {a['seed']}")
    model = rebuild(cks["A"]).to(device)
    T = cks["A"]["args"]["seq_len"]; batch = max(1, 8192 // T)
    configs = []
    for to, frm in (("A", "B"), ("B", "A")):
        configs.append((f"{to}", cks[to]["state"]))
        for parts in (("heads",), ("local",), ("heads", "local")):
            configs.append((f"{to}+{'+'.join(parts)}({frm})", take(cks[to]["state"], cks[frm]["state"], parts)))
    rows = []
    print(f"{'model':22} {'A-corpus bpb':>12} {'B-corpus bpb':>12}   (copyable / non on A-corpus)")
    for name, state in configs:
        model.load_state_dict(state)
        r = {tag: evaluate(model, vals[tag], T, batch, args.n_batches, device) for tag in ("A", "B")}
        rows.append({"model": name, "A": r["A"], "B": r["B"]})
        print(f"{name:22} {r['A']['bpb']:12.3f} {r['B']['bpb']:12.3f}   ({r['A']['bpb_copyable']:.3f} / {r['A']['bpb_noncopyable']:.3f})", flush=True)
    out = RESULTS / f"{args.run_name}_swap.json"
    out.write_text(json.dumps({"a": args.a, "b": args.b, "n_batches": args.n_batches, "T": T, "rows": rows}, indent=2) + "\n")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
