#!/usr/bin/env python3
"""Division of labor (2026-11-05): score a saved checkpoint with one component knocked out at a time, on its own validation
bytes and on a shifted corpus, no gradient steps. The bits/byte lost per knockout says where the work is done today:
in the memory heads (in-context tables), in the minGRU layers (weights), or in the local-attention layer.

Knockouts: each memory head's output zeroed (the residual stays); all heads at once; each minGRU layer replaced by the
identity; the local-attention layer zeroed. The model is otherwise untouched, so a knockout is a lower bound on the
component's worth (nothing retrains around the hole)."""

from __future__ import annotations

import argparse
import contextlib
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
CORPORA = Path("/games/ailab/lab/flybrain-lab/corpora")  # the main checkout's corpora; worktrees have none


@contextlib.contextmanager
def knocked(model, what: str):
    """Temporarily replace one component; `what` is none | head:<i> | heads | layer:<i> | local."""
    patched = []

    def zero(mod, name="memory"):
        setattr(mod, name, lambda x, *a, **k: torch.zeros_like(x)); patched.append((mod, name))

    def identity(mod):
        mod.forward = lambda x, *a, **k: x; patched.append((mod, "forward"))

    heads = model.all_heads()
    if what == "none":
        pass
    elif what == "heads":
        for h in heads: zero(h)
    elif what.startswith("head:"):
        zero(heads[int(what[5:])])
    elif what.startswith("layer:"):
        identity(model.core.layers[int(what[6:])])
    elif what == "local":
        zero(model.local, "forward")
    else:
        raise ValueError(what)
    try:
        yield
    finally:
        for mod, name in patched:
            delattr(mod, name)  # the instance attribute shadowed the class method; dropping it restores the original


def knockouts(model) -> list[str]:
    out = ["none"] + [f"head:{i}" for i in range(len(model.all_heads()))] + ["heads"]
    out += [f"layer:{i}" for i in range(len(model.core.layers))]
    if getattr(model, "local_attn", 0):
        out.append("local")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-batches", type=int, default=16, help="batches of 8192 tokens per corpus (16 = 131k bytes)")
    p.add_argument("--shift-file", default="", help="the shifted corpus (default: cpython.bytes, or linux_full.bytes for a CPython checkpoint)")
    p.add_argument("--shift-offset", type=int, default=9_100_000)
    p.add_argument("--shift-bytes", type=int, default=600_000)
    p.add_argument("--run-name", default="")
    args = p.parse_args()
    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    own = load_bytes(Path(a["corpus_file"]), max_bytes=a["val_bytes"], offset=a["train_bytes"] + 100_000)
    shift_file = Path(args.shift_file) if args.shift_file else (CORPORA / ("linux_full.bytes" if "cpython" in a["corpus_file"] else "cpython.bytes"))
    shift = load_bytes(shift_file, max_bytes=args.shift_bytes, offset=args.shift_offset)
    model = rebuild(ck).to(device)
    T = a["seq_len"]; batch = max(1, 8192 // T)
    rows = []
    base = None
    print(f"{'knockout':10} {'own bpb':>8} {'Δ':>7} {'copy Δ':>7} {'non Δ':>7} | {'shift bpb':>9} {'Δ':>7} {'copy Δ':>7} {'non Δ':>7}")
    for what in knockouts(model):
        with knocked(model, what):
            r_own = evaluate(model, own, T, batch, args.n_batches, device)
            r_shift = evaluate(model, shift, T, batch, args.n_batches, device)
        row = {"knockout": what, "own": r_own, "shift": r_shift}
        rows.append(row)
        if base is None:
            base = row
        d = lambda side, key: row[side][key] - base[side][key]
        print(f"{what:10} {r_own['bpb']:8.3f} {d('own', 'bpb'):+7.3f} {d('own', 'bpb_copyable'):+7.3f} {d('own', 'bpb_noncopyable'):+7.3f} | "
              f"{r_shift['bpb']:9.3f} {d('shift', 'bpb'):+7.3f} {d('shift', 'bpb_copyable'):+7.3f} {d('shift', 'bpb_noncopyable'):+7.3f}", flush=True)
    name = args.run_name or Path(args.ckpt).stem.replace("_mingru_sparse_post", "")
    out = RESULTS / f"{name}_knockout.json"
    out.write_text(json.dumps({"ckpt": args.ckpt, "arm": ck["arm"], "own_corpus": a["corpus_file"], "shift_corpus": str(shift_file),
                               "shift_offset": args.shift_offset, "n_batches": args.n_batches, "T": T, "rows": rows}, indent=2) + "\n")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
