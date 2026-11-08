"""Per-window scores of a token reader (model #2) on the bits/byte bar's byte windows: score_tables.py's job on a token stream.
usage: SCORE_CODE=<the tok worktree> score_tokens.py --stream DIR --offset O --n 1024 --T 2048 [--windows FILE] [--warm-bytes 2048]
                                                    --device cuda:0 --out-dir DIR name=ckpt@featdir ...
A byte window [a, a + T) (window i: a = offset + i * T) is scored as the tokens whose first byte lies in it (tok_common.TokenStream.span):
cold = the model reads the token before the window and the window's tokens, every window token predicted; warm = the tokens of the
--warm-bytes bytes before the window are read first (the same reach in bytes as the byte reader's warm half), the window's tokens
scored. bits/byte = the summed bits over the summed byte lengths of the scored tokens (byte-weighted over the windows; the mean of the
per-window bits/byte is reported too). Writes DIR/<name>.json like score_tables.py's (bpb, bpb_warm, per_window_cold/warm in bits per
byte) plus bpt, bpt_warm, bytes, tokens; skips a name whose file exists."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tok_common import TokenStream  # noqa: E402

WT = os.environ.get("SCORE_CODE", "/games/ailab/lab/flybrain-lab/.worktrees/tok")
sys.path.insert(0, WT)
from flybrain.data.memfeat import FeatureStore  # noqa: E402
from flybrain.model.inputs import channel_kwargs  # noqa: E402
from scripts.cross_eval import rebuild  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--stream", required=True); p.add_argument("--offset", type=int, required=True)
p.add_argument("--n", type=int, default=1024); p.add_argument("--T", type=int, default=2048)
p.add_argument("--windows", default=""); p.add_argument("--warm-bytes", type=int, default=2048)
p.add_argument("--device", default="cuda:0"); p.add_argument("--out-dir", required=True)
p.add_argument("jobs", nargs="+")
args = p.parse_args()
device = torch.device(args.device)
out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
st = TokenStream(args.stream)
sel = list(range(args.n))
if args.windows:
    sel = [int(l) for l in Path(args.windows).read_text().split()]
    if not sel or min(sel) < 0 or max(sel) >= args.n:
        sys.exit(f"--windows {args.windows}: indices must lie in [0, {args.n})")
spans = [st.span(args.offset + i * args.T, args.offset + (i + 1) * args.T) for i in sel]
warm0 = [st.span(max(0, args.offset + i * args.T - args.warm_bytes), args.offset + i * args.T)[0] for i in sel]
tokens = st.tokens
tl = st.tok_len


def run(m, store, s, t1, score_from):
    """Inputs tokens[s : t1 - 1], targets tokens[s + 1 : t1]; returns the bits of the targets from index score_from on and their byte lengths."""
    x = torch.from_numpy(np.asarray(tokens[s:t1 - 1], dtype=np.int64))[None].to(device)
    y = torch.from_numpy(np.asarray(tokens[s + 1:t1], dtype=np.int64))[None].to(device)
    kw = {}
    if store is not None:
        pr, cf = store.windows(torch.tensor([s]), t1 - 1 - s)
        kw = {"mem_pred": pr.to(device), "mem_conf": cf.to(device)}
    logits, _ = m(x, **kw)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none") / math.log(2)
    k = score_from - (s + 1)
    return ce[k:].double().cpu(), np.asarray(tl[score_from:t1], dtype=np.int64)


for job in args.jobs:
    name, spec = job.split("=", 1)
    ckpt, _, feat = spec.partition("@")
    out = out_dir / f"{name}.json"
    if out.exists():
        print(f"skip {name}", flush=True); continue
    t0 = time.time()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    orders = channel_kwargs(ck["args"]).get("mem_orders", ())
    if bool(orders) != bool(feat):
        sys.exit(f"{name}: the checkpoint's table orders are {list(orders)} but the feature directory is {feat!r}")
    store = None
    if feat:
        store = FeatureStore(feat, orders)
        if store.unit != "token" or store.size != st.n:
            sys.exit(f"{name}: the features in {feat} are not this stream's ({store.unit}, {store.size} positions vs {st.n} tokens)")
    m = rebuild(ck).to(device).eval()
    cold_bits, warm_bits, nbytes, ntok = [], [], [], []
    with torch.no_grad():
        for (t0_, t1), tw in zip(spans, warm0):
            if t1 - t0_ < 1 or t0_ < 1:
                cold_bits.append(0.0); warm_bits.append(0.0); nbytes.append(0); ntok.append(0); continue
            cb, nb = run(m, store, t0_ - 1, t1, t0_)
            wb, _ = run(m, store, max(0, min(tw, t0_ - 1)), t1, t0_)
            cold_bits.append(float(cb.sum())); warm_bits.append(float(wb.sum())); nbytes.append(int(nb.sum())); ntok.append(t1 - t0_)
    cold_bits, warm_bits, nbytes, ntok = (np.array(v, dtype=np.float64) for v in (cold_bits, warm_bits, nbytes, ntok))
    ok = nbytes > 0
    res = {"name": name, "ckpt": ckpt, "feat": feat, "stream": args.stream, "corpus": st.layout["corpus"], "offset": args.offset, "n": args.n, "T": args.T,
           "warm_bytes": args.warm_bytes, "arm": ck["arm"], "unit": "token",
           "bpb": float(cold_bits.sum() / nbytes.sum()), "bpb_warm": float(warm_bits.sum() / nbytes.sum()),
           "bpt": float(cold_bits.sum() / ntok.sum()), "bpt_warm": float(warm_bits.sum() / ntok.sum()),
           "bpb_window_mean": float((cold_bits[ok] / nbytes[ok]).mean()), "bytes": int(nbytes.sum()), "tokens": int(ntok.sum()),
           "per_window_cold": (cold_bits / np.maximum(nbytes, 1)).tolist(), "per_window_warm": (warm_bits / np.maximum(nbytes, 1)).tolist(),
           "per_window_bytes": nbytes.astype(int).tolist(), "windows": sel, "windows_file": args.windows, "seconds": time.time() - t0}
    out.write_text(json.dumps(res))
    print(f"{name:28s} cold {res['bpb']:.4f}  warm {res['bpb_warm']:.4f}  ({res['bpt']:.3f} bits/token, {res['bytes']:,} bytes, {res['seconds']:.0f} s)", flush=True)
    del m
    if device.type == "cuda":
        torch.cuda.empty_cache()
