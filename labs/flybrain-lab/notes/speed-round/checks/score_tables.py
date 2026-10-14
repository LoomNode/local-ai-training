"""Per-window scores of checkpoints on fixed windows, with or without table features (the tables follow-up).
usage: score_tables.py --corpus C --offset O --n 1024 --T 2048 --device cuda:0 --out-dir DIR name=ckpt[@featdir] ...
Writes DIR/<name>.json: bits/byte cold (whole window) and warm (second half), the mean and per window; skips a name whose file exists.
The mean cold score equals cross_eval.py --fixed on the same windows (equal-length windows, every position weighted once)."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

import os

WT = os.environ.get("SCORE_CODE", "/games/ailab/lab/flybrain-lab/.worktrees/speed-run4")  # the code the checkpoint was trained with: the skip arms (bet C) need speed-run5
sys.path.insert(0, WT)
from flybrain.data.memfeat import FeatureStore  # noqa: E402
from flybrain.data.text import fixed_windows, load_bytes  # noqa: E402
from flybrain.model.inputs import channel_kwargs  # noqa: E402
from scripts.cross_eval import rebuild  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--corpus", required=True); p.add_argument("--offset", type=int, required=True)
p.add_argument("--n", type=int, default=1024); p.add_argument("--T", type=int, default=2048)
p.add_argument("--device", default="cuda:0"); p.add_argument("--out-dir", required=True)
p.add_argument("--windows", default="", help="a text file of window indices (one per line, of the n from --offset): score only those "
               "(the clean subset of a leak mask); the per-window lists then follow that order and 'windows' records it")
p.add_argument("--score-from", type=int, default=0, help="also score window positions from this index on (the content of a block-built window, "
               "checks/term_corpus.py: the preamble before it is read, not scored): 'bpb_from' and 'per_window_from'")
p.add_argument("jobs", nargs="+")
args = p.parse_args()
device = torch.device(args.device)
out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
data = load_bytes(args.corpus, max_bytes=args.n * args.T + 1, offset=args.offset)
xs, ys = fixed_windows(data, args.T, args.n)
sel = None
if args.windows:
    sel = torch.tensor([int(l) for l in Path(args.windows).read_text().split()], dtype=torch.long)
    if len(sel) == 0 or int(sel.min()) < 0 or int(sel.max()) >= args.n:
        sys.exit(f"--windows {args.windows}: indices must lie in [0, {args.n})")
    xs, ys = xs[sel], ys[sel]
N = len(xs)
B = max(1, 8192 // args.T)
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
    if feat:
        store = FeatureStore(feat, orders); store.check_corpus(args.corpus)
        pred, conf = store.fixed(args.offset, args.n, args.T)
        if sel is not None:
            pred, conf = pred[sel], conf[sel]
    m = rebuild(ck).to(device).eval()
    if getattr(m, "skip_frac", 0.0):                 # bet C: the stack runs at explicit rotary positions, which need the rope table
        for a in m.core.attn:
            a.use_rope_table(args.T)
    cold = torch.zeros(N, dtype=torch.float64); warm = torch.zeros_like(cold); part = torch.zeros_like(cold)
    with torch.no_grad():
        for i in range(0, N, B):
            x, y = xs[i:i + B].to(device), ys[i:i + B].to(device)
            kw = {"mem_pred": pred[i:i + B].to(device), "mem_conf": conf[i:i + B].to(device)} if feat else {}
            logits, _ = m(x, **kw)
            ce = F.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none").view(-1, args.T) / math.log(2)
            cold[i:i + B] = ce.mean(1).double().cpu(); warm[i:i + B] = ce[:, args.T // 2:].mean(1).double().cpu()
            part[i:i + B] = ce[:, args.score_from:].mean(1).double().cpu()
    res = {"name": name, "ckpt": ckpt, "feat": feat, "corpus": args.corpus, "offset": args.offset, "n": args.n, "T": args.T, "arm": ck["arm"],
           "bpb": float(cold.mean()), "bpb_warm": float(warm.mean()), "per_window_cold": cold.tolist(), "per_window_warm": warm.tolist(),
           "seconds": time.time() - t0, **({"windows": sel.tolist(), "windows_file": args.windows} if sel is not None else {}),
           **({"score_from": args.score_from, "bpb_from": float(part.mean()), "per_window_from": part.tolist()} if args.score_from else {})}
    out.write_text(json.dumps(res))
    print(f"{name:28s} cold {res['bpb']:.4f}  warm {res['bpb_warm']:.4f}" + (f"  from {args.score_from}: {res['bpb_from']:.4f}" if args.score_from else "")
          + f"  ({res['seconds']:.0f} s)", flush=True)
    del m; torch.cuda.empty_cache()
