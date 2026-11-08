"""How much of a table-trained model lives in its weights (Kyle, 2026-10-09: "the tables are like bulky speed training ... regular
training would just be like a quant/compression tech"): the dev windows scored with the memory inputs withdrawn after training, so the
model sees what it saw at a miss. Conditions: full; tables_off (corpus-table inputs all misses); match_off (the in-window lookup all
misses); both_off (the weights alone). usage:
  ablate_inputs.py --feat DIR --out JSON name=ckpt ...     (SCORE_CODE picks the code, as in score_tables.py)"""
import argparse
import json
import math
import os
import sys
import time

import torch
from torch.nn import functional as F

WT = os.environ.get("SCORE_CODE", "/games/ailab/lab/flybrain-lab/.worktrees/speed-run4")
sys.path.insert(0, WT)
from flybrain.data.memfeat import FeatureStore  # noqa: E402
from flybrain.data.text import fixed_windows, load_bytes  # noqa: E402
from flybrain.model import hybrid  # noqa: E402
from flybrain.model.inputs import channel_kwargs, match_features  # noqa: E402
from scripts.cross_eval import rebuild  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--corpus", default="/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"); p.add_argument("--offset", type=int, default=202117357)
p.add_argument("--n", type=int, default=1024); p.add_argument("--T", type=int, default=2048); p.add_argument("--feat", required=True)
p.add_argument("--device", default="cuda:0"); p.add_argument("--out", required=True); p.add_argument("jobs", nargs="+")
args = p.parse_args()
device = torch.device(args.device)
data = load_bytes(args.corpus, max_bytes=args.n * args.T + 1, offset=args.offset)
xs, ys = fixed_windows(data, args.T, args.n)
B = max(1, 8192 // args.T)


def no_match(tokens, orders):  # every position a miss: (pred 0, class 0) at every order
    z = torch.zeros(*tokens.shape, len(tuple(orders)), dtype=torch.long, device=tokens.device)
    return z, z.clone()


def score(m, pred, conf, tables_off: bool, match_off: bool):
    hybrid.match_features = no_match if match_off else match_features
    cold = torch.zeros(args.n, dtype=torch.float64); warm = torch.zeros_like(cold)
    with torch.no_grad():
        for i in range(0, args.n, B):
            x, y = xs[i:i + B].to(device), ys[i:i + B].to(device)
            kw = {}
            if pred is not None:
                pr, cf = pred[i:i + B].to(device), conf[i:i + B].to(device)
                if tables_off:
                    pr, cf = torch.zeros_like(pr), torch.zeros_like(cf)
                kw = {"mem_pred": pr, "mem_conf": cf}
            logits, _ = m(x, **kw)
            ce = F.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none").view(-1, args.T) / math.log(2)
            cold[i:i + B] = ce.mean(1).double().cpu(); warm[i:i + B] = ce[:, args.T // 2:].mean(1).double().cpu()
    hybrid.match_features = match_features
    return cold, warm


res = {}
for job in args.jobs:
    name, ckpt = job.split("=", 1)
    t0 = time.time()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    orders = channel_kwargs(ck["args"]).get("mem_orders", ())
    pred = conf = None
    if orders:
        store = FeatureStore(args.feat, orders); pred, conf = store.fixed(args.offset, args.n, args.T)
    m = rebuild(ck).to(device).eval()
    has_match = m.match is not None
    conds = {"full": (False, False)}
    if orders: conds["tables_off"] = (True, False)
    if has_match: conds["match_off"] = (False, True)
    if orders and has_match: conds["both_off"] = (True, True)
    res[name] = {"ckpt": ckpt, "orders": list(orders), "match": has_match}
    for cond, (t_off, m_off) in conds.items():
        cold, warm = score(m, pred, conf, t_off, m_off)
        res[name][cond] = {"bpb": float(cold.mean()), "bpb_warm": float(warm.mean()), "per_window_cold": cold.tolist()}
        print(f"{name:12s} {cond:11s} cold {cold.mean():.4f}  warm {warm.mean():.4f}", flush=True)
    print(f"  ({time.time() - t0:.0f} s)", flush=True)
    del m; torch.cuda.empty_cache()
json.dump(res, open(args.out, "w"))
