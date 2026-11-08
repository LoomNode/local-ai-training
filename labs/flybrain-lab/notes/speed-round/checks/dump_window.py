"""Per-byte costs of a trained model on dev windows, for the board's Reading view (tools/board): every byte of a window scored with the
model's memories on, and with each withdrawn as ablate_inputs.py withdraws them (the corpus table's inputs all misses; only the project
tier's orders as misses; the in-window lookup all misses; both), plus the model's top-3 guesses with everything on. The viewer colours
each byte by the memory whose removal hurts it most. One JSON per (model, window), T entries per condition, bits per byte. usage (from
the speed worktree, on the CPU; seconds per window):
  CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 0-5,8-13 env OMP_NUM_THREADS=4 /games/ailab/lab/flybrain-lab/.venv/bin/python \
    notes/speed-round/checks/dump_window.py --feat results/mem/feat_all28_fly --name l4dir \
    --ckpt /games/ailab/lab/flybrain-lab/.worktrees/speed-run4/results/ckpt/l4dir_r_transformer.pt --windows 0,100,250 --out-dir tools/board/data
SCORE_CODE picks the model code, as in ablate_inputs.py (the run worktree)."""
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
p.add_argument("--n", type=int, default=1024, help="the dev set's window count (the scorer's); --windows index into it")
p.add_argument("--T", type=int, default=2048); p.add_argument("--feat", default="", help="feature directory; needed when the checkpoint reads a corpus table")
p.add_argument("--ckpt", required=True); p.add_argument("--name", required=True, help="the model's label in the viewer (file prefix)")
p.add_argument("--windows", required=True, help="comma-separated window indices"); p.add_argument("--out-dir", required=True)
p.add_argument("--device", default="cpu")
args = p.parse_args()
device = torch.device(args.device)
ws = [int(w) for w in args.windows.split(",")]
data = load_bytes(args.corpus, max_bytes=args.n * args.T + 1, offset=args.offset)
xs, ys = fixed_windows(data, args.T, args.n)
os.makedirs(args.out_dir, exist_ok=True)


def no_match(tokens, orders):  # every position a miss: (pred 0, class 0) at every order
    z = torch.zeros(*tokens.shape, len(tuple(orders)), dtype=torch.long, device=tokens.device)
    return z, z.clone()


ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
orders = tuple(channel_kwargs(ck["args"]).get("mem_orders", ()))
tier_cols = [i for i, o in enumerate(orders) if 2000 <= o < 3000]
pred = conf = None
if orders:
    if not args.feat:
        raise SystemExit("the checkpoint reads a corpus table: pass --feat")
    store = FeatureStore(args.feat, orders)  # only the requested windows (the scorer's fixed(offset, n, T) = windows at offset + T * i)
    pred, conf = store.windows(torch.tensor([args.offset + args.T * w for w in ws]), args.T)
m = rebuild(ck).to(device).eval()
has_match = m.match is not None
conds = {"full": ("none", False)}
if orders: conds["tables_off"] = ("all", False)
if tier_cols: conds["tier_off"] = ("tier", False)
if has_match: conds["match_off"] = ("none", True)
if orders and has_match: conds["both_off"] = ("all", True)


def run(w: int, tables: str, match_off: bool):
    hybrid.match_features = no_match if match_off else match_features
    x, y = xs[w:w + 1].to(device), ys[w:w + 1].to(device)
    kw = {}
    if pred is not None:
        j = ws.index(w)
        pr, cf = pred[j:j + 1].to(device).clone(), conf[j:j + 1].to(device).clone()
        if tables == "all":
            pr.zero_(); cf.zero_()
        elif tables == "tier":
            pr[..., tier_cols] = 0; cf[..., tier_cols] = 0
        kw = {"mem_pred": pr, "mem_conf": cf}
    with torch.no_grad():
        logits, _ = m(x, **kw)
    hybrid.match_features = match_features
    logits = logits.reshape(-1, 256).float()
    bits = F.cross_entropy(logits, y.reshape(-1), reduction="none") / math.log(2)
    return bits, logits


for w in ws:
    t0 = time.time()
    out = {"model": args.name, "ckpt": args.ckpt, "window": w, "offset": args.offset + w * args.T, "T": args.T, "orders": list(orders),
           "match": has_match, "text": bytes(ys[w].tolist()).decode("latin-1"), "cost": {}}
    for cond, (tables, m_off) in conds.items():
        bits, logits = run(w, tables, m_off)
        out["cost"][cond] = [round(float(b), 4) for b in bits]
        if cond == "full":
            prob, byte = F.softmax(logits, -1).topk(3, -1)
            out["top3"] = [[int(byte[i, 0]), round(float(prob[i, 0]), 4), int(byte[i, 1]), round(float(prob[i, 1]), 4), int(byte[i, 2]), round(float(prob[i, 2]), 4)] for i in range(byte.shape[0])]
    path = os.path.join(args.out_dir, f"{args.name}_w{w}.json")
    json.dump(out, open(path, "w"))
    means = "  ".join(f"{c} {sum(v) / len(v):.4f}" for c, v in out["cost"].items())
    print(f"{args.name} window {w}: {means}  ({time.time() - t0:.1f} s) -> {path}", flush=True)
