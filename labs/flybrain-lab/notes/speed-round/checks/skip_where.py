"""Where a --skip-frac model (bet C) loses: on the first N dev windows, its gate's keep mask at the frozen threshold, and the bits per byte
on the kept and the skipped positions for it and for plain models scored on the same windows. usage:
  SCORE_CODE=<run worktree with the skip code> skip_where.py --feat DIR --n 128 --out JSON skip=<ckpt> plain=<ckpt> [plain2=<ckpt>]"""
import argparse
import json
import math
import os
import sys

import torch
from torch.nn import functional as F

WT = os.environ.get("SCORE_CODE", "/games/ailab/lab/flybrain-lab/.worktrees/speed-run5")
sys.path.insert(0, WT)
from flybrain.data.memfeat import FeatureStore  # noqa: E402
from flybrain.data.text import fixed_windows, load_bytes  # noqa: E402
from flybrain.model.inputs import channel_kwargs  # noqa: E402
from scripts.cross_eval import rebuild  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--corpus", default="/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"); p.add_argument("--offset", type=int, default=202117357)
p.add_argument("--n", type=int, default=128); p.add_argument("--T", type=int, default=2048); p.add_argument("--feat", required=True)
p.add_argument("--device", default="cuda:0"); p.add_argument("--out", required=True); p.add_argument("jobs", nargs="+")
args = p.parse_args()
device = torch.device(args.device)
data = load_bytes(args.corpus, max_bytes=args.n * args.T + 1, offset=args.offset)
xs, ys = fixed_windows(data, args.T, args.n)
B = max(1, 8192 // args.T)


def per_position(ckpt):
    """bits per byte at every position [n, T], and for a skip model its keep mask and gate entropy (else None)."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    orders = channel_kwargs(ck["args"]).get("mem_orders", ())
    store = FeatureStore(args.feat, orders); pred, conf = store.fixed(args.offset, args.n, args.T)
    m = rebuild(ck).to(device).eval()
    skip = bool(getattr(m, "skip_frac", 0.0))
    if skip:
        for a in m.core.attn:
            a.use_rope_table(args.T)
    bits = torch.zeros(args.n, args.T, dtype=torch.float64); keep = torch.zeros(args.n, args.T, dtype=torch.bool); ent = torch.zeros(args.n, args.T)
    with torch.no_grad():
        for i in range(0, args.n, B):
            x, y = xs[i:i + B].to(device), ys[i:i + B].to(device)
            kw = {"mem_pred": pred[i:i + B].to(device), "mem_conf": conf[i:i + B].to(device)}
            if skip:
                logits, gate = m.forward_skip(x, **kw)
                lp = F.log_softmax(gate.float(), -1); e = -(lp.exp() * lp).sum(-1)
                k = e > m.skip_tau; k[:, 0] = True
                keep[i:i + B] = k.cpu(); ent[i:i + B] = e.cpu()
            else:
                logits, _ = m(x, **kw)
            bits[i:i + B] = (F.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none").view(-1, args.T) / math.log(2)).cpu()
    return bits, (keep if skip else None), (ent if skip else None), float(m.skip_tau) if skip else None


res = {}
masks = None
for job in args.jobs:
    name, ckpt = job.split("=", 1)
    bits, keep, ent, tau = per_position(ckpt)
    res[name] = {"ckpt": ckpt, "bpb": bits.mean().item(), "bpb_warm": bits[:, args.T // 2:].mean().item()}
    if keep is not None:
        masks = keep
        res[name].update(tau_nats=tau, keep_frac=keep.double().mean().item(), gate_entropy_mean=ent.mean().item())
    res[name]["_bits"] = bits
for name, r in res.items():
    bits = r.pop("_bits")
    if masks is not None:
        kept, skipped = bits[masks], bits[~masks]
        r.update(bpb_kept=kept.mean().item(), bpb_skipped=skipped.mean().item(), loss_share_skipped=(skipped.sum() / bits.sum()).item())
    print(f"{name:10s} bpb {r['bpb']:.4f} warm {r['bpb_warm']:.4f}" + (f"  kept {r['bpb_kept']:.4f} skipped {r['bpb_skipped']:.4f} (share of loss {r['loss_share_skipped']:.1%})" if masks is not None else "")
          + (f"  keep_frac {r['keep_frac']:.3f} tau {r['tau_nats']:.3f} nats" if "keep_frac" in r else ""))
json.dump(res, open(args.out, "w"), indent=1)
