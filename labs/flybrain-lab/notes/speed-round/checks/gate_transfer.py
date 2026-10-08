"""Bet C, the number it stands on: gate the big model's positions by a CHEAP model's predictive entropy (BLT's construction, with the
0-layer table mixer as the cheap model). For each skip fraction f, over the dev windows: the positions with the lowest cheap-model
entropy; the big model's loss share there (what the global stack would stop computing); the cheap model's own loss there (what the
skipped positions would cost instead); and the estimate for C at that f: big total - big loss there + cheap loss there, before any
gain from the steps the saved compute buys.
usage: gate_transfer.py <cheap ckpt> <feat dir> <big losses .npy from skip_ceiling.py> <out json>"""
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed-run4"
sys.path.insert(0, SPEED)
from flybrain.data.memfeat import FeatureStore  # noqa: E402
from flybrain.data.text import fixed_windows, load_bytes  # noqa: E402
from flybrain.model.inputs import channel_kwargs  # noqa: E402
from scripts.cross_eval import rebuild  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
OFFSET, N, T = 202117357, 1024, 2048
ckpt, feat, big_path, out_path = sys.argv[1:5]
device = torch.device("cuda:0")
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
orders = list(channel_kwargs(ck["args"]).get("mem_orders", ()))
store = FeatureStore(feat, orders); store.check_corpus(CORPUS)
data = load_bytes(CORPUS, max_bytes=N * T + 1, offset=OFFSET)
xs, ys = fixed_windows(data, T, N)
pred, conf = store.fixed(OFFSET, N, T)
m = rebuild(ck).to(device).eval()
B = 8
cheap = np.zeros((N, T), dtype=np.float32); ent = np.zeros((N, T), dtype=np.float32)
with torch.no_grad():
    for i in range(0, N, B):
        x, y = xs[i:i + B].to(device), ys[i:i + B].to(device)
        logits, _ = m(x, mem_pred=pred[i:i + B].to(device), mem_conf=conf[i:i + B].to(device))
        lp = F.log_softmax(logits.float(), dim=-1)
        cheap[i:i + B] = (F.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none").view(-1, T) / math.log(2)).cpu().numpy()
        ent[i:i + B] = (-(lp.exp() * lp).sum(-1) / math.log(2)).cpu().numpy()
big = np.load(big_path)
res = {"cheap_ckpt": ckpt, "cheap_bpb": float(cheap.mean()), "big_bpb": float(big.mean()), "fractions": {}}
print(f"cheap model {cheap.mean():.4f} bits/byte, big model {big.mean():.4f}")
for f in (0.3, 0.4, 0.5, 0.6, 0.7):
    sel = ent <= np.quantile(ent, f)
    est = big.mean() - big[sel].sum() / big.size + cheap[sel].sum() / big.size
    r = {"big_loss_share": float(big[sel].sum() / big.sum()), "big_loss_there": float(big[sel].mean()), "cheap_loss_there": float(cheap[sel].mean()),
         "estimate_no_step_gain": float(est)}
    res["fractions"][str(f)] = r
    print(f"skip {f:.0%} by the cheap model's entropy: big loss share there {r['big_loss_share']:.3f} (mean {r['big_loss_there']:.3f}), cheap loss there {r['cheap_loss_there']:.3f}: "
          f"C would read {est:.4f} before any step gain ({est - big.mean():+.4f})")
json.dump(res, open(out_path, "w"), indent=1)
