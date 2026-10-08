"""Bet C's ceiling before it is built: per-position loss of a trained table-fed checkpoint on the dev windows against a "sure" score from
the table's own features (P(right | order, confidence class), measured on train bytes). For each skip fraction f: the share of the
model's total loss at the f most-sure positions, the model's mean loss there, and how often the table's byte is right there. A skip
mechanism can only lose what sits at those positions; it saves the global stack's compute in proportion to f.
usage: skip_ceiling.py <ckpt> <feat dir> <out json>"""
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed-run4"   # the pinned run worktree (spike/speed-int 3a812b5)
sys.path.insert(0, SPEED)
from flybrain.data.memfeat import FeatureStore  # noqa: E402
from flybrain.data.text import fixed_windows, load_bytes  # noqa: E402
from flybrain.model.inputs import channel_kwargs, conf_class  # noqa: E402
from scripts.cross_eval import rebuild  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
OFFSET, N, T = 202117357, 1024, 2048
ckpt, feat, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
device = torch.device("cuda:0")
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
orders = list(channel_kwargs(ck["args"]).get("mem_orders", ()))
store = FeatureStore(feat, orders); store.check_corpus(CORPUS)

# P(right | order, class) from 4 MB of train bytes
raw = np.fromfile(CORPUS, dtype=np.uint8)
tr_off, tr_n = 4 << 20, 4 << 20
tgt = raw[tr_off: tr_off + tr_n]
P = np.stack([np.load(f"{feat}/pred_n{o}.npy", mmap_mode="r")[tr_off: tr_off + tr_n] for o in orders], 1)
K = conf_class(torch.from_numpy(np.stack([np.load(f"{feat}/conf_n{o}.npy", mmap_mode="r")[tr_off: tr_off + tr_n] for o in orders], 1))).numpy()
prob = np.zeros((len(orders), 5))
for i in range(len(orders)):
    for c in range(1, 5):
        m = K[:, i] == c
        prob[i, c] = float((P[m, i] == tgt[m]).mean()) if m.sum() > 100 else 0.0

data = load_bytes(CORPUS, max_bytes=N * T + 1, offset=OFFSET)
xs, ys = fixed_windows(data, T, N)
pred, conf = store.fixed(OFFSET, N, T)
m = rebuild(ck).to(device).eval()
B = 4
losses = np.zeros((N, T), dtype=np.float32); ent = np.zeros((N, T), dtype=np.float32)
with torch.no_grad():
    for i in range(0, N, B):
        x, y = xs[i:i + B].to(device), ys[i:i + B].to(device)
        logits, _ = m(x, mem_pred=pred[i:i + B].to(device), mem_conf=conf[i:i + B].to(device))
        lp = F.log_softmax(logits.float(), dim=-1)
        ce = F.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1), reduction="none").view(-1, T) / math.log(2)
        losses[i:i + B] = ce.cpu().numpy(); ent[i:i + B] = (-(lp.exp() * lp).sum(-1) / math.log(2)).cpu().numpy()
np.save(out_path.replace(".json", "_losses.npy"), losses)
cls = conf_class(conf).numpy()                                   # [N, T, orders]
S = prob[np.arange(len(orders))[None, None, :], cls]              # [N, T, orders]
score = S.max(-1); best = S.argmax(-1)
tbl = np.take_along_axis(pred.numpy(), best[..., None], -1)[..., 0]
right = tbl == ys.numpy()
res = {"ckpt": ckpt, "feat": feat, "bpb": float(losses.mean()), "prob_right": prob.tolist(), "orders": orders, "fractions": {}}
print(f"model {losses.mean():.4f} bits/byte on the dev windows")
for f in (0.2, 0.3, 0.35, 0.4, 0.5, 0.6):
    thr = np.quantile(score, 1 - f); sel = score >= thr
    r = {"covered": float(sel.mean()), "loss_share": float(losses[sel].sum() / losses.sum()), "loss_there": float(losses[sel].mean()),
         "loss_rest": float(losses[~sel].mean()), "table_right_there": float(right[sel].mean()), "threshold": float(thr)}
    res["fractions"][str(f)] = r
    print(f"skip {f:.0%}: covered {r['covered']:.2f}, loss share {r['loss_share']:.3f}, loss there {r['loss_there']:.3f} vs rest {r['loss_rest']:.3f} bits, table right there {r['table_right_there']:.3f}")
# two other gates: the oracle (the lowest-loss positions themselves) and the model's own predictive entropy at p (what a cheap model of the
# same kind could approximate, BLT-style)
for f in (0.2, 0.3, 0.35, 0.4, 0.5):
    o = losses <= np.quantile(losses, f); e = ent <= np.quantile(ent, f)
    res["fractions"][str(f)].update({"oracle_loss_share": float(losses[o].sum() / losses.sum()), "entropy_gate_loss_share": float(losses[e].sum() / losses.sum()),
                                     "entropy_gate_loss_there": float(losses[e].mean())})
    print(f"skip {f:.0%}: oracle loss share {losses[o].sum() / losses.sum():.3f}; the model's own entropy as the gate: loss share {losses[e].sum() / losses.sum():.3f}, loss there {losses[e].mean():.3f}")
json.dump(res, open(out_path, "w"), indent=1)
