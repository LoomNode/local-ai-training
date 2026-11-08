"""What a feature directory's extra orders add over the byte table's 8/16/32 on the dev windows, before any training: per order the
hit rate and precision, and the coverage added (positions where no base order hits but this one does, with its precision there) and
the disagreement (positions where a base order hits and this one hits with another byte, and who is right).
usage: key_coverage.py <feat dir> [base feat dir]"""
import json
import sys

import numpy as np

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
W0, NW, T = 202_117_357, 1024, 2048
lo, hi = W0 + 1, W0 + NW * T + 1                      # the positions the windows predict
feat = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "/games/ailab/lab/flybrain-lab/.worktrees/speed/results/mem/feat"
data = np.fromfile(CORPUS, dtype=np.uint8, count=hi)[lo:hi]
info = json.load(open(f"{feat}/features.json"))
BASE = (8, 16, 32)


def load(d, n):
    return (np.load(f"{d}/pred_n{n}.npy", mmap_mode="r")[lo:hi], np.load(f"{d}/conf_n{n}.npy", mmap_mode="r")[lo:hi])


bp = {n: load(base, n) for n in BASE}
base_hit = np.zeros(len(data), dtype=bool)
base_pred = np.full(len(data), -1, dtype=np.int16)           # the longest hitting base order's prediction
for n in BASE:                                                # ascending: the longest order wins
    p, c = bp[n]
    h = c > 0
    base_hit |= h
    base_pred[h] = p[h]
base_right = base_hit & (base_pred == data)
print(f"{feat}: {len(data):,} dev positions; base 8/16/32 hits {base_hit.mean():.3f}, right {base_right.sum() / max(1, base_hit.sum()):.3f} of hits, {base_right.mean():.3f} of all")
print(f"{'order':>6s} {'hits':>6s} {'right':>6s} | {'added':>6s} {'right':>6s} | {'overlap':>7s} {'agree':>6s} {'it right':>8s} {'base right':>10s}")
for n in info["orders"]:
    if n in BASE:
        continue
    p, c = load(feat, n)
    h = c > 0
    right = h & (p == data)
    added = h & ~base_hit
    both = h & base_hit
    agree = both & (p == base_pred)
    dis = both & ~agree
    print(f"{n:6d} {h.mean():6.3f} {right.sum() / max(1, h.sum()):6.3f} | {added.mean():6.3f} {(right & added).sum() / max(1, added.sum()):6.3f} | "
          f"{both.mean():7.3f} {agree.sum() / max(1, both.sum()):6.3f} {(right & dis).sum() / max(1, dis.sum()):8.3f} {(base_right & dis).sum() / max(1, dis.sum()):10.3f}")
print("hits/right: share of positions the order hits, and the share of those it predicts right; added: hits where no base order hits;")
print("overlap: hits where a base order also hits; agree: same byte; it right / base right: among the disagreements, who is right.")
