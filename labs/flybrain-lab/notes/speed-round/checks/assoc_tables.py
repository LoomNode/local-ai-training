"""Two-way set-associative tables (cache design, not fly design): each context may live in either of two buddy slots (slot, slot ^ 1).
A write confirms a buddy with the same tag, else fills an empty buddy, else replaces the buddy with the lower confidence unless both
are confirmed (confidence >= 2; then the newcomer is dropped). A lookup checks both buddies. Compared with the direct-mapped table
of the same memory. usage: assoc_tables.py <out root> <name>:<log2 slots> ...  (features for the dev window region only, sparse files)."""
import json
import os
import sys
import time

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import CONF_MAX, SLOT, context_hashes, slot_and_tag  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
TRAIN_END, ORDERS, CHUNK = 200_000_000, (8, 16, 32), 1 << 20
N, T = 1024, 2048
W0 = 202_117_357
W1 = W0 + N * T + 1
root = sys.argv[1]
conds = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[2:]]
data = np.memmap(CORPUS, dtype=np.uint8, mode="r")
size = len(data)


def write(table, slot, tag, nxt):
    """One chunk's (slot, tag, next byte) triples into a 2-way table. Duplicate contexts inside a chunk resolve by numpy's last write."""
    slot = slot.astype(np.int64)
    a, b = slot & ~1, slot | 1
    ra, rb = table[a], table[b]
    hit_a = (ra["conf"] > 0) & (ra["tag"] == tag)
    hit_b = ~hit_a & (rb["conf"] > 0) & (rb["tag"] == tag)
    # confirm or decay the matching buddy
    for hit, idx, rec in ((hit_a, a, ra), (hit_b, b, rb)):
        if hit.any():
            r = rec[hit].copy(); same = r["byte"] == nxt[hit]
            c = r["conf"].astype(np.int32)
            new_c = np.where(same, np.minimum(c + 1, CONF_MAX), c - 1)
            r["byte"] = np.where(new_c <= 0, nxt[hit], r["byte"]); r["conf"] = np.where(new_c <= 0, 1, new_c).astype(np.uint8)
            table[idx[hit]] = r
    miss = ~hit_a & ~hit_b
    if not miss.any():
        return
    a, b, tag, nxt, ra, rb = a[miss], b[miss], tag[miss], nxt[miss], ra[miss], rb[miss]
    empty_a, empty_b = ra["conf"] == 0, rb["conf"] == 0
    target = np.where(empty_a, a, np.where(empty_b, b, np.where(ra["conf"] <= rb["conf"], a, b)))
    both_confirmed = (ra["conf"] >= 2) & (rb["conf"] >= 2)
    ok = ~both_confirmed
    rec = np.empty(int(ok.sum()), dtype=SLOT); rec["tag"] = tag[ok]; rec["byte"] = nxt[ok]; rec["conf"] = 1
    table[target[ok]] = rec


def lookup(table, slot, tag):
    slot = slot.astype(np.int64)
    a, b = slot & ~1, slot | 1
    ra, rb = table[a], table[b]
    hit_a = (ra["conf"] > 0) & (ra["tag"] == tag)
    hit_b = ~hit_a & (rb["conf"] > 0) & (rb["tag"] == tag)
    pred = np.where(hit_a, ra["byte"], np.where(hit_b, rb["byte"], 0)).astype(np.uint8)
    conf = np.where(hit_a, ra["conf"], np.where(hit_b, rb["conf"], 0)).astype(np.uint8)
    return pred, conf


for name, log2 in conds:
    t0 = time.time()
    tables = {n: np.zeros(1 << log2, dtype=SLOT) for n in ORDERS}
    for lo in range(0, TRAIN_END, CHUNK):
        hi = min(lo + CHUNK, TRAIN_END)
        hashes = context_hashes(data, lo, hi, ORDERS)
        nxt = np.asarray(data[lo:hi], dtype=np.uint8)
        for n in ORDERS:
            first = min(max(lo, n) - lo, hi - lo)
            slot, tag = slot_and_tag(hashes[n][first:], log2)
            write(tables[n], slot, tag, nxt[first:])
    hashes = context_hashes(data, W0, W1, ORDERS)
    pred = np.zeros((len(ORDERS), W1 - W0), dtype=np.uint8); conf = np.zeros_like(pred)
    for k, n in enumerate(ORDERS):
        slot, tag = slot_and_tag(hashes[n], log2)
        pred[k], conf[k] = lookup(tables[n], slot, tag)
    d = f"{root}/{name}"
    os.makedirs(d, exist_ok=True)
    for k, n in enumerate(ORDERS):
        for kind, arr in (("pred", pred), ("conf", conf)):
            mm = np.lib.format.open_memmap(f"{d}/{kind}_n{n}.npy", mode="w+", dtype=np.uint8, shape=(size,))
            mm[W0:W1] = arr[k]; mm.flush(); del mm
    occ = {n: float((t["conf"] > 0).mean()) for n, t in tables.items()}
    json.dump({"kind": "features", "corpus": CORPUS, "corpus_bytes": size, "train_end": TRAIN_END, "orders": list(ORDERS),
               "files": {"pred": "pred_n<order>.npy", "conf": "conf_n<order>.npy"}, "log2_slots": log2, "assoc": 2, "occupancy": occ,
               "note": "assoc_tables.py: two-way set-associative tables, confirmed entries stay; features for the dev window region only"}, open(f"{d}/features.json", "w"), indent=2)
    hits = (conf > 0).mean(1)
    right = [float((pred[k][conf[k] > 0] == data[W0:W1][conf[k] > 0]).mean()) if hits[k] else 0.0 for k in range(len(ORDERS))]
    print(f"{name:10s} 2^{log2} slots, 2-way: occupancy " + ", ".join(f"{n}: {o:.2f}" for n, o in occ.items()) + "; hits "
          + ", ".join(f"{n}: {h:.3f} (right {r:.3f})" for n, h, r in zip(ORDERS, hits, right)) + f"; {time.time() - t0:.0f} s", flush=True)
    del tables
