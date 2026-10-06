"""Tables written only where the model was surprised (the fly's dopamine-gated write), at several sizes.
usage: gated_tables.py <loss.npy> <out root> cond ...   with cond = <name>:<log2 slots>:<min bits>[:keep]   (min bits 0 = every position;
keep = a slot whose entry was confirmed (confidence 2 or more) is never taken by a different context: confirmed entries stay)
For each condition: the tables of orders 8, 16, 32 written in one pass over the training bytes, but a position is written only
where the plain model's loss at it was at or above <min bits>; then the dev window region looked up and written as a feature
directory <out root>/<name>/ (sparse files: only the region is stored). Prints the share of positions written and the hit rates."""
import json
import os
import sys
import time

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import CorpusMemory, context_hashes, next_state, slot_and_tag  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
TRAIN_END, ORDERS, CHUNK = 200_000_000, (8, 16, 32), 1 << 20
N, T = 1024, 2048
W0 = 202_117_357
W1 = W0 + N * T + 1
loss_path, root = sys.argv[1], sys.argv[2]
conds = [(a.split(":")[0], int(a.split(":")[1]), float(a.split(":")[2]), len(a.split(":")) > 3 and a.split(":")[3] == "keep") for a in sys.argv[3:]]
data = np.memmap(CORPUS, dtype=np.uint8, mode="r")
size = len(data)
loss = np.load(loss_path, mmap_mode="r")


def fit_masked(mem, keep, keep_confirmed=False):
    """One pass over [0, TRAIN_END): CorpusMemory.update's rule and its 'last position of a slot wins' order, on the kept positions only."""
    for lo in range(0, TRAIN_END, CHUNK):
        hi = min(lo + CHUNK, TRAIN_END)
        k = keep[lo:hi]
        if not k.any():
            continue
        hashes = context_hashes(data, lo, hi, ORDERS)
        nxt = np.asarray(data[lo:hi], dtype=np.uint8)
        m = hi - lo
        rel = np.arange(m, dtype=np.int64)
        posbits = max(1, (m - 1).bit_length())
        for n in ORDERS:
            first = min(max(lo, n) - lo, m)
            sel = np.nonzero(k[first:])[0] + first              # kept positions with a full context
            if not len(sel):
                continue
            slot, tag = slot_and_tag(hashes[n][sel], mem.log2_slots)
            key = (slot << posbits) | rel[sel]
            order = np.argsort(key, kind="stable")
            s = slot[order]
            last = np.ones(len(s), dtype=bool)
            np.not_equal(s[1:], s[:-1], out=last[:-1])
            idx = order[last]
            table = mem.tables[n]
            old = table[slot[idx]]
            new, _ = next_state(old, tag[idx], nxt[sel][idx])
            if keep_confirmed:
                stay = (old["conf"] >= 2) & (old["tag"] != tag[idx])
                new[stay] = old[stay]
            table[slot[idx]] = new


for name, log2, min_bits, keep_confirmed in conds:
    t0 = time.time()
    keep = (loss[:TRAIN_END] >= int(round(min_bits * 16))) if min_bits > 0 else np.ones(TRAIN_END, dtype=bool)
    mem = CorpusMemory(orders=ORDERS, log2_slots=log2)
    fit_masked(mem, keep, keep_confirmed)
    pred, conf = mem.lookup(data, W0, W1)
    d = f"{root}/{name}"
    os.makedirs(d, exist_ok=True)
    for k, n in enumerate(ORDERS):
        for kind, arr in (("pred", pred), ("conf", conf)):
            mm = np.lib.format.open_memmap(f"{d}/{kind}_n{n}.npy", mode="w+", dtype=np.uint8, shape=(size,))
            mm[W0:W1] = arr[k]
            mm.flush(); del mm
    info = {"kind": "features", "corpus": CORPUS, "corpus_bytes": size, "train_end": TRAIN_END, "orders": list(ORDERS),
            "files": {"pred": "pred_n<order>.npy", "conf": "conf_n<order>.npy"}, "log2_slots": log2, "min_bits": min_bits, "keep_confirmed": keep_confirmed,
            "share_written": float(keep.mean()), "occupancy": mem.occupancy(),
            "note": f"gated_tables.py: tables of 2^{log2} slots written only at positions where the plain model's loss was >= {min_bits} bits; features for [{W0}, {W1}) only"}
    json.dump(info, open(f"{d}/features.json", "w"), indent=2)
    hits = (conf > 0).mean(1)
    right = [float((pred[k][conf[k] > 0] == data[W0:W1][conf[k] > 0]).mean()) if hits[k] else 0.0 for k in range(len(ORDERS))]
    print(f"{name:10s} 2^{log2} slots, >= {min_bits} bits{', confirmed entries stay' if keep_confirmed else ''}: {keep.mean():.1%} of positions written; occupancy "
          + ", ".join(f"{n}: {o:.2f}" for n, o in mem.occupancy().items()) + "; hits " + ", ".join(f"{n}: {h:.3f} (right {r:.3f})" for n, h, r in zip(ORDERS, hits, right))
          + f"; {time.time() - t0:.0f} s", flush=True)
    del mem
