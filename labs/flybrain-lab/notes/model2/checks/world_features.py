#!/usr/bin/env python
"""The table inputs of Kyle's world file for a fine-tune (2026-10-18, lead 1: the register in the weights): the world's files laid out as a
corpus (index_project.py's index.bytes) read through the prose corpus's saved tables ALONE (t28, no world indexed in), by the use
layer's own path (use_features.text_features), written as a feature directory the runner reads like a corpus build's (pred_n<order>.npy,
conf_n<order>.npy of the file's length, features.json). Through the plain tables the world is unseen text to the table, as his turns
are at test, so the reader learns the register from the bytes and keeps an honest trust in the table. Prints counts only.
usage: world_features.py --corpus /games/ailab/corpora/kyle/world.bytes --tables /games/ailab/corpora/talk-tables/t28 --out /games/ailab/corpora/kyle/world-feat [--orders 4,8,...]"""
import argparse, json, os, sys, time
import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, f"{SPEED}/notes/speed-round/checks")
from use_features import Tables, text_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True); ap.add_argument("--tables", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--orders", default="4,8,12,16,32,64,116,132,202,203,301,1001,1002,1003,1004,1005,1006,1007")
ap.add_argument("--chunk", type=int, default=1 << 22, help="bytes per pass (each pass recomputes from the last file header at least 64 KB before it)")
a = ap.parse_args()
os.umask(0o077)
orders = [int(n) for n in a.orders.split(",")]
tables = Tables(a.tables)
data = np.fromfile(a.corpus, dtype=np.uint8); N = len(data)
os.makedirs(a.out, exist_ok=True)
pred = {n: np.lib.format.open_memmap(f"{a.out}/pred_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in orders}
conf = {n: np.lib.format.open_memmap(f"{a.out}/conf_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in orders}
HEADER = b"\n\n// ==== "; t0 = time.time(); lo = 0
while lo < N:
    hi = min(N, lo + a.chunk)
    back = max(0, lo - (1 << 16)); H = bytes(data[max(0, back - (1 << 17)):back]).rfind(HEADER)
    start = (max(0, back - (1 << 17)) + H) if H >= 0 else 0            # a file header at least 64 KB before the chunk (or the file's start)
    p, c = text_features(tables, data[start:hi], orders)                 # rows 0..hi-start: position start+r's inputs
    for j, n in enumerate(orders):
        pred[n][lo:hi] = p[lo - start:hi - start, j]; conf[n][lo:hi] = c[lo - start:hi - start, j]
    print(f"  {hi:,} / {N:,} bytes ({time.time() - t0:.0f} s)", flush=True); lo = hi
for n in orders: pred[n].flush(); conf[n].flush()
hits = {n: float((conf[n][:] > 0).mean()) for n in orders}
json.dump({"kind": "features", "corpus": a.corpus, "corpus_bytes": N, "train_end": N, "orders": orders, "log2_slots": tables.log2, "two_way": True,
           "tables": a.tables, "files": {"pred": "pred_n<order>.npy", "conf": "conf_n<order>.npy"},
           "alignment": "feature[p] is the table's prediction for the byte at absolute position p from the bytes before it (conf 0 = miss); every position read the saved tables (the world is unseen by them)",
           "stats": {str(n): {"hit_rate": hits[n]} for n in orders}, "note": "world_features.py: the world file's inputs through the saved tables alone, by use_features.text_features"},
          open(f"{a.out}/features.json", "w"), indent=2)
print(f"done: {N:,} bytes, {len(orders)} orders, hit rates " + ", ".join(f"{n}:{hits[n]:.2f}" for n in orders if n < 1000) + f" ({time.time() - t0:.0f} s)")
