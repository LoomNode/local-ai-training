"""Two-way tables (checks/assoc_tables.py's rule: a context may live in either of two buddy slots; a write confirms a matching buddy, else
fills an empty one, else replaces the weaker one unless both are confirmed) at 2^<log2> slots per order, for every lookup order of the
combined arm (byte orders 4, 8, 12, 16, 32, 64; keys 116, 132, 202, 203, 301 from feat_keys/key_n<id>.npy, built by key_tables.py
--keep-keys), as full-corpus features for training: one leak-free pass (a chunk's lookups before its writes, chunks cut at train_end,
positions past train_end read the frozen tables). The counters are linked from feat_counters. usage: assoc_all.py <log2> <out dir>
With 11 orders at 2^28 the tables take 11.8 GB of RAM; about half an hour of one CPU."""
import json
import os
import sys
import time

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import CONF_MAX, SLOT, context_hashes, slot_and_tag  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
MEM = SPEED + "/results/mem"
TRAIN_END, CHUNK = 200_000_000, 1 << 20
BYTE_ORDERS = (4, 8, 12, 16, 32, 64)
KEY_IDS = (116, 132, 202, 203, 301)
COUNTERS = (1001, 1002, 1003, 1004, 1005, 1006, 1007)
log2, OUT = int(sys.argv[1]), sys.argv[2]
data = np.fromfile(CORPUS, dtype=np.uint8)
N = len(data)
keys = {kid: np.load(f"{MEM}/feat_keys/key_n{kid}.npy", mmap_mode="r") for kid in KEY_IDS}


def write(table, slot, tag, nxt):
    """One chunk's (slot, tag, next byte) triples into a 2-way table. Duplicate contexts inside a chunk resolve by numpy's last write."""
    slot = slot.astype(np.int64)
    a, b = slot & ~1, slot | 1
    ra, rb = table[a], table[b]
    hit_a = (ra["conf"] > 0) & (ra["tag"] == tag)
    hit_b = ~hit_a & (rb["conf"] > 0) & (rb["tag"] == tag)
    for hit, idx, rec in ((hit_a, a, ra), (hit_b, b, rb)):          # confirm or decay the matching buddy
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
    ok = ~((ra["conf"] >= 2) & (rb["conf"] >= 2))
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


t0 = time.time()
orders = (*BYTE_ORDERS, *KEY_IDS)
tables = {n: np.zeros(1 << log2, dtype=SLOT) for n in orders}
os.makedirs(OUT, exist_ok=True)
pred = {n: np.lib.format.open_memmap(f"{OUT}/pred_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in orders}
conf = {n: np.lib.format.open_memmap(f"{OUT}/conf_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in orders}
bounds = sorted(set(range(0, N, CHUNK)) | {TRAIN_END, N})
for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
    hashes = context_hashes(data, lo, hi, BYTE_ORDERS)
    for kid in KEY_IDS:
        hashes[kid] = np.asarray(keys[kid][lo:hi])
    nxt = data[lo:hi]
    for n in orders:
        first = min(max(lo, n) - lo, hi - lo)                        # positions with a full context (keys: the first n positions skipped, harmless)
        slot, tag = slot_and_tag(hashes[n][first:], log2)
        p, c = lookup(tables[n], slot, tag)
        pred[n][lo + first:hi], conf[n][lo + first:hi] = p, c
        if hi <= TRAIN_END:
            write(tables[n], slot, tag, nxt[first:])
    if i % 20 == 0:
        print(f"{hi:,} / {N:,}  {time.time() - t0:.0f} s", flush=True)
stats = {}
held = slice(TRAIN_END, N)
for n in orders:
    pred[n].flush(); conf[n].flush()
    h = conf[n][held] > 0
    stats[n] = {"occupancy": float((tables[n]["conf"] > 0).mean()), "heldout_hit_rate": float(h.mean()),
                "heldout_right_given_hit": float((pred[n][held][h] == data[held][h]).mean()) if h.any() else 0.0}
    print(f"order {n}: occupancy {stats[n]['occupancy']:.3f}, held-out hits {stats[n]['heldout_hit_rate']:.3f} (right {stats[n]['heldout_right_given_hit']:.3f})", flush=True)
for n in COUNTERS:
    for kind in ("pred", "conf"):
        dst = f"{OUT}/{kind}_n{n}.npy"
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(f"{MEM}/feat_counters/{kind}_n{n}.npy"), dst)
info = json.load(open(f"{MEM}/feat/features.json"))
info.update(orders=[*orders, *COUNTERS], log2_slots=log2, two_way=True, stats=stats,
            note=f"assoc_all.py: two-way tables of 2^{log2} slots per order for byte orders {list(BYTE_ORDERS)} and keys {list(KEY_IDS)}, one leak-free pass; counters linked from feat_counters")
info.pop("feature_digest", None); info.pop("compare", None)
json.dump(info, open(f"{OUT}/features.json", "w"), indent=2)
print(f"done in {time.time() - t0:.0f} s")
