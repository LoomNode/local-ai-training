"""Top-two slots (idea 4 of 2026-10-07, second half): the table keeps two continuations per context with their counts instead of the
last one. 29 % of the round's table hits are wrong, and the second candidate is often the right one. Direct-mapped tables of 2^<log2>
slots per order for the combined arm's eleven lookup orders (byte orders 4..64; keys 116..301 from feat_keys/key_n<id>.npy), served as
two pseudo-orders per order: <o> = the first candidate with its count, 5000 + <o> = the second. Write rule: a different tag evicts the
slot (the round's rule); a hit confirms the matching candidate (and the second takes first place when it is counted higher); a hit on
neither decays the second candidate, or replaces it when it is weak (count at most 1) or empty. One leak-free pass, as the byte tables.
usage: top2_tables.py <log2> <out dir>; at 2^27 the eleven tables take 11.8 GB of RAM."""
import json
import os
import sys
import time

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import CONF_MAX, context_hashes, slot_and_tag  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
MEM = SPEED + "/results/mem"
TRAIN_END, CHUNK = 200_000_000, 1 << 20
BYTE_ORDERS = (4, 8, 12, 16, 32, 64)
KEY_IDS = (116, 132, 202, 203, 301)
SECOND = 5000
SLOT2 = np.dtype([("tag", "<u2"), ("b1", "u1"), ("c1", "u1"), ("b2", "u1"), ("c2", "u1")])
log2, OUT = int(sys.argv[1]), sys.argv[2]
if os.path.exists(f"{OUT}/features.json"):
    sys.exit(f"{OUT} exists: not rewriting a feature directory (a trainer may have it mapped)")
data = np.fromfile(CORPUS, dtype=np.uint8)
N = len(data)
keys = {kid: np.load(f"{MEM}/feat_keys/key_n{kid}.npy", mmap_mode="r") for kid in KEY_IDS}


def write(table, slot, tag, nxt):
    """One chunk's (slot, tag, next byte) triples. Duplicate contexts inside a chunk resolve by numpy's last write."""
    r = table[slot]
    empty = r["c1"] == 0
    same = ~empty & (r["tag"] == tag)
    hit1 = same & (r["b1"] == nxt)
    hit2 = same & ~hit1 & (r["c2"] > 0) & (r["b2"] == nxt)
    neither = same & ~hit1 & ~hit2
    new = np.empty(len(slot), dtype=SLOT2)
    new["tag"] = tag
    c1 = r["c1"].astype(np.int32); c2 = r["c2"].astype(np.int32)
    # eviction or an empty slot: the new byte is the only candidate
    new["b1"] = np.where(same, r["b1"], nxt); new["c1"] = np.where(same, c1, 1)
    new["b2"] = np.where(same, r["b2"], 0); new["c2"] = np.where(same, c2, 0)
    # hit on the first candidate
    new["c1"] = np.where(hit1, np.minimum(c1 + 1, CONF_MAX), new["c1"])
    # hit on the second: count it, and swap when it overtakes the first
    c2n = np.minimum(c2 + 1, CONF_MAX)
    swap = hit2 & (c2n > c1)
    new["b1"] = np.where(swap, r["b2"], new["b1"]); new["c1"] = np.where(swap, c2n, new["c1"])
    new["b2"] = np.where(swap, r["b1"], new["b2"]); new["c2"] = np.where(swap, c1, np.where(hit2, c2n, new["c2"]))
    # neither: the second candidate decays, or is replaced when weak or empty
    weak = neither & (c2 <= 1)
    new["b2"] = np.where(weak, nxt, new["b2"]); new["c2"] = np.where(weak, 1, np.where(neither, c2 - 1, new["c2"]))
    table[slot] = new


def lookup(table, slot, tag):
    r = table[slot]
    hit = (r["c1"] > 0) & (r["tag"] == tag)
    z = np.zeros(len(slot), dtype=np.uint8)
    return (np.where(hit, r["b1"], z), np.where(hit, r["c1"], z), np.where(hit, r["b2"], z), np.where(hit, r["c2"], z))


t0 = time.time()
orders = (*BYTE_ORDERS, *KEY_IDS)
tables = {n: np.zeros(1 << log2, dtype=SLOT2) for n in orders}
os.makedirs(OUT, exist_ok=True)
pred = {n: np.lib.format.open_memmap(f"{OUT}/pred_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in orders}
conf = {n: np.lib.format.open_memmap(f"{OUT}/conf_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in orders}
pred2 = {n: np.lib.format.open_memmap(f"{OUT}/pred_n{SECOND + n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in orders}
conf2 = {n: np.lib.format.open_memmap(f"{OUT}/conf_n{SECOND + n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in orders}
bounds = sorted(set(range(0, N, CHUNK)) | {TRAIN_END, N})
for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
    hashes = context_hashes(data, lo, hi, BYTE_ORDERS)
    for kid in KEY_IDS:
        hashes[kid] = np.asarray(keys[kid][lo:hi])
    nxt = data[lo:hi]
    for n in orders:
        first = min(max(lo, n) - lo, hi - lo)
        slot, tag = slot_and_tag(hashes[n][first:], log2)
        p1, c1, p2, c2 = lookup(tables[n], slot, tag)
        pred[n][lo + first:hi], conf[n][lo + first:hi] = p1, c1
        pred2[n][lo + first:hi], conf2[n][lo + first:hi] = p2, c2
        if hi <= TRAIN_END:
            write(tables[n], slot, tag, nxt[first:])
    if i % 20 == 0:
        print(f"{hi:,} / {N:,}  {time.time() - t0:.0f} s", flush=True)
stats = {}
held = slice(TRAIN_END, N)
d = data[held]
for n in orders:
    for m in (pred, conf, pred2, conf2):
        m[n].flush()
    h1 = conf[n][held] > 0; h2 = conf2[n][held] > 0
    r1 = pred[n][held] == d; r2 = pred2[n][held] == d
    stats[n] = {"occupancy": float((tables[n]["c1"] > 0).mean()), "heldout_hit_rate": float(h1.mean()),
                "heldout_right_given_hit": float(r1[h1].mean()) if h1.any() else 0.0,
                "second_present_given_hit": float(h2[h1].mean()) if h1.any() else 0.0,
                "second_right_given_first_wrong": float(r2[h1 & ~r1].mean()) if (h1 & ~r1).any() else 0.0}
    s = stats[n]
    print(f"order {n}: occupancy {s['occupancy']:.3f}, held-out hits {s['heldout_hit_rate']:.3f} (right {s['heldout_right_given_hit']:.3f}); "
          f"a second candidate at {s['second_present_given_hit']:.3f} of hits, right at {s['second_right_given_first_wrong']:.3f} of the first's misses", flush=True)
info = json.load(open(f"{MEM}/feat/features.json"))
info.update(orders=[*orders, *(SECOND + n for n in orders)], log2_slots=log2, top2=True, stats=stats,
            note=f"top2_tables.py: direct-mapped tables of 2^{log2} slots with two candidates per slot, served as <o> (first) and {SECOND}+<o> (second); one leak-free pass")
info.pop("feature_digest", None); info.pop("compare", None)
json.dump(info, open(f"{OUT}/features.json", "w"), indent=2)
print(f"done in {time.time() - t0:.0f} s")
