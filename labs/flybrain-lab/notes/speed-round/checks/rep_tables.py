"""Test 2's second half (the fly's slow compartment admits a memory only after repetition: spaced training, Aso & Rubin 2016): the
six byte orders of feat_all28 rebuilt at 2^28 two-way slots with a repetition-gated write. A context's first sighting sets a bit in a
2^30-bit filter and writes nothing; its second sighting writes. Confirming and decaying an entry that is there is unchanged. Same
slots, same key, a different admission rule: one-off contexts no longer take slots from repeated ones. Keys and counters are
symlinked from feat_all28. usage: rep_tables.py <out dir>  (about 7 GB of RAM, half an hour of one CPU)"""
import json
import os
import sys
import time

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import CONF_MAX, SLOT, _U64, context_hashes, slot_and_tag  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
MEM = SPEED + "/results/mem"
TRAIN_END, CHUNK = 200_000_000, 1 << 20
LOG2, SEEN_BITS = 28, 30
BYTE_ORDERS = (4, 8, 12, 16, 32, 64)
OTHERS = (116, 132, 202, 203, 301, 1001, 1002, 1003, 1004, 1005, 1006, 1007)
OUT = sys.argv[1]


def write_gated(table, seen, slot, tag, h, nxt):
    """assoc_all.py's two-way write, with the miss branch gated by the seen filter (bit = the top SEEN_BITS of the hash)."""
    slot = slot.astype(np.int64)
    a, b = slot & ~1, slot | 1
    ra, rb = table[a], table[b]
    hit_a = (ra["conf"] > 0) & (ra["tag"] == tag)
    hit_b = ~hit_a & (rb["conf"] > 0) & (rb["tag"] == tag)
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
    bit = (h[miss] >> _U64(64 - SEEN_BITS)).astype(np.int64)
    admit = seen[bit]                                                 # seen before: write; else remember it and write nothing
    seen[bit[~admit]] = True
    if not admit.any():
        return
    sel = np.flatnonzero(miss)[admit]
    a, b, tag, nxt, ra, rb = a[sel], b[sel], tag[sel], nxt[sel], ra[sel], rb[sel]
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
    return (np.where(hit_a, ra["byte"], np.where(hit_b, rb["byte"], 0)).astype(np.uint8),
            np.where(hit_a, ra["conf"], np.where(hit_b, rb["conf"], 0)).astype(np.uint8))


t0 = time.time()
data = np.fromfile(CORPUS, dtype=np.uint8); N = len(data)
tables = {n: np.zeros(1 << LOG2, dtype=SLOT) for n in BYTE_ORDERS}
seen = {n: np.zeros(1 << SEEN_BITS, dtype=bool) for n in BYTE_ORDERS}
os.makedirs(OUT, exist_ok=True)
pred = {n: np.lib.format.open_memmap(f"{OUT}/pred_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in BYTE_ORDERS}
conf = {n: np.lib.format.open_memmap(f"{OUT}/conf_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in BYTE_ORDERS}
bounds = sorted(set(range(0, N, CHUNK)) | {TRAIN_END, N})
for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
    hashes = context_hashes(data, lo, hi, BYTE_ORDERS)
    nxt = data[lo:hi]
    for n in BYTE_ORDERS:
        first = min(max(lo, n) - lo, hi - lo)
        h = hashes[n][first:]
        slot, tag = slot_and_tag(h, LOG2)
        p, c = lookup(tables[n], slot, tag)
        pred[n][lo + first:hi], conf[n][lo + first:hi] = p, c
        if hi <= TRAIN_END:
            write_gated(tables[n], seen[n], slot, tag, h, nxt[first:])
    if i % 20 == 0:
        print(f"{hi:,} / {N:,}  {time.time() - t0:.0f} s", flush=True)
stats = {}
held = slice(TRAIN_END, N)
for n in BYTE_ORDERS:
    pred[n].flush(); conf[n].flush()
    hh = conf[n][held] > 0
    stats[n] = {"occupancy": float((tables[n]["conf"] > 0).mean()), "seen_bits_set": float(seen[n].mean()), "heldout_hit_rate": float(hh.mean()),
                "heldout_right_given_hit": float((pred[n][held][hh] == data[held][hh]).mean()) if hh.any() else 0.0}
    print(f"order {n}: occupancy {stats[n]['occupancy']:.3f}, seen bits {stats[n]['seen_bits_set']:.3f}, held-out hits {stats[n]['heldout_hit_rate']:.3f} (right {stats[n]['heldout_right_given_hit']:.3f})", flush=True)
for n in OTHERS:
    for kind in ("pred", "conf"):
        dst = f"{OUT}/{kind}_n{n}.npy"
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(f"{MEM}/feat_all28/{kind}_n{n}.npy"), dst)
info = json.load(open(f"{MEM}/feat_all28/features.json"))
info.update(orders=[*BYTE_ORDERS, *OTHERS], stats={**info.get("stats", {}), **{str(k): v for k, v in stats.items()}}, repetition_gated=True,
            note=info.get("note", "") + f" | rep_tables.py: byte orders rebuilt at 2^{LOG2} with a repetition-gated write (2^{SEEN_BITS}-bit seen filter); keys and counters linked from feat_all28")
json.dump(info, open(f"{OUT}/features.json", "w"), indent=2)
print(f"done in {time.time() - t0:.0f} s")
