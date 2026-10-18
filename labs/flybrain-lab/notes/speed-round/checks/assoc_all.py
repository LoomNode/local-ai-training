"""Two-way tables (checks/assoc_tables.py's rule: a context may live in either of two buddy slots; a write confirms a matching buddy, else
fills an empty one, else replaces the weaker one unless both are confirmed) at 2^<log2> slots per order, for every lookup order of the
combined arm (byte orders 4, 8, 12, 16, 32, 64; keys 116, 132, 202, 203, 301 from feat_keys/key_n<id>.npy, built by key_tables.py
--keep-keys), as full-corpus features for training: one leak-free pass (a chunk's lookups before its writes, chunks cut at train_end,
positions past train_end read the frozen tables). The counters are linked from feat_counters. usage: assoc_all.py <log2> <out dir>
With 11 orders at 2^28 the tables take 11.8 GB of RAM; about half an hour of one CPU.

Parameters (environment; the defaults are the original constants): CORPUS, TRAIN_END, MEM, CHUNK (the table pass's chunk, 2^20), KEYS
(dir of key_n<id>.npy, default MEM/feat_keys), COUNTERS (dir of the counter features, default MEM/feat_counters), OUT (default the
positional <out dir>; the positional argument wins when given), ORDERS (a comma-separated subset of the 11 table orders, default all).
ORDERS splits a big build into passes into one OUT (2^30 x 11 orders is 44 GB of tables): a pass writes only its own orders' pred/conf
files, never another's, and features.json lists every order whose pred and conf files exist in OUT (stats merged with the earlier
pass's). Without MEM/feat/features.json for this corpus (a new corpus) features.json is written from scratch.
TABLES_ONLY=1 (the use layer, 2026-10-11): build the same tables in the same order but write no features (no lookups, no pred/conf
files, no features.json) and save each order's final table as OUT/table_n<order>.npy (the SLOT records, 4 bytes per slot) with
OUT/tables.json; the final table is the one every position past train_end read, so a lookup in it reproduces the stored held-out
features exactly (checks/check_tables.py tests that)."""
import json
import os
import sys
import time

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import CONF_MAX, SLOT, context_hashes, slot_and_tag  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_common import base_info, env_int, env_str, refuse_if_in_use  # noqa: E402

CORPUS = env_str("CORPUS", "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes")
MEM = env_str("MEM", SPEED + "/results/mem")
TRAIN_END, CHUNK = env_int("TRAIN_END", 200_000_000), env_int("CHUNK", 1 << 20)
KEYS, COUNTER_DIR = env_str("KEYS", MEM + "/feat_keys"), env_str("COUNTERS", MEM + "/feat_counters")
ALL_BYTE_ORDERS = (4, 8, 12, 16, 32, 64)
ALL_KEY_IDS = (116, 132, 202, 203, 301, 200, 201, 104, 108)   # 200/201/104/108: the dictionary and the short template keys (2026-10-18)
COUNTERS = (1001, 1002, 1003, 1004, 1005, 1006, 1007)
log2 = int(sys.argv[1])
OUT = sys.argv[2] if len(sys.argv) > 2 else os.environ["OUT"]
_sel = [int(x) for x in os.environ["ORDERS"].split(",")] if os.environ.get("ORDERS") else [*ALL_BYTE_ORDERS, *ALL_KEY_IDS]
if not _sel or not set(_sel) <= {*ALL_BYTE_ORDERS, *ALL_KEY_IDS}:
    sys.exit(f"ORDERS must be a subset of {[*ALL_BYTE_ORDERS, *ALL_KEY_IDS]}, got {_sel}")
BYTE_ORDERS = tuple(n for n in ALL_BYTE_ORDERS if n in _sel)
KEY_IDS = tuple(n for n in ALL_KEY_IDS if n in _sel)
data = np.fromfile(CORPUS, dtype=np.uint8)
N = len(data)
keys = {kid: np.load(f"{KEYS}/key_n{kid}.npy", mmap_mode="r") for kid in KEY_IDS}


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
TABLES_ONLY = os.environ.get("TABLES_ONLY") == "1"
orders = (*BYTE_ORDERS, *KEY_IDS)
tables = {n: np.zeros(1 << log2, dtype=SLOT) for n in orders}
os.makedirs(OUT, exist_ok=True)
if TABLES_ONLY:
    refuse_if_in_use(OUT, [f"table_n{n}.npy" for n in orders])
    tbounds = sorted(set(range(0, TRAIN_END, CHUNK)) | {TRAIN_END})
    for i, (lo, hi) in enumerate(zip(tbounds[:-1], tbounds[1:])):
        hashes = context_hashes(data, lo, hi, BYTE_ORDERS) if BYTE_ORDERS else {}
        for kid in KEY_IDS:
            hashes[kid] = np.asarray(keys[kid][lo:hi])
        nxt = data[lo:hi]
        for n in orders:
            first = min(max(lo, n) - lo, hi - lo)
            slot, tag = slot_and_tag(hashes[n][first:], log2)
            write(tables[n], slot, tag, nxt[first:])
        if i % 50 == 0:
            print(f"{hi:,} / {TRAIN_END:,}  {time.time() - t0:.0f} s", flush=True)
    for n in orders:
        np.save(f"{OUT}/table_n{n}.npy", tables[n])
        print(f"order {n}: occupancy {float((tables[n]['conf'] > 0).mean()):.3f}, saved", flush=True)
    meta = {"kind": "tables", "corpus": CORPUS, "corpus_bytes": N, "train_end": TRAIN_END, "chunk": CHUNK, "log2_slots": log2,
            "orders": list(orders), "slot_dtype": [list(f) for f in SLOT.descr], "two_way": True, "keys": KEYS,
            "rule": "assoc_all.py write(): buddy slots (slot & ~1, slot | 1); confirm or decay a matching tag, else fill an empty slot, "
                    "else replace the weaker unless both are confirmed (conf >= 2); slot = top log2 bits of the 64-bit context hash, "
                    "tag = its low 16 bits",
            "files": "table_n<order>.npy"}
    if os.path.exists(f"{OUT}/tables.json"):                     # a pass of a split build: keep the other orders in the list
        meta["orders"] = sorted(set(json.load(open(f"{OUT}/tables.json"))["orders"]) | set(orders))
    json.dump(meta, open(f"{OUT}/tables.json", "w"), indent=2)
    print(f"tables done in {time.time() - t0:.0f} s")
    sys.exit(0)
refuse_if_in_use(OUT, [f"{k}_n{n}.npy" for n in orders for k in ("pred", "conf")])
pred = {n: np.lib.format.open_memmap(f"{OUT}/pred_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in orders}
conf = {n: np.lib.format.open_memmap(f"{OUT}/conf_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in orders}
bounds = sorted(set(range(0, N, CHUNK)) | {TRAIN_END, N})
for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
    hashes = context_hashes(data, lo, hi, BYTE_ORDERS) if BYTE_ORDERS else {}
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
            os.symlink(os.path.realpath(f"{COUNTER_DIR}/{kind}_n{n}.npy"), dst)
info, _ = base_info(f"{MEM}/feat", CORPUS, N, TRAIN_END, CHUNK)
if os.path.exists(f"{OUT}/features.json") and set(orders) != {*ALL_BYTE_ORDERS, *ALL_KEY_IDS}:     # a pass of a split build: keep the others' stats
    stats = {**json.load(open(f"{OUT}/features.json")).get("stats", {}), **{str(n): v for n, v in stats.items()}}
canon = [*ALL_BYTE_ORDERS, *ALL_KEY_IDS, *COUNTERS]
present = [n for n in canon if all(os.path.exists(f"{OUT}/{k}_n{n}.npy") for k in ("pred", "conf"))]
stats = {str(n): stats[n] if n in stats else stats[str(n)] for n in canon if n in stats or str(n) in stats}
built = [n for n in present if n not in COUNTERS]
info.update(orders=present, log2_slots=log2, two_way=True, stats=stats,
            note=f"assoc_all.py: two-way tables of 2^{log2} slots per order for byte orders {[n for n in built if n in ALL_BYTE_ORDERS]} and keys {[n for n in built if n in ALL_KEY_IDS]}, one leak-free pass; counters linked from feat_counters")
info["slots_per_order"] = 1 << log2               # the base dir's value was its own 2^27, not this build's
info.pop("feature_digest", None); info.pop("compare", None)
json.dump(info, open(f"{OUT}/features.json", "w"), indent=2)
print(f"done in {time.time() - t0:.0f} s")
