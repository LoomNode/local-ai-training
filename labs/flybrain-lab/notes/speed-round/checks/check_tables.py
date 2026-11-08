"""Checks a saved table dump (assoc_all.py with TABLES_ONLY=1) against the features of a normal build of the same corpus and size: every
position past train_end read the final tables, so a lookup in the dump must reproduce pred and conf there exactly, for every order.
usage: check_tables.py <tables dir> <features dir> [--ranges LO:HI ...]   (default: the first 2,097,153 positions past train_end, the
fixed dev windows of the package corpus). The corpus and the key files come from the dump's tables.json. CPU and memory-mapped files
only; exits 1 on any difference."""
import argparse
import json
import sys

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import context_hashes, slot_and_tag  # noqa: E402

BYTE_ORDERS = (4, 8, 12, 16, 32, 64)


def lookup(table, slot, tag):
    """assoc_all.py's lookup: the matching buddy of the two slots, else nothing (pred 0, conf 0)."""
    slot = slot.astype(np.int64)
    a, b = slot & ~1, slot | 1
    ra, rb = table[a], table[b]
    hit_a = (ra["conf"] > 0) & (ra["tag"] == tag)
    hit_b = ~hit_a & (rb["conf"] > 0) & (rb["tag"] == tag)
    pred = np.where(hit_a, ra["byte"], np.where(hit_b, rb["byte"], 0)).astype(np.uint8)
    conf = np.where(hit_a, ra["conf"], np.where(hit_b, rb["conf"], 0)).astype(np.uint8)
    return pred, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables"); ap.add_argument("features")
    ap.add_argument("--ranges", nargs="*", default=None, help="LO:HI position ranges (all past train_end)")
    a = ap.parse_args()
    meta = json.load(open(f"{a.tables}/tables.json"))
    log2, train_end = meta["log2_slots"], meta["train_end"]
    data = np.memmap(meta["corpus"], dtype=np.uint8, mode="r")
    ranges = [tuple(int(x) for x in r.split(":")) for r in a.ranges] if a.ranges else [(train_end, min(len(data), train_end + 1024 * 2048 + 1))]
    if any(lo < train_end or hi > len(data) or lo >= hi for lo, hi in ranges):
        sys.exit(f"ranges must lie in [{train_end}, {len(data)}): {ranges}")
    bad = 0
    for n in meta["orders"]:
        table = np.load(f"{a.tables}/table_n{n}.npy", mmap_mode="r")
        fp, fc = np.load(f"{a.features}/pred_n{n}.npy", mmap_mode="r"), np.load(f"{a.features}/conf_n{n}.npy", mmap_mode="r")
        key = None if n in BYTE_ORDERS else np.load(f"{meta['keys']}/key_n{n}.npy", mmap_mode="r")
        diff = total = 0
        for lo, hi in ranges:
            h = context_hashes(data, lo, hi, (n,))[n] if key is None else np.asarray(key[lo:hi])
            first = min(max(lo, n) - lo, hi - lo)
            slot, tag = slot_and_tag(h[first:], log2)
            p, c = lookup(table, slot, tag)
            diff += int((p != fp[lo + first:hi]).sum() + (c != fc[lo + first:hi]).sum()); total += hi - lo - first
        print(f"order {n}: {total:,} positions, {diff} differences", flush=True)
        bad += diff
    print("tables reproduce the features" if bad == 0 else f"MISMATCH: {bad} differences")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
