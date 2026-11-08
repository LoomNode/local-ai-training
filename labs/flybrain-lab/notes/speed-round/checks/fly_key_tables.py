"""Tests 1 and 2 from the fly's circuit (Kyle, 2026-10-09: "queue tests 1 and 2 aswell"), as corpus features beside feat_all28's 18:

  3001  the fly's key: a similarity-preserving address over the last 16 bytes. FlyHash on a positional one-hot of the context is a
        locality-sensitive hash for Hamming distance (Dasgupta 2017's l2 on one-hot bits), and its exact, cheap form is bit sampling:
        K random masks of R of the 16 positions, each hashed into its own two-way table (an exact table over its sub-context);
        a context one byte off shares each mask with probability 1 - R/16, so the K slots are shared in proportion to the distance
        (the fly's graded overlap). Read as the fly reads its K active slots: the byte most of the hitting masks agree on, with the
        agreement count as the confidence (the mean of the k slots, Dasgupta 2018). No mask favours the last byte: the sampling is
        the fly's, random. K = 8, R = 6, 2^26 slots per mask.
  2008, 2016  the project tier (item 6; the fly's parallel compartments): the order-8 and order-16 context hashed together with the
        directory of the file the position lies in (flybrain.data.files), its own two-way tables of 2^26: a memory of what this
        directory's files did, beside the corpus-wide one. Dev files have their directories in the train split for 353 of 500.
One leak-free pass as in assoc_all.py (a chunk's lookups before its writes; positions past train_end read frozen tables); the other
18 orders are symlinked from feat_all28, so one feature directory serves both arms. usage: fly_key_tables.py <out dir>"""
import json
import os
import sys
import time

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import _FNV_OFFSET, _FNV_PRIME, _U64, _finalise, CONF_MAX, SLOT, context_hashes, slot_and_tag  # noqa: E402
from flybrain.data.files import file_table  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
MEM = SPEED + "/results/mem"
TRAIN_END, CHUNK = 200_000_000, 1 << 20
LOG2 = 26
N_CTX, K_MASKS, R_POS, SEED = 16, 8, 6, 7
FLY_ID, DIR_IDS = 3001, {8: 2008, 16: 2016}
ALL18 = (4, 8, 12, 16, 32, 64, 116, 132, 202, 203, 301, 1001, 1002, 1003, 1004, 1005, 1006, 1007)
OUT = sys.argv[1]


def write(table, slot, tag, nxt):
    """(assoc_all.py's two-way rule, copied: that script builds at import.) One chunk's (slot, tag, next byte) triples into a 2-way table. Duplicate contexts inside a chunk resolve by numpy's last write."""
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


def masked_hashes(data, lo, hi, masks):
    """For every p in [lo, hi) and every mask (a sorted tuple of offsets j: the j-th byte back from p), the hash of those bytes, salted
    by the mask's index. Reads nothing at or after p."""
    K, m = N_CTX, hi - lo
    left = max(0, lo - K)
    buf = np.zeros(K + m, dtype=np.uint64)
    buf[K - (lo - left):] = data[left:hi]
    out = []
    for i, mask in enumerate(masks):
        h = np.full(m, _FNV_OFFSET, dtype=np.uint64)
        for j in mask:
            h ^= buf[K - j: K - j + m]
            h *= _U64(_FNV_PRIME)
        out.append(_finalise(h ^ _U64(0x9E3779B97F4A7C15 * (i + 1) & 0xFFFFFFFFFFFFFFFF)))
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    masks = [tuple(sorted(rng.choice(np.arange(1, N_CTX + 1), size=R_POS, replace=False).tolist())) for _ in range(K_MASKS)]
    print("masks (bytes back from the position):", masks, flush=True)
    data = np.fromfile(CORPUS, dtype=np.uint8); N = len(data)
    files = file_table(CORPUS)
    dirs = sorted({f.dirname for f in files}); dir_index = {d: i for i, d in enumerate(dirs)}
    starts = np.array([f.start for f in files] + [N]); ids = np.array([dir_index[f.dirname] for f in files], dtype=np.int64)
    dir_of_pos = np.repeat(ids, np.diff(starts)); dir_of_pos = np.concatenate([np.full(starts[0], ids[0]), dir_of_pos])[:N]
    dir_salt = _finalise((dir_of_pos.astype(np.uint64) + _U64(1)) * _U64(0xD6E8FEB86659FD93))
    print(f"{len(files):,} files in {len(dirs):,} directories", flush=True)
    fly_tables = [np.zeros(1 << LOG2, dtype=SLOT) for _ in range(K_MASKS)]
    dir_tables = {n: np.zeros(1 << LOG2, dtype=SLOT) for n in DIR_IDS}
    os.makedirs(OUT, exist_ok=True)
    out_ids = [FLY_ID, *DIR_IDS.values()]
    pred = {n: np.lib.format.open_memmap(f"{OUT}/pred_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in out_ids}
    conf = {n: np.lib.format.open_memmap(f"{OUT}/conf_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in out_ids}
    bounds = sorted(set(range(0, N, CHUNK)) | {TRAIN_END, N})
    agree_hist = np.zeros(K_MASKS + 1, dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        nxt = data[lo:hi]
        first = min(max(lo, N_CTX) - lo, hi - lo)                   # positions with the full 16-byte context
        # the fly key: K masked lookups, then the vote
        hs = masked_hashes(data, lo, hi, masks)
        preds = np.zeros((K_MASKS, hi - lo - first), dtype=np.uint8); confs = np.zeros_like(preds)
        for k in range(K_MASKS):
            slot, tag = slot_and_tag(hs[k][first:], LOG2)
            preds[k], confs[k] = lookup(fly_tables[k], slot, tag)
            if hi <= TRAIN_END:
                write(fly_tables[k], slot, tag, nxt[first:])
        hit = confs > 0
        votes = np.zeros((256, preds.shape[1]), dtype=np.int16)
        cols = np.broadcast_to(np.arange(preds.shape[1]), preds.shape)
        np.add.at(votes, (preds[hit].astype(np.int64), cols[hit]), 1)
        best = votes.argmax(0); agreement = votes[best, np.arange(preds.shape[1])]
        pred[FLY_ID][lo + first:hi] = best.astype(np.uint8); conf[FLY_ID][lo + first:hi] = agreement.astype(np.uint8)
        if lo >= TRAIN_END:
            agree_hist += np.bincount(agreement, minlength=K_MASKS + 1)[:K_MASKS + 1]
        # the project tier: order-n context hashed with the directory
        ctx = context_hashes(data, lo, hi, tuple(DIR_IDS))
        for n, oid in DIR_IDS.items():
            f2 = min(max(lo, n) - lo, hi - lo)
            h = _finalise(ctx[n][f2:] ^ dir_salt[lo + f2:hi])
            slot, tag = slot_and_tag(h, LOG2)
            p, c = lookup(dir_tables[n], slot, tag)
            pred[oid][lo + f2:hi], conf[oid][lo + f2:hi] = p, c
            if hi <= TRAIN_END:
                write(dir_tables[n], slot, tag, nxt[f2:])
        if i % 20 == 0:
            print(f"{hi:,} / {N:,}  {time.time() - t0:.0f} s", flush=True)
    stats = {}
    held = slice(TRAIN_END, N)
    for oid in out_ids:
        pred[oid].flush(); conf[oid].flush()
        h = conf[oid][held] > 0
        stats[oid] = {"heldout_hit_rate": float(h.mean()), "heldout_right_given_hit": float((pred[oid][held][h] == data[held][h]).mean()) if h.any() else 0.0}
        print(f"order {oid}: held-out hits {stats[oid]['heldout_hit_rate']:.3f} (right {stats[oid]['heldout_right_given_hit']:.3f})", flush=True)
    for k in range(1, K_MASKS + 1):
        sel = conf[FLY_ID][held] == k
        if sel.any():
            print(f"  fly key: {k} of {K_MASKS} masks agree at {sel.mean():.3f} of held-out positions, right {float((pred[FLY_ID][held][sel] == data[held][sel]).mean()):.3f}", flush=True)
    stats[FLY_ID].update(masks=masks, agreement_hist_heldout=agree_hist.tolist(), occupancy=[float((t["conf"] > 0).mean()) for t in fly_tables])
    for oid, n in zip(DIR_IDS.values(), DIR_IDS):
        stats[oid]["occupancy"] = float((dir_tables[n]["conf"] > 0).mean())
    for n in ALL18:
        for kind in ("pred", "conf"):
            dst = f"{OUT}/{kind}_n{n}.npy"
            if not os.path.exists(dst):
                os.symlink(os.path.realpath(f"{MEM}/feat_all28/{kind}_n{n}.npy"), dst)
    info = json.load(open(f"{MEM}/feat_all28/features.json"))
    info.update(orders=[*ALL18, *out_ids], stats={**info.get("stats", {}), **{str(k): v for k, v in stats.items()}},
                note=info.get("note", "") + f" | fly_key_tables.py: 3001 = fly key (bit-sampling LSH, {K_MASKS} masks of {R_POS} of the last {N_CTX} bytes, 2^{LOG2} each, agreement vote); 2008/2016 = directory-keyed orders 8/16, 2^{LOG2} each")
    json.dump(info, open(f"{OUT}/features.json", "w"), indent=2)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
