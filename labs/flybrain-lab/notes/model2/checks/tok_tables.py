#!/usr/bin/env python
"""Model #2's tables on tokens (spec docs/superpowers/specs/2026-10-16-model2-tokens-design.md): assoc_all.py's two-way tables with a
token position instead of a byte position, as full-stream features for training, one leak-free pass (a chunk's lookups before its
writes, chunks cut at train_end_tok, positions past it read the frozen tables).

A position is a token i. Its inputs: the n-gram orders 1, 2, 3, 4, 6, 8, 12, 16 (flybrain.corpus_memory.context_hashes on the ids:
the last n tokens before i); the byte-defined keys 116, 132, 202, 203, 301 (key_tables.compute_keys on the chunk's bytes with its
halo, read at the token's first byte: the template, words and lines of the bytes before the token); the counters 1001-1007
(counter_tables.counters on the chunk's bytes, the string/comment state carried from chunk to chunk, read at the token's first byte,
stored as the value itself, clipped to 0..127, conf 1: `value_orders` in features.json). The stored symbol is the next token:
slot = tag u2 | tok u2 | conf u1 | pad u1 (6 bytes). Features: pred_n<o>.npy (uint16, n_tokens) and conf_n<o>.npy (uint8), feature[i]
= the prediction for token i (count orders) or the state before token i (keys, counters); every input of position i is a function of
tokens[:i] / bytes[:tok_off[i]] (the prefix-honest definition, R107; tests/test_tokens.py checks it against a prefix recompute).

Chunks are groups of whole file units of at least CHUNK tokens (a unit longer than that is a chunk of its own), cut at train_end_tok;
the keys and counters of a batch of chunks are computed by JOBS workers, the table pass consumes them in corpus order. 13 tables of
2^LOG2 slots: 20 GB of RAM at 2^28; about an hour for the package corpus on 6 cores.
usage: tok_tables.py --stream DIR --out DIR [--log2 28] [--chunk 262144] [--jobs 6] [--save-tables]
       --save-tables also writes table_n<order>.npy (the final tables, for the use layer and the tests)."""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SR = os.path.normpath(os.path.join(HERE, "..", "..", "speed-round", "checks"))
SPEED = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, SR); sys.path.insert(0, SPEED); sys.path.insert(0, HERE)
from flybrain.corpus_memory import CONF_MAX, context_hashes, slot_and_tag  # noqa: E402
from counter_tables import CODE, counters  # noqa: E402
from key_tables import HALO, compute_keys  # noqa: E402
from table_common import file_cuts, halo_start, refuse_if_in_use  # noqa: E402
from tok_common import TokenStream  # noqa: E402

NGRAM = (1, 2, 3, 4, 6, 8, 12, 16)
KEYS = (116, 132, 202, 203, 301)
COUNTERS = (1001, 1002, 1003, 1004, 1005, 1006, 1007)
SLOT_TOK = np.dtype([("tag", "<u2"), ("tok", "<u2"), ("conf", "u1"), ("pad", "u1")])


def write(table, slot, tag, nxt):
    """One chunk's (slot, tag, next token) triples into a 2-way table (assoc_all.py's rule). Duplicates inside a chunk: numpy's last write."""
    slot = slot.astype(np.int64)
    a, b = slot & ~1, slot | 1
    ra, rb = table[a], table[b]
    hit_a = (ra["conf"] > 0) & (ra["tag"] == tag)
    hit_b = ~hit_a & (rb["conf"] > 0) & (rb["tag"] == tag)
    for hit, idx, rec in ((hit_a, a, ra), (hit_b, b, rb)):
        if hit.any():
            r = rec[hit].copy(); same = r["tok"] == nxt[hit]
            c = r["conf"].astype(np.int32)
            new_c = np.where(same, np.minimum(c + 1, CONF_MAX), c - 1)
            r["tok"] = np.where(new_c <= 0, nxt[hit], r["tok"]); r["conf"] = np.where(new_c <= 0, 1, new_c).astype(np.uint8)
            table[idx[hit]] = r
    miss = ~hit_a & ~hit_b
    if not miss.any():
        return
    a, b, tag, nxt, ra, rb = a[miss], b[miss], tag[miss], nxt[miss], ra[miss], rb[miss]
    empty_a, empty_b = ra["conf"] == 0, rb["conf"] == 0
    target = np.where(empty_a, a, np.where(empty_b, b, np.where(ra["conf"] <= rb["conf"], a, b)))
    ok = ~((ra["conf"] >= 2) & (rb["conf"] >= 2))
    rec = np.zeros(int(ok.sum()), dtype=SLOT_TOK); rec["tag"] = tag[ok]; rec["tok"] = nxt[ok]; rec["conf"] = 1
    table[target[ok]] = rec


def lookup(table, slot, tag):
    slot = slot.astype(np.int64)
    a, b = slot & ~1, slot | 1
    ra, rb = table[a], table[b]
    hit_a = (ra["conf"] > 0) & (ra["tag"] == tag)
    hit_b = ~hit_a & (rb["conf"] > 0) & (rb["tag"] == tag)
    pred = np.where(hit_a, ra["tok"], np.where(hit_b, rb["tok"], 0)).astype(np.uint16)
    conf = np.where(hit_a, ra["conf"], np.where(hit_b, rb["conf"], 0)).astype(np.uint8)
    return pred, conf


class Chunker:
    """Chunk bounds in tokens: whole file units grouped to at least `chunk` tokens, cut at train_end_tok."""

    def __init__(self, stream: TokenStream, chunk: int):
        starts = np.fromfile(os.path.join(stream.dir, "unit_starts.u64"), dtype=np.uint64).astype(np.int64)
        bounds = [0]
        for s in starts.tolist():
            if s - bounds[-1] >= chunk and s < stream.n:
                bounds.append(s)
        bounds = sorted(set(bounds) | {stream.train_end_tok, stream.n})
        self.bounds = [b for b in bounds if 0 <= b <= stream.n]
        self.chunks = list(zip(self.bounds[:-1], self.bounds[1:]))


_G = {}   # the workers' globals (fork)


def _inputs(job):
    """One chunk [lo, hi) in tokens: its byte keys and counter values at the token starts, entered in counter state `entry`.
    Returns (index, keys {kid: uint64[m]}, values {cid: uint8[m]}, exit state)."""
    i, lo, hi, entry = job
    st, cuts, m = _G["stream"], _G["cuts"], _G["corpus"]
    off = np.asarray(st.tok_off[lo:hi + 1], dtype=np.int64)
    blo, bhi = int(off[0]), int(off[-1])
    ctx = halo_start(cuts, blo, HALO)
    data = np.asarray(m[ctx:bhi])
    at = off[:-1]
    keys = compute_keys(data, check=blo - ctx)
    kout = {kid: np.ascontiguousarray(keys[kid][at - ctx]) for kid in KEYS}
    vals, state = counters(np.asarray(m[blo:bhi]), entry)
    vout = {cid: np.minimum(np.asarray(vals[cid])[at - blo], 127).astype(np.uint8) for cid in COUNTERS}
    return i, kout, vout, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--log2", type=int, default=28); ap.add_argument("--chunk", type=int, default=1 << 18)
    ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--save-tables", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    st = TokenStream(a.stream)
    corpus = st.layout["corpus"]
    m = np.memmap(corpus, dtype=np.uint8, mode="r")
    cuts = file_cuts(corpus)
    _G.update(stream=st, cuts=cuts, corpus=m)
    ch = Chunker(st, a.chunk)
    TE, N = st.train_end_tok, st.n
    orders = (*NGRAM, *KEYS)
    os.makedirs(a.out, exist_ok=True)
    refuse_if_in_use(a.out, [f"{k}_n{n}.npy" for n in (*orders, *COUNTERS) for k in ("pred", "conf")])
    tables = {n: np.zeros(1 << a.log2, dtype=SLOT_TOK) for n in orders}
    pred = {n: np.lib.format.open_memmap(f"{a.out}/pred_n{n}.npy", mode="w+", dtype=np.uint16, shape=(N,)) for n in (*orders, *COUNTERS)}
    conf = {n: np.lib.format.open_memmap(f"{a.out}/conf_n{n}.npy", mode="w+", dtype=np.uint8, shape=(N,)) for n in (*orders, *COUNTERS)}
    print(f"{len(ch.chunks):,} chunks of >= {a.chunk:,} tokens, {N:,} tokens, train_end_tok {TE:,}, {len(orders)} tables of 2^{a.log2} ({time.time() - t0:.0f} s)", flush=True)
    state = CODE
    B = max(1, 4 * a.jobs)
    with get_context("fork").Pool(a.jobs) as pool:
        for b0 in range(0, len(ch.chunks), B):
            batch = ch.chunks[b0:b0 + B]
            res = {i: r for i, *r in pool.map(_inputs, [(b0 + k, lo, hi, CODE) for k, (lo, hi) in enumerate(batch)])}
            for k, (lo, hi) in enumerate(batch):
                i = b0 + k
                keys, vals, exit_state = res[i]
                if state != CODE:                                        # the previous chunk ended inside a string or comment: redo this one from that state
                    _, keys, vals, exit_state = _inputs((i, lo, hi, state))
                state = exit_state
                hashes = context_hashes(st.tokens, lo, hi, NGRAM)
                hashes.update(keys)
                nxt = np.asarray(st.tokens[lo:hi], dtype=np.uint16)
                for n in orders:
                    first = min(max(lo, n) - lo, hi - lo) if n in NGRAM else 0
                    slot, tag = slot_and_tag(hashes[n][first:], a.log2)
                    p, c = lookup(tables[n], slot, tag)
                    pred[n][lo + first:hi], conf[n][lo + first:hi] = p, c
                    if hi <= TE:
                        write(tables[n], slot, tag, nxt[first:])
                for cid in COUNTERS:
                    pred[cid][lo:hi] = vals[cid]; conf[cid][lo:hi] = 1
            if (b0 // B) % 10 == 0:
                print(f"{batch[-1][1]:,} / {N:,} tokens ({time.time() - t0:.0f} s)", flush=True)
    stats = {}
    held = slice(TE, N)
    nxt_held = np.asarray(st.tokens[held], dtype=np.uint16)
    for n in orders:
        pred[n].flush(); conf[n].flush()
        h = conf[n][held] > 0
        stats[str(n)] = {"occupancy": float((tables[n]["conf"] > 0).mean()), "heldout_hit_rate": float(h.mean()),
                         "heldout_right_given_hit": float((pred[n][held][h] == nxt_held[h]).mean()) if h.any() else 0.0}
        print(f"order {n}: occupancy {stats[str(n)]['occupancy']:.3f}, held-out hits {stats[str(n)]['heldout_hit_rate']:.3f} (right {stats[str(n)]['heldout_right_given_hit']:.3f})", flush=True)
    for cid in COUNTERS:
        pred[cid].flush(); conf[cid].flush()
        v = np.asarray(pred[cid][held], dtype=np.int64)
        stats[str(cid)] = {"heldout_mean": float(v.mean()) if len(v) else 0.0, "share_nonzero": float((pred[cid][:] > 0).mean())}
    if a.save_tables:
        for n in orders:
            np.save(f"{a.out}/table_n{n}.npy", tables[n])
    info = {"kind": "features", "unit": "token", "corpus": corpus, "corpus_bytes": int(st.layout["corpus_bytes"]), "stream": os.path.abspath(a.stream),
            "n_tokens": N, "train_end_tok": TE, "train_end": int(st.layout["train_end"]), "heldout_positions": N - TE, "vocab": int(st.layout["vocab"]),
            "tokenizer": st.layout.get("tokenizer"), "tokenizer_sha256": st.layout.get("tokenizer_sha256"), "chunk": a.chunk,
            "orders": [*orders, *COUNTERS], "ngram_orders": list(NGRAM), "key_orders": list(KEYS), "value_orders": list(COUNTERS),
            "log2_slots": a.log2, "slots_per_order": 1 << a.log2, "two_way": True, "slot_dtype": [list(f) for f in SLOT_TOK.descr],
            "files": {"pred": "pred_n<order>.npy (uint16)", "conf": "conf_n<order>.npy (uint8)", **({"tables": "table_n<order>.npy"} if a.save_tables else {})},
            "alignment": "feature[i] is the table's prediction for token i from the inputs before it (conf 0 = miss); value orders hold the counter's "
                         "value at token i's first byte (conf 1). i < train_end_tok: the lookup made before the chunk holding i was written; "
                         "i >= train_end_tok: the frozen table's lookup", "stats": stats, "seconds": time.time() - t0,
            "note": "tok_tables.py: two-way tables per order for the n-gram orders (ids) and the byte keys 116/132/202/203/301 read at token starts; "
                    "counters 1001-1007 as values at token starts"}
    json.dump(info, open(f"{a.out}/features.json", "w"), indent=2)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
