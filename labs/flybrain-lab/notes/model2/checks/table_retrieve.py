#!/usr/bin/env python
"""The tables choose what the big model reads (Kyle, 2026-10-20: "lets try" the table-driven retrieval): the chooser of qwen_memory.py's
memory condition, redone with the lab's own contexts instead of 12-byte spans. The same chunk pool (Kyle's files, half-overlapping
1,500-byte chunks in the same order, so chunk ids agree), the same budget (top 2 chunks, 3,200 bytes, the window's own text excluded),
the same prompt; only the vote changes:
  every position of the window's first half votes for the chunks holding its LONGEST matching byte context (orders 64 down to 4, the
  tables' context_hashes; the vote is worth the order's length) and, separately, its longest matching KEY (the tables' key families:
  lines 301, templates 132/116/108/104, words 203/202/201/200, weighted by the context each summarises); a vote is split among the chunks
  the context occurs in, and a context found in more than --max-chunks chunks is no cue at all (without that cap the first choices were
  chunks that merely shared the window's style: hundreds of order-4/8 and template hits, two shell scripts for a turn of prose, +0.34
  on the smoke windows). The top two chunks by vote are the memory.
Runs on the CPU with the lab venv, once, before Qwen: writes <out>/choices.json (chunk ids per window, by set) and prints counts only:
how many windows get a memory, bytes, and the agreement with the span chooser (same top chunk, Jaccard of the chosen sets), so the test's
room to differ is known before GPU time. qwen_memory.py --choices <out>/choices.json then scores Qwen with these chunks.
usage: table_retrieve.py --out DIR [--world DIR] [--heldout F --json F] [--kyle-every 1 --dlg-every 4] [--top 2 --max-mem 3200 --min-vote 24]"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.abspath(os.path.join(HERE, "..", "..", "speed-round", "checks"))
sys.path.insert(0, CK)
from table_common import SPEED  # noqa: E402
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import context_hashes  # noqa: E402
from key_tables import compute_keys  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--world", default="/games/ailab/corpora/kyle/world/files")
ap.add_argument("--heldout", default="/games/ailab/corpora/kyle/heldout.bytes"); ap.add_argument("--json", default="/games/ailab/corpora/kyle/heldout.json")
ap.add_argument("--T", type=int, default=2048); ap.add_argument("--kyle-every", type=int, default=1); ap.add_argument("--dlg-every", type=int, default=4)
ap.add_argument("--chunk", type=int, default=1500); ap.add_argument("--top", type=int, default=2); ap.add_argument("--max-mem", type=int, default=3200)
ap.add_argument("--min-vote", type=float, default=24.0, help="a window with less total vote for its best chunk gets no memory (the span chooser's min_hits=3)")
ap.add_argument("--span", type=int, default=12); ap.add_argument("--min-hits", type=int, default=3)
ap.add_argument("--max-chunks", type=int, default=16, help="a context found in more chunks than this is no cue: it neither votes nor takes its position")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
os.umask(0o077); os.makedirs(a.out, exist_ok=True)

BYTE_ORDERS = (64, 32, 16, 12, 8, 4)                                           # longest first
KEY_WEIGHT = ((301, 64), (132, 32), (116, 16), (203, 16), (202, 11), (108, 8), (201, 6), (104, 4), (200, 3))   # longest first

# ---- the chunk pool: qwen_memory.py's, byte for byte ------------------------------------------------------------------------------------
t0 = time.time(); chunks = []; chunk_file = []; chunk_start = []; files = []
for p in sorted(glob.glob(os.path.join(a.world, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    b = open(p, "rb").read(); fi = len(files); files.append(b)
    for s in range(0, max(1, len(b) - a.chunk // 2), a.chunk // 2 if len(b) > a.chunk else a.chunk):
        c = b[s:s + a.chunk]
        if len(c) < 64: continue
        chunks.append(c); chunk_file.append(fi); chunk_start.append(s)
pool_digest = hashlib.sha256(b"".join(hashlib.sha256(c).digest() for c in chunks)).hexdigest()
print(f"pool: {len(chunks):,} chunks of <= {a.chunk} bytes from {len(files):,} files ({time.time() - t0:.0f} s), digest {pool_digest[:16]}", flush=True)

# ---- the span chooser (qwen_memory.py's, for the agreement count) --------------------------------------------------------------------------
MUL = np.uint64(0x9E3779B97F4A7C15)
def span_hashes(b: bytes, S: int) -> np.ndarray:
    d = np.frombuffer(b, dtype=np.uint8).astype(np.uint64)
    if len(d) < S: return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(d) - S + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(S): h = h * MUL + d[j:len(d) - S + 1 + j] + np.uint64(1)
    return h
def sorted_index(parts_h, parts_i):
    H = np.concatenate(parts_h); I = np.concatenate(parts_i); o = np.argsort(H, kind="stable"); return H[o], I[o]
t0 = time.time(); hs = []; ids = []
for i, c in enumerate(chunks):
    h = np.unique(span_hashes(c, a.span)); hs.append(h); ids.append(np.full(len(h), i, dtype=np.int32))
SH, SI = sorted_index(hs, ids); del hs, ids
print(f"span chooser: {len(SH):,} span hashes ({time.time() - t0:.0f} s)", flush=True)

def span_choose(prefix: bytes, W: bytes) -> list[int]:
    q = np.unique(span_hashes(prefix, a.span)); lo = np.searchsorted(SH, q, "left"); hi = np.searchsorted(SH, q, "right")
    counts = np.zeros(len(chunks), dtype=np.int32)
    for l_, h_ in zip(lo, hi):
        if h_ > l_: np.add.at(counts, SI[l_:h_], 1)
    best = np.argsort(-counts)[: a.top * 3]; out = []; total = 0
    for i in best:
        if counts[i] < a.min_hits or len(out) >= a.top: break
        c = chunks[i]
        if c[:200] in W or W[:200] in c: continue
        out.append(int(i)); total += len(c)
        if total >= a.max_mem: break
    return out

# ---- the tables' chooser: every file hashed with the tables' contexts and keys, rows assigned to its chunks --------------------------------
t0 = time.time()
ORDERS = [*BYTE_ORDERS, *(k for k, _ in KEY_WEIGHT)]
parts = {n: ([], []) for n in ORDERS}
by_file = {}
for ci, (fi, s) in enumerate(zip(chunk_file, chunk_start)): by_file.setdefault(fi, []).append((ci, s))
for fi, b in enumerate(files):
    if fi not in by_file: continue
    data = np.frombuffer(b, dtype=np.uint8); N = len(data)
    hb = context_hashes(data, 0, N, BYTE_ORDERS); hk = compute_keys(data)
    for ci, s in by_file[fi]:
        e = min(N, s + a.chunk)
        for n in BYTE_ORDERS:
            lo = max(s, n)                                                          # rows with a full context only
            if e > lo:
                u = np.unique(hb[n][lo:e]); parts[n][0].append(u); parts[n][1].append(np.full(len(u), ci, dtype=np.int32))
        for n, _ in KEY_WEIGHT:
            u = np.unique(np.asarray(hk[n][s:e])); parts[n][0].append(u); parts[n][1].append(np.full(len(u), ci, dtype=np.int32))
    if fi % 500 == 0: print(f"  hashed {fi:,} / {len(files):,} files ({time.time() - t0:.0f} s)", flush=True)
IDX = {n: sorted_index(*parts[n]) for n in ORDERS}; del parts
print(f"tables' chooser: " + ", ".join(f"{n}:{len(IDX[n][0]) / 1e6:.1f}M" for n in ORDERS) + f" rows ({time.time() - t0:.0f} s)", flush=True)

def vote(q_hashes: dict, valid_from: dict, family, counts: np.ndarray) -> None:
    """Longest-first per position: the first order of `family` whose hash at that position occurs in the pool takes the position's vote,
    worth its weight, split among the chunks it occurs in. valid_from[n]: the first row with a full context of order n."""
    rows = len(next(iter(q_hashes.values()))); taken = np.zeros(rows, dtype=bool)
    for n, w in family:
        H, I = IDX[n]; q = q_hashes[n]
        lo = np.searchsorted(H, q, "left"); hi = np.searchsorted(H, q, "right")
        hit = (hi > lo) & (hi - lo <= a.max_chunks) & ~taken; hit[:valid_from.get(n, 0)] = False
        if hit.any():
            starts, lens = lo[hit], (hi - lo)[hit]; total = int(lens.sum())
            idx = np.arange(total) + np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)   # the ragged ranges [lo, hi)
            np.add.at(counts, I[idx], np.repeat(w / lens, lens))
        taken |= hit

def table_choose(prefix: bytes, W: bytes) -> tuple[list[int], float]:
    data = np.frombuffer(prefix, dtype=np.uint8); ext = np.concatenate([data, np.zeros(1, np.uint8)]); M = len(ext)
    hb = context_hashes(ext, 0, M, BYTE_ORDERS); hk = compute_keys(ext)
    counts = np.zeros(len(chunks), dtype=np.float64)
    vote({n: hb[n] for n in BYTE_ORDERS}, {n: n for n in BYTE_ORDERS}, [(n, n) for n in BYTE_ORDERS], counts)
    vote({n: np.asarray(hk[n]) for n, _ in KEY_WEIGHT}, {}, KEY_WEIGHT, counts)
    best = np.argsort(-counts)[: a.top * 3]; out = []; total = 0
    for i in best:
        if counts[i] < a.min_vote or len(out) >= a.top: break
        c = chunks[i]
        if c[:200] in W or W[:200] in c: continue
        out.append(int(i)); total += len(c)
        if total >= a.max_mem: break
    return out, float(counts[best[0]])

# ---- every window's choice, both choosers ---------------------------------------------------------------------------------------------------
held = open(a.heldout, "rb").read(); meta = json.load(open(a.json)); T = a.T
sets = {"kyle": (meta["kyle"]["offset"], meta["kyle"]["windows"], a.kyle_every), "dlg": (meta["dialogue"]["offset"], meta["dialogue"]["windows"], a.dlg_every)}
out = {"pool": {"world": a.world, "chunk": a.chunk, "chunks": len(chunks), "digest": pool_digest}, "chooser": {"byte_orders": BYTE_ORDERS, "keys": KEY_WEIGHT,
       "top": a.top, "max_mem": a.max_mem, "min_vote": a.min_vote, "max_chunks": a.max_chunks}, "choices": {}, "span_choices": {}, "stats": {}}
for name, (off, nwin, every) in sets.items():
    t0 = time.time(); ch = []; sp = []; same_top = 0; jac = []; mem_bytes = 0; got = 0; both = 0; best_votes = []
    for w in range(0, nwin, every):
        if a.limit and len(ch) >= a.limit: break
        W = held[off + w * T: off + (w + 1) * T]; first = W[:T // 2]
        t, v = table_choose(first, W); s = span_choose(first, W); ch.append(t); sp.append(s); best_votes.append(v)
        if t: got += 1; mem_bytes += sum(len(chunks[i]) for i in t)
        if t and s:
            both += 1; same_top += int(t[0] == s[0]); jac.append(len(set(t) & set(s)) / len(set(t) | set(s)))
    st = {"windows": len(ch), "memory_found": got, "memory_bytes_mean": mem_bytes / got if got else 0, "best_vote_median": float(np.median(best_votes)),
          "both_found": both, "same_top_chunk": same_top, "jaccard_mean": float(np.mean(jac)) if jac else 0, "identical_sets": int(sum(j == 1.0 for j in jac)),
          "seconds": round(time.time() - t0)}
    out["choices"][name] = ch; out["span_choices"][name] = sp; out["stats"][name] = st
    print(f"{name}: {st['windows']} windows; the tables choose a memory for {got} ({st['memory_bytes_mean']:.0f} bytes on average, best vote median {st['best_vote_median']:.0f}); "
          f"vs the span chooser: same top chunk {same_top}/{both}, identical sets {st['identical_sets']}, Jaccard {st['jaccard_mean']:.2f} ({st['seconds']} s)", flush=True)
json.dump(out, open(os.path.join(a.out, "choices.json"), "w"))
print("wrote", os.path.join(a.out, "choices.json"))
