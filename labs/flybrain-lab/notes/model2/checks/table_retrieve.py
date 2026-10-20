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
Three modes (the follow-ups of 2026-10-20, Kyle: "Queue up those 3"): --mode longest is the vote above; --mode breadth is the span count
redone with the tables' hashes (every distinct context of --families that occurs in the pool adds one to each chunk holding it: no
weights, no longest-first; the same families and cap as the longest vote by default, the keys as vocabularies), to tell whether the cue was the whole
difference; --mode spans is the 12-byte span chooser itself, which with --dedup never takes two chunks of one text (a chunk whose first
200 bytes lie in a chosen chunk, or the reverse, is skipped: overlapping neighbours, copies of a file; the span run spent a quarter of
its budget on them). Every mode also writes span_scores (the plain span chooser's hit counts) for the gate sweep (gate_sweep.py).
Runs on the CPU with the lab venv, once, before Qwen: writes <out>/choices.json (chunk ids per window, by set) and prints counts only:
how many windows get a memory, bytes, and the agreement with the span chooser (same top chunk, Jaccard of the chosen sets), so the test's
room to differ is known before GPU time. qwen_memory.py --choices <out>/choices.json then scores Qwen with these chunks.
usage: table_retrieve.py --out DIR [--world DIR] [--heldout F --json F] [--kyle-every 1 --dlg-every 4] [--top 2 --max-mem 3200 --min-vote 24]
       [--mode longest|breadth|spans] [--families 16,301,203,202,132,116] [--dedup] [--max-chunks N (longest and breadth: 16; 0 = none)]
The sweep (Kyle, 2026-10-20: "exhaustively test all 3 separately"): --vote one|order|rarity weights the breadth vote (one per distinct
context; the family's length; one over the number of chunks holding it); --dedup-len L is the containment test's length (200); --dedup-frac F
also skips a candidate sharing more than F of its 12-byte spans with a chosen chunk; --chunk, --top and --max-mem set the pool and the
budget (qwen_memory.py must be given the same --chunk: the digest checks it). choices.json also carries the pool's file index and extension
per chunk, the first half's distinct span count and the best vote per window, for the gate sweep."""
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
ap.add_argument("--span", type=int, default=12); ap.add_argument("--min-hits", type=float, default=3, help="the floor on the best chunk's vote (spans and breadth; the rarity vote is fractional: use 1; the length vote: 36)")
ap.add_argument("--max-chunks", type=int, default=None, help="a context found in more chunks than this is no cue (longest: it neither votes nor takes its position; default 16); breadth: default 0, no cap")
ap.add_argument("--mode", default="longest", choices=["longest", "breadth", "spans"], help="the chooser: the longest-first weighted vote, the breadth vote, or the span chooser (see above)")
ap.add_argument("--families", default="64,32,16,12,8,4,301,132,116,203,202,108,201,104,200", help="breadth mode: the byte orders and key ids whose distinct contexts vote, one each (default: every family of the longest vote)")
ap.add_argument("--dedup", action="store_true", help="never two chunks of one text (see above)")
ap.add_argument("--dedup-len", type=int, default=200, help="dedup: a candidate whose first L bytes lie in a chosen chunk, or the reverse, is skipped")
ap.add_argument("--dedup-frac", type=float, default=0.0, help="dedup: also skip a candidate sharing more than this fraction of its distinct spans with a chosen chunk (0 = off)")
ap.add_argument("--vote", default="one", choices=["one", "order", "rarity"], help="breadth: a distinct context's vote: one; its family's length; one over the chunks holding it")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
if a.max_chunks is None: a.max_chunks = 16 if a.mode in ("longest", "breadth") else 0
FAMS = [int(x) for x in a.families.split(",")] if a.mode == "breadth" else []
os.umask(0o077); os.makedirs(a.out, exist_ok=True)

BYTE_ORDERS = (64, 32, 16, 12, 8, 4)                                           # longest first
KEY_WEIGHT = ((301, 64), (132, 32), (116, 16), (203, 16), (202, 11), (108, 8), (201, 6), (104, 4), (200, 3))   # longest first
FAM_WEIGHT = {**{n: n for n in BYTE_ORDERS}, **dict(KEY_WEIGHT)}

# ---- the chunk pool: qwen_memory.py's, byte for byte ------------------------------------------------------------------------------------
t0 = time.time(); chunks = []; chunk_file = []; chunk_start = []; files = []; file_paths = []
for p in sorted(glob.glob(os.path.join(a.world, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    b = open(p, "rb").read(); fi = len(files); files.append(b); file_paths.append(p)
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
SH, SI = sorted_index(hs, ids); CH_SPANS = hs; del ids
print(f"span chooser: {len(SH):,} span hashes ({time.time() - t0:.0f} s)", flush=True)

def pick(counts: np.ndarray, floor: float, W: bytes, dedup: bool) -> list[int]:
    """The top chunks by count: at least `floor`, never the window's own text, with dedup never a second chunk of one text, up to --top
    chunks and --max-mem bytes."""
    best = np.argsort(-counts)[: a.top * 8]; out = []; total = 0
    for i in best:
        if counts[i] < floor or len(out) >= a.top: break
        c = chunks[i]
        if c[:200] in W or W[:200] in c: continue
        if dedup and any(same_text(int(i), j) for j in out): continue
        out.append(int(i)); total += len(c)
        if total >= a.max_mem: break
    return out

def same_text(i: int, j: int) -> bool:
    """Two chunks of one text: one's first --dedup-len bytes lie in the other (overlapping neighbours, copies of a file), or they share
    more than --dedup-frac of the first's distinct spans."""
    L = a.dedup_len; ci, cj = chunks[i], chunks[j]
    if ci[:L] in cj or cj[:L] in ci: return True
    if a.dedup_frac > 0 and len(CH_SPANS[i]) and np.isin(CH_SPANS[i], CH_SPANS[j], assume_unique=True).mean() > a.dedup_frac: return True
    return False

def span_choose(prefix: bytes, W: bytes, dedup: bool = False) -> tuple[list[int], list[int], int]:
    """The span chooser's chunks, their hit counts and the query's distinct span count (the plain chooser without dedup is the reference
    and the gate's signal)."""
    q = np.unique(span_hashes(prefix, a.span)); lo = np.searchsorted(SH, q, "left"); hi = np.searchsorted(SH, q, "right")
    counts = np.zeros(len(chunks), dtype=np.int32)
    for l_, h_ in zip(lo, hi):
        if h_ > l_: np.add.at(counts, SI[l_:h_], 1)
    out = pick(counts, a.min_hits, W, dedup); return out, [int(counts[i]) for i in out], int(len(q))

# ---- the tables' chooser: every file hashed with the tables' contexts and keys, rows assigned to its chunks --------------------------------
t0 = time.time()
ORDERS = [*BYTE_ORDERS, *(k for k, _ in KEY_WEIGHT)] if a.mode == "longest" else FAMS
IDX_BO = [n for n in BYTE_ORDERS if n in ORDERS]; IDX_KEYS = [n for n in ORDERS if n not in BYTE_ORDERS]
parts = {n: ([], []) for n in ORDERS}
by_file = {}
for ci, (fi, s) in enumerate(zip(chunk_file, chunk_start)): by_file.setdefault(fi, []).append((ci, s))
for fi, b in enumerate(files):
    if fi not in by_file or not ORDERS: continue
    data = np.frombuffer(b, dtype=np.uint8); N = len(data)
    hb = context_hashes(data, 0, N, IDX_BO) if IDX_BO else {}; hk = compute_keys(data) if IDX_KEYS else {}
    for ci, s in by_file[fi]:
        e = min(N, s + a.chunk)
        for n in IDX_BO:
            lo = max(s, n)                                                          # rows with a full context only
            if e > lo:
                u = np.unique(hb[n][lo:e]); parts[n][0].append(u); parts[n][1].append(np.full(len(u), ci, dtype=np.int32))
        for n in IDX_KEYS:
            u = np.unique(np.asarray(hk[n][s:e])); parts[n][0].append(u); parts[n][1].append(np.full(len(u), ci, dtype=np.int32))
    if fi % 500 == 0: print(f"  hashed {fi:,} / {len(files):,} files ({time.time() - t0:.0f} s)", flush=True)
IDX = {n: sorted_index(*parts[n]) for n in ORDERS}; del parts
if ORDERS: print(f"tables' index ({a.mode}): " + ", ".join(f"{n}:{len(IDX[n][0]) / 1e6:.1f}M" for n in ORDERS) + f" rows ({time.time() - t0:.0f} s)", flush=True)

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
    return pick(counts, a.min_vote, W, a.dedup), float(counts.max())

def breadth_choose(prefix: bytes, W: bytes) -> tuple[list[int], float]:
    """The breadth vote: every distinct context of each family in FAMS that occurs in the pool adds one to each chunk holding it (a
    context in more than --max-chunks chunks, if a cap is set, adds nothing). The span count with the tables' hashes."""
    data = np.frombuffer(prefix, dtype=np.uint8); ext = np.concatenate([data, np.zeros(1, np.uint8)]); M = len(ext)
    bo = [n for n in FAMS if n in BYTE_ORDERS]
    hb = context_hashes(ext, 0, M, bo) if bo else {}; hk = compute_keys(ext) if len(bo) < len(FAMS) else {}
    counts = np.zeros(len(chunks), dtype=np.float64)
    for n in FAMS:
        H, I = IDX[n]; q = np.unique(hb[n][n:] if n in BYTE_ORDERS else np.asarray(hk[n]))
        lo = np.searchsorted(H, q, "left"); hi = np.searchsorted(H, q, "right")
        hit = (hi > lo) if not a.max_chunks else (hi > lo) & (hi - lo <= a.max_chunks)
        if hit.any():
            starts, lens = lo[hit], (hi - lo)[hit]; total = int(lens.sum())
            idx = np.arange(total) + np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
            w = 1.0 if a.vote == "one" else float(FAM_WEIGHT[n]) if a.vote == "order" else None
            np.add.at(counts, I[idx], w if w is not None else np.repeat(1.0 / lens, lens))
    return pick(counts, a.min_hits, W, a.dedup), float(counts.max())

def choose(prefix: bytes, W: bytes) -> tuple[list[int], float]:
    if a.mode == "longest": return table_choose(prefix, W)
    if a.mode == "breadth": return breadth_choose(prefix, W)
    out, sc, _ = span_choose(prefix, W, a.dedup); return out, float(sc[0]) if sc else 0.0

# ---- every window's choice, both choosers ---------------------------------------------------------------------------------------------------
held = open(a.heldout, "rb").read(); meta = json.load(open(a.json)); T = a.T
sets = {"kyle": (meta["kyle"]["offset"], meta["kyle"]["windows"], a.kyle_every), "dlg": (meta["dialogue"]["offset"], meta["dialogue"]["windows"], a.dlg_every)}
chooser = {"mode": a.mode, "top": a.top, "max_mem": a.max_mem, "dedup": a.dedup, "max_chunks": a.max_chunks, "chunk": a.chunk,
           "dedup_len": a.dedup_len, "dedup_frac": a.dedup_frac, "vote": a.vote}
if a.mode == "longest": chooser.update({"byte_orders": BYTE_ORDERS, "keys": KEY_WEIGHT, "min_vote": a.min_vote})
elif a.mode == "breadth": chooser.update({"families": FAMS, "min_hits": a.min_hits})
else: chooser.update({"span": a.span, "min_hits": a.min_hits})
out = {"pool": {"world": a.world, "chunk": a.chunk, "chunks": len(chunks), "digest": pool_digest, "file": chunk_file, "ext": [os.path.splitext(fp)[1] for fp in file_paths],
               "file_bytes": [len(b) for b in files]},
       "chooser": chooser, "choices": {}, "span_choices": {}, "span_scores": {}, "query_spans": {}, "best_vote": {}, "stats": {}}
for name, (off, nwin, every) in sets.items():
    t0 = time.time(); ch = []; sp = []; ssc = []; nqs = []; same_top = 0; jac = []; mem_bytes = 0; got = 0; both = 0; best_votes = []
    for w in range(0, nwin, every):
        if a.limit and len(ch) >= a.limit: break
        W = held[off + w * T: off + (w + 1) * T]; first = W[:T // 2]
        t, v = choose(first, W); s, sc, nq = span_choose(first, W); ch.append(t); sp.append(s); ssc.append(sc); nqs.append(nq); best_votes.append(v)
        if t: got += 1; mem_bytes += sum(len(chunks[i]) for i in t)
        if t and s:
            both += 1; same_top += int(t[0] == s[0]); jac.append(len(set(t) & set(s)) / len(set(t) | set(s)))
    st = {"windows": len(ch), "memory_found": got, "memory_bytes_mean": mem_bytes / got if got else 0, "best_vote_median": float(np.median(best_votes)),
          "both_found": both, "same_top_chunk": same_top, "jaccard_mean": float(np.mean(jac)) if jac else 0, "identical_sets": int(sum(j == 1.0 for j in jac)),
          "seconds": round(time.time() - t0)}
    out["choices"][name] = ch; out["span_choices"][name] = sp; out["span_scores"][name] = ssc; out["query_spans"][name] = nqs; out["best_vote"][name] = best_votes; out["stats"][name] = st
    print(f"{name}: {st['windows']} windows; the {a.mode} chooser{' with dedup' if a.dedup else ''} finds a memory for {got} ({st['memory_bytes_mean']:.0f} bytes on average, best vote median {st['best_vote_median']:.0f}); "
          f"vs the span chooser: same top chunk {same_top}/{both}, identical sets {st['identical_sets']}, Jaccard {st['jaccard_mean']:.2f} ({st['seconds']} s)", flush=True)
json.dump(out, open(os.path.join(a.out, "choices.json"), "w"))
print("wrote", os.path.join(a.out, "choices.json"))
