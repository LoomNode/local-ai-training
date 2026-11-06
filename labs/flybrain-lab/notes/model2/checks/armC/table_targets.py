#!/usr/bin/env python
"""Arm C (2026-11-06): the table's next-token distribution at every loss position of Arm B's train.npz, the target a Memory Decoder
imitates (0.5 KL(p_table || p_mem) + 0.5 NLL). The table is the Arm A one (orders 1..N over the world files and his earlier sessions,
the same rolling hash) but kept as (context hash, next token, date, document) so that continuations can be enumerated. At a position the
target is the longest order whose context was seen at least --cmin times in documents other than the position's own and created before
its date (the deployed table's view), the counts over the next tokens, truncated to the --top most frequent and renormalised; positions
with no match get no table term. Writes targets/ids.npy [P, top] int32, p.npy [P, top] float16 (zero-padded), n_eff.npy [P] int8, and
targets.json. Numbers and ids only.
usage: table_targets.py --data DIR --out DIR [--before 2026-09-01] [--N 12] [--cmin 1] [--top 16]"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "armA")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "armB"))
from ngram_index import hash_ngrams, day
from prompts import Tokenizer, WORLD, ASOF

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--before", default="2026-09-01"); ap.add_argument("--N", type=int, default=12)
ap.add_argument("--cmin", type=int, default=1); ap.add_argument("--top", type=int, default=16); ap.add_argument("--session-dates", default="/games/ailab/corpora/kyle/session_dates.json")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); os.umask(0o077); os.makedirs(a.out, exist_ok=True); t0 = time.time(); tok = Tokenizer(); cut = day(a.before)

# ---- the documents, in make_train.py's order (files sorted, then kyle sessions, then dialogue sessions), so document ids line up ----
docs = []   # (tokens, day)
asof = json.load(open(ASOF))["files"]
import glob
for p in sorted(glob.glob(os.path.join(WORLD, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    d = day(asof.get(os.path.relpath(p, WORLD), ""))
    if cut and d and d >= cut: continue
    b = open(p, "rb").read(); t = np.asarray(tok(b), dtype=np.int32) if b else np.zeros(0, np.int32); docs.append((t, d))
nfiles = len(docs)
sd = json.load(open(a.session_dates))
for name, fn in (("kyle", "kyle.bytes"), ("dialogue", "dialogue.bytes")):
    b = open(os.path.join(os.path.dirname(a.session_dates), fn), "rb").read()
    for s, e, dt in sd[name]:
        if not dt or (cut and day(dt) >= cut): continue
        docs.append((np.asarray(tok(b[s:e]), dtype=np.int32), day(dt)))
T = np.concatenate([d[0] for d in docs]); D = np.concatenate([np.full(len(d[0]), d[1], np.int32) for d in docs]); F = np.repeat(np.arange(len(docs), dtype=np.int32), [len(d[0]) for d in docs]); L = len(T)
print(f"table: {nfiles} files + {len(docs) - nfiles} sessions, {L:,} tokens ({time.time() - t0:.0f} s)", flush=True)
H = hash_ngrams(T, a.N)                                                 # H[n-1, i] = hash of the n tokens ending at i
tables = []                                                             # per order n: (ctx hashes sorted, next tokens, dates, docs) keyed by the context hash of the n tokens before the next token
for n in range(1, a.N + 1):
    i = np.arange(n - 1, L - 1); ok = F[i - (n - 1)] == F[i + 1]       # context and next token inside one document
    i = i[ok]; h = H[n - 1, i]; nxt = T[i + 1]; order = np.lexsort((nxt, h))
    tables.append((h[order], nxt[order], D[i + 1][order], F[i + 1][order]))
print(f"  orders 1..{a.N} keyed ({time.time() - t0:.0f} s)", flush=True)

# ---- the training positions: make_train.py cut documents into byte segments; here the segment's tokens are the loss tokens of each sequence, and the
# ---- context for the table is the segment's own earlier tokens (the memory prefix is not his text). Document id and date come from the sequence.
z = np.load(os.path.join(a.data, "train.npz")); tokens, seq_start, loss_start, kind, sday = z["tokens"], z["seq_start"], z["loss_start"], z["kind"], z["day"]
S = len(loss_start); nloss = seq_start[1:] - seq_start[:-1] - loss_start; pos_start = np.concatenate([[0], np.cumsum(nloss)]); P = int(pos_start[-1])
# which document each sequence came from: make_train.py walks the same documents in the same order; recover by matching the segment's first tokens... simpler: rebuild the mapping by re-cutting
seq_doc = np.full(S, -1, np.int32); si = 0
for di, (t, d) in enumerate(docs):
    pass
# the mapping by re-cutting is identical to make_train.py's loop (seg bytes); rather than re-tokenise, use exclusion by *date and content*: exclude entries whose document holds the same n-gram at the same hash AND the same document id is unknown.
# To stay exact, make_train.py's order gives: sequences of kind 0 in file order, then kind 1 in session order, each document contributing its segments in order; a document with no segment contributes none.
# Re-cut the documents' bytes to count their segments:
seg_counts = []
for p in sorted(glob.glob(os.path.join(WORLD, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    d = day(asof.get(os.path.relpath(p, WORLD), ""))
    if cut and d and d >= cut: continue
    seg_counts.append(open(p, "rb").read())
for name, fn in (("kyle", "kyle.bytes"), ("dialogue", "dialogue.bytes")):
    b = open(os.path.join(os.path.dirname(a.session_dates), fn), "rb").read()
    for s, e, dt in sd[name]:
        if not dt or (cut and day(dt) >= cut): continue
        seg_counts.append(b[s:e])
assert len(seg_counts) == len(docs)
his = np.nonzero(kind < 2)[0]; q = 0
for di, b in enumerate(seg_counts):
    for s in range(0, len(b), 6000):
        seg = b[s:s + 6000]
        if len(seg) < 200: continue
        st = tok(seg)
        # make_train dropped sequences over --max-tokens; such a segment has no sequence: detect by comparing the next sequence's loss tokens
        if q < len(his) and nloss[his[q]] == len(st) and np.array_equal(tokens[seq_start[his[q]] + loss_start[his[q]]:seq_start[his[q] + 1]], np.asarray(st, np.int32)):
            seq_doc[his[q]] = di; q += 1
assert q == len(his), f"matched {q} of {len(his)} sequences to documents"
print(f"  {len(his)} his sequences mapped to {len(set(seq_doc[his].tolist()))} documents ({time.time() - t0:.0f} s)", flush=True)

ids = np.lib.format.open_memmap(os.path.join(a.out, "ids.npy"), mode="w+", dtype=np.int32, shape=(P, a.top)); ids[:] = 0
pw = np.lib.format.open_memmap(os.path.join(a.out, "p.npy"), mode="w+", dtype=np.float16, shape=(P, a.top)); pw[:] = 0
neff = np.lib.format.open_memmap(os.path.join(a.out, "n_eff.npy"), mode="w+", dtype=np.int8, shape=(P,)); neff[:] = 0
stats = {"positions": P, "his_positions": int(nloss[his].sum()), "seen": 0, "by_order": [0] * (a.N + 1), "p_true_sum": 0.0}
for c, si_ in enumerate(his):
    if a.limit and c >= a.limit: break
    st = tokens[seq_start[si_] + loss_start[si_]:seq_start[si_ + 1]]; d = int(sday[si_]); doc = int(seq_doc[si_]); p0 = int(pos_start[si_])
    Hs = hash_ngrams(st, a.N)                                            # context hashes within the segment
    LO = np.zeros((a.N, len(st)), np.int64); HI = np.zeros((a.N, len(st)), np.int64)
    for n in range(1, a.N + 1):                                          # every order's range for every position at once
        hh = tables[n - 1][0]; LO[n - 1] = np.searchsorted(hh, Hs[n - 1], "left"); HI[n - 1] = np.searchsorted(hh, Hs[n - 1], "right")
    for j in range(1, len(st)):                                          # position j predicts st[j] from st[:j]; j = 0 has no context
        for n in range(min(a.N, j), 0, -1):
            lo = LO[n - 1, j - 1]; hi = HI[n - 1, j - 1]
            if hi - lo < a.cmin: continue
            hh, nx, dd, ff = tables[n - 1]
            ok = (ff[lo:hi] != doc) & ((dd[lo:hi] < d) | (dd[lo:hi] == 0)) if d else (ff[lo:hi] != doc)
            if ok.sum() < a.cmin: continue
            nx_ = nx[lo:hi][ok]; u, cnt = np.unique(nx_, return_counts=True)
            if len(u) > a.top: k = np.argsort(-cnt)[:a.top]; u, cnt = u[k], cnt[k]
            pr = cnt / cnt.sum(); ids[p0 + j, :len(u)] = u; pw[p0 + j, :len(u)] = pr.astype(np.float16); neff[p0 + j] = n
            stats["seen"] += 1; stats["by_order"][n] += 1; m = u == st[j]; stats["p_true_sum"] += float(pr[m][0]) if m.any() else 0.0
            break
    if (c + 1) % 200 == 0: print(f"  {c + 1}/{len(his)} sequences, seen {stats['seen']:,}, mean p_table(true) over seen {stats['p_true_sum'] / max(1, stats['seen']):.3f} ({time.time() - t0:.0f} s)", flush=True)
ids.flush(); pw.flush(); neff.flush()
stats["p_true_mean_seen"] = stats["p_true_sum"] / max(1, stats["seen"]); stats["seconds"] = round(time.time() - t0); stats.update({"N": a.N, "cmin": a.cmin, "top": a.top, "before": a.before})
json.dump(stats, open(os.path.join(a.out, "targets.json"), "w"), indent=1); open(os.path.join(a.out, "targets.done"), "w").close(); print(json.dumps(stats))
