#!/usr/bin/env python
"""The leak mask of the prose corpus (2026-10-21; a side agent's point: the slice builder dropped 59 of 4,693 train-region documents for
holding a 64-byte span of the held-out text, so mixB-trained readers have seen some of Kyle's held-out sets verbatim, the 10-11 lesson
again). Scans the whole train region [0, train_end) of the corpus for every 64-byte span of the tail (the held-out sets) and reports,
per held-out set, the windows touched (any span found) with their covered bytes, plus where the spans sit in the train region (source
and site of the holding documents, from their headers). Spans of fewer than --min-distinct distinct byte values (rules, blank runs) are
left out and counted. A 64-bit rolling hash; a sample of the pairs is verified byte for byte. Prints and writes counts and indices only,
never text. CPU, ~10 min on the quiet box.
usage: leak_scan.py --corpus /games/ailab/corpora/talkB.bytes --layout /games/ailab/corpora/talkB/layout.json --heldout-json /games/ailab/corpora/kyle/heldout.json --out results/tables/leak/leak_windows_talkB.json"""
import argparse, json, os, re, time
from collections import Counter

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="/games/ailab/corpora/talkB.bytes"); ap.add_argument("--layout", default="/games/ailab/corpora/talkB/layout.json")
ap.add_argument("--heldout-json", default="/games/ailab/corpora/kyle/heldout.json"); ap.add_argument("--out", required=True)
ap.add_argument("--span", type=int, default=64); ap.add_argument("--min-distinct", type=int, default=8)
ap.add_argument("--train-end", type=int, default=0, help="scan only this much of the train region (a test)")
ap.add_argument("--chunk", type=int, default=1 << 26); ap.add_argument("--max-train-hits", type=int, default=5_000_000)
a = ap.parse_args()
t0 = time.time(); S = a.span; M = np.uint64(0x9E3779B97F4A7C15)
lay = json.load(open(a.layout)); hj = json.load(open(a.heldout_json)); T = int(hj["T"])
train_end = a.train_end or int(lay["train_end"]); tail_off = int(lay["tail_offset"]); tail_bytes = int(lay["tail_bytes"])
corpus = np.memmap(a.corpus, dtype=np.uint8, mode="r")
sets = {k: (int(v["offset"]), int(v["windows"])) for k, v in hj.items() if isinstance(v, dict) and "windows" in v}


def rolling(x: np.ndarray) -> np.ndarray:
    """h[i] = sum_j x[i+j] * M^(S-1-j) mod 2^64 for i = 0 .. len(x)-S."""
    n = len(x) - S + 1
    x = x.astype(np.uint64)
    h = np.zeros(n, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(S):
            h *= M; h += x[j:j + n]
    return h


held = np.asarray(corpus[tail_off:tail_off + tail_bytes])
H = rolling(held); n_h = len(H)
rows = np.lib.stride_tricks.sliding_window_view(held, S)
distinct = (np.diff(np.sort(rows, axis=1), axis=1) != 0).sum(1) + 1
keep = distinct >= a.min_distinct
pos_kept = np.nonzero(keep)[0]; Hk = H[keep]
order = np.argsort(Hk, kind="stable"); Hs = Hk[order]; Ps = pos_kept[order]
print(f"held-out: {tail_bytes:,} bytes, {n_h:,} spans of {S}, {int((~keep).sum()):,} with under {a.min_distinct} distinct bytes left out; "
      f"{len(np.unique(Hs)):,} distinct hashes ({time.time() - t0:.0f} s)", flush=True)

held_hits = np.zeros(n_h, np.int64); train_hits = []; n_train_hits = 0; n_pairs = 0
for lo in range(0, train_end, a.chunk):
    hi = min(train_end, lo + a.chunk + S - 1)
    if hi - lo < S: break
    h = rolling(np.asarray(corpus[lo:hi]))
    left = np.searchsorted(Hs, h, "left"); right = np.searchsorted(Hs, h, "right")
    cnt = right - left; hit = np.nonzero(cnt)[0]
    if len(hit):
        n_train_hits += len(hit); n_pairs += int(cnt[hit].sum())
        # every (train position, held position) pair: expand the ranges [left, right)
        c = cnt[hit]; starts = left[hit]
        rep = np.repeat(starts - np.concatenate([[0], np.cumsum(c)[:-1]]), c) + np.arange(c.sum())
        np.add.at(held_hits, Ps[rep], 1)
        if len(train_hits) * 0 + sum(len(t) for t in train_hits) < a.max_train_hits: train_hits.append(lo + hit)
    print(f"  {hi / 1e6:8.0f} MB: {n_train_hits:,} train positions hit so far ({time.time() - t0:.0f} s)", flush=True)
train_hits = np.concatenate(train_hits) if train_hits else np.zeros(0, np.int64)

# verify a sample of pairs byte for byte
rng = np.random.default_rng(0); bad = 0; checked = 0
if len(train_hits):
    for tp in rng.choice(train_hits, size=min(200, len(train_hits)), replace=False):
        h1 = rolling(np.asarray(corpus[tp:tp + S]))[0]; l, r = np.searchsorted(Hs, h1, "left"), np.searchsorted(Hs, h1, "right")
        for q in Ps[l:r]:
            checked += 1; bad += int(not np.array_equal(corpus[tp:tp + S], held[q:q + S]))
print(f"train region {train_end:,}: {n_train_hits:,} positions hold a held-out span ({n_pairs:,} pairs); sample check {checked} pairs, {bad} mismatches ({time.time() - t0:.0f} s)", flush=True)

# coverage of the held-out text and the windows of each set
d = np.zeros(tail_bytes + 1, np.int64); q = np.nonzero(held_hits)[0]
np.add.at(d, q, 1); np.add.at(d, np.minimum(q + S, tail_bytes), -1)
covered = np.cumsum(d)[:tail_bytes] > 0
out = {"corpus": a.corpus, "train_end": train_end, "span": S, "min_distinct": a.min_distinct, "T": T, "held_spans": int(n_h),
       "held_spans_left_out": int((~keep).sum()), "held_spans_hit": int((held_hits > 0).sum()), "train_positions_hit": int(n_train_hits),
       "pairs": int(n_pairs), "verified_pairs": checked, "mismatches": bad, "held_hit_positions": q.tolist(), "sets": {}}   # positions in the tail, no text
for name, (off, nwin) in sets.items():
    cov = covered[off:off + nwin * T].reshape(nwin, T).sum(1); hits = np.add.reduceat(np.concatenate([(held_hits > 0)[off:off + nwin * T], [False]]).astype(np.int64)[:nwin * T], np.arange(0, nwin * T, T))
    touched = np.nonzero(cov > 0)[0]
    out["sets"][name] = {"offset": off, "windows": nwin, "touched": touched.tolist(), "covered_bytes": {int(w): int(cov[w]) for w in touched},
                         "span_starts": {int(w): int(hits[w]) for w in touched}, "clean": np.nonzero(cov == 0)[0].tolist(),
                         "touched_over_128": int((cov >= 128).sum()), "touched_over_512": int((cov >= 512).sum()), "touched_over_1024": int((cov >= 1024).sum())}
    print(f"{name}: {len(touched)} of {nwin} windows touched ({100 * len(touched) / nwin:.1f} %); covered bytes >= 128 / 512 / 1024 on "
          f"{out['sets'][name]['touched_over_128']} / {out['sets'][name]['touched_over_512']} / {out['sets'][name]['touched_over_1024']}; "
          f"median covered bytes of a touched window {int(np.median(cov[touched])) if len(touched) else 0}, max {int(cov.max())}", flush=True)
if q.size:
    dh = distinct[q]; print(f"hit spans' distinct bytes: median {int(np.median(dh))} (all spans {int(np.median(distinct))}); held bytes covered {int(covered.sum()):,} of {tail_bytes:,}", flush=True)

# where the holding documents are: headers of the train region, the hit positions attributed to their document
HDR = re.compile(rb"\n// ==== (.*?) ====\n"); heads = []; paths = []; BLOCK = 1 << 26
for lo in range(0, train_end, BLOCK):
    b = bytes(corpus[lo:min(train_end, lo + BLOCK + 4096)])
    for m in HDR.finditer(b):
        s = lo + m.start()
        if s >= min(train_end, lo + BLOCK): break
        if heads and s <= heads[-1]: continue
        heads.append(s); paths.append(m.group(1).decode("utf-8", "replace"))
heads = np.asarray(heads)
if len(train_hits) and len(heads):
    doc = np.searchsorted(heads, train_hits, "right") - 1; doc = doc[doc >= 0]
    docs_hit = np.unique(doc); by_src = Counter(); by_site = Counter()
    for i in docs_hit:
        parts = paths[i].split("/"); by_src[parts[0]] += 1; by_site["/".join(parts[:2])] += 1
    out["train_documents"] = len(heads); out["documents_hit"] = int(len(docs_hit)); out["documents_hit_by_source"] = dict(by_src)
    out["documents_hit_by_site"] = dict(by_site.most_common(12)); out["train_hits_attributed"] = int(len(doc))
    print(f"documents of the train region: {len(heads):,}; holding a held-out span: {len(docs_hit):,} ({dict(by_src)}); top sites {by_site.most_common(6)}", flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=0)
print(f"wrote {a.out} ({time.time() - t0:.0f} s)")
