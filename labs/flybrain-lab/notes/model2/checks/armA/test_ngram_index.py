"""ngram_index: hashes, counts and the as-of rule against brute force on a toy world."""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ngram_index as ni

def brute(files, context, targets, N, d):
    seq = list(context) + list(targets); c = len(context); ctx = np.zeros((N, len(targets)), int); tru = np.zeros((N, len(targets)), int)
    for j in range(len(targets)):
        for n in range(1, N + 1):
            if c + j - n < 0: continue
            g = tuple(seq[c + j - n: c + j]); gt = g + (seq[c + j],)
            for toks, fd in files:
                if d and fd and fd >= d: continue
                for i in range(len(toks)):
                    if tuple(toks[i:i + n]) == g and i + n <= len(toks): ctx[n - 1, j] += 1
                    if tuple(toks[i:i + n + 1]) == gt: tru[n - 1, j] += 1
    return ctx, tru

def make_index(files, N):
    toks = [np.asarray(t, np.int32) for t, _ in files]; D = np.concatenate([np.full(len(t), fd, np.int32) for t, (_, fd) in zip(toks, files)])
    T = np.concatenate(toks); L = len(T); fid = np.repeat(np.arange(len(toks)), [len(t) for t in toks]); H = ni.hash_ngrams(T, N + 1); out = {"N": N, "tokens": L}
    for n in range(1, N + 2):
        valid = np.arange(L) - (n - 1) >= 0; valid &= fid[np.maximum(np.arange(L) - (n - 1), 0)] == fid
        idx = np.nonzero(valid)[0]; order = np.lexsort((D[idx], H[n - 1, idx])); idx = idx[order]; out[f"h{n}"] = H[n - 1, idx]; out[f"d{n}"] = D[idx]
    p = os.path.join(tempfile.mkdtemp(), "i.npz"); np.savez(p, **out); return ni.Index(p)

def test_counts_match_brute_force():
    rng = np.random.default_rng(0); N = 4
    files = [(rng.integers(0, 5, size=60).tolist(), 100), (rng.integers(0, 5, size=40).tolist(), 200), (rng.integers(0, 5, size=30).tolist(), 0)]
    ix = make_index(files, N); context = rng.integers(0, 5, size=7); targets = rng.integers(0, 5, size=9)
    for d in (0, 150, 250):
        ctx, tru = ix.features(context, targets, d); bc, bt = brute(files, context, targets, N, d)
        assert (ctx == bc).all(), (d, ctx, bc); assert (tru == bt).all(), (d, tru, bt)

def test_hash_is_order_sensitive_and_consistent():
    a = ni.hash_ngrams(np.array([1, 2, 3]), 3); b = ni.hash_ngrams(np.array([3, 2, 1]), 3)
    assert a[2, 2] != b[2, 2]; assert a[0, 1] == ni.hash_ngrams(np.array([2]), 1)[0, 0]

def test_day():
    assert ni.day("") == 0 and ni.day("2026-09-20") == ni.day("2026-09-19") + 1
