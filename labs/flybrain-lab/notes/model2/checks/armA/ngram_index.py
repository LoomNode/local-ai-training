#!/usr/bin/env python
"""Arm A (2026-11-05): a token-level n-gram table over Kyle's world, queried as-of a date. The Infini-gram idea on Qwen token ids:
for a context's longest suffix the world has seen, the table's probability of the true next token is count(suffix + token) / count(suffix).
Build once (tokens of every world file, the file's creation date beside each position, rolling hashes of every n-gram ending at each
position for n = 1..N+1, each order sorted by (hash, date)); query a window's positions all at once (searchsorted on hash, then on date).
Counts never include files created on or after the window's session date (table_retrieve.py's rule; an undated file is always allowed,
an undated window sees everything). Hashes are 64-bit polynomial, collisions ignored. Prints counts and numbers only, never text.
usage: ngram_index.py build --model GGUF --world DIR --asof asof_dates.json --out index.npz [--N 12]"""
import argparse, glob, json, os, sys, time
import numpy as np

M = np.uint64(0x9E3779B97F4A7C15)
DAY0 = np.datetime64("2000-01-01")
def day(d: str) -> int:
    """'2026-09-19' -> days since 2000; '' (undated) -> 0, so an undated file counts for every window."""
    return int((np.datetime64(d) - DAY0).astype(int)) if d else 0

def roll(prev: np.ndarray, tok: np.ndarray) -> np.ndarray:
    """h_n[i] = h_{n-1}[i-1] * M + tok[i] + 1 (prev is h_{n-1} aligned to i-1, i.e. prev[i] already shifted)."""
    with np.errstate(over="ignore"):
        return prev * M + tok.astype(np.uint64) + np.uint64(1)

def hash_ngrams(tokens: np.ndarray, N: int) -> np.ndarray:
    """H[n-1, i] = hash of the n-gram ending at i (n = 1..N); positions with i < n-1 get hash 0 and are invalid."""
    t = np.asarray(tokens, dtype=np.uint64); L = len(t); H = np.zeros((N, L), dtype=np.uint64)
    H[0] = roll(np.zeros(L, dtype=np.uint64), t)
    for n in range(2, N + 1):
        H[n - 1, n - 1:] = roll(H[n - 2, n - 2:L - 1], t[n - 1:])
    return H

class Index:
    def __init__(self, path: str):
        z = np.load(path); self.N = int(z["N"]); self.tokens_n = int(z["tokens"])
        self.Hs = [z[f"h{n}"] for n in range(1, self.N + 2)]; self.Ds = [z[f"d{n}"] for n in range(1, self.N + 2)]
    def counts(self, hashes: np.ndarray, n: int, d: int) -> np.ndarray:
        """How many n-grams with these hashes the world holds in files dated before day d (d = 0: all)."""
        Hs, Ds = self.Hs[n - 1], self.Ds[n - 1]
        lo = np.searchsorted(Hs, hashes, "left"); hi = np.searchsorted(Hs, hashes, "right"); out = hi - lo
        if d:
            for j in np.nonzero(out)[0]:
                out[j] = np.searchsorted(Ds[lo[j]:hi[j]], d, "left")      # dates sorted within a hash's run
        return out
    def features(self, context: np.ndarray, targets: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
        """For each target position j (the true next token targets[j] after context + targets[:j]): count of the n-gram context ending
        just before it and of that context + the target, for n = 1..N. Returns (ctx[N, k], tru[N, k]) as int32."""
        seq = np.concatenate([np.asarray(context, dtype=np.int64), np.asarray(targets, dtype=np.int64)]); c = len(context); k = len(targets)
        H = hash_ngrams(seq, self.N + 1)
        ctx = np.zeros((self.N, k), dtype=np.int32); tru = np.zeros((self.N, k), dtype=np.int32)
        for n in range(1, self.N + 1):
            pos = np.arange(c, c + k)                         # the target positions; the context n-gram ends at pos - 1
            ok = pos - 1 >= n - 1
            if not ok.any(): break
            ctx[n - 1, ok] = self.counts(H[n - 1, pos[ok] - 1], n, d)
            tru[n - 1, ok] = self.counts(H[n, pos[ok]], n + 1, d)
        return ctx, tru

def build(a):
    from llama_cpp import Llama
    llm = Llama(model_path=a.model, vocab_only=True, verbose=False)
    asof = json.load(open(a.asof))["files"] if a.asof else {}
    toks = []; days = []; files = 0; t0 = time.time()
    for p in sorted(glob.glob(os.path.join(a.world, "**", "*"), recursive=True)):
        if not os.path.isfile(p): continue
        b = open(p, "rb").read()
        if not b: continue
        t = np.asarray(llm.tokenize(b, add_bos=False, special=False), dtype=np.int32)
        if not len(t): continue
        toks.append(t); days.append(np.full(len(t), day(asof.get(os.path.relpath(p, a.world), "")), dtype=np.int32)); files += 1
    if a.before:                                                         # the time split: only files created before the cutoff (undated files stay)
        keep = [i for i, d in enumerate(days) if d[0] == 0 or d[0] < day(a.before)]; toks = [toks[i] for i in keep]; days = [days[i] for i in keep]; files = len(keep)
    sessions = 0
    if a.sessions_before:                                                # his earlier sessions: every held-out session document dated before the cutoff
        sd = json.load(open(a.session_dates)); cut = day(a.sessions_before)
        for name, fn in (("kyle", "kyle.bytes"), ("dialogue", "dialogue.bytes")):
            b = open(os.path.join(os.path.dirname(a.session_dates), fn), "rb").read()
            for s, e, d in sd[name]:
                if not d or day(d) >= cut: continue
                t = np.asarray(llm.tokenize(b[s:e], add_bos=False, special=False), dtype=np.int32)
                if len(t): toks.append(t); days.append(np.full(len(t), day(d), dtype=np.int32)); sessions += 1
    T = np.concatenate(toks); D = np.concatenate(days); L = len(T)
    starts = np.cumsum([0] + [len(t) for t in toks[:-1]]); fid = np.repeat(np.arange(len(toks)), [len(t) for t in toks])
    print(f"world: {files:,} files" + (f" created before {a.before}" if a.before else "") + (f" + {sessions:,} held-out sessions before {a.sessions_before}" if a.sessions_before else "")
          + f", {L:,} tokens ({time.time() - t0:.0f} s)", flush=True)
    H = hash_ngrams(T, a.N + 1); out = {"N": a.N, "tokens": L}
    for n in range(1, a.N + 2):
        valid = np.arange(L) - (n - 1) >= 0
        valid &= fid[np.maximum(np.arange(L) - (n - 1), 0)] == fid        # an n-gram stays inside one file
        idx = np.nonzero(valid)[0]; order = np.lexsort((D[idx], H[n - 1, idx])); idx = idx[order]
        out[f"h{n}"] = H[n - 1, idx]; out[f"d{n}"] = D[idx]
        print(f"  order {n}: {len(idx):,} positions, {len(np.unique(out[f'h{n}'])):,} distinct", flush=True)
    np.savez(a.out, **out); print(f"wrote {a.out} ({os.path.getsize(a.out) / 2**20:.0f} MB, {time.time() - t0:.0f} s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True); b = sub.add_parser("build")
    b.add_argument("--model", required=True); b.add_argument("--world", default="/games/ailab/corpora/kyle/world/files")
    b.add_argument("--asof", default="/games/ailab/corpora/kyle/asof_dates.json"); b.add_argument("--out", required=True); b.add_argument("--N", type=int, default=12)
    b.add_argument("--before", default="", help="time split: only world files created before this date (YYYY-MM-DD)")
    b.add_argument("--sessions-before", default="", help="add the held-out session documents (his turns, the dialogue) dated before this date")
    b.add_argument("--session-dates", default="/games/ailab/corpora/kyle/session_dates.json")
    a = ap.parse_args(); os.umask(0o077)
    if a.cmd == "build": build(a)
