#!/usr/bin/env python
"""Qwen as the yardstick, pointed at the lab's memory (Kyle, 2026-10-19: "go ahead and run the qwen tests"): how much is Kyle's world worth
to a model that already knows the language? The shape of the reader's measurements: every 2,048-byte held-out window's SECOND half is
scored given its first half (warm), three ways:
  cold      the whole window alone (the reader's cold number's shape)
  warm      the second half given the first half
  memory    the same, with text retrieved from Kyle's world by the first half in front of it: the top chunks of his files sharing the
            most 12-byte spans with the first half (the lab's match idea as a retriever; the memory sees the context, never the target)
Bits per byte of the scored bytes, by Qwen's own token log-probabilities (llama-cpp-python, logits for every position, the GGUF on one
card). Tokens are cut at the half boundary (the halves tokenised separately), so the scored tokens are exactly the second half's bytes.
Prints counts and numbers only, never text. usage:
  qwen_memory.py --model GGUF --world DIR --heldout heldout.bytes --json heldout.json --out DIR [--kyle-every 1] [--dlg-every 4] [--n-ctx 6144] [--top 2]"""
import argparse, glob, json, math, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--world", default="/games/ailab/corpora/kyle/world/files")
ap.add_argument("--heldout", default="/games/ailab/corpora/kyle/heldout.bytes"); ap.add_argument("--json", default="/games/ailab/corpora/kyle/heldout.json")
ap.add_argument("--out", required=True); ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--kyle-every", type=int, default=1); ap.add_argument("--dlg-every", type=int, default=4)
ap.add_argument("--n-ctx", type=int, default=6144); ap.add_argument("--top", type=int, default=2); ap.add_argument("--chunk", type=int, default=1500)
ap.add_argument("--span", type=int, default=12); ap.add_argument("--min-hits", type=int, default=3); ap.add_argument("--max-mem", type=int, default=3200)
ap.add_argument("--conditions", default="cold,warm,memory"); ap.add_argument("--limit", type=int, default=0, help="debug: score only this many windows a set")
a = ap.parse_args()
os.umask(0o077); os.makedirs(a.out, exist_ok=True)
from llama_cpp import Llama  # noqa: E402

# ---- the retriever: 12-byte spans of Kyle's world, in chunks ---------------------------------------------------------------------------
M = np.uint64(0x9E3779B97F4A7C15)
def span_hashes(b: bytes, S: int) -> np.ndarray:
    d = np.frombuffer(b, dtype=np.uint8).astype(np.uint64)
    if len(d) < S: return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(d) - S + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(S): h = h * M + d[j:len(d) - S + 1 + j] + np.uint64(1)
    return h
t0 = time.time(); chunks = []; hs = []; ids = []
for p in sorted(glob.glob(os.path.join(a.world, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    b = open(p, "rb").read()
    for s in range(0, max(1, len(b) - a.chunk // 2), a.chunk // 2 if len(b) > a.chunk else a.chunk):   # half-overlapping chunks
        c = b[s:s + a.chunk]
        if len(c) < 64: continue
        h = np.unique(span_hashes(c, a.span)); hs.append(h); ids.append(np.full(len(h), len(chunks), dtype=np.int32)); chunks.append(c)
H = np.concatenate(hs); I = np.concatenate(ids); order = np.argsort(H, kind="stable"); H = H[order]; I = I[order]
print(f"retriever: {len(chunks):,} chunks of <= {a.chunk} bytes from {a.world}, {len(H):,} span hashes ({time.time() - t0:.0f} s)", flush=True)
def retrieve(prefix: bytes, exclude_bytes: bytes) -> list[bytes]:
    q = np.unique(span_hashes(prefix, a.span))
    lo = np.searchsorted(H, q, "left"); hi = np.searchsorted(H, q, "right")
    counts = np.zeros(len(chunks), dtype=np.int32)
    for l_, h_ in zip(lo, hi):
        if h_ > l_: np.add.at(counts, I[l_:h_], 1)
    best = np.argsort(-counts)[: a.top * 3]; out = []; total = 0
    for i in best:
        if counts[i] < a.min_hits or len(out) >= a.top: break
        c = chunks[i]
        if exclude_bytes and (c[:200] in exclude_bytes or exclude_bytes[:200] in c): continue   # never the window's own text
        out.append(c); total += len(c)
        if total >= a.max_mem: break
    return out

# ---- Qwen -------------------------------------------------------------------------------------------------------------------------------
llm = Llama(model_path=a.model, n_gpu_layers=-1, n_ctx=a.n_ctx, n_batch=1024, logits_all=True, verbose=False)
BOS = llm.token_bos()
def score(prefix: bytes, target: bytes) -> tuple[float, int]:
    """Bits of `target` given `prefix` (the halves tokenised separately): (bits, tokens)."""
    tp = llm.tokenize(prefix, add_bos=True, special=False) if prefix else [BOS]
    tt = llm.tokenize(target, add_bos=False, special=False)
    toks = (tp + tt)[-a.n_ctx:]; k = len(tt)
    llm.reset(); llm.eval(toks)
    logits = np.asarray(llm.scores[:len(toks)], dtype=np.float32)              # logits at each position predict the NEXT token
    pred = logits[len(toks) - k - 1:len(toks) - 1]                              # the positions whose next token is a target token
    targ = np.asarray(toks[len(toks) - k:], dtype=np.int64)
    m = pred.max(axis=1, keepdims=True); lse = m[:, 0] + np.log(np.exp(pred - m).sum(axis=1))
    lp = pred[np.arange(k), targ] - lse
    return float(-lp.sum() / math.log(2)), k

held = open(a.heldout, "rb").read(); meta = json.load(open(a.json)); T = a.T
sets = {"kyle": (meta["kyle"]["offset"], meta["kyle"]["windows"], a.kyle_every), "dlg": (meta["dialogue"]["offset"], meta["dialogue"]["windows"], a.dlg_every)}
conds = a.conditions.split(",")
summary = {}
for name, (off, nwin, every) in sets.items():
    res = {c: [] for c in conds}; ntok = {c: 0 for c in conds}; mem_used = 0; mem_bytes = 0; t0 = time.time(); n = 0
    for w in range(0, nwin, every):
        if a.limit and n >= a.limit: break
        W = held[off + w * T: off + (w + 1) * T]; first, second = W[:T // 2], W[T // 2:]
        if "cold" in conds:
            b, k = score(b"", W); res["cold"].append(b / T); ntok["cold"] += k
        if "warm" in conds:
            b, k = score(first, second); res["warm"].append(b / (T // 2)); ntok["warm"] += k
        if "memory" in conds:
            mem = retrieve(first, W)
            if mem: mem_used += 1; mem_bytes += sum(len(c) for c in mem)
            pre = b"".join(c + b"\n\n" for c in mem) + first if mem else first
            b, k = score(pre, second); res["memory"].append(b / (T // 2)); ntok["memory"] += k
        n += 1
        if n % 50 == 0: print(f"  {name}: {n} windows, " + ", ".join(f"{c} {np.mean(res[c]):.4f}" for c in conds) + f" ({time.time() - t0:.0f} s)", flush=True)
    summary[name] = {c: {"bpb": float(np.mean(res[c])), "n": len(res[c]), "tokens": ntok[c], "per_window": [round(x, 4) for x in res[c]]} for c in conds}
    summary[name]["memory_used_windows"] = mem_used; summary[name]["memory_bytes_mean"] = (mem_bytes / mem_used) if mem_used else 0
    print(f"{name}: {n} windows; " + "; ".join(f"{c} {summary[name][c]['bpb']:.4f} bits/byte ({summary[name][c]['tokens'] / (n * (T if c == 'cold' else T // 2)):.2f} tok/byte)" for c in conds)
          + f"; memory found for {mem_used} windows, {summary[name]['memory_bytes_mean']:.0f} bytes on average ({time.time() - t0:.0f} s)", flush=True)
summary["model"] = a.model; summary["world"] = a.world; summary["T"] = T; summary["n_ctx"] = a.n_ctx; summary["retriever"] = {"chunk": a.chunk, "span": a.span, "top": a.top, "min_hits": a.min_hits}
json.dump(summary, open(os.path.join(a.out, "qwen_memory.json"), "w"), indent=1)
k, d = summary["kyle"], summary["dlg"]
line = (f"Qwen ({os.path.basename(a.model)}) on the second half of 2 KB windows given the first: Kyle's turns ({k['warm']['n']}) warm {k['warm']['bpb']:.4f}, + his files retrieved {k['memory']['bpb']:.4f} ({k['memory']['bpb'] - k['warm']['bpb']:+.4f}); "
        f"the dialogue ({d['warm']['n']}) warm {d['warm']['bpb']:.4f}, + the files {d['memory']['bpb']:.4f} ({d['memory']['bpb'] - d['warm']['bpb']:+.4f}); cold (whole windows) {k['cold']['bpb']:.4f} / {d['cold']['bpb']:.4f}")
print(line); open(os.path.join(a.out, "qwen_memory.txt"), "a").write(line + "\n")
