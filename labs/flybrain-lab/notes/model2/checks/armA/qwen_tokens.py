#!/usr/bin/env python
"""Arm A (2026-11-05): qwen_memory.py's memory condition, keeping every scored token. Same prompts as qwen_memory.py (the span retriever
or --choices, --headers, --sep, --order; the halves tokenised separately, left-truncated to n_ctx), same bits-per-byte line, plus a
tokens.jsonl with one record per window: the set, the window index, the session date, the first half's tokens on their own (the n-gram
context a deployed table would see), the target tokens and each one's log-probability under Qwen (nats). Numbers and token ids only,
never text. usage: qwen_tokens.py --model GGUF --out DIR [--choices choices.json --headers] [--limit 3] [--dlg-every 4]"""
import argparse, codecs, glob, hashlib, json, math, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--world", default="/games/ailab/corpora/kyle/world/files")
ap.add_argument("--heldout", default="/games/ailab/corpora/kyle/heldout.bytes"); ap.add_argument("--json", default="/games/ailab/corpora/kyle/heldout.json")
ap.add_argument("--asof", default="/games/ailab/corpora/kyle/asof_dates.json"); ap.add_argument("--out", required=True); ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--kyle-every", type=int, default=1); ap.add_argument("--dlg-every", type=int, default=4)
ap.add_argument("--n-ctx", type=int, default=6144); ap.add_argument("--top", type=int, default=2); ap.add_argument("--chunk", type=int, default=1500)
ap.add_argument("--span", type=int, default=12); ap.add_argument("--min-hits", type=int, default=3); ap.add_argument("--max-mem", type=int, default=3200)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--choices", default=None); ap.add_argument("--label", default="the tables")
ap.add_argument("--order", default="top-first", choices=["top-first", "top-last"]); ap.add_argument("--sep", default="\\n\\n"); ap.add_argument("--headers", action="store_true")
a = ap.parse_args()
os.umask(0o077); os.makedirs(a.out, exist_ok=True)
SEP = codecs.decode(a.sep, "unicode_escape").encode()
from llama_cpp import Llama  # noqa: E402

M = np.uint64(0x9E3779B97F4A7C15)
def span_hashes(b: bytes, S: int) -> np.ndarray:
    d = np.frombuffer(b, dtype=np.uint8).astype(np.uint64)
    if len(d) < S: return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(d) - S + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(S): h = h * M + d[j:len(d) - S + 1 + j] + np.uint64(1)
    return h
t0 = time.time(); chunks = []; hs = []; ids = []; paths = []
for p in sorted(glob.glob(os.path.join(a.world, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    b = open(p, "rb").read(); rel = os.path.relpath(p, a.world)
    for s in range(0, max(1, len(b) - a.chunk // 2), a.chunk // 2 if len(b) > a.chunk else a.chunk):
        c = b[s:s + a.chunk]
        if len(c) < 64: continue
        h = np.unique(span_hashes(c, a.span)); hs.append(h); ids.append(np.full(len(h), len(chunks), dtype=np.int32)); chunks.append(c); paths.append(rel)
H = np.concatenate(hs); I = np.concatenate(ids); order = np.argsort(H, kind="stable"); H = H[order]; I = I[order]
print(f"retriever: {len(chunks):,} chunks of <= {a.chunk} bytes from {a.world}, {len(H):,} span hashes ({time.time() - t0:.0f} s)", flush=True)
choices = None; chooser_info = None
if a.choices:
    ch = json.load(open(a.choices)); digest = hashlib.sha256(b"".join(hashlib.sha256(c).digest() for c in chunks)).hexdigest()
    if ch["pool"]["digest"] != digest or ch["pool"]["chunks"] != len(chunks): raise SystemExit(f"{a.choices} was chosen from another chunk pool")
    choices = ch["choices"]; chooser_info = ch["chooser"]; print(f"memory chunks from {a.choices} ({a.label}: {chooser_info})", flush=True)
def retrieve(prefix: bytes, exclude_bytes: bytes) -> list[int]:
    q = np.unique(span_hashes(prefix, a.span))
    lo = np.searchsorted(H, q, "left"); hi = np.searchsorted(H, q, "right")
    counts = np.zeros(len(chunks), dtype=np.int32)
    for l_, h_ in zip(lo, hi):
        if h_ > l_: np.add.at(counts, I[l_:h_], 1)
    best = np.argsort(-counts)[: a.top * 3]; out = []; total = 0
    for i in best:
        if counts[i] < a.min_hits or len(out) >= a.top: break
        c = chunks[i]
        if exclude_bytes and (c[:200] in exclude_bytes or exclude_bytes[:200] in c): continue
        out.append(int(i)); total += len(c)
        if total >= a.max_mem: break
    return out

llm = Llama(model_path=a.model, n_gpu_layers=-1, n_ctx=a.n_ctx, n_batch=1024, logits_all=True, verbose=False)
BOS = llm.token_bos()
def score(prefix: bytes, target: bytes):
    """(bits, target tokens, log-probs in nats per target token) of `target` given `prefix`, the halves tokenised separately."""
    tp = llm.tokenize(prefix, add_bos=True, special=False) if prefix else [BOS]
    tt = llm.tokenize(target, add_bos=False, special=False)
    toks = (tp + tt)[-a.n_ctx:]; k = len(tt)
    llm.reset(); llm.eval(toks)
    logits = np.asarray(llm.scores[:len(toks)], dtype=np.float32)
    pred = logits[len(toks) - k - 1:len(toks) - 1]; targ = np.asarray(toks[len(toks) - k:], dtype=np.int64)
    m = pred.max(axis=1, keepdims=True); lse = m[:, 0] + np.log(np.exp(pred - m).sum(axis=1))
    lp = pred[np.arange(k), targ] - lse
    return float(-lp.sum() / math.log(2)), targ, lp

held = open(a.heldout, "rb").read(); meta = json.load(open(a.json)); T = a.T
asof = json.load(open(a.asof)) if a.asof else {}
sets = {"kyle": (meta["kyle"]["offset"], meta["kyle"]["windows"], a.kyle_every, asof.get("kyle_windows", [])), "dlg": (meta["dialogue"]["offset"], meta["dialogue"]["windows"], a.dlg_every, asof.get("dialogue_windows", []))}
summary = {}; rec = open(os.path.join(a.out, "tokens.jsonl"), "w")
for name, (off, nwin, every, dates) in sets.items():
    res = []; ntok = 0; mem_used = 0; mem_bytes = 0; t0 = time.time(); n = 0
    for w in range(0, nwin, every):
        if a.limit and n >= a.limit: break
        W = held[off + w * T: off + (w + 1) * T]; first, second = W[:T // 2], W[T // 2:]
        ids_ = list(choices[name][n]) if choices is not None else retrieve(first, W)
        if a.order == "top-last": ids_ = ids_[::-1]
        mem = [((b"## " + paths[i].encode("utf-8", "replace") + b"\n") if a.headers else b"") + chunks[i] for i in ids_]
        if mem: mem_used += 1; mem_bytes += sum(len(c) for c in mem)
        pre_mem = b"".join(c + SEP for c in mem)
        b, targ, lp = score(pre_mem + first, second); res.append(b / (T // 2)); ntok += len(targ)
        ctx = llm.tokenize(first, add_bos=False, special=False)
        rec.write(json.dumps({"set": name, "w": w, "date": (dates[w] if w < len(dates) else ""), "chunks": len(mem), "bits": round(b, 4),
                              "ctx": [int(t) for t in ctx], "tgt": [int(t) for t in targ], "lp": [round(float(x), 5) for x in lp]}) + "\n"); rec.flush()
        n += 1
        if n % 50 == 0: print(f"  {name}: {n} windows, memory {np.mean(res):.4f} ({time.time() - t0:.0f} s)", flush=True)
    summary[name] = {"memory": {"bpb": float(np.mean(res)), "n": len(res), "tokens": ntok, "per_window": [round(x, 4) for x in res]}, "memory_used_windows": mem_used,
                     "memory_bytes_mean": (mem_bytes / mem_used) if mem_used else 0}
    print(f"{name}: {n} windows; memory {summary[name]['memory']['bpb']:.4f} bits/byte ({ntok / max(1, n * (T // 2)):.2f} tok/byte); memory found for {mem_used} windows, "
          f"{summary[name]['memory_bytes_mean']:.0f} bytes on average ({time.time() - t0:.0f} s)", flush=True)
summary.update({"model": a.model, "world": a.world, "T": T, "n_ctx": a.n_ctx, "retriever": ({"chooser": chooser_info, "label": a.label, "choices": a.choices} if choices is not None
                else {"chooser": "spans", "chunk": a.chunk, "span": a.span, "top": a.top, "min_hits": a.min_hits}) | {"order": a.order, "sep": a.sep, "headers": a.headers}})
json.dump(summary, open(os.path.join(a.out, "qwen_memory.json"), "w"), indent=1)
line = (f"Qwen ({os.path.basename(a.model)}) on the second half of 2 KB windows given the first (" + (f"{a.label}'s chunks" if choices is not None else "his files retrieved")
        + (", file headers" if a.headers else "") + f"): Kyle's turns ({summary['kyle']['memory']['n']}) memory {summary['kyle']['memory']['bpb']:.4f}; "
        f"the dialogue ({summary['dlg']['memory']['n']}) memory {summary['dlg']['memory']['bpb']:.4f}; tokens kept in tokens.jsonl")
print(line); open(os.path.join(a.out, "qwen_memory.txt"), "a").write(line + "\n")
