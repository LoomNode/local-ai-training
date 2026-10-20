#!/usr/bin/env python
"""Qwen as the yardstick, pointed at the lab's memory (Kyle, 2026-10-19: "go ahead and run the qwen tests"): how much is Kyle's world worth
to a model that already knows the language? The shape of the reader's measurements: every 2,048-byte held-out window's SECOND half is
scored given its first half (warm), three ways:
  cold      the whole window alone (the reader's cold number's shape)
  warm      the second half given the first half
  first     the first half alone (the easy-window gate's signal, 2026-10-20: a memory hurts the windows Qwen already reads easily)
  memory    the same, with text retrieved from Kyle's world by the first half in front of it: the top chunks of his files sharing the
            most 12-byte spans with the first half (the lab's match idea as a retriever; the memory sees the context, never the target)
Bits per byte of the scored bytes, by Qwen's own token log-probabilities (llama-cpp-python, logits for every position, the GGUF on one
card). Tokens are cut at the half boundary (the halves tokenised separately), so the scored tokens are exactly the second half's bytes.
Prints counts and numbers only, never text. usage:
  qwen_memory.py --model GGUF --world DIR --heldout heldout.bytes --json heldout.json --out DIR [--kyle-every 1] [--dlg-every 4] [--n-ctx 6144] [--top 2]
  --choices choices.json (table_retrieve.py's, 2026-10-20): the memory condition takes the chunks the tables chose for each window instead of the
  span retriever's (the same pool, checked by digest); --compare qwen_memory.json: paired per-window differences against an earlier run's
  warm and memory conditions; --conditions memory runs only that one; --label names the chooser in the paired lines (default "the tables").
The sweep (Kyle, 2026-10-20: "exhaustively test all 3 separately"): --cache scores.jsonl keeps every prompt's bits by content hash (the model,
the context size, the prefix and the target), so a prompt any run scored before is not scored again (the scorer is deterministic: identical
sets in the same order scored identically to four decimals, twice); the cache holds hashes and numbers, never text. --order top-last puts the
best chunk adjacent to the window; --sep sets the separator after each chunk; --headers puts a '## <file path>' line before each chunk;
--conditions first_mem scores the FIRST half given the memory (the memory's effect on the query itself, a gate signal known at use time)."""
import argparse, codecs, glob, hashlib, json, math, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--world", default="/games/ailab/corpora/kyle/world/files")
ap.add_argument("--heldout", default="/games/ailab/corpora/kyle/heldout.bytes"); ap.add_argument("--json", default="/games/ailab/corpora/kyle/heldout.json")
ap.add_argument("--out", required=True); ap.add_argument("--T", type=int, default=2048)
ap.add_argument("--kyle-every", type=int, default=1); ap.add_argument("--dlg-every", type=int, default=4)
ap.add_argument("--n-ctx", type=int, default=6144); ap.add_argument("--top", type=int, default=2); ap.add_argument("--chunk", type=int, default=1500)
ap.add_argument("--span", type=int, default=12); ap.add_argument("--min-hits", type=int, default=3); ap.add_argument("--max-mem", type=int, default=3200)
ap.add_argument("--conditions", default="cold,warm,memory"); ap.add_argument("--limit", type=int, default=0, help="debug: score only this many windows a set")
ap.add_argument("--choices", default=None, help="table_retrieve.py's choices.json: the memory chunks per window, by chunk id into the same pool")
ap.add_argument("--compare", default=None, help="an earlier run's qwen_memory.json: paired per-window differences against its warm and memory")
ap.add_argument("--label", default="the tables", help="the chooser's name in the paired lines (--choices runs)")
ap.add_argument("--cache", default=None, help="a scores.jsonl: every prompt's bits by content hash; a prompt scored before, by any run, is not scored again")
ap.add_argument("--order", default="top-first", choices=["top-first", "top-last"], help="the memory's chunk order: the best chunk first (farthest from the window) or last (adjacent to it)")
ap.add_argument("--sep", default="\\n\\n", help="the separator after each chunk (python escapes; default a blank line)")
ap.add_argument("--headers", action="store_true", help="a '## <file path>' line before each chunk")
a = ap.parse_args()
os.umask(0o077); os.makedirs(a.out, exist_ok=True)
SEP = codecs.decode(a.sep, "unicode_escape").encode()
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
t0 = time.time(); chunks = []; hs = []; ids = []; paths = []
for p in sorted(glob.glob(os.path.join(a.world, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    b = open(p, "rb").read(); rel = os.path.relpath(p, a.world)
    for s in range(0, max(1, len(b) - a.chunk // 2), a.chunk // 2 if len(b) > a.chunk else a.chunk):   # half-overlapping chunks
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
    """The span chooser's chunk ids (top-first)."""
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
        out.append(int(i)); total += len(c)
        if total >= a.max_mem: break
    return out

# ---- the cache: every prompt's bits by content hash, shared by every run ---------------------------------------------------------------------
CACHE = {}; cache_f = None; cache_hits = 0
if a.cache:
    os.makedirs(os.path.dirname(os.path.abspath(a.cache)), exist_ok=True)
    if os.path.exists(a.cache):
        for line in open(a.cache):
            try: d = json.loads(line)
            except Exception: continue
            CACHE[d["k"]] = (d["b"], d["t"])
    cache_f = open(a.cache, "a"); os.chmod(a.cache, 0o600)
    print(f"cache: {len(CACHE):,} prompts scored before ({a.cache})", flush=True)
KEY0 = (os.path.basename(a.model) + f"|{a.n_ctx}|").encode()
def ckey(prefix: bytes, target: bytes) -> str: return hashlib.sha256(KEY0 + prefix + b"\x00<|target|>\x00" + target).hexdigest()[:32]

# ---- Qwen -------------------------------------------------------------------------------------------------------------------------------
llm = Llama(model_path=a.model, n_gpu_layers=-1, n_ctx=a.n_ctx, n_batch=1024, logits_all=True, verbose=False)
BOS = llm.token_bos()
def score(prefix: bytes, target: bytes) -> tuple[float, int]:
    """Bits of `target` given `prefix` (the halves tokenised separately): (bits, tokens)."""
    global cache_hits
    k_ = ckey(prefix, target) if a.cache else None
    if k_ is not None and k_ in CACHE: cache_hits += 1; return CACHE[k_]
    tp = llm.tokenize(prefix, add_bos=True, special=False) if prefix else [BOS]
    tt = llm.tokenize(target, add_bos=False, special=False)
    toks = (tp + tt)[-a.n_ctx:]; k = len(tt)
    llm.reset(); llm.eval(toks)
    logits = np.asarray(llm.scores[:len(toks)], dtype=np.float32)              # logits at each position predict the NEXT token
    pred = logits[len(toks) - k - 1:len(toks) - 1]                              # the positions whose next token is a target token
    targ = np.asarray(toks[len(toks) - k:], dtype=np.int64)
    m = pred.max(axis=1, keepdims=True); lse = m[:, 0] + np.log(np.exp(pred - m).sum(axis=1))
    lp = pred[np.arange(k), targ] - lse
    r = (float(-lp.sum() / math.log(2)), k)
    if k_ is not None: CACHE[k_] = r; cache_f.write(json.dumps({"k": k_, "b": r[0], "t": r[1]}) + "\n"); cache_f.flush()
    return r

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
        if "first" in conds:
            b, k = score(b"", first); res["first"].append(b / (T // 2)); ntok["first"] += k
        if "memory" in conds or "first_mem" in conds:
            ids_ = list(choices[name][n]) if choices is not None else retrieve(first, W)
            if a.order == "top-last": ids_ = ids_[::-1]
            mem = [((b"## " + paths[i].encode("utf-8", "replace") + b"\n") if a.headers else b"") + chunks[i] for i in ids_]
            if mem: mem_used += 1; mem_bytes += sum(len(c) for c in mem)
            pre_mem = b"".join(c + SEP for c in mem)
            if "memory" in conds:
                b, k = score(pre_mem + first, second); res["memory"].append(b / (T // 2)); ntok["memory"] += k
            if "first_mem" in conds:
                b, k = score(pre_mem, first); res["first_mem"].append(b / (T // 2)); ntok["first_mem"] += k
        n += 1
        if n % 50 == 0: print(f"  {name}: {n} windows, " + ", ".join(f"{c} {np.mean(res[c]):.4f}" for c in conds) + f" ({time.time() - t0:.0f} s, {cache_hits} from the cache)", flush=True)
    summary[name] = {c: {"bpb": float(np.mean(res[c])), "n": len(res[c]), "tokens": ntok[c], "per_window": [round(x, 4) for x in res[c]]} for c in conds}
    summary[name]["memory_used_windows"] = mem_used; summary[name]["memory_bytes_mean"] = (mem_bytes / mem_used) if mem_used else 0
    print(f"{name}: {n} windows; " + "; ".join(f"{c} {summary[name][c]['bpb']:.4f} bits/byte ({summary[name][c]['tokens'] / (n * (T if c == 'cold' else T // 2)):.2f} tok/byte)" for c in conds)
          + f"; memory found for {mem_used} windows, {summary[name]['memory_bytes_mean']:.0f} bytes on average ({time.time() - t0:.0f} s)", flush=True)
summary["model"] = a.model; summary["world"] = a.world; summary["T"] = T; summary["n_ctx"] = a.n_ctx
summary["retriever"] = {"chooser": chooser_info, "label": a.label, "choices": a.choices} if choices is not None else {"chooser": "spans", "chunk": a.chunk, "span": a.span, "top": a.top, "min_hits": a.min_hits}
summary["retriever"].update({"order": a.order, "sep": a.sep, "headers": a.headers}); summary["cache_hits"] = cache_hits
json.dump(summary, open(os.path.join(a.out, "qwen_memory.json"), "w"), indent=1)
k, d = summary["kyle"], summary["dlg"]; who = f"{a.label}{chr(39) if a.label.endswith(chr(115)) else chr(39) + chr(115)} chunks" if choices is not None else "his files retrieved" if "memory" in conds else "no memory"
fmt = ("" if a.order == "top-first" else ", best chunk last") + ("" if a.sep == "\\n\\n" else f", sep {a.sep!r}") + (", file headers" if a.headers else "")
parts = [f"Qwen ({os.path.basename(a.model)}) on " + ("the first half alone (the gate's signal)" if conds == ["first"] else f"the first half given the memory ({who}{fmt})" if conds == ["first_mem"]
         else f"the second half of 2 KB windows given the first ({who}{fmt})") + ":"]
for s_, nm in ((k, "Kyle's turns"), (d, "the dialogue")):
    parts.append(f"{nm} ({s_[conds[0]]['n']}) " + ", ".join(f"{c} {s_[c]['bpb']:.4f}" for c in conds) + (f" ({s_['memory']['bpb'] - s_['warm']['bpb']:+.4f})" if "memory" in conds and "warm" in conds else "") + ";")
line = " ".join(parts); print(line); open(os.path.join(a.out, "qwen_memory.txt"), "a").write(line + "\n")
if a.compare and "memory" in conds:
    ref = json.load(open(a.compare)); lines = []
    for s_, nm in (("kyle", "Kyle's turns"), ("dlg", "the dialogue")):
        new = np.asarray(summary[s_]["memory"]["per_window"]); n_ = len(new)
        warm = np.asarray(ref[s_]["warm"]["per_window"][:n_]); old = np.asarray(ref[s_]["memory"]["per_window"][:n_])
        if len(warm) != n_ or len(old) != n_: lines.append(f"{nm}: {a.compare} holds other windows; no pairing"); continue
        g_new, g_old = new - warm, old - warm; dd = new - old; srt = np.sort(g_new); tenth = max(1, n_ // 10)
        L = a.label; P = L + ("'" if L.endswith("s") else "'s")
        lines.append(f"{nm} ({n_} windows, paired): warm {warm.mean():.4f}; the spans' memory {old.mean():.4f} ({g_old.mean():+.4f}); {P} memory {new.mean():.4f} ({g_new.mean():+.4f}); "
                     f"{L} vs the spans {dd.mean():+.4f} (median {np.median(dd):+.4f}; {L} better by >0.02 on {int((dd < -0.02).sum())}, worse on {int((dd > 0.02).sum())}); "
                     f"{P} gain: median {np.median(g_new):+.4f}, improves >0.02 on {int((g_new < -0.02).sum())}, worsens on {int((g_new > 0.02).sum())}, best tenth carries {srt[:tenth].sum() / srt.sum() if srt.sum() < 0 else float('nan'):.0%}")
    for l_ in lines: print(l_)
    open(os.path.join(a.out, "qwen_memory.txt"), "a").write("\n".join(lines) + "\n")
