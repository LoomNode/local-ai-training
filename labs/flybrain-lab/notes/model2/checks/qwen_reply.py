#!/usr/bin/env python
"""The product end to end (Kyle, 2026-10-21: "we can queue 3 up"): a real turn of his from the held-out dialogue, the memory his files
give for it, and the local Qwen generating the assistant's reply with and without that memory, next to the reply that was actually
given, for Kyle to read. Points: the first 'Assistant:' turn after a 'User:' turn of --min-user..--max-user bytes, --n of them spread
over the dialogue; the context is up to --context bytes before it starting at a turn boundary; the memory is the span chooser's
(dedup, --chunk x --top, the window's own text excluded) for the context's last 1,024 bytes, labelled ('## <path>') with a rule
separator, as the sweep found best. The prompt is ChatML with an empty think block (no thinking); greedy decoding, --max-tokens.
Writes <out>/replies.md (0600: his text, both replies, the actual reply; for Kyle's eyes, never committed) and <out>/points.json
(offsets, sizes, paths; no text). Prints counts only. --dry selects and retrieves without the model.
usage: qwen_reply.py --out DIR --model GGUF [--n 24 --context 3000 --chunk 750 --top 4 --max-mem 3200 --max-tokens 350 --dry]"""
import argparse, glob, hashlib, json, os, re, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--model", default="/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf")
ap.add_argument("--dialogue", default="/games/ailab/corpora/kyle/dialogue.bytes"); ap.add_argument("--world", default="/games/ailab/corpora/kyle/world/files")
ap.add_argument("--n", type=int, default=24); ap.add_argument("--context", type=int, default=3000); ap.add_argument("--min-user", type=int, default=120); ap.add_argument("--max-user", type=int, default=2500)
ap.add_argument("--reply", type=int, default=900, help="bytes of the actual reply shown"); ap.add_argument("--query", type=int, default=1024)
ap.add_argument("--chunk", type=int, default=750); ap.add_argument("--top", type=int, default=4); ap.add_argument("--max-mem", type=int, default=3200); ap.add_argument("--min-hits", type=int, default=3)
ap.add_argument("--span", type=int, default=12); ap.add_argument("--sep", default="\n\n---\n\n"); ap.add_argument("--max-tokens", type=int, default=350); ap.add_argument("--n-ctx", type=int, default=8192)
ap.add_argument("--dry", action="store_true"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); os.umask(0o077); os.makedirs(a.out, exist_ok=True); os.chmod(a.out, 0o700); t0 = time.time()

# ---- the points -----------------------------------------------------------------------------------------------------------------------------
D = open(a.dialogue, "rb").read()
users = [m.start() for m in re.finditer(rb"\nUser:", D)]; assists = [m.start() for m in re.finditer(rb"\nAssistant:", D)]
turns = [m.start() for m in re.finditer(rb"\n(User|Assistant):", D)]; turns_a = np.asarray(turns)
cands = []
for u in users:
    i = np.searchsorted(turns_a, u, "right")
    if i >= len(turns): continue
    nxt = turns[i]
    if not D.startswith(b"\nAssistant:", nxt): continue
    if a.min_user <= nxt - u <= a.max_user: cands.append((u, nxt))
rng = np.random.default_rng(a.seed); idx = np.linspace(0, len(cands) - 1, a.n).round().astype(int) if len(cands) >= a.n else np.arange(len(cands))
points = []
for k in idx:
    u, m = cands[k]; lo = max(0, m - a.context); j = np.searchsorted(turns_a, lo, "left"); start = turns[j] + 1 if j < len(turns) and turns[j] < u else lo
    ctx = D[start:m]; e = np.searchsorted(turns_a, m, "right"); end = turns[e] if e < len(turns) else len(D)
    actual = D[m + 1:min(end, m + 1 + a.reply)]; user_text = D[u + 1:m]
    points.append({"user_at": int(u), "reply_at": int(m), "context_start": int(start), "context_bytes": len(ctx), "user_bytes": len(user_text), "actual_bytes": len(actual),
                   "_ctx": ctx, "_user": user_text, "_actual": actual})
print(f"dialogue {len(D):,} bytes: {len(users):,} user turns, {len(cands):,} candidate points (a user turn of {a.min_user}-{a.max_user} bytes answered next), {len(points)} chosen; "
      f"context {np.mean([p['context_bytes'] for p in points]):.0f} bytes on average, the user's turn {np.mean([p['user_bytes'] for p in points]):.0f}, the actual reply shown {np.mean([p['actual_bytes'] for p in points]):.0f}", flush=True)

# ---- the memory (the span chooser with dedup, labelled) -------------------------------------------------------------------------------------
chunks = []; paths = []
for p in sorted(glob.glob(os.path.join(a.world, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    b = open(p, "rb").read(); rel = os.path.relpath(p, a.world)
    for s in range(0, max(1, len(b) - a.chunk // 2), a.chunk // 2 if len(b) > a.chunk else a.chunk):
        c = b[s:s + a.chunk]
        if len(c) < 64: continue
        chunks.append(c); paths.append(rel)
MUL = np.uint64(0x9E3779B97F4A7C15)
def span_hashes(b: bytes, S: int) -> np.ndarray:
    d = np.frombuffer(b, dtype=np.uint8).astype(np.uint64)
    if len(d) < S: return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(d) - S + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(S): h = h * MUL + d[j:len(d) - S + 1 + j] + np.uint64(1)
    return h
hs = []; ids = []
for i, c in enumerate(chunks):
    h = np.unique(span_hashes(c, a.span)); hs.append(h); ids.append(np.full(len(h), i, dtype=np.int32))
H = np.concatenate(hs); I = np.concatenate(ids); o = np.argsort(H, kind="stable"); SH = H[o]; SI = I[o]
def choose(query: bytes, W: bytes) -> list[int]:
    q = np.unique(span_hashes(query, a.span)); lo = np.searchsorted(SH, q, "left"); hi = np.searchsorted(SH, q, "right")
    counts = np.zeros(len(chunks), dtype=np.int32)
    for l_, h_ in zip(lo, hi):
        if h_ > l_: np.add.at(counts, SI[l_:h_], 1)
    out = []; total = 0
    for i in np.argsort(-counts)[: a.top * 8]:
        if counts[i] < a.min_hits or len(out) >= a.top: break
        c = chunks[i]
        if c[:200] in W or W[:200] in c: continue
        if any(c[:200] in chunks[j] or chunks[j][:200] in c for j in out): continue
        out.append(int(i)); total += len(c)
        if total >= a.max_mem: break
    return out
for p in points:
    ids_ = choose(p["_ctx"][-a.query:], p["_ctx"]); p["memory_ids"] = ids_; p["memory_paths"] = [paths[i] for i in ids_]
    p["_mem"] = b"".join(b"## " + paths[i].encode("utf-8", "replace") + b"\n" + chunks[i] + a.sep.encode() for i in ids_); p["memory_bytes"] = len(p["_mem"])
got = sum(1 for p in points if p["memory_ids"])
print(f"pool {len(chunks):,} chunks of <= {a.chunk}; a memory for {got} of {len(points)} points, {np.mean([p['memory_bytes'] for p in points if p['memory_ids']]) if got else 0:.0f} bytes, "
      f"{np.mean([len(p['memory_ids']) for p in points if p['memory_ids']]) if got else 0:.1f} chunks, {len({pa for p in points for pa in p['memory_paths']})} distinct files ({time.time() - t0:.0f} s)", flush=True)
json.dump({"model": a.model, "dialogue": a.dialogue, "chunk": a.chunk, "top": a.top, "points": [{k: v for k, v in p.items() if not k.startswith("_")} for p in points]},
          open(os.path.join(a.out, "points.json"), "w"), indent=0)
if a.dry: print("dry: no model"); raise SystemExit(0)

# ---- the replies ----------------------------------------------------------------------------------------------------------------------------
from llama_cpp import Llama  # noqa: E402
llm = Llama(model_path=a.model, n_gpu_layers=-1, n_ctx=a.n_ctx, n_batch=1024, verbose=False)
SYSTEM = ("You are the assistant in a working session with Kyle, who runs a homelab and an AI lab on his own machines. The transcript so far "
          "is given; reply to his last message as the session's assistant would, in the same register, concretely and briefly.")
def prompt(ctx: bytes, mem: bytes) -> bytes:
    sysm = SYSTEM.encode() + (b"\n\nNotes from Kyle's own files that may bear on this (each under its path):\n\n" + mem if mem else b"")
    return b"<|im_start|>system\n" + sysm + b"<|im_end|>\n<|im_start|>user\n" + ctx + b"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
def reply(ctx: bytes, mem: bytes) -> tuple[str, int, float]:
    t = time.time(); toks = llm.tokenize(prompt(ctx, mem), add_bos=False, special=True)
    r = llm.create_completion(prompt=toks, max_tokens=a.max_tokens, temperature=0.0, stop=["<|im_end|>", "\nUser:"], seed=a.seed)
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", r["choices"][0]["text"], flags=re.S).strip()
    return text, len(toks), time.time() - t
md = [f"# The product end to end: {len(points)} turns of the held-out dialogue, the local Qwen replying with and without the memory of Kyle's files\n",
      f"model {os.path.basename(a.model)}; memory = the span chooser (dedup, {a.top} chunks of {a.chunk} bytes, labelled); greedy, {a.max_tokens} tokens at most; context {a.context} bytes.\n"]
lens = {"without": [], "with": []}; secs = 0.0; ptoks = []
for k, p in enumerate(points):
    r0, n0, s0 = reply(p["_ctx"], b""); r1, n1, s1 = reply(p["_ctx"], p["_mem"]) if p["memory_ids"] else (r0, n0, 0.0)
    secs += s0 + s1; ptoks += [n0, n1]; lens["without"].append(len(r0)); lens["with"].append(len(r1))
    p["reply_without_chars"] = len(r0); p["reply_with_chars"] = len(r1); p["prompt_tokens"] = [n0, n1]
    md.append(f"\n## {k + 1}. dialogue offset {p['reply_at']:,} (context {p['context_bytes']} bytes, memory {p['memory_bytes']} bytes from {len(p['memory_ids'])} chunks)\n")
    md.append("**Memory files:** " + (", ".join(f"`{x}`" for x in p["memory_paths"]) if p["memory_paths"] else "none found") + "\n")
    md.append("\n**His last message:**\n\n```text\n" + p["_user"].decode("utf-8", "replace").strip()[:2500] + "\n```\n")
    md.append("\n**Reply without memory:**\n\n```text\n" + r0[:3000] + "\n```\n")
    md.append("\n**Reply with memory:**\n\n```text\n" + (r1[:3000] if p["memory_ids"] else "(no memory found: same as above)") + "\n```\n")
    md.append("\n**The reply actually given:**\n\n```text\n" + p["_actual"].decode("utf-8", "replace").strip() + "\n```\n")
    print(f"  {k + 1}/{len(points)}: prompts {n0}/{n1} tokens, replies {len(r0)}/{len(r1)} chars ({s0 + s1:.0f} s)", flush=True)
open(os.path.join(a.out, "replies.md"), "w").write("".join(md)); os.chmod(os.path.join(a.out, "replies.md"), 0o600)
json.dump({"model": a.model, "dialogue": a.dialogue, "chunk": a.chunk, "top": a.top, "seconds": secs, "points": [{k: v for k, v in p.items() if not k.startswith("_")} for p in points]},
          open(os.path.join(a.out, "points.json"), "w"), indent=0)
print(f"wrote {a.out}/replies.md: {len(points)} points, replies {np.mean(lens['without']):.0f} / {np.mean(lens['with']):.0f} chars on average (without / with memory), prompts {np.mean(ptoks):.0f} tokens, {secs:.0f} s of generation ({time.time() - t0:.0f} s)")
