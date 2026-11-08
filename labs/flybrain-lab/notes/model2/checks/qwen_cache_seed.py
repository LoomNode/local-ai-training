#!/usr/bin/env python
"""Seed qwen_memory.py's prompt cache from runs scored before it existed (the sweep, 2026-10-20): every run's per-window numbers are the
bits of a prompt that can be rebuilt byte for byte (the window's halves, the chunks a choices.json names in top-first order, a blank line
after each chunk), so the cache gets them at the numbers' stored precision (four decimals of bits per byte) and later runs with the same
prompt for a window score it from the cache. Hashes and numbers only, never text. usage:
  qwen_cache_seed.py --cache scores.jsonl --model GGUF [--n-ctx 6144] [--world DIR --chunk 1500] RUN=COND[:CHOICES.json[:KEY]] ...
  e.g. results/tables/qwen=warm,cold  results/tables/qwen=memory:results/tables/qwen-dedup/choices.json:span_choices
       results/tables/qwen-gate=first  results/tables/qwen-breadth=memory:results/tables/qwen-breadth/choices.json"""
import argparse, glob, hashlib, json, os, time
ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--model", required=True); ap.add_argument("--n-ctx", type=int, default=6144)
ap.add_argument("--world", default="/games/ailab/corpora/kyle/world/files"); ap.add_argument("--chunk", type=int, default=1500)
ap.add_argument("--heldout", default="/games/ailab/corpora/kyle/heldout.bytes"); ap.add_argument("--json", default="/games/ailab/corpora/kyle/heldout.json")
ap.add_argument("--T", type=int, default=2048); ap.add_argument("--kyle-every", type=int, default=1); ap.add_argument("--dlg-every", type=int, default=4)
ap.add_argument("runs", nargs="+")
a = ap.parse_args(); os.umask(0o077)
t0 = time.time(); chunks = []
for p in sorted(glob.glob(os.path.join(a.world, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    b = open(p, "rb").read()
    for s in range(0, max(1, len(b) - a.chunk // 2), a.chunk // 2 if len(b) > a.chunk else a.chunk):
        c = b[s:s + a.chunk]
        if len(c) < 64: continue
        chunks.append(c)
digest = hashlib.sha256(b"".join(hashlib.sha256(c).digest() for c in chunks)).hexdigest()
KEY0 = (os.path.basename(a.model) + f"|{a.n_ctx}|").encode()
def ckey(prefix: bytes, target: bytes) -> str: return hashlib.sha256(KEY0 + prefix + b"\x00<|target|>\x00" + target).hexdigest()[:32]
have = {}
if os.path.exists(a.cache):
    for line in open(a.cache):
        try: d = json.loads(line); have[d["k"]] = d
        except Exception: pass
held = open(a.heldout, "rb").read(); meta = json.load(open(a.json)); T = a.T
sets = {"kyle": (meta["kyle"]["offset"], meta["kyle"]["windows"], a.kyle_every), "dlg": (meta["dialogue"]["offset"], meta["dialogue"]["windows"], a.dlg_every)}
f = open(a.cache, "a"); os.chmod(a.cache, 0o600); added = 0; skipped = 0; clashes = 0
for spec in a.runs:
    run, rest = spec.split("=", 1); parts = rest.split(":"); conds = parts[0].split(","); choices = None
    if len(parts) > 1:
        ch = json.load(open(parts[1])); key = parts[2] if len(parts) > 2 else "choices"
        if ch["pool"]["digest"] != digest: raise SystemExit(f"{parts[1]}: another chunk pool")
        choices = ch[key]
    R = json.load(open(os.path.join(run, "qwen_memory.json")))
    for name, (off, nwin, every) in sets.items():
        n = 0
        for w in range(0, nwin, every):
            W = held[off + w * T: off + (w + 1) * T]; first, second = W[:T // 2], W[T // 2:]
            for c in conds:
                pw = R[name][c]["per_window"]
                if n >= len(pw): continue
                if c == "cold": k_, bits = ckey(b"", W), pw[n] * T
                elif c == "warm": k_, bits = ckey(first, second), pw[n] * (T // 2)
                elif c == "first": k_, bits = ckey(b"", first), pw[n] * (T // 2)
                elif c == "memory":
                    mem = [chunks[i] for i in choices[name][n]]; pre = b"".join(x + b"\n\n" for x in mem) + first if mem else first
                    k_, bits = ckey(pre, second), pw[n] * (T // 2)
                else: raise SystemExit(f"condition {c}?")
                if k_ in have:
                    if abs(have[k_]["b"] - bits) > 0.06: clashes += 1
                    skipped += 1; continue
                d = {"k": k_, "b": float(bits), "t": 0}; have[k_] = d; f.write(json.dumps(d) + "\n"); added += 1
            n += 1
    print(f"{run} {conds}: cache now {len(have):,} ({time.time() - t0:.0f} s)", flush=True)
f.close()
print(f"seeded {added:,} prompts, {skipped:,} were there already ({clashes} with a different number)")
