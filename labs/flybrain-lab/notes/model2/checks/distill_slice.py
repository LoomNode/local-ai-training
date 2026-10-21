#!/usr/bin/env python
"""The distillation slice (T2 of the plan, 2026-10-20): 32 MB of text that looks like Kyle's use, the same bytes for every arm.
Half from mixB's train region (oasst2 documents first, then Stack Exchange, in corpus order; never the dev, test or tail regions),
half from his world (every assistant reply, then his own files in sorted order, to the budget; never his held-out turns). Every
document is checked for 64-byte spans of the held-out text and dropped if it holds one. The lab's corpus format (a header line
"\\n// ==== path ====\\n" before each document). Output mode 0600 (it carries his world's text); prints counts only, never text.
usage: distill_slice.py --out /games/ailab/corpora/distill.bytes --layout-out /games/ailab/corpora/distill [--mb-corpus 16 --mb-world 16]"""
import argparse, hashlib, json, os, re, time
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="/games/ailab/corpora/talkB.bytes"); ap.add_argument("--layout", default="/games/ailab/corpora/talkB/layout.json")
ap.add_argument("--world", default="/games/ailab/corpora/kyle/world"); ap.add_argument("--heldout", default="/games/ailab/corpora/kyle/heldout.bytes")
ap.add_argument("--out", required=True); ap.add_argument("--layout-out", required=True)
ap.add_argument("--mb-corpus", type=float, default=16); ap.add_argument("--mb-world", type=float, default=16); ap.add_argument("--span", type=int, default=64)
ap.add_argument("--sources", default="oasst2,stackexchange", help="the corpus sources, in order; the corpus half is split evenly among them")
a = ap.parse_args(); os.umask(0o077)
HDR = re.compile(rb"\n// ==== ([^\n]*) ====\n")
t0 = time.time()
held = open(a.heldout, "rb").read(); S = a.span
spans = {hash(held[i:i + S]) for i in range(0, len(held) - S + 1)}
def leaks(b: bytes) -> bool: return any(hash(b[i:i + S]) in spans for i in range(0, len(b) - S + 1))
print(f"held-out spans: {len(spans):,} ({time.time() - t0:.0f} s)", flush=True)

# ---- the corpus half: documents of the train region, by source ---------------------------------------------------------------------------
lay = json.load(open(a.layout)); train_end = int(lay["train_end"])
corpus = np.memmap(a.corpus, dtype=np.uint8, mode="r")
docs = []                                                                   # (start, end, source, name) in corpus order
BLOCK = 1 << 26; last = None
for lo in range(0, train_end, BLOCK):
    hi = min(train_end, lo + BLOCK + 4096); b = bytes(corpus[lo:hi])
    for m in HDR.finditer(b):
        s = lo + m.start()
        if s >= min(train_end, lo + BLOCK): break
        if last is not None and s <= last[0]: continue
        path = m.group(1).decode("utf-8", "replace"); src = path.split("/", 1)[0]
        if last is not None: docs.append((last[0], s, last[1], last[2]))
        last = (s, src, path)
if last is not None: docs.append((last[0], train_end, last[1], last[2]))
print(f"train region: {len(docs):,} documents in {train_end:,} bytes ({time.time() - t0:.0f} s)", flush=True)
sources = a.sources.split(","); per_source = a.mb_corpus * 1e6 / len(sources)
out = open(a.out, "wb"); os.chmod(a.out, 0o600); ranges = []; pos = 0; stats = Counter(); taken_bytes = Counter()
for src in sources:
    for s, e, src_, path in docs:
        if src_ != src: continue
        if taken_bytes[src] >= per_source: break
        b = bytes(corpus[s:e]); stats[(src, "seen")] += 1
        if leaks(b): stats[(src, "dropped: held-out span")] += 1; continue
        out.write(b); ranges.append([pos, pos + len(b), src, path]); pos += len(b); taken_bytes[src] += len(b); stats[(src, "kept")] += 1
# ---- the world half: the replies, then his files ----------------------------------------------------------------------------------------
budget = a.mb_world * 1e6
for sub in ("replies", "files"):
    root = os.path.join(a.world, sub); paths = []
    for dp, dn, fn in os.walk(root):
        dn.sort(); paths += [os.path.join(dp, f) for f in sorted(fn)]
    for p in paths:
        if taken_bytes["world"] >= budget: break
        b = open(p, "rb").read(); rel = "world/" + os.path.relpath(p, a.world); stats[(sub, "seen")] += 1
        if not b.strip(): stats[(sub, "empty")] += 1; continue
        if leaks(b): stats[(sub, "dropped: held-out span")] += 1; continue
        doc = b"\n// ==== " + rel.encode("utf-8", "replace") + b" ====\n" + b
        out.write(doc); ranges.append([pos, pos + len(doc), sub, rel]); pos += len(doc); taken_bytes["world"] += len(doc); stats[(sub, "kept")] += 1
out.close()
digest = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
os.makedirs(a.layout_out, exist_ok=True); os.chmod(a.layout_out, 0o700)
json.dump({"slice": a.out, "bytes": pos, "sha256": digest, "built": time.strftime("%F"), "corpus": a.corpus, "train_end": train_end, "world": a.world,
           "span": S, "per_source_bytes": dict(taken_bytes), "counts": {f"{k[0]}:{k[1]}": v for k, v in stats.items()}, "ranges": ranges},
          open(os.path.join(a.layout_out, "layout.json"), "w"), indent=1)
os.chmod(os.path.join(a.layout_out, "layout.json"), 0o600)
print(f"slice: {pos:,} bytes, {len(ranges):,} documents, sha256 {digest[:16]} ({time.time() - t0:.0f} s)")
for k in sorted(stats): print(f"  {k[0]:14s} {k[1]:26s} {stats[k]:6d}")
print("bytes by source:", {k: f"{v:,}" for k, v in taken_bytes.items()})
