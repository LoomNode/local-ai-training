#!/usr/bin/env python
"""The prose corpus ("talk") from its sources, in the lab's corpus format (documents headed "\\n// ==== path ====\\n").
Two uses:
  the mix:     build_corpus_talk.py --out /games/ailab/corpora/talk.bytes --layout-dir /games/ailab/corpora/talk \\
                   stackexchange=1000000000 ubuntu-irc=300000000 oasst2=0 wikipedia=500000000 docs=200000000
               per source: shuffle its documents (seed), take whole documents up to the byte budget (0 = all), hold out --holdout of them
               for dev and as many again for test; the selected documents of all sources are shuffled together; file = train, dev, test.
  one source with a given dev file (for scoring a source on Kyle's held-out set): --dev-file /games/ailab/corpora/kyle/kyle.bytes <src>=0
               the whole source is train (cut at a document header), the dev region is the given file, no test.
Sources are read from --src-dir/<name>.bytes. Writes <out>, <layout-dir>/layout.json and manifest.json (documents and bytes per source and split).
"""
import argparse, json, os, random, sys, time
import numpy as np

HDR = b"\n// ==== "
ap = argparse.ArgumentParser()
ap.add_argument("--src-dir", default="/vault/ailab/corpora-src/talk")
ap.add_argument("--out", required=True); ap.add_argument("--layout-dir", required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--holdout", type=float, default=0.05)
ap.add_argument("--dev-file", default="", help="use this file as the dev region (the sources become train only)")
ap.add_argument("--T", type=int, default=2048); ap.add_argument("--max-dev-windows", type=int, default=1024)
ap.add_argument("sources", nargs="+", help="name=bytes (0 = all)")
a = ap.parse_args()

def doc_spans(m):
    """start offsets of every document header in the memmap, as an int64 array (+ the end)."""
    starts = []; n = len(m); chunk = 64 << 20; pos = 0
    while pos < n:
        buf = m[pos:min(n, pos + chunk + len(HDR))].tobytes(); i = buf.find(HDR)
        while i >= 0 and i < chunk:
            starts.append(pos + i); i = buf.find(HDR, i + 1)
        pos += chunk
    if not starts: sys.exit(f"no document headers in a source")
    starts = np.array(starts, dtype=np.int64)
    return starts, np.append(starts[1:], n)

rng = random.Random(a.seed)
plan = {}  # name -> dict(split -> list of (start, end))
for spec in a.sources:
    name, budget = spec.split("="); budget = int(budget)
    path = os.path.join(a.src_dir, f"{name}.bytes")
    m = np.memmap(path, dtype=np.uint8, mode="r"); starts, ends = doc_spans(m)
    idx = list(range(len(starts))); rng.shuffle(idx)
    sel = []; total = 0
    for i in idx:
        L = int(ends[i] - starts[i])
        if budget and total + L > budget: continue
        sel.append(i); total += L
        if budget and total >= budget: break
    if a.dev_file: splits = {"train": sel, "dev": [], "test": []}
    else:
        k = max(1, int(len(sel) * a.holdout)) if len(sel) >= 40 else 0
        splits = {"dev": sel[:k], "test": sel[k:2 * k], "train": sel[2 * k:]}
    plan[name] = {"path": path, "m": m, "starts": starts, "ends": ends, "splits": splits,
                  "docs_in_source": len(starts), "bytes_in_source": int(len(m))}
    print(f"{name}: {len(starts):,} docs, {len(m):,} bytes in the source; selected {len(sel):,} docs, {total:,} bytes "
          f"(train {len(splits['train']):,}, dev {len(splits['dev']):,}, test {len(splits['test']):,})", flush=True)

os.makedirs(a.layout_dir, exist_ok=True); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
manifest = {"sources": {}, "splits": {}}
offsets = {}
with open(a.out, "wb") as f:
    for split in ("train", "dev", "test"):
        items = [(name, i) for name, p in plan.items() for i in p["splits"][split]]
        rng.shuffle(items); start = f.tell(); per = {}
        for name, i in items:
            p = plan[name]; s, e = int(p["starts"][i]), int(p["ends"][i]); d = p["m"][s:e].tobytes()
            if not d.endswith(b"\n"): d += b"\n"
            f.write(d); per[name] = per.get(name, 0) + len(d)
        if split == "dev" and a.dev_file:
            d = open(a.dev_file, "rb").read(); f.write(d if d.endswith(b"\n") else d + b"\n"); per["dev_file"] = len(d)
        offsets[split] = (start, f.tell() - start)
        manifest["splits"][split] = {"offset": start, "bytes": f.tell() - start, "docs": len(items), "per_source_bytes": per}
    total = f.tell()
for name, p in plan.items():
    manifest["sources"][name] = {"path": p["path"], "docs_in_source": p["docs_in_source"], "bytes_in_source": p["bytes_in_source"],
                                 "docs_selected": {k: len(v) for k, v in p["splits"].items()}}
train_end = offsets["train"][0] + offsets["train"][1]
dev_off, dev_n = offsets["dev"]; test_off, test_n = offsets["test"]
layout = {"corpus": os.path.abspath(a.out), "built": time.strftime("%F"), "builder": "build_corpus_talk.py", "seed": a.seed,
          "flags": " ".join(sys.argv[1:]), "corpus_bytes": total, "train_end": train_end, "dev_offset": dev_off, "dev_bytes": dev_n,
          "test_offset": test_off, "test_bytes": test_n, "T": a.T, "dev_windows": min(a.max_dev_windows, dev_n // a.T),
          "note": ("the dev region is " + a.dev_file) if a.dev_file else
                  f"documents shuffled across sources (seed {a.seed}); dev and test hold out {a.holdout:.0%} of each source's selected documents"}
json.dump(layout, open(os.path.join(a.layout_dir, "layout.json"), "w"), indent=1)
json.dump(manifest, open(os.path.join(a.layout_dir, "manifest.json"), "w"), indent=1)
print(f"wrote {a.out}: {total:,} bytes; train {train_end:,}, dev {dev_n:,} ({layout['dev_windows']} windows), test {test_n:,}")
