#!/usr/bin/env python
"""Teach the reader to read memory (Kyle, 2026-10-21: "do 1 now"): his world laid out as pieces with retrieved memory in front of each,
for a fine-tune of the 4 KB reader (the combo recipe). The pool and the chooser are qwen_memory.py's span chooser with dedup (the same
files, half-overlapping chunks of --chunk bytes, 12-byte spans); the memory's format is the sweep's winner (a '## <path>' line before
each chunk, a rule separator after it). For every file of the world, in the pool's order: pieces of --piece bytes; each piece's memory
is retrieved by its FIRST HALF (as at use, where the query is the window's first half), never from the piece's own file nor from a copy
of it (a chunk whose first 200 bytes lie in the file, or the reverse), never two chunks of one text; the top --top chunks with at
least --min-hits spans, up to --max-mem bytes. Each document written is the lab format's header line, the memory, then the piece; a
piece whose query finds nothing is written without a memory (as at use). Then the replay world-mix used: the same slice of the corpus's
train region. Writes <out> (0600) and <layout>/layout.json (counts); prints counts only, never text.
usage: rag_corpus.py --out /games/ailab/corpora/kyle/world-rag.bytes --layout /games/ailab/corpora/kyle/world-rag-layout
       [--world .../kyle/world/files --chunk 900 --top 2 --max-mem 1800 --min-hits 3 --piece 2048 --replay /games/ailab/corpora/talk.bytes:1200000000:1233554432]"""
import argparse, glob, hashlib, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--layout", required=True); ap.add_argument("--world", default="/games/ailab/corpora/kyle/world/files")
ap.add_argument("--chunk", type=int, default=900); ap.add_argument("--top", type=int, default=2); ap.add_argument("--max-mem", type=int, default=1800)
ap.add_argument("--min-hits", type=int, default=3); ap.add_argument("--span", type=int, default=12); ap.add_argument("--dedup-len", type=int, default=200)
ap.add_argument("--piece", type=int, default=2048); ap.add_argument("--min-piece", type=int, default=256); ap.add_argument("--query", type=int, default=1024)
ap.add_argument("--sep", default="\n\n---\n\n"); ap.add_argument("--replay", default="/games/ailab/corpora/talk.bytes:1200000000:1233554432")
a = ap.parse_args(); os.umask(0o077); t0 = time.time()

# ---- the pool (qwen_memory.py's, byte for byte, at this chunk size) -------------------------------------------------------------------------
chunks = []; chunk_file = []; paths = []; files = []; rels = []
for p in sorted(glob.glob(os.path.join(a.world, "**", "*"), recursive=True)):
    if not os.path.isfile(p): continue
    b = open(p, "rb").read(); rel = os.path.relpath(p, a.world); fi = len(files); files.append(b); rels.append(rel)
    for s in range(0, max(1, len(b) - a.chunk // 2), a.chunk // 2 if len(b) > a.chunk else a.chunk):
        c = b[s:s + a.chunk]
        if len(c) < 64: continue
        chunks.append(c); chunk_file.append(fi); paths.append(rel)
chunk_file = np.asarray(chunk_file)
digest = hashlib.sha256(b"".join(hashlib.sha256(c).digest() for c in chunks)).hexdigest()
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
H = np.concatenate(hs); I = np.concatenate(ids); o = np.argsort(H, kind="stable"); SH = H[o]; SI = I[o]; del H, I, o, ids
print(f"pool: {len(chunks):,} chunks of <= {a.chunk} bytes from {len(files):,} files, digest {digest[:16]}; {len(SH):,} span hashes ({time.time() - t0:.0f} s)", flush=True)


def same_text(i: int, j: int) -> bool:
    L = a.dedup_len; ci, cj = chunks[i], chunks[j]; return ci[:L] in cj or cj[:L] in ci


def choose(query: bytes, fi: int, fb: bytes) -> list[int]:
    q = np.unique(span_hashes(query, a.span)); lo = np.searchsorted(SH, q, "left"); hi = np.searchsorted(SH, q, "right")
    counts = np.zeros(len(chunks), dtype=np.int32)
    for l_, h_ in zip(lo, hi):
        if h_ > l_: np.add.at(counts, SI[l_:h_], 1)
    counts[chunk_file == fi] = 0                                                 # never the piece's own file
    out = []; total = 0
    for i in np.argsort(-counts)[: a.top * 8]:
        if counts[i] < a.min_hits or len(out) >= a.top: break
        c = chunks[i]
        if c[:a.dedup_len] in fb or fb[:a.dedup_len] in c: continue             # a copy of the file
        if any(same_text(int(i), j) for j in out): continue
        out.append(int(i)); total += len(c)
        if total >= a.max_mem: break
    return out


# ---- the pieces, each behind its memory ------------------------------------------------------------------------------------------------------
os.makedirs(a.layout, exist_ok=True); os.chmod(a.layout, 0o700)
out = open(a.out, "wb"); os.chmod(a.out, 0o600); sep = a.sep.encode()
n_pieces = n_mem = 0; mem_bytes = 0; mem_chunks = 0; rag_bytes = 0; same_file_skipped = 0; docs = []
t1 = time.time()
for fi, (b, rel) in enumerate(zip(files, rels)):
    for k, s in enumerate(range(0, len(b), a.piece)):
        piece = b[s:s + a.piece]
        if len(piece) < a.min_piece and k > 0: continue
        ids_ = choose(piece[:a.query], fi, b); n_pieces += 1
        mem = b"".join(b"## " + paths[i].encode("utf-8", "replace") + b"\n" + chunks[i] + sep for i in ids_)
        if ids_: n_mem += 1; mem_bytes += len(mem); mem_chunks += len(ids_)
        doc = b"\n// ==== rag/" + rel.encode("utf-8", "replace") + f"#{k}".encode() + b" ====\n" + mem + piece
        docs.append((rag_bytes, len(doc), len(mem), len(ids_))); out.write(doc); rag_bytes += len(doc)
    if (fi + 1) % 500 == 0: print(f"  {fi + 1:,} files, {n_pieces:,} pieces ({time.time() - t1:.0f} s)", flush=True)
replay = {}
if a.replay:
    f, lo, hi = a.replay.split(":"); lo, hi = int(lo), int(hi)
    src = np.memmap(f, dtype=np.uint8, mode="r"); out.write(bytes(src[lo:hi])); replay = {"file": f, "slice": [lo, hi], "bytes": hi - lo}
out.close(); total = os.path.getsize(a.out)
lay = {"corpus": a.out, "built": time.strftime("%Y-%m-%d"), "world": a.world, "pool": {"chunk": a.chunk, "chunks": len(chunks), "digest": digest, "files": len(files)},
       "chooser": {"mode": "spans", "dedup": True, "top": a.top, "max_mem": a.max_mem, "min_hits": a.min_hits, "span": a.span, "dedup_len": a.dedup_len, "query": a.query,
                   "exclude": "the piece's own file and copies of it", "format": "'## <path>' + chunk + sep", "sep": a.sep},
       "piece": a.piece, "pieces": n_pieces, "pieces_with_memory": n_mem, "memory_bytes_mean": mem_bytes / max(1, n_mem), "chunks_per_memory": mem_chunks / max(1, n_mem),
       "rag_bytes": rag_bytes, "replay": replay, "corpus_bytes": total, "docs": "start, length, memory bytes, chunks per piece", "doc_table": docs}
json.dump(lay, open(os.path.join(a.layout, "layout.json"), "w"))
print(f"pieces: {n_pieces:,} of {a.piece} bytes from {len(files):,} files; with a memory {n_mem:,} ({100 * n_mem / max(1, n_pieces):.0f} %), {mem_bytes / max(1, n_mem):.0f} memory bytes and "
      f"{mem_chunks / max(1, n_mem):.2f} chunks on average; the rag part {rag_bytes:,} bytes, the replay {replay.get('bytes', 0):,}, the corpus {total:,} ({time.time() - t0:.0f} s)")
print("wrote", a.out, "and", os.path.join(a.layout, "layout.json"))
