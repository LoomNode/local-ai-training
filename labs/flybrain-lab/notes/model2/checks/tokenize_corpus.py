#!/usr/bin/env python
"""Model #2's token stream of a byte corpus (tok_common.py has the rules): tokens.u16, tok_len.u8, tok_off.u64 and layout.json in --out.
Files are encoded in chunks of whole file units (--chunk bytes), each chunk through one encode_batch (the Rust thread pool; set
RAYON_NUM_THREADS), the ids appended to tokens.u16 as they come; then one pass for the lengths and offsets. The layout adds n_tokens,
train_end_tok (the first token at or after train_end; it must start a file unit), dev_tok and test_tok (token spans of the byte
regions), vocab and the tokenizer's path and sha256. About 20 minutes for 5.5 GB on 12 cores.
usage: tokenize_corpus.py --corpus C --layout L --tokenizer T --out DIR [--chunk 67108864]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tok_common import byte_token_ids, encode_files, file_units, id_byte_len, sha256_file  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True); ap.add_argument("--layout", required=True); ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--chunk", type=int, default=64 << 20)
a = ap.parse_args()
t0 = time.time()
tok = Tokenizer.from_file(a.tokenizer)
byte_ids, lens = byte_token_ids(tok), id_byte_len(tok)
layout = json.load(open(a.layout))
N = os.path.getsize(a.corpus)
if int(layout.get("corpus_bytes", N)) != N and "corpus_bytes" in layout:
    sys.exit(f"layout says {layout['corpus_bytes']} bytes, the corpus has {N}")
units = file_units(a.corpus)
os.makedirs(a.out, exist_ok=True)
m = np.memmap(a.corpus, dtype=np.uint8, mode="r")
starts = []                         # the first token of every unit
n_tok = 0
with open(os.path.join(a.out, "tokens.u16"), "wb") as f:
    i = 0
    while i < len(units):
        j, size = i, 0
        while j < len(units) and (size < a.chunk or j == i):
            size += units[j][1] - units[j][0]; j += 1
        files = [m[lo:hi].tobytes() for lo, hi in units[i:j]]
        for ids in encode_files(tok, files, byte_ids):
            starts.append(n_tok); n_tok += len(ids); f.write(ids.astype("<u2").tobytes())
        i = j; chunks = locals().get("chunks", 0) + 1
        done = units[j - 1][1]
        if chunks % 20 == 0 or j == len(units):
            print(f"{done:,} / {N:,} bytes, {n_tok:,} tokens ({time.time() - t0:.0f} s)", flush=True)
tokens = np.memmap(os.path.join(a.out, "tokens.u16"), dtype="<u2", mode="r", shape=(n_tok,))
tok_len = np.memmap(os.path.join(a.out, "tok_len.u8"), dtype=np.uint8, mode="w+", shape=(n_tok,))
tok_off = np.memmap(os.path.join(a.out, "tok_off.u64"), dtype=np.uint64, mode="w+", shape=(n_tok + 1,))
off, B = 0, 1 << 24
for lo in range(0, n_tok, B):
    hi = min(lo + B, n_tok)
    ln = lens[np.asarray(tokens[lo:hi], dtype=np.int64)]
    tok_len[lo:hi] = ln
    c = np.cumsum(ln, dtype=np.int64)
    tok_off[lo:hi] = off + c - ln
    off += int(c[-1])
tok_off[n_tok] = off
tok_len.flush(); tok_off.flush()
if off != N:
    sys.exit(f"the tokens cover {off} bytes, the corpus has {N}: the stream does not round-trip")
bad = [k for k, s in enumerate(starts) if int(tok_off[s]) != units[k][0]]   # every unit's first token starts at the unit's first byte
if bad:
    sys.exit(f"{len(bad)} file units do not start on a token boundary (first: unit {bad[0]})")
train_end = int(layout["train_end"])
train_end_tok = int(np.searchsorted(tok_off, np.uint64(train_end), side="left"))
if int(tok_off[train_end_tok]) != train_end:
    sys.exit(f"train_end {train_end} is inside a token (token {train_end_tok} starts at {int(tok_off[train_end_tok])}): it must start a file unit")
def span(x, y):
    return [int(np.searchsorted(tok_off, np.uint64(x), side="left")), int(np.searchsorted(tok_off, np.uint64(y), side="left"))]
out = dict(layout)
out.update(corpus=a.corpus, corpus_bytes=N, n_tokens=n_tok, train_end_tok=train_end_tok, vocab=tok.get_vocab_size(),
           tokenizer=a.tokenizer, tokenizer_sha256=sha256_file(a.tokenizer), units=len(units), unit_starts_file="unit_starts.u64",
           dev_tok=span(int(layout["dev_offset"]), int(layout["dev_offset"]) + int(layout["dev_bytes"])) if "dev_offset" in layout else None,
           test_tok=span(int(layout["test_offset"]), int(layout["test_offset"]) + int(layout["test_bytes"])) if "test_offset" in layout else None,
           built=time.strftime("%Y-%m-%d %H:%M"), rule="tok_common.py: per file unit, UTF-8 runs through the BPE, invalid bytes as single-byte tokens")
np.asarray(starts, dtype=np.uint64).tofile(os.path.join(a.out, "unit_starts.u64"))
json.dump(out, open(os.path.join(a.out, "layout.json"), "w"), indent=1)
print(f"done: {n_tok:,} tokens over {N:,} bytes ({N / n_tok:.3f} bytes/token), train_end_tok {train_end_tok:,}, {len(units):,} units, {time.time() - t0:.0f} s", flush=True)
