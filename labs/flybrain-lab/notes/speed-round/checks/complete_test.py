"""The completion test of model #1's viability (Kyle, 2026-10-13; item 2: "complete a line usefully, judged on real spots"). Spots are
line starts in a project's held-out files (index_project.py's heldout.bytes: files the model and the index never saw), each with its
file read from its header up to the spot; the model writes the rest of the line greedily (temperature 0, no guard) through the
streaming reader, and the written line is compared with the real one: exact (after trailing whitespace), the shared prefix as a share
of the real line, and whether the first token (the first run of identifier bytes) is right. The same spots, seed and files for every
model and table set, so two runs compare (the corpus tables against the project's index, the snapshots against each other).

usage: complete_test.py --ckpt CKPT --tables DIR --stream heldout.bytes [--spots 40] [--seed 0] [--min-prefix 256] [--max-line 120]
                        [--device cpu] [--out results.json] [--show 8]
Spots: lines of at least 12 non-blank bytes that are not comment-only and have at least --min-prefix bytes of their file before them,
sampled without replacement across files with the seed. The file prefix is fed whole (the table inputs need the header and the scanners
need the file; the reader keeps its last 2,048 bytes of keys)."""
import argparse
import json
import os
import random
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_tables import load  # noqa: E402
from stream_reader import Reader  # noqa: E402
from use_features import HEADER, Tables  # noqa: E402

IDENT = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")


def files_of(stream: bytes) -> list[tuple[bytes, int]]:
    """(file bytes including its header, offset in the stream) per file."""
    out = []
    starts = [m.start() for m in re.finditer(re.escape(HEADER), stream)]
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(stream)
        out.append((stream[s:e], s))
    return out


def file_name(f: bytes) -> str:
    return f[len(HEADER):f.find(b" ====\n", len(HEADER))].decode("latin-1")


def spots_of(stream: bytes, n: int, seed: int, min_prefix: int, exts=()) -> list[dict]:
    """n (file index, line start, line end) spots: real lines of code with enough file before them, in files of the extensions given
    (all files when none)."""
    cands = []
    for fi, (f, off) in enumerate(files_of(stream)):
        if exts and file_name(f).rsplit(".", 1)[-1].lower() not in exts:
            continue
        hdr_end = f.find(b"\n", len(HEADER)) + 1
        p = hdr_end
        while p < len(f):
            q = f.find(b"\n", p)
            if q < 0:
                q = len(f)
            line = f[p:q]
            body = line.strip()
            if p - hdr_end >= min_prefix and len(body) >= 12 and not body.startswith((b"#", b"//", b"/*", b"*")):
                cands.append({"file": fi, "start": p, "end": q})
            p = q + 1
    rng = random.Random(seed)
    rng.shuffle(cands)
    chosen = []
    per_file = {}
    for c in cands:                                        # spread over files: at most three spots per file
        if per_file.get(c["file"], 0) < 3:
            chosen.append(c); per_file[c["file"]] = per_file.get(c["file"], 0) + 1
        if len(chosen) >= n:
            break
    return chosen


def first_ident(b: bytes) -> bytes:
    m = IDENT.search(b)
    return m.group(0) if m else b""


@torch.no_grad()
def complete(reader: Reader, prefix: bytes, max_line: int) -> bytes:
    reader.feed(prefix)
    out = bytearray()
    while len(out) < max_line:
        b = int(reader.logits.argmax())
        if b == 10:
            break
        out.append(b)
        reader.feed(bytes([b]))
    return bytes(out)


def run(model, orders, tables, stream: bytes, spots: list[dict], max_line: int, device, show: int = 0) -> dict:
    files = files_of(stream)
    rows = []
    t0 = time.time()
    for k, sp in enumerate(spots):
        f, _ = files[sp["file"]]
        prefix, real = f[:sp["start"]], f[sp["start"]:sp["end"]]
        reader = Reader(model, orders, tables, device)
        got = complete(reader, prefix, max_line)
        real_s, got_s = real.rstrip(), got.rstrip()
        common = 0
        for x, y in zip(real_s, got_s):
            if x != y:
                break
            common += 1
        row = {"file": file_name(f), "line_at": sp["start"], "real": real.decode("latin-1"),
               "got": got.decode("latin-1"), "exact": real_s == got_s, "prefix_share": common / max(len(real_s), 1),
               "first_token": first_ident(real_s.lstrip()) == first_ident(got_s.lstrip()) and bool(first_ident(real_s.lstrip()))}
        rows.append(row)
        if show and k < show:
            verdict = "exact" if row["exact"] else f"{row['prefix_share']:.0%} prefix"
            print(f"  [{row['file']}] real: {row['real'].strip()[:70]!r}\n      got: {row['got'].strip()[:70]!r}  ({verdict})", flush=True)
    n = len(rows)
    return {"n": n, "exact": sum(r["exact"] for r in rows) / n, "prefix_share": sum(r["prefix_share"] for r in rows) / n,
            "half_or_more": sum(r["prefix_share"] >= 0.5 for r in rows) / n, "first_token": sum(r["first_token"] for r in rows) / n,
            "seconds": round(time.time() - t0, 1), "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tables", required=True); ap.add_argument("--stream", required=True)
    ap.add_argument("--spots", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-prefix", type=int, default=256); ap.add_argument("--max-line", type=int, default=120)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--out", default=None); ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--ext", default="", help="spots only in files with these extensions, comma-separated (default: every file)")
    a = ap.parse_args()
    device = torch.device(a.device)
    tables = Tables(a.tables)
    model, orders, _ = load(a.ckpt, device)
    stream = open(a.stream, "rb").read()
    spots = spots_of(stream, a.spots, a.seed, a.min_prefix, set(e for e in a.ext.lower().split(",") if e))
    print(f"{len(spots)} spots in {len({s['file'] for s in spots})} files of {os.path.basename(a.stream)}; {os.path.basename(a.ckpt)}, tables {a.tables}", flush=True)
    res = run(model, orders, tables, stream, spots, a.max_line, device, a.show)
    print(f"exact {res['exact']:.0%}, first token right {res['first_token']:.0%}, half the line or more {res['half_or_more']:.0%}, "
          f"mean shared prefix {res['prefix_share']:.0%} ({res['n']} spots, {res['seconds']} s)")
    if a.out:
        json.dump({"ckpt": a.ckpt, "tables": a.tables, "stream": a.stream, "spots": a.spots, "seed": a.seed, "ext": a.ext, **res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
