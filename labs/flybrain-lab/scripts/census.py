#!/usr/bin/env python3
"""Exact-match census of the held-out targets. CPU only.

Per target (a held-out file with >= 4 same-directory training files), n and P: how many of its first scored body bytes are
A  (continue an n-gram seen earlier in the target itself),
B  (not A; the n-gram and its continuation occur in the related prefix: its same-directory training files, path order, cut at P bytes),
B' (not A; the same test against an equal-size random prefix from other directories),
C  (neither). --never-trained adds a second pass whose related prefix comes from same-directory files of the target's own split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.census import A, B, C, bins
from flybrain.data.files import DEV_FILES, MIN_NEVER, TAIL_START, TRAIN_END, file_table, hash_table, read_stream, related, split_range, splits, targets
from flybrain.data.prefixes import prefix_pairs

RESULTS = ROOT / "results"
NS = (8, 12, 16)
PS = (8192, 16384, 32768)
KEYS = ("A", "B", "Bp", "C", "Cp")


def count_cell(stream: bytes, body_start: int, n_scored: int, rel: bytes, rnd: bytes, n: int) -> dict[str, int]:
    """Counts of scored bytes in A, B (related prefix), Bp (random prefix), C = not A or B, Cp = not A or Bp."""
    r, q = bins(stream, body_start, n_scored, rel, n), bins(stream, body_start, n_scored, rnd, n)
    return {"A": r.count(A), "B": r.count(B), "Bp": q.count(B), "C": r.count(C), "Cp": q.count(C), "total": len(r)}


def count_target(stream: bytes, body_start: int, n_scored: int, pairs: dict, ns) -> dict[str, dict[str, dict[str, int]]]:
    return {str(n): {str(p): count_cell(stream, body_start, n_scored, rel, rnd, n) for p, (rel, rnd) in pairs.items()} for n in ns}


def pool(recs: list[dict], key: str, ns, ps) -> dict:
    """Sum the counts of every record whose `key` is not None; shares are byte-weighted (counts / total)."""
    used = [r for r in recs if r[key] is not None]
    cells = {str(n): {str(p): dict.fromkeys((*KEYS, "total"), 0) for p in ps} for n in ns}
    for r in used:
        for n, row in cells.items():
            for p, c in row.items():
                for k in c:
                    c[k] += r[key][n][p][k]
    for row in cells.values():
        for c in row.values():
            c["share"] = {k: (c[k] / c["total"] if c["total"] else None) for k in KEYS}
    return {"n_targets": len(used), "n_scored_bytes": sum(r["n_scored"] for r in used), "cells": cells}


def run_census(path, split: str, ns=NS, ps=PS, seed: int = 0, never_trained: bool = False, n_scored: int = 4096,
               train_end: int = TRAIN_END, tail_start: int = TAIL_START, dev_files: int = DEV_FILES, limit: int = 0) -> dict:
    table = file_table(path)
    hashes = hash_table(path, table)
    idxs = splits(table, tail_start, dev_files)[split]
    if not idxs:
        raise ValueError(f"split {split!r} holds no files")
    lo, hi = split_range(table, idxs)  # the split's own byte range, for the never-trained pass
    recs = []
    for i in targets(table, idxs, hashes, 4, train_end)[: limit or None]:
        stream, body_start, scored = read_stream(path, table[i], n_scored)
        if scored == 0:
            continue
        pairs = prefix_pairs(path, table, i, hashes, ps, seed, 0, train_end, train_end)
        rec = {"index": i, "path": table[i].path, "n_scored": scored, "related_files": len(related(i, table, 0, train_end, hashes)),
               "cells": count_target(stream, body_start, scored, pairs, ns), "never_files": None, "never_cells": None}
        k = len(related(i, table, lo, hi, hashes)) if never_trained else 0
        if k >= MIN_NEVER:
            npairs = prefix_pairs(path, table, i, hashes, ps, seed, lo, hi, train_end)
            rec["never_files"], rec["never_cells"] = k, count_target(stream, body_start, scored, npairs, ns)
        recs.append(rec)
    trained = pool(recs, "cells", ns, ps)
    never = None
    if never_trained:
        sub = [r for r in recs if r["never_cells"] is not None]
        never = {**pool(recs, "never_cells", ns, ps), "trained_cells": pool(sub, "cells", ns, ps)["cells"]}
    return {"split": split, "ns": list(ns), "ps": list(ps), "n_targets": trained["n_targets"], "n_scored_bytes": trained["n_scored_bytes"],
            "trained": trained, "never_trained": never, "targets": recs}


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render(res: dict) -> str:
    """Markdown tables of the pooled shares: one table per n, one row per P."""
    lines = [f"{res.get('run_name', '')} {res['split']}: {res['n_targets']} targets, {res['n_scored_bytes']} scored bytes"]
    blocks = [("related prefix = same-directory training files", res["trained"])]
    nt = res["never_trained"]
    if nt:
        blocks += [("same targets, trained related files", {**nt, "cells": nt["trained_cells"]}), ("never trained: related files from the target's own split", nt)]
    for title, blk in blocks:
        if not blk["n_targets"]:
            continue
        lines.append(f"\n{title} ({blk['n_targets']} targets, {blk['n_scored_bytes']} bytes)")
        for n, row in blk["cells"].items():
            lines += [f"\nn = {n}", "| P | A | B | B' | C | C' |", "|---:|---:|---:|---:|---:|---:|"]
            lines += [f"| {p} | " + " | ".join(_f(c["share"][k]) for k in KEYS) + " |" for p, c in row.items()]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--corpus", default=str(ROOT / "corpora" / "linux_full.bytes"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--ps", default=",".join(map(str, PS)))
    ap.add_argument("--n-scored", type=int, default=4096)
    ap.add_argument("--never-trained", action="store_true", help="second pass: related prefix from the target's own split")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N targets only")
    ap.add_argument("--train-end", type=int, default=TRAIN_END, help="tiny-corpus tests only")
    ap.add_argument("--tail-start", type=int, default=TAIL_START, help="tiny-corpus tests only")
    ap.add_argument("--dev-files", type=int, default=DEV_FILES, help="tiny-corpus tests only")
    a = ap.parse_args(argv)
    out = RESULTS / f"{a.run_name}.json"
    if out.exists():
        sys.exit(f"{out} exists: new run names only")
    res = run_census(a.corpus, a.split, [int(x) for x in a.ns.split(",")], [int(x) for x in a.ps.split(",")], a.seed, a.never_trained,
                     a.n_scored, a.train_end, a.tail_start, a.dev_files, a.limit)
    res = {"run_name": a.run_name, "args": vars(a), **res}
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(render(res))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
