"""The parser test's leak check: parser_features.py parses every file ONCE, whole, so the node kind and depth it gives byte p-1
(orders 2001, 2002) come from a tree that has seen the bytes after p (whether `foo` is a call or a declaration is decided by what
follows it). A reader at use time only has the prefix. This recomputes 2001 and 2002 for the scored windows from the prefix alone:
for every position p of a window, the file's bytes [file start, p) are parsed (tree-sitter's incremental parse, one byte appended at a
time; the same tree a fresh parse gives), and the feature is the smallest named node holding the prefix's last byte, as the build
defines it. 2003 and 2004 (name completions) already use only earlier bytes and are kept. Writes a feature directory for scoring:
the base orders and 2003/2004 linked to the full-parse directory, 2001/2002 copied with the window positions overwritten, plus the
share of positions where the honest feature differs from the full-parse one (the size of the leak in input terms).
usage: parser_prefix.py --feat results/mem/feat_parser --out results/mem/feat_parser_prefix --corpus C --offset O --T 2048 --n 1024
                        --windows results/tables/big/clean_kernel.txt [--jobs 4] [--check 64] [--limit K]
Then: score_tables.py ... k_pparse_prefix=results/ckpt/pparse_r_transformer.pt@<out>   (checks/parser_prefix.sh)"""
import argparse
import json
import os
import shutil
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parser_features as pf  # noqa: E402

t0 = time.time()
CORPUS = None
FILES = None      # (name, start, end) per file, sorted by start
KINDS = None


def file_of(pos: int):
    """The file (name, start, end) holding pos, or None (a header byte)."""
    starts = FILES_START
    i = int(np.searchsorted(starts, pos, side="right")) - 1
    if i < 0:
        return None
    name, s, e = FILES[i]
    return (name, s, e) if pos < e else None


def kind_depth(root, q: int, kmap: dict) -> tuple[int, int]:
    """The build's 2001/2002 for the byte q of a tree: the smallest named node holding it (pre-order last = innermost), its depth."""
    node = root.named_descendant_for_byte_range(q, q + 1)
    if node is None:
        return 0, 0
    d = 0
    n = node
    while n.parent is not None:
        n = n.parent; d += 1
    return kmap.get(node.type, 127), min(d, 127)


def point_of(buf: bytes, rel: int) -> tuple[int, int]:
    row = buf.count(b"\n", 0, rel)
    nl = buf.rfind(b"\n", 0, rel)
    return row, rel - (nl + 1)


def window_features(mm, w0: int, w1: int, kinds: dict):
    """pred/conf for 2001 and 2002 at corpus positions [w0, w1), from prefix-only parses. Returns (pred[2, T], conf[2, T])."""
    T = w1 - w0
    pred = np.zeros((2, T), np.uint8); conf = np.zeros((2, T), np.uint8)
    p = w0
    while p < w1:
        f = file_of(p)
        if f is None:
            p += 1; continue
        name, fs, fe = f
        b = min(fe, w1)
        lang = pf.lang_of(name)
        if lang is None:
            p = b; continue
        parser = pf.parser_for(lang)
        kmap = kinds.get(lang, {})
        data = bytes(mm[fs:b])                   # the file's bytes up to the segment end; prefixes are slices of it
        rel = p - fs
        tree = parser.parse(data[:rel]) if rel > 0 else None
        row, col = point_of(data, rel)
        while p < b:
            if rel > 0:
                k, d = kind_depth(tree.root_node, rel - 1, kmap)
                pred[0, p - w0] = k; conf[0, p - w0] = 1
                pred[1, p - w0] = 128 + d; conf[1, p - w0] = 1
            byte = data[rel:rel + 1]
            new_point = (row + 1, 0) if byte == b"\n" else (row, col + 1)
            if tree is None:
                tree = parser.parse(data[:rel + 1])
            else:
                tree.edit(start_byte=rel, old_end_byte=rel, new_end_byte=rel + 1, start_point=(row, col), old_end_point=(row, col),
                          new_end_point=new_point)
                tree = parser.parse(data[:rel + 1], tree)
            row, col = new_point
            rel += 1; p += 1
    return pred, conf


def check_window(mm, w0: int, w1: int, kinds: dict, pred, conf, n: int, seed: int = 0) -> int:
    """Fresh prefix parses at n sampled positions, features by the build's own walk: must equal the incremental ones."""
    rng = np.random.default_rng(seed)
    bad = 0
    for p in sorted(rng.choice(np.arange(w0, w1), size=min(n, w1 - w0), replace=False).tolist()):
        f = file_of(p)
        exp = (0, 0, 0, 0)
        if f is not None and pf.lang_of(f[0]) is not None and p > f[1]:
            lang = pf.lang_of(f[0]); kmap = kinds.get(lang, {})
            prefix = bytes(mm[f[1]:p])
            tree = pf.parser_for(lang).parse(prefix)
            N = len(prefix)
            kind_at = np.zeros(N + 1, np.uint8); depth_at = np.zeros(N + 1, np.uint8)
            for node, depth in pf.walk(tree.root_node):
                s, e = node.start_byte, min(node.end_byte, N)
                if e <= s:
                    continue
                kind_at[s:e] = kmap.get(node.type, 127); depth_at[s:e] = min(depth, 127)
            exp = (int(kind_at[N - 1]), 1, 128 + int(depth_at[N - 1]), 1)
        got = (int(pred[0, p - w0]), int(conf[0, p - w0]), int(pred[1, p - w0]), int(conf[1, p - w0]))
        if got != exp:
            bad += 1
            if bad <= 5:
                print(f"  mismatch at {p}: incremental {got} vs fresh {exp}", flush=True)
    return bad


def job(args):
    i, w, w0, w1, check = args
    mm = np.memmap(CORPUS, dtype=np.uint8, mode="r")
    pred, conf = window_features(mm, w0, w1, KINDS)
    bad = check_window(mm, w0, w1, KINDS, pred, conf, check) if check else -1
    return i, w, pred, conf, bad


def main():
    global CORPUS, FILES, FILES_START, KINDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True); ap.add_argument("--out", required=True); ap.add_argument("--corpus", required=True)
    ap.add_argument("--offset", type=int, required=True); ap.add_argument("--T", type=int, default=2048); ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--windows", required=True); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--check", type=int, default=64, help="fresh-parse positions checked per window for the first --check-windows windows")
    ap.add_argument("--check-windows", type=int, default=4); ap.add_argument("--limit", type=int, default=0, help="only the first K windows (a timing run)")
    a = ap.parse_args()
    CORPUS = a.corpus
    mm = np.memmap(CORPUS, dtype=np.uint8, mode="r")
    heads = [(m.start(), m.end(), m.group(1).decode("latin-1")) for m in pf.HEADER_RE.finditer(mm)]
    N = len(mm)
    FILES = [(name, e, heads[i + 1][0] if i + 1 < len(heads) else N) for i, (s, e, name) in enumerate(heads)]
    FILES_START = np.array([f[1] for f in FILES])
    print(f"{len(FILES):,} files ({time.time() - t0:.0f} s)", flush=True)
    KINDS = pf.kind_table(FILES, mm)
    print(f"kind tables for {sorted(KINDS)} ({time.time() - t0:.0f} s)", flush=True)
    sel = [int(l) for l in open(a.windows).read().split()]
    if a.limit:
        sel = sel[:a.limit]
    tasks = [(i, w, a.offset + w * a.T, a.offset + (w + 1) * a.T, a.check if i < a.check_windows else 0) for i, w in enumerate(sel)]
    os.makedirs(a.out, exist_ok=True)
    full = {kid: np.load(f"{a.feat}/pred_n{kid}.npy", mmap_mode="r") for kid in (2001, 2002)}
    fullc = {kid: np.load(f"{a.feat}/conf_n{kid}.npy", mmap_mode="r") for kid in (2001, 2002)}
    outp, outc = {}, {}
    for j, kid in enumerate((2001, 2002)):
        for kind, src, dst in (("pred", full, outp), ("conf", fullc, outc)):
            path = f"{a.out}/{kind}_n{kid}.npy"
            if not os.path.exists(path):
                shutil.copyfile(f"{a.feat}/{kind}_n{kid}.npy", path)
            dst[kid] = np.load(path, mmap_mode="r+")
    for kid in (4, 8, 12, 16, 32, 64, 116, 132, 202, 203, 301, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 2003, 2004):
        for kind in ("pred", "conf"):
            d = f"{a.out}/{kind}_n{kid}.npy"
            if not os.path.lexists(d):
                os.symlink(os.path.realpath(f"{a.feat}/{kind}_n{kid}.npy"), d)
    print(f"{len(tasks)} windows, {a.jobs} jobs ({time.time() - t0:.0f} s)", flush=True)
    differ = np.zeros(2, np.int64); covered = np.zeros(2, np.int64); total = 0; bad = 0; done = 0
    with get_context("fork").Pool(a.jobs) as pool:
        for i, w, pred, conf, nb in pool.imap_unordered(job, tasks):
            w0, w1 = a.offset + w * a.T, a.offset + (w + 1) * a.T
            for j, kid in enumerate((2001, 2002)):
                differ[j] += int((pred[j] != full[kid][w0:w1]).sum()); covered[j] += int((conf[j] > 0).sum())
                outp[kid][w0:w1] = pred[j]; outc[kid][w0:w1] = conf[j]
            total += w1 - w0; done += 1
            if nb > 0:
                bad += nb; print(f"  window {w}: {nb} of the checked positions differ from a fresh parse", flush=True)
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} windows: honest kind differs from the full parse at {differ[0] / total:.3f} of positions, "
                      f"depth at {differ[1] / total:.3f} ({time.time() - t0:.0f} s)", flush=True)
    for arr in (*outp.values(), *outc.values()):
        arr.flush()
    info = json.load(open(f"{a.feat}/features.json"))
    info.update(prefix_check={"windows": len(tasks), "positions": total, "kind_differs": float(differ[0] / total), "depth_differs": float(differ[1] / total),
                              "with_feature": [float(c / total) for c in covered], "fresh_parse_mismatches": bad, "windows_file": a.windows},
                note=info.get("note", "") + " | parser_prefix.py: 2001/2002 recomputed from prefix-only parses on the scored windows")
    json.dump(info, open(f"{a.out}/features.json", "w"), indent=2)
    print(f"done in {time.time() - t0:.0f} s: kind differs at {differ[0] / total:.3f}, depth at {differ[1] / total:.3f} of {total:,} positions; "
          f"{bad} fresh-parse mismatches", flush=True)


if __name__ == "__main__":
    main()
