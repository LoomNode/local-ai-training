"""The terminal built in, step 1 (Kyle, 2026-10-14: "training from scratch locally with the terminal built in. We should be able to test
it at our size"): a corpus pair from the package corpus, block-aligned so every training window is one block of PRE + CONTENT bytes:
  term:   [header line][a terminal preamble: `$ ls <dir>` of the file's directory, then `$ grep -n <name>` over the project's OTHER
           files for names the file used in its previous block, the hits pasted as path:line: text][1,280 bytes of the file]
  plain:  [header line][the file's own previous bytes, as many as the preamble had room for][the same 1,280 bytes]
The two streams have the same files, the same blocks in the same order, and the same content bytes at the same offsets; only the
preamble area differs, so an arm trained on each and scored on the content bytes (score_tables.py --score-from PRE) answers: is a
terminal's output worth more to the reader than the same room filled with the file's own earlier text? The header line opens every
block (the counters restart their depths at a header, in both streams alike). The grep's names come from the previous block only
(nothing from the block being predicted), ranked by rarity across the project's files; a name with no hit in another file is skipped.
Files shorter than one block and a file's last partial block are dropped (the same in both streams). The dev part is built from the
held-out packages' files the leak mask calls clean (results/tables/leak_windows2_2026-10-11.json: files with no code line found in
the training text), their greps over the other held-out files of their package; the dev windows are its first blocks.

usage: term_corpus.py --out DIR [--content-bytes 300000000] [--seed 0] [--pre 768] [--block 1280] [--max-package 50000000]
       writes DIR/term.bytes, DIR/plain.bytes, DIR/layout.json (train_end, dev_offset, dev_windows, pre, block), DIR/blocks_dev.json.
Then the feature pipeline on each (checks/term_tables.sh)."""
import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np

HEADER = b"\n// ==== "
HEADER_RE = re.compile(rb"\n// ==== ([^\n]*?) ====\n")
IDENT = re.compile(rb"[A-Za-z_][A-Za-z0-9_]{3,}")
PKGS = "/games/ailab/corpora/pkgs.bytes"
LAYOUT = "/games/ailab/corpora/pkgs/layout.json"
DEV = "/games/ailab/corpora/pkgs/dev.bytes"
MASK = "/games/ailab/lab/flybrain-lab/.worktrees/speed/results/tables/leak_windows2_2026-10-11.json"
COMMON_DF = 40          # a name in more files of a project than this is not worth a grep
MAX_HITS = 3            # lines pasted per grep
MAX_GREPS = 8           # greps per block, at most; the preamble stops when its room is full
SYMBOL = re.compile(rb"_|\d|[a-z][A-Z]")   # a name that looks like code (snake_case, digits, camelCase), not a word from a comment


def scan_files(mm, lo: int, hi: int) -> list[tuple[str, int, int]]:
    """(name, content start, content end) of every file whose header lies in [lo, hi): name = '<pkg>/<path>'."""
    out = []
    heads = [(m.start(), m.end(), m.group(1)) for m in HEADER_RE.finditer(mm, lo, hi)]
    for i, (s, e, name) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else hi
        out.append((name.decode("latin-1"), e, end))
    return out


class Project:
    """One package's files: the identifier index for the greps and the directory listings for ls."""

    def __init__(self, files: list[tuple[str, bytes]]):
        self.files = files                                  # (name, content)
        self.dirs = defaultdict(list)
        for name, _ in files:
            d, base = name.rsplit("/", 1) if "/" in name else ("", name)
            self.dirs[d].append(base)
        self.index = defaultdict(list)                      # ident -> [(name, lineno, line)]
        df = Counter()
        for name, content in files:
            seen = set()
            for ln, line in enumerate(content.split(b"\n"), 1):
                for m in IDENT.finditer(line):
                    w = m.group(0)
                    if w not in seen:
                        seen.add(w)
                    lst = self.index[w]
                    if len(lst) < 8:
                        lst.append((name, ln, line.strip()[:160]))
            df.update(seen)
        self.df = df

    def ls(self, name: str) -> bytes:
        d = name.rsplit("/", 1)[0] if "/" in name else ""
        return b"$ ls " + (d.encode() or b".") + b"\n" + b"  ".join(b.encode() for b in sorted(self.dirs[d])[:60]) + b"\n"

    def greps(self, name: str, prev: bytes, room: int) -> bytes:
        """`$ grep -n <name>` lines for names of the previous block, the most symbol-like and rarest first, until the room is full."""
        cands = {}
        for m in IDENT.finditer(prev):
            w = m.group(0)
            if 1 < self.df.get(w, 0) <= COMMON_DF and w not in cands:
                called = prev[m.end():m.end() + 1] == b"("
                cands[w] = (-(2 * bool(SYMBOL.search(w)) + called + (len(w) >= 8)), self.df[w], w)
        out = b""; n_greps = 0
        for w, _ in sorted(cands.items(), key=lambda kv: kv[1]):
            hits = [(n, ln, line) for n, ln, line in self.index.get(w, ()) if n != name][:MAX_HITS]
            if not hits:
                continue
            g = b"$ grep -n " + w + b"\n" + b"".join(n.encode() + b":" + str(ln).encode() + b": " + line + b"\n" for n, ln, line in hits)
            if len(out) + len(g) > room:
                continue
            out += g; n_greps += 1
            if n_greps >= MAX_GREPS:
                break
        return out


def blocks_of(name: str, content: bytes, proj: Project, pre: int, block: int):
    """Yields (term block, plain block, k) for the file: PRE + block bytes each."""
    hdr = HEADER + name.encode() + b" ====\n"
    room = pre - len(hdr)
    if room < 64:
        return
    n = len(content) // block
    for k in range(n):
        body = content[k * block:(k + 1) * block]
        prev = content[max(0, k * block - block):k * block]
        ls = proj.ls(name)
        if len(ls) > min(room, 240):                           # a long listing would crowd out the greps: cut it at a name
            ls = ls[:min(room, 240)].rsplit(b"  ", 1)[0] + b"  ...\n"
        text = ls + proj.greps(name, prev if k else name.encode(), room - len(ls))
        if len(text) < room:
            text = text + b"\n" * (room - len(text))
        own = content[max(0, k * block - room):k * block]
        own = b"\n" * (room - len(own)) + own
        yield hdr + text + body, hdr + own + body, k


def build(out: str, content_bytes: int, seed: int, pre: int, block: int, max_package: int) -> None:
    t0 = time.time()
    lay = json.load(open(LAYOUT))
    mm = np.memmap(PKGS, dtype=np.uint8, mode="r")
    train_end = int(lay["train_end"])
    files = scan_files(mm, 0, train_end)
    print(f"{len(files):,} training files scanned in {time.time() - t0:.0f} s", flush=True)
    by_pkg = defaultdict(list)
    for name, s, e in files:
        by_pkg[name.split("/", 1)[0]].append((name, s, e))
    pkgs = [p for p in by_pkg if sum(e - s for _, s, e in by_pkg[p]) <= max_package]
    random.Random(seed).shuffle(pkgs)
    chosen, total = [], 0
    for p in pkgs:
        if total >= content_bytes:
            break
        chosen.append(p); total += sum(e - s for _, s, e in by_pkg[p])
    print(f"{len(chosen)} packages, {total:,} bytes of files", flush=True)
    os.makedirs(out, exist_ok=True)
    ft, fp = open(f"{out}/term.bytes", "wb"), open(f"{out}/plain.bytes", "wb")
    nblk = 0; rng = random.Random(seed + 1)
    for pi, p in enumerate(chosen):
        proj = Project([(name, bytes(mm[s:e])) for name, s, e in by_pkg[p]])
        order = list(proj.files); rng.shuffle(order)
        for name, content in order:
            for tb, pb, _ in blocks_of(name, content, proj, pre, block):
                ft.write(tb); fp.write(pb); nblk += 1
        if pi % 50 == 0:
            print(f"  {pi} packages, {nblk:,} blocks, {time.time() - t0:.0f} s", flush=True)
    train_end_out = nblk * (pre + block)
    # the dev part: clean held-out files, greps over the other held-out files of their package
    mask = json.load(open(MASK)); clean = set(mask["dev_clean_windows"])
    dm = np.memmap(DEV, dtype=np.uint8, mode="r")
    dfiles = scan_files(dm, 0, len(dm))
    T = 2048
    clean_names = set()
    for name, s, e in dfiles:
        hs = s - 1
        for i in clean:
            a = i * T
            if hs <= a < e or (hs < a + T <= e):
                clean_names.add(name); break
    dby = defaultdict(list)
    for name, s, e in dfiles:
        dby[name.split("/", 1)[0]].append((name, s, e))
    dev_blocks = []
    for p in sorted({n.split("/", 1)[0] for n in clean_names}):
        proj = Project([(name, bytes(dm[s:e])) for name, s, e in dby[p]])
        for name, content in proj.files:
            if name in clean_names:
                for tb, pb, k in blocks_of(name, content, proj, pre, block):
                    dev_blocks.append((tb, pb, name, k))
    dev_blocks.sort(key=lambda b: (b[2], b[3]))
    for tb, pb, _, _ in dev_blocks:
        ft.write(tb); fp.write(pb)
    ft.write(b"\n"); fp.write(b"\n")                          # one byte past the last window's target
    ft.close(); fp.close()
    layout = {"built": time.strftime("%F"), "builder": "checks/term_corpus.py", "pre": pre, "block": block, "window": pre + block,
              "packages": len(chosen), "content_bytes": total, "train_blocks": nblk, "train_end": train_end_out, "dev_offset": train_end_out,
              "dev_blocks": len(dev_blocks), "dev_windows": len(dev_blocks), "dev_files": len(clean_names), "seed": seed,
              "note": "block-aligned windows of pre + block bytes: a header line, the preamble area (term: ls + greps; plain: the file's own "
                      "previous bytes), the content; score the content only (score_tables.py --score-from pre). Dev = clean held-out files."}
    json.dump(layout, open(f"{out}/layout.json", "w"), indent=1)
    json.dump([{"i": i, "file": n, "k": k} for i, (_, _, n, k) in enumerate(dev_blocks)], open(f"{out}/blocks_dev.json", "w"))
    print(f"done in {time.time() - t0:.0f} s: {nblk:,} train blocks ({train_end_out:,} bytes), {len(dev_blocks):,} dev blocks from "
          f"{len(clean_names)} clean files; {out}/term.bytes and plain.bytes", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True); ap.add_argument("--content-bytes", type=int, default=300_000_000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--pre", type=int, default=768); ap.add_argument("--block", type=int, default=1280)
    ap.add_argument("--max-package", type=int, default=50_000_000)
    a = ap.parse_args()
    build(a.out, a.content_bytes, a.seed, a.pre, a.block, a.max_package)


if __name__ == "__main__":
    main()
