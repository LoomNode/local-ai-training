"""Parser features: what a real parser knows at every position, served through the table channel (Kyle, 2026-10-14: "test a real
parser too"; the fly's lesson: learning is a thin layer on innate machinery, so wire in what the tools already compute). tree-sitter
(the language pack, chosen by the file's extension from the corpus headers) parses every file once; four pseudo-orders, each a function
of the bytes before the position (nothing from the byte itself), in the counters' format (pred_n<id>.npy, conf_n<id>.npy, uint8 per
corpus position; features.json; the base dir's orders linked in, so one --mem-feat directory feeds all):
  2001  node kind: the smallest named syntax node holding the previous byte, as a class (per language, the 126 commonest kinds by
        count in the corpus's first files; others 127; no parser for the language: conf 0)
  2002  node depth: that node's depth in the tree, clipped to 127 (pred 128 + depth)
  2003  defined-name completion: the identifier being typed (the identifier bytes just before the position) completed by the most
        recently DEFINED name of the file so far that starts with it and is longer (function, class, type, parameter and variable names:
        nodes with a `name` field, and C-style declarators): pred = that name's next byte, conf = how many bytes are typed (1..7, +1)
  2004  used-name completion: the same over every identifier that has appeared earlier in the file
The scope tracker in its simplest form: 2003 says "a name you declared starts like this; its next byte is X". The match lookup copies
earlier bytes by 8/16/32-byte contexts; 2003 and 2004 fire from the first byte of a name and only on names.
usage: CORPUS=... TRAIN_END=... BASE=<feat dir to link> OUT=<out dir> [JOBS=4] parser_features.py
Hour-arm test: the 4-layer reader on the kernel corpus with --mem-feat OUT --mem-orders $ALL18,2001,2002,2003,2004 against the same
arm on the base features (checks/parser_gpu0.sh)."""
import json
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_common import base_info, env_int, env_str, file_cuts, link_base  # noqa: E402

CORPUS = env_str("CORPUS", "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes")
TRAIN_END = env_int("TRAIN_END", 200_000_000)
BASE = env_str("BASE", "/games/ailab/lab/flybrain-lab/.worktrees/speed/results/mem/feat_all28")
OUT = env_str("OUT", "/games/ailab/lab/flybrain-lab/.worktrees/speed/results/mem/feat_parser")
JOBS = env_int("JOBS", 4)
IDS = (2001, 2002, 2003, 2004)
BASE_ORDERS = (4, 8, 12, 16, 32, 64, 116, 132, 202, 203, 301, 1001, 1002, 1003, 1004, 1005, 1006, 1007)
HEADER_RE = re.compile(rb"\n// ==== ([^\n]*?) ====\n")
IDENT = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")
LANG = {".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".py": "python", ".js": "javascript",
        ".mjs": "javascript", ".ts": "typescript", ".go": "go", ".rs": "rust", ".java": "java", ".rb": "ruby", ".lua": "lua", ".sh": "bash",
        ".bash": "bash", ".pl": "perl", ".pm": "perl", ".gd": "gdscript", ".mk": "make", ".cmake": "cmake", ".toml": "toml", ".S": "c", ".s": "c"}
NAMES = {"Makefile": "make", "Kbuild": "make", "CMakeLists.txt": "cmake"}
DECL_KINDS = ("init_declarator", "parameter_declaration", "function_declarator", "pointer_declarator", "array_declarator", "declaration",
              "field_declaration", "type_definition")
_parsers = {}
_kinds = {}
t0 = time.time()


def lang_of(path: str) -> str | None:
    base = path.rsplit("/", 1)[-1]
    if base in NAMES:
        return NAMES[base]
    ext = "." + base.rsplit(".", 1)[-1] if "." in base else ""
    return LANG.get(ext)


def parser_for(lang: str):
    if lang not in _parsers:
        from tree_sitter_language_pack import get_parser
        _parsers[lang] = get_parser(lang)
    return _parsers[lang]


def walk(root):
    """Pre-order over named nodes: (node, depth)."""
    cur = root.walk()
    depth = 0
    while True:
        if cur.node.is_named:
            yield cur.node, depth
        if cur.goto_first_child():
            depth += 1; continue
        while not cur.goto_next_sibling():
            if not cur.goto_parent():
                return
            depth -= 1


def def_name(node):
    """The identifier a definition-like node names, or None."""
    nm = node.child_by_field_name("name")
    if nm is not None and "identifier" in nm.type:
        return nm
    if node.type in DECL_KINDS:
        d = node.child_by_field_name("declarator")
        while d is not None and "identifier" not in d.type:
            nxt = d.child_by_field_name("declarator")
            if nxt is None:
                nxt = next((c for c in d.children if c.is_named), None)
            d = nxt
        if d is not None and "identifier" in d.type:
            return d
    return None


def file_features(data: bytes, lang: str | None, kinds: dict) -> tuple[np.ndarray, np.ndarray]:
    """pred, conf: uint8 [len(data), 4] for the file's bytes (position p's feature from bytes before p)."""
    N = len(data)
    pred = np.zeros((N, 4), np.uint8); conf = np.zeros_like(pred)
    runs = [(m.start(), m.end()) for m in IDENT.finditer(data)]
    defs = []                                                      # (start, name bytes), in order of start
    if lang is not None:
        tree = parser_for(lang).parse(data)
        kind_at = np.zeros(N + 1, np.uint8); depth_at = np.zeros(N + 1, np.uint8)
        kmap = kinds.get(lang, {})
        for node, depth in walk(tree.root_node):
            s, e = node.start_byte, min(node.end_byte, N)
            if e <= s:
                continue
            kind_at[s:e] = kmap.get(node.type, 127); depth_at[s:e] = min(depth, 127)
            nm = def_name(node)
            if nm is not None:
                defs.append((nm.start_byte, data[nm.start_byte:nm.end_byte]))
        pred[1:, 0] = kind_at[:N - 1]; conf[1:, 0] = 1                  # the node holding the previous byte
        pred[1:, 1] = 128 + depth_at[:N - 1]; conf[1:, 1] = 1
        defs.sort()
    # completion by defined names (2003) and by used names (2004): the partial identifier [s, p) completed by the latest earlier name
    by_prefix_def, by_prefix_use = {}, {}
    di = 0
    for s, e in runs:
        while di < len(defs) and defs[di][0] < s:
            name = defs[di][1]
            for k in range(1, len(name)):
                by_prefix_def[name[:k]] = name
            di += 1
        run = data[s:e]
        for k in range(1, e - s):
            p = s + k; part = run[:k]
            name = by_prefix_def.get(part)
            if name is not None and len(name) > k:
                pred[p, 2] = name[k]; conf[p, 2] = min(k, 7) + 1
            name = by_prefix_use.get(part)
            if name is not None and len(name) > k:
                pred[p, 3] = name[k]; conf[p, 3] = min(k, 7) + 1
        for k in range(1, len(run)):
            by_prefix_use[run[:k]] = run
    return pred, conf


def kind_table(files, mm, per_lang_files: int = 60) -> dict:
    """The 126 commonest node kinds per language, from the first files of each."""
    counts, seen = {}, Counter()
    for path, s, e in files:
        lang = lang_of(path)
        if lang is None or seen[lang] >= per_lang_files:
            continue
        seen[lang] += 1
        c = counts.setdefault(lang, Counter())
        try:
            for node, _ in walk(parser_for(lang).parse(bytes(mm[s:e])).root_node):
                c[node.type] += 1
        except Exception:
            pass
    return {lang: {k: i + 1 for i, (k, _) in enumerate(c.most_common(126))} for lang, c in counts.items()}


def job(args):
    i, files, kinds = args
    mm = np.memmap(CORPUS, dtype=np.uint8, mode="r")
    out = []
    for path, s, e in files:
        lang = lang_of(path)
        try:
            pred, conf = file_features(bytes(mm[s:e]), lang, kinds)
        except Exception as ex:                                  # a parser failure leaves the file without parser features
            pred = np.zeros((e - s, 4), np.uint8); conf = np.zeros_like(pred)
            out.append((s, e, pred, conf, f"{type(ex).__name__}: {ex}"[:80])); continue
        out.append((s, e, pred, conf, None))
    P = {kid: np.load(f"{OUT}/pred_n{kid}.npy", mmap_mode="r+") for kid in IDS}
    C = {kid: np.load(f"{OUT}/conf_n{kid}.npy", mmap_mode="r+") for kid in IDS}
    errs = 0; n_bytes = 0; fired = np.zeros(4, np.int64)
    for s, e, pred, conf, err in out:
        for j, kid in enumerate(IDS):
            P[kid][s:e] = pred[:, j]; C[kid][s:e] = conf[:, j]
        errs += err is not None; n_bytes += e - s; fired += (conf > 0).sum(0)
    for a in (*P.values(), *C.values()):
        a.flush()
    return i, n_bytes, errs, fired


def main():
    N = os.path.getsize(CORPUS)
    os.makedirs(OUT, exist_ok=True)
    info, base_ok = base_info(BASE, CORPUS, N, TRAIN_END, 1 << 20)
    linked = link_base(BASE, OUT, BASE_ORDERS, base_ok)
    mm = np.memmap(CORPUS, dtype=np.uint8, mode="r")
    heads = [(m.start(), m.end(), m.group(1).decode("latin-1")) for m in HEADER_RE.finditer(mm)]
    files = [(name, e, heads[i + 1][0] if i + 1 < len(heads) else N) for i, (s, e, name) in enumerate(heads)]
    langs = Counter(lang_of(p) for p, _, _ in files)
    print(f"{len(files):,} files; languages {dict(langs.most_common(8))} ({time.time() - t0:.0f} s)", flush=True)
    kinds = kind_table(files, mm)
    print(f"kind tables for {sorted(kinds)} ({time.time() - t0:.0f} s)", flush=True)
    for kid in IDS:
        for kind in ("pred", "conf"):
            a = np.lib.format.open_memmap(f"{OUT}/{kind}_n{kid}.npy", mode="w+", dtype=np.uint8, shape=(N,)); del a
    batches = [files[i:i + 200] for i in range(0, len(files), 200)]
    done = 0; errs = 0; fired = np.zeros(4, np.int64)
    with get_context("fork").Pool(JOBS) as pool:
        for i, nb, ne, f in pool.imap_unordered(job, [(i, b, kinds) for i, b in enumerate(batches)]):
            done += nb; errs += ne; fired += f
            if i % 20 == 0:
                print(f"  {done:,} / {N:,} bytes, {errs} parser failures ({time.time() - t0:.0f} s)", flush=True)
    stats = {str(kid): {"share_with_feature": float(fired[j] / N)} for j, kid in enumerate(IDS)}
    info.update(orders=[*linked, *IDS], parser_stats=stats, parser_failures=errs, languages={str(k): v for k, v in langs.items()},
                kinds={lang: len(k) for lang, k in kinds.items()},
                note="parser_features.py: the base features linked, plus tree-sitter pseudo-orders 2001 (node kind), 2002 (node depth), "
                     "2003 (defined-name completion), 2004 (used-name completion); position p's feature from the bytes before p")
    info.pop("feature_digest", None); info.pop("compare", None)
    json.dump(info, open(f"{OUT}/features.json", "w"), indent=2)
    print(f"done in {time.time() - t0:.0f} s: {errs} parser failures; share of positions with a feature " +
          ", ".join(f"{kid} {fired[j] / N:.3f}" for j, kid in enumerate(IDS)), flush=True)


if __name__ == "__main__":
    main()
