"""Shared pieces of the table builders (key_tables.py, counter_tables.py, assoc_all.py): parameters from the environment, the file-header
cut points a corpus can be split at, the grouping of whole files into chunks for a worker pool, the features.json every builder writes,
and the guard against rewriting memory-mapped feature files.

Cut points. A corpus is a run of files, each preceded by a header line "\n// ==== path ====\n"; a cut point is the position of the
header's first "/" when the two bytes before it are newlines (the previous file's last newline and the header's own). Cutting there
leaves no identifier run, line or token across the cut, the previous line of the header line is the empty one, and brace/paren depths
reset there anyway, which is what lets each chunk be computed as a small corpus of its own (see the builders for the exceptions and
what they do about them)."""
import json
import mmap
import os
import sys
from bisect import bisect_right

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
HEADER = b"\n\n// ==== "


def env_str(name, default):
    return os.environ.get(name) or default


def env_int(name, default):
    v = os.environ.get(name)
    return int(v.replace("_", "")) if v else default


def file_cuts(path):
    """Positions of the "/" of every header line preceded by an empty line, ascending (a few hundred thousand for 5 GB: one find each)."""
    cuts = []
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        i = mm.find(HEADER)
        while i >= 0:
            cuts.append(i + 2)
            i = mm.find(HEADER, i + 1)
    return cuts


def chunk_bounds(cuts, n, chunk):
    """[0, b1, ..., n]: whole files grouped greedily into chunks of at least `chunk` bytes (a file longer than that is a chunk of its own)."""
    bounds = [0]
    for c in cuts:
        if c - bounds[-1] >= chunk and c < n:
            bounds.append(c)
    bounds.append(n)
    return bounds


def halo_start(cuts, lo, halo):
    """The last cut point at or before lo - halo (0 if none): where a chunk starting at lo begins reading its context."""
    if lo == 0:
        return 0
    i = bisect_right(cuts, lo - halo)
    return cuts[i - 1] if i else 0


ALIGNMENT = ("feature[p] is the table's prediction for the byte at absolute position p from the n bytes before it (conf 0 = miss). "
             "p < train_end: the lookup made before the chunk holding p was written; p >= train_end: the frozen table's lookup")


def base_info(base, corpus, n, train_end, chunk):
    """The features.json a builder starts from: the base byte-feature dir's (as the builders always did) when it exists and is for a
    corpus of this size, else a fresh one with what flybrain.data.memfeat.FeatureStore reads (corpus_bytes, train_end, orders, the
    pred_n/conf_n file naming). Returns (info, base_ok)."""
    path = f"{base}/features.json"
    if os.path.exists(path):
        info = json.load(open(path))
        if int(info.get("corpus_bytes", -1)) == n:
            return info, True
        print(f"{path} is for a corpus of {info.get('corpus_bytes')} bytes, not {n}: not used", flush=True)
    return {"kind": "features", "corpus": corpus, "corpus_bytes": n, "train_end": train_end, "heldout_positions": n - train_end,
            "chunk": chunk, "files": {"pred": "pred_n<order>.npy", "conf": "conf_n<order>.npy"}, "alignment": ALIGNMENT}, False


def link_base(base, out, orders, ok):
    """Symlink the base dir's byte-table features of `orders` into out (skipped without a usable base dir)."""
    if not ok:
        return []
    linked = []
    for n in orders:
        if not all(os.path.exists(f"{base}/{k}_n{n}.npy") for k in ("pred", "conf")):
            continue
        for kind in ("pred", "conf"):
            dst = f"{out}/{kind}_n{n}.npy"
            if not os.path.lexists(dst):
                os.symlink(f"{base}/{kind}_n{n}.npy", dst)
        linked.append(n)
    return linked


def refuse_if_in_use(directory, names=None):
    """A feature file that a running process has memory-mapped must not be rewritten (np.save truncates it: the process dies of SIGBUS,
    as a training run did on 2026-10-07). Checks every process's maps for files of the directory (or only `names` in it), by their
    real paths."""
    import glob
    files = glob.glob(f"{directory}/*.npy") if names is None else [f"{directory}/{x}" for x in names if os.path.exists(f"{directory}/{x}")]
    real = {os.path.realpath(f) for f in files}
    if not real:
        return
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            maps = open(f"/proc/{pid}/maps").read()
        except OSError:
            continue
        for f in real:
            if f in maps:
                sys.exit(f"refusing to rewrite {f}: mapped by pid {pid} ({open(f'/proc/{pid}/comm').read().strip()})")
