"""The table inputs of any text (the use layer, 2026-10-11). A table-fed reader takes, at every position, 18 inputs computed from the
bytes before it: the two-way tables' answers for the byte orders 4..64 and the template / word / line keys (116, 132, 202, 203, 301),
and seven counters (1001-1007: column, indent, previous indent, brace and paren depth, in-string, in-comment). The corpus build
computed them once for the whole corpus (assoc_all.py, key_tables.py, counter_tables.py); a model reading new text needs the same
functions on that text plus the saved tables (assoc_all.py TABLES_ONLY=1). text_features() calls the build's own functions
(context_hashes, compute_keys, counters, the two-way lookup), so the inputs of a corpus stretch computed from its text alone equal the
stored features wherever the stretch carries the build's context: from a file header, with enough before it for the template keys'
32 symbols (--check reads 64 KB).

Row p of the result holds position p's inputs, the answers for the byte at p from the bytes before p (the features' alignment), for
p = 0..N with N = len(text): row N is the next byte's, the one a sampler predicts. A byte order gets no answer (pred 0, conf 0) where
fewer bytes than the order precede the position, as at the corpus's start; the keys and counters answer everywhere.

usage (the exactness check): use_features.py --check <tables dir> <features dir> [--offset O] [--n 1024] [--T 2048]
(default offset: the tables' train_end, the package corpus's fixed dev windows). CPU and memory-mapped tables only."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_common import SPEED  # noqa: E402

sys.path.insert(0, SPEED)
from flybrain.corpus_memory import context_hashes, slot_and_tag  # noqa: E402
from check_tables import lookup  # noqa: E402
from counter_tables import IDS as COUNTER_IDS, counters  # noqa: E402
from key_tables import KEY_IDS, compute_keys  # noqa: E402

BYTE_ORDERS = (4, 8, 12, 16, 32, 64)
HEADER = b"\n// ==== "


class Tables:
    """The saved two-way tables of a build (table_n<order>.npy and tables.json), memory-mapped read-only."""

    def __init__(self, directory):
        self.dir = directory
        self.meta = json.load(open(f"{directory}/tables.json"))
        self.log2 = int(self.meta["log2_slots"])
        self.t = {int(n): np.load(f"{directory}/table_n{n}.npy", mmap_mode="r") for n in self.meta["orders"]}


def text_features(tables: Tables, text, orders) -> tuple[np.ndarray, np.ndarray]:
    """pred, conf: uint8 [len(text) + 1, len(orders)], the inputs of positions 0..len(text) in the order given."""
    data = text if isinstance(text, np.ndarray) else np.frombuffer(bytes(text), dtype=np.uint8)
    ext = np.concatenate([data.astype(np.uint8), np.zeros(1, np.uint8)])     # a dummy last byte: no input of any row reads it
    M = len(ext)
    orders = [int(n) for n in orders]
    unknown = [n for n in orders if n not in BYTE_ORDERS and n not in KEY_IDS and n not in COUNTER_IDS]
    missing = [n for n in orders if (n in BYTE_ORDERS or n in KEY_IDS) and n not in tables.t]
    if unknown or missing:
        raise ValueError(f"orders {unknown} are unknown and {missing} have no saved table in {tables.dir}")
    pred = np.zeros((M, len(orders)), np.uint8)
    conf = np.zeros_like(pred)
    bo = [n for n in orders if n in BYTE_ORDERS]
    hashes = context_hashes(ext, 0, M, bo) if bo else {}
    keys = compute_keys(ext) if any(n in KEY_IDS for n in orders) else {}
    vals = counters(ext)[0] if any(n in COUNTER_IDS for n in orders) else {}
    for j, n in enumerate(orders):
        if n in BYTE_ORDERS:
            first = min(n, M)
            slot, tag = slot_and_tag(hashes[n][first:], tables.log2)
            pred[first:, j], conf[first:, j] = lookup(tables.t[n], slot, tag)
        elif n in KEY_IDS:
            slot, tag = slot_and_tag(np.asarray(keys[n]), tables.log2)
            pred[:, j], conf[:, j] = lookup(tables.t[n], slot, tag)
        else:
            v = np.asarray(vals[n])
            pred[:, j] = (128 + np.minimum(v, 127).astype(np.int16)).astype(np.uint8)
            conf[:, j] = 1
    return pred, conf


def check(tables_dir, features_dir, offset=None, n=1024, T=2048, back=1 << 16) -> int:
    """The inputs of the n windows of T bytes from `offset`, computed from the corpus text alone (from the last file header at least
    `back` bytes before the windows), against the stored features of the same positions. Returns the number of differences."""
    tables = Tables(tables_dir)
    corpus = np.memmap(tables.meta["corpus"], dtype=np.uint8, mode="r")
    A = int(tables.meta["train_end"] if offset is None else offset)
    B = A + n * T + 1
    lo = max(0, A - back - (1 << 17))                    # files are 2 to 60 KB: a header lies in any 128 KB
    H = bytes(corpus[lo:max(lo, A - back)]).rfind(HEADER)
    if H < 0:
        raise SystemExit(f"no file header in the {back:,} bytes before {A:,}")
    H += lo
    info = json.load(open(f"{features_dir}/features.json"))
    orders = [int(x) for x in info["orders"]]
    pred, conf = text_features(tables, np.asarray(corpus[H:B]), orders)
    bad = 0
    for j, o in enumerate(orders):
        fp = np.load(f"{features_dir}/pred_n{o}.npy", mmap_mode="r")[A:B]
        fc = np.load(f"{features_dir}/conf_n{o}.npy", mmap_mode="r")[A:B]
        d = int((pred[A - H:B - H, j] != fp).sum() + (conf[A - H:B - H, j] != fc).sum())
        print(f"order {o}: {B - A:,} positions, {d} differences", flush=True)
        bad += d
    print(f"text features from {H:,} reproduce the stored features of [{A:,}, {B:,})" if bad == 0 else f"MISMATCH: {bad} differences")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs=2, metavar=("TABLES", "FEATURES"), required=True)
    ap.add_argument("--offset", type=int, default=None); ap.add_argument("--n", type=int, default=1024); ap.add_argument("--T", type=int, default=2048)
    a = ap.parse_args()
    sys.exit(1 if check(*a.check, offset=a.offset, n=a.n, T=a.T) else 0)


if __name__ == "__main__":
    main()
