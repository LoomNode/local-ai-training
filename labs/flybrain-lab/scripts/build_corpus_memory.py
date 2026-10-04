#!/usr/bin/env python3
"""Fill the corpus memory in one pass over the training bytes, then emit its lookups for the held-out bytes. CPU only, no gradient.

The training bytes [0, TRAIN_END) are read once and written into the tables of flybrain.corpus_memory (the build). The tables are then
frozen (read-only) and, for every position p in [TRAIN_END, end of file), the prediction of each order from the n bytes before p is
stored as two uint8 arrays per order, pred_n<n>.npy and conf_n<n>.npy, of the file's length and indexed by absolute position:
feature[p] is the prediction for the byte at p (conf 0 = miss). Only positions >= TRAIN_END are filled; the rest read 0 and are not
features (the files are sparse on disk). build.json holds the build's wall and CPU seconds and the occupancy of every table.

The memory is built from data[:TRAIN_END] alone (the held-out bytes are not even read until the tables are frozen) and its
`boundary` makes any update that would reach byte TRAIN_END raise.

--features (build_features) emits the prediction of every order for EVERY position of the file, training positions included, as
pred_n<n>.npy / conf_n<n>.npy (uint8, indexed by absolute position) and features.json. A training position p gets the lookup made
BEFORE the chunk that holds p is written into the table (the same one pass): it sees nothing from its own chunk, so never its own
continuation. A held-out position (p >= TRAIN_END) gets the lookup in the frozen table, as above."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.corpus_memory import CHUNK, LOG2_SLOTS, ORDERS, CorpusMemory, context_hashes
from flybrain.data.files import TRAIN_END

RESULTS = ROOT / "results"
CORPUS = ROOT / "corpora" / "linux_full.bytes"
FEATURE_ORDERS = (8, 16, 32)   # orders 16 and 32 carry most of test a's gain, 8 is the short-context one; order 4 added nothing
TENTHS = 10                    # the training positions are reported by tenth of the range, to show the table filling


def _log(msg: str) -> None:
    print(msg, flush=True)


def build(corpus: str | Path, out_dir: str | Path, orders=ORDERS, log2_slots: int = LOG2_SLOTS, chunk: int = CHUNK,
          train_end: int = TRAIN_END, save_tables: bool = False, log=_log) -> dict:
    """Build the memory from corpus[:train_end], freeze it, write the held-out lookups under out_dir. Returns what build.json holds."""
    corpus, out_dir = Path(corpus), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = corpus.stat().st_size
    if not 0 < train_end <= size:
        raise ValueError(f"train_end {train_end} must be in (0, {size}]")
    load0 = os.getloadavg()
    wall0, cpu0 = time.perf_counter(), time.process_time()

    train = np.fromfile(corpus, dtype=np.uint8, count=train_end)          # the training bytes, and nothing after them
    assert len(train) == train_end
    read_seconds = time.perf_counter() - wall0
    mem = CorpusMemory(orders, log2_slots, boundary=train_end)

    def progress(hi: int, end: int) -> None:
        if (hi // chunk) % 20 == 0 or hi == end:
            log(f"  build {hi / end:6.1%}  {time.perf_counter() - wall0:7.1f} s")

    mem.fit(train, 0, train_end, chunk, progress=progress)
    build_wall, build_cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    assert mem.next_pos == train_end                                      # every training byte was read once, none after
    mem.freeze()
    occupancy, histogram = mem.occupancy(), mem.conf_histogram()
    log(f"build: {build_wall:.1f} s wall ({read_seconds:.1f} s of it reading), {build_cpu:.1f} s CPU; occupancy "
        + ", ".join(f"n={n}: {o:.3f}" for n, o in occupancy.items()))
    del train

    wall1 = time.perf_counter()
    full = np.fromfile(corpus, dtype=np.uint8)                            # held-out bytes are read only now, with the tables frozen
    assert len(full) == size
    files = {}
    for n in mem.orders:
        for kind in ("pred", "conf"):
            files[(kind, n)] = np.lib.format.open_memmap(out_dir / f"{kind}_n{n}.npy", mode="w+", dtype=np.uint8, shape=(size,))
    hits = dict.fromkeys(mem.orders, 0)
    for lo in range(train_end, size, chunk):
        hi = min(lo + chunk, size)
        pred, conf = mem.lookup(full, lo, hi)
        for k, n in enumerate(mem.orders):
            files[("pred", n)][lo:hi] = pred[k]
            files[("conf", n)][lo:hi] = conf[k]
            hits[n] += int(np.count_nonzero(conf[k]))
    for f in files.values():
        f.flush()
    del files, full
    assert mem.next_pos == train_end and mem.frozen                       # the lookups wrote nothing (the tables are read-only)
    lookup_seconds = time.perf_counter() - wall1
    n_held = size - train_end
    digests = {str(n): hashlib.blake2b(t.view(np.uint8), digest_size=16).hexdigest() for n, t in mem.tables.items()}   # two builds agree iff these do
    if save_tables:
        for n, t in mem.tables.items():
            np.save(out_dir / f"table_n{n}.npy", t)

    rec = {
        "corpus": str(corpus), "corpus_bytes": size, "train_end": train_end, "heldout_positions": n_held, "valid_from": train_end,
        "orders": list(mem.orders), "log2_slots": log2_slots, "slots_per_order": 1 << log2_slots, "slot_bytes": 4,
        "table_bytes_per_order": 4 << log2_slots, "chunk": chunk,
        "build_seconds": build_wall, "build_cpu_seconds": build_cpu, "read_seconds": read_seconds,
        "build_mb_per_second": train_end / 1e6 / build_wall,
        "lookup_seconds": lookup_seconds, "lookup_positions": n_held,
        "occupancy": {str(n): o for n, o in occupancy.items()},
        "update_counts": {str(n): s for n, s in mem.stats.items()}, "table_digest": digests,
        "confidence_histogram": {str(n): h for n, h in histogram.items()},
        "heldout_hit_rate": {str(n): hits[n] / max(n_held, 1) for n in mem.orders},
        "affinity_cpus": len(os.sched_getaffinity(0)), "loadavg_start": list(load0), "loadavg_end": list(os.getloadavg()),
        "numpy": np.__version__, "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
        "note": "build_seconds = reading bytes [0, train_end) + all table writes, one process, numpy; lookup_seconds is the held-out emission and file writes",
    }
    (out_dir / "build.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def _digest_array(a: np.ndarray, piece: int = 1 << 26) -> str:
    h = hashlib.blake2b(digest_size=16)
    for lo in range(0, len(a), piece):
        h.update(np.ascontiguousarray(a[lo:lo + piece]).tobytes())
    return h.hexdigest()


def _compare_with_plain(compare: Path, mem: CorpusMemory, out_dir: Path, train_end: int, size: int, chunk: int, digests: dict) -> dict:
    """The held-out part [train_end, size) of this build's arrays against the arrays of the plain build in `compare` (test a's layout:
    build.json plus pred_n<n>.npy / conf_n<n>.npy), order by order, and the final tables' digests."""
    info = json.loads((compare / "build.json").read_text())
    for key, mine in (("train_end", train_end), ("log2_slots", mem.log2_slots), ("corpus_bytes", size)):
        if info[key] != mine:
            raise ValueError(f"cannot compare with {compare}: its {key} is {info[key]}, this build's is {mine}")
    equal, digest_equal = {}, {}
    for n in mem.orders:
        if not (compare / f"pred_n{n}.npy").exists():
            continue
        same = True
        for kind in ("pred", "conf"):
            a = np.load(out_dir / f"{kind}_n{n}.npy", mmap_mode="r")
            b = np.load(compare / f"{kind}_n{n}.npy", mmap_mode="r")
            same = same and all(np.array_equal(a[lo:min(lo + chunk, size)], b[lo:min(lo + chunk, size)]) for lo in range(train_end, size, chunk))
        equal[str(n)] = same
        digest_equal[str(n)] = info.get("table_digest", {}).get(str(n)) == digests[str(n)]
    return {"dir": str(compare), "heldout_arrays_equal": equal, "table_digest_equal": digest_equal, "all_equal": bool(equal) and all(equal.values()) and all(digest_equal.values())}


def build_features(corpus: str | Path, out_dir: str | Path, orders=FEATURE_ORDERS, log2_slots: int = LOG2_SLOTS, chunk: int = CHUNK,
                   train_end: int = TRAIN_END, compare: str | Path | None = None, log=_log) -> dict:
    """The table's prediction for every position of the corpus, uint8 arrays indexed by absolute position under out_dir, plus
    features.json (also returned). Training positions [0, train_end): the lookup made before the chunk holding the position is written
    (one pass, chunks of `chunk`). Held-out positions [train_end, end): the lookup in the frozen table. `compare`: a directory written by
    build() whose held-out arrays and table digests must equal this build's (recorded under "compare")."""
    corpus, out_dir = Path(corpus), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = corpus.stat().st_size
    if not 0 < train_end <= size:
        raise ValueError(f"train_end {train_end} must be in (0, {size}]")
    load0 = os.getloadavg()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    sec = dict.fromkeys(("read", "hash", "lookup", "table_update", "feature_files"), 0.0)

    train = np.fromfile(corpus, dtype=np.uint8, count=train_end)          # the training bytes, and nothing after them
    assert len(train) == train_end
    sec["read"] = time.perf_counter() - wall0
    mem = CorpusMemory(orders, log2_slots, boundary=train_end)
    files = {(kind, n): np.lib.format.open_memmap(out_dir / f"{kind}_n{n}.npy", mode="w+", dtype=np.uint8, shape=(size,)) for n in mem.orders for kind in ("pred", "conf")}
    edges = [train_end * i // TENTHS for i in range(TENTHS + 1)]
    hits = {n: np.zeros(TENTHS, dtype=np.int64) for n in mem.orders}
    right = {n: np.zeros(TENTHS, dtype=np.int64) for n in mem.orders}

    def emit(lo: int, hi: int, pred: np.ndarray, conf: np.ndarray, data: np.ndarray, tally: bool) -> None:
        t0 = time.perf_counter()
        for k, n in enumerate(mem.orders):
            files[("pred", n)][lo:hi] = pred[k]
            files[("conf", n)][lo:hi] = conf[k]
            if tally:
                for i in range(TENTHS):
                    a, b = max(lo, edges[i]), min(hi, edges[i + 1])
                    if a < b:
                        h = conf[k, a - lo:b - lo] > 0
                        hits[n][i] += np.count_nonzero(h)
                        right[n][i] += np.count_nonzero(h & (pred[k, a - lo:b - lo] == data[a:b]))
        sec["feature_files"] += time.perf_counter() - t0

    for lo in range(0, train_end, chunk):
        hi = min(lo + chunk, train_end)
        t0 = time.perf_counter()
        hashes = context_hashes(train, lo, hi, mem.orders)                # hashed once, for the lookup and for the write
        t1 = time.perf_counter()
        pred, conf = mem.lookup(train, lo, hi, hashes=hashes)             # the table as it stands BEFORE this chunk is written
        t2 = time.perf_counter()
        sec["hash"] += t1 - t0
        sec["lookup"] += t2 - t1
        emit(lo, hi, pred, conf, train, tally=True)
        t3 = time.perf_counter()
        mem.update(train, lo, hi, hashes=hashes)
        sec["table_update"] += time.perf_counter() - t3
        if (hi // chunk) % 20 == 0 or hi == train_end:
            log(f"  features {hi / train_end:6.1%}  {time.perf_counter() - wall0:7.1f} s")
    build_wall, build_cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    assert mem.next_pos == train_end                                      # every training byte was read once, none after
    mem.freeze()
    occupancy = mem.occupancy()
    log(f"features, training pass: {build_wall:.1f} s wall ({sec['read']:.1f} s reading), {build_cpu:.1f} s CPU; occupancy "
        + ", ".join(f"n={n}: {o:.3f}" for n, o in occupancy.items()))
    del train

    wall1 = time.perf_counter()
    full = np.fromfile(corpus, dtype=np.uint8)                            # held-out bytes are read only now, with the tables frozen
    assert len(full) == size
    n_held = size - train_end
    held_hits = dict.fromkeys(mem.orders, 0)
    held_right = dict.fromkeys(mem.orders, 0)
    for lo in range(train_end, size, chunk):
        hi = min(lo + chunk, size)
        pred, conf = mem.lookup(full, lo, hi)
        emit(lo, hi, pred, conf, full, tally=False)
        for k, n in enumerate(mem.orders):
            h = conf[k] > 0
            held_hits[n] += int(np.count_nonzero(h))
            held_right[n] += int(np.count_nonzero(h & (pred[k] == full[lo:hi])))
    for f in files.values():
        f.flush()
    del files, full
    assert mem.next_pos == train_end and mem.frozen                       # the held-out lookups wrote nothing
    heldout_seconds = time.perf_counter() - wall1

    wall2 = time.perf_counter()
    table_digest = {str(n): hashlib.blake2b(t.view(np.uint8), digest_size=16).hexdigest() for n, t in mem.tables.items()}
    feature_digest = {f"{kind}_n{n}": _digest_array(np.load(out_dir / f"{kind}_n{n}.npy", mmap_mode="r")) for n in mem.orders for kind in ("pred", "conf")}
    cmp = _compare_with_plain(Path(compare), mem, out_dir, train_end, size, chunk, table_digest) if compare else None
    check_seconds = time.perf_counter() - wall2

    width = np.maximum(np.diff(edges), 1)
    rec = {
        "kind": "features", "corpus": str(corpus), "corpus_bytes": size, "train_end": train_end, "heldout_positions": n_held,
        "orders": list(mem.orders), "log2_slots": log2_slots, "slots_per_order": 1 << log2_slots, "chunk": chunk,
        "files": {"pred": "pred_n<order>.npy", "conf": "conf_n<order>.npy"},
        "alignment": "feature[p] is the table's prediction for the byte at absolute position p from the n bytes before it (conf 0 = miss). "
                     "p < train_end: the lookup made before the chunk holding p was written; p >= train_end: the frozen table's lookup",
        "build_seconds": build_wall, "build_cpu_seconds": build_cpu, "seconds_by_stage": sec,
        "heldout_lookup_seconds": heldout_seconds, "digest_and_compare_seconds": check_seconds,
        "build_mb_per_second": train_end / 1e6 / build_wall,
        "occupancy": {str(n): o for n, o in occupancy.items()}, "table_digest": table_digest, "feature_digest": feature_digest,
        "train_hit_rate_by_tenth": {str(n): (hits[n] / width).tolist() for n in mem.orders},
        "train_right_given_hit_by_tenth": {str(n): (right[n] / np.maximum(hits[n], 1)).tolist() for n in mem.orders},
        "train_hit_rate": {str(n): float(hits[n].sum()) / train_end for n in mem.orders},
        "heldout_hit_rate": {str(n): held_hits[n] / max(n_held, 1) for n in mem.orders},
        "heldout_right_given_hit": {str(n): held_right[n] / max(held_hits[n], 1) for n in mem.orders},
        "compare": cmp,
        "affinity_cpus": len(os.sched_getaffinity(0)), "loadavg_start": list(load0), "loadavg_end": list(os.getloadavg()),
        "numpy": np.__version__, "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
        "note": "build_seconds = reading bytes [0, train_end) + the one pass: hashing, lookups, feature-file writes and table writes, one process, numpy; "
                "heldout_lookup_seconds is the frozen-table emission for positions >= train_end",
    }
    (out_dir / "features.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--out", default="", help="output directory (default results/mem; with --features results/mem/feat)")
    p.add_argument("--orders", default="", help=f"comma list of context orders (default {','.join(map(str, ORDERS))}; with --features {','.join(map(str, FEATURE_ORDERS))})")
    p.add_argument("--log2-slots", type=int, default=LOG2_SLOTS, help="slots per order = 2 ** this")
    p.add_argument("--chunk", type=int, default=CHUNK, help="positions per vectorised update")
    p.add_argument("--train-end", type=int, default=TRAIN_END)
    p.add_argument("--save-tables", action="store_true", help="also save the frozen tables (4 bytes a slot) next to the lookups")
    p.add_argument("--features", action="store_true", help="emit the prediction for EVERY position: training positions from the lookup made before their chunk is written, held-out ones from the frozen table")
    p.add_argument("--compare", default="", help="with --features: a plain build's directory (build.json + pred/conf arrays) whose held-out arrays must equal this build's; exit 1 if not")
    a = p.parse_args()
    if a.compare and not a.features:
        p.error("--compare belongs to --features")
    if a.save_tables and a.features:
        p.error("--save-tables belongs to the plain mode")
    orders = [int(n) for n in a.orders.split(",")] if a.orders else list(FEATURE_ORDERS if a.features else ORDERS)
    if a.features:
        rec = build_features(a.corpus, a.out or RESULTS / "mem" / "feat", orders, a.log2_slots, a.chunk, a.train_end, a.compare or None)
        print(f"features: {rec['build_seconds']:.1f} s wall for the training pass ({rec['build_cpu_seconds']:.1f} s CPU), "
              f"{rec['heldout_lookup_seconds']:.1f} s for the held-out lookups; stages {json.dumps({k: round(v, 1) for k, v in rec['seconds_by_stage'].items()})}")
        print("training hit rate by tenth of the corpus\n| order | " + " | ".join(f"{i + 1}/10" for i in range(TENTHS)) + " | held-out |\n|---:|" + "---:|" * (TENTHS + 1))
        for n in rec["orders"]:
            print(f"| {n} | " + " | ".join(f"{v:.4f}" for v in rec["train_hit_rate_by_tenth"][str(n)]) + f" | {rec['heldout_hit_rate'][str(n)]:.4f} |")
        if rec["compare"] is not None:
            print(f"compare with {rec['compare']['dir']}: held-out arrays equal {rec['compare']['heldout_arrays_equal']}, table digests equal {rec['compare']['table_digest_equal']}")
            if not rec["compare"]["all_equal"]:
                sys.exit("MISMATCH: the held-out features differ from the plain build's")
        return
    rec = build(a.corpus, a.out or RESULTS / "mem", orders, a.log2_slots, a.chunk, a.train_end, a.save_tables)
    print(f"build {rec['build_seconds']:.1f} s wall, {rec['build_cpu_seconds']:.1f} s CPU; lookups {rec['lookup_seconds']:.1f} s")
    print("| order | occupancy | held-out positions with a hit |\n|---:|---:|---:|")
    for n in rec["orders"]:
        print(f"| {n} | {rec['occupancy'][str(n)]:.4f} | {rec['heldout_hit_rate'][str(n)]:.4f} |")


if __name__ == "__main__":
    main()
