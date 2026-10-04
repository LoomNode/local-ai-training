#!/usr/bin/env python3
"""Fill the corpus memory in one pass over the training bytes, then emit its lookups for the held-out bytes. CPU only, no gradient.

The training bytes [0, TRAIN_END) are read once and written into the tables of flybrain.corpus_memory (the build). The tables are then
frozen (read-only) and, for every position p in [TRAIN_END, end of file), the prediction of each order from the n bytes before p is
stored as two uint8 arrays per order, pred_n<n>.npy and conf_n<n>.npy, of the file's length and indexed by absolute position:
feature[p] is the prediction for the byte at p (conf 0 = miss). Only positions >= TRAIN_END are filled; the rest read 0 and are not
features (the files are sparse on disk). build.json holds the build's wall and CPU seconds and the occupancy of every table.

The memory is built from data[:TRAIN_END] alone (the held-out bytes are not even read until the tables are frozen) and its
`boundary` makes any update that would reach byte TRAIN_END raise."""

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

from flybrain.corpus_memory import CHUNK, LOG2_SLOTS, ORDERS, CorpusMemory
from flybrain.data.files import TRAIN_END

RESULTS = ROOT / "results"
CORPUS = ROOT / "corpora" / "linux_full.bytes"


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--out", default=str(RESULTS / "mem"))
    p.add_argument("--orders", default=",".join(map(str, ORDERS)))
    p.add_argument("--log2-slots", type=int, default=LOG2_SLOTS, help="slots per order = 2 ** this")
    p.add_argument("--chunk", type=int, default=CHUNK, help="positions per vectorised update")
    p.add_argument("--train-end", type=int, default=TRAIN_END)
    p.add_argument("--save-tables", action="store_true", help="also save the frozen tables (4 bytes a slot) next to the lookups")
    a = p.parse_args()
    rec = build(a.corpus, a.out, [int(n) for n in a.orders.split(",")], a.log2_slots, a.chunk, a.train_end, a.save_tables)
    print(f"build {rec['build_seconds']:.1f} s wall, {rec['build_cpu_seconds']:.1f} s CPU; lookups {rec['lookup_seconds']:.1f} s")
    print("| order | occupancy | held-out positions with a hit |\n|---:|---:|---:|")
    for n in rec["orders"]:
        print(f"| {n} | {rec['occupancy'][str(n)]:.4f} | {rec['heldout_hit_rate'][str(n)]:.4f} |")


if __name__ == "__main__":
    main()
