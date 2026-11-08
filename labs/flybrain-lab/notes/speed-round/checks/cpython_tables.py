"""Tables follow-up, probe 2: a model trained with the kernel table, read on CPython text with four tables.

Feature directories under results/mem_cpython/, all for corpora/cpython.bytes (45,522,125 bytes), features filled for the window
region [C0, C1) only (the scorer reads nothing else):
  none     every lookup a miss (the model with no table at all);
  kernel   the kernel training tables, looked up on CPython text (the table the model was trained with, on the wrong corpus);
  own      a table written in one pass over CPython bytes [0, C_TRAIN) and nothing else;
  merged   the kernel tables extended with the same CPython bytes (a new project added to the old table).
The file that straddles C_TRAIN is not written (its window part would see its own earlier bytes), nor is any file whose content
equals a window-region file's. CPU only; well under a minute."""
import json
import os
import sys
import time

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import SLOT, CorpusMemory  # noqa: E402
from flybrain.data.files import content_hash, file_table  # noqa: E402

CPY = "/games/ailab/lab/flybrain-lab/corpora/cpython.bytes"
MEM = SPEED + "/results/mem"
OUT = SPEED + "/results/mem_cpython"
ORDERS = (8, 16, 32)
N, T = 1024, 2048
C_TRAIN = 40_000_000
C0, C1 = C_TRAIN, C_TRAIN + N * T + 1
CHUNK = 1 << 20

t0 = time.time()
data = np.memmap(CPY, dtype=np.uint8, mode="r")
size = len(data)
assert C1 <= size, (C1, size)
table = file_table(CPY)
hashes = [content_hash(CPY, e) for e in table]
win_files = [i for i, e in enumerate(table) if e.start < C1 and e.end > C0]
win_hash = {hashes[i] for i in win_files}
straddle = next((i for i, e in enumerate(table) if e.start < C_TRAIN < e.end), None)
train_files = [i for i, e in enumerate(table) if e.end <= C_TRAIN and hashes[i] not in win_hash]
skipped = [table[i].path for i, e in enumerate(table) if e.end <= C_TRAIN and hashes[i] in win_hash]
print(f"cpython: {size:,} bytes, {len(table)} files; window region [{C0:,}, {C1:,}) holds {len(win_files)} files; table material: {len(train_files)} files "
      f"({sum(table[i].end - table[i].start for i in train_files):,} bytes); skipped as copies of window files: {len(skipped)}; straddling file "
      f"{'none' if straddle is None else table[straddle].path} not written", flush=True)


def load_kernel():
    mem = CorpusMemory(orders=ORDERS)
    for n in ORDERS:
        t = np.load(f"{MEM}/table_n{n}.npy")
        assert t.dtype == SLOT and t.shape == mem.tables[n].shape
        mem.tables[n] = t
    return mem


def fit_cpython(mem):
    mem.next_pos = 0
    for i in train_files:
        e = table[i]
        for a in range(e.start, e.end, CHUNK):
            mem.update(data, a, min(a + CHUNK, e.end))


def emit(name, mem, note):
    d = f"{OUT}/{name}"
    os.makedirs(d, exist_ok=True)
    pred = np.zeros((len(ORDERS), size), dtype=np.uint8)
    conf = np.zeros_like(pred)
    if mem is not None:
        p, c = mem.lookup(data, C0, C1)
        pred[:, C0:C1] = p; conf[:, C0:C1] = c
    for k, n in enumerate(ORDERS):
        np.save(f"{d}/pred_n{n}.npy", pred[k]); np.save(f"{d}/conf_n{n}.npy", conf[k])
    info = {"kind": "features", "corpus": CPY, "corpus_bytes": size, "train_end": 0, "orders": list(ORDERS), "files": {"pred": "pred_n<order>.npy", "conf": "conf_n<order>.npy"},
            "alignment": "feature[p] is the table's prediction for the byte at absolute position p from the n bytes before it (conf 0 = miss); filled for [C0, C1) only",
            "C0": C0, "C1": C1, "note": note, "occupancy": mem.occupancy() if mem is not None else {}}
    json.dump(info, open(f"{d}/features.json", "w"), indent=2)
    hits = (conf[:, C0:C1] > 0).mean(1)
    right = [float((pred[k, C0:C1][conf[k, C0:C1] > 0] == data[C0:C1][conf[k, C0:C1] > 0]).mean()) if hits[k] else 0.0 for k in range(len(ORDERS))]
    print(f"{name:7s}: hit rate " + ", ".join(f"{n}: {h:.3f} (right {r:.3f})" for n, h, r in zip(ORDERS, hits, right)) + f"; {time.time() - t0:.0f} s", flush=True)


emit("none", None, "every lookup a miss")
mem = load_kernel()
emit("kernel", mem, "the kernel training tables (results/mem/table_n*.npy) looked up on CPython text")
fit_cpython(mem)
emit("merged", mem, f"the kernel training tables extended with CPython bytes [0, {C_TRAIN}) by file, copies of window files and the straddling file skipped")
del mem
own = CorpusMemory(orders=ORDERS)
fit_cpython(own)
emit("own", own, f"a table written over CPython bytes [0, {C_TRAIN}) by file only, copies of window files and the straddling file skipped")
json.dump({"window_files": [table[i].path for i in win_files], "skipped_copies": skipped, "train_files": len(train_files)}, open(f"{OUT}/files.json", "w"), indent=1)
print(f"done in {time.time() - t0:.0f} s")
