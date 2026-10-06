"""Tables follow-up, probe 1: the corpus table extended AFTER training with held-out kernel files the model never saw.

Writes results/mem/feat_ext (next to the base features in results/mem/feat): the base features with the dev window region
[W0, W1) replaced by lookups in the extended table, so cross_eval.py / score_tables.py can feed them to the table-trained checkpoints.

Material, written in this order into the frozen training tables (orders 8, 16, 32, loaded from results/mem/table_n*.npy):
  phase 1a  the bytes from the end of the training range (200,000,000) up to W0: the legacy validation region, never trained on;
  phase 1b  every dev file that lies entirely after the window region;
  phase 2   the files of the window region, in order: each is LOOKED UP first (its features), then written, so a position sees
            every file written before it and never its own file or a later one.
A file whose content equals another window-region file's is never written (an exact copy would be an oracle). Test files are not touched.
Also writes results/mem/feat_ext/windows.json: per window, the file, its directory, the hits per order under the base and the
extended features, and how much same-directory material the extended table holds for it.
CPU only; a minute or two."""
import json
import sys
import time

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import SLOT, CorpusMemory, context_hashes, next_state, slot_and_tag  # noqa: E402
from flybrain.data.files import TRAIN_END, content_hash, file_table, splits  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
MEM = SPEED + "/results/mem"
POLICY = sys.argv[1] if len(sys.argv) > 1 else "evict"     # evict = the build rule (a new context takes an occupied slot); noevict = only empty or same-tag slots
BASE, OUT = MEM + "/feat", MEM + ("/feat_ext" if POLICY == "evict" else "/feat_ext_noevict")
ORDERS = (8, 16, 32)
N, T = 1024, 2048
W0 = 202_117_357                 # the dev yardstick windows: inputs [W0, W0 + N * T), features of positions [W0 + 1, W0 + N * T + 1)
W1 = W0 + N * T + 1
CHUNK = 1 << 20

t0 = time.time()
data = np.memmap(CORPUS, dtype=np.uint8, mode="r")
table = file_table(CORPUS)
sp = splits(table)
dev = sp["dev"]
assert table[dev[0]].start == W0, (table[dev[0]].start, W0)
dev_end = table[dev[-1]].end
tail = [i for i, e in enumerate(table) if e.end > TRAIN_END and e.start < dev_end]          # every file with bytes after the training range, before test
hashes = {i: content_hash(CORPUS, table[i]) for i in tail}
win_files = [i for i in dev if table[i].start < W1 and table[i].end > W0]
win_hashes = {}
for i in win_files:
    win_hashes.setdefault(hashes[i], []).append(i)
dup = {h for h, idx in win_hashes.items() if len(idx) > 1}                                  # content shared by two window files
never = {i for i in tail if hashes[i] in win_hashes and (i not in win_files or hashes[i] in dup)}
print(f"files: {len(table)} in the corpus, {len(tail)} after the training range before test, {len(win_files)} in the window region; "
      f"{len(never)} never written (content of a window file)", flush=True)

mem = CorpusMemory(orders=ORDERS)
for n in ORDERS:
    t = np.load(f"{MEM}/table_n{n}.npy")
    assert t.dtype == SLOT and t.shape == mem.tables[n].shape
    mem.tables[n] = t
occ0 = mem.occupancy()


def write(lo, hi):
    """Write bytes [lo, hi) in chunks. evict: the build's rule through CorpusMemory.update. noevict: the same rule, but a slot that
    holds a different context keeps it (the extension adds knowledge and never removes any; a context repeated inside one chunk is
    written once, the last occurrence winning)."""
    mem.next_pos = 0                                    # the build's in-order rule does not apply to an extension written out of order
    for a in range(lo, hi, CHUNK):
        b = min(a + CHUNK, hi)
        if POLICY == "evict":
            mem.update(data, a, b); continue
        hashes = context_hashes(data, a, b, ORDERS)
        nxt = np.asarray(data[a:b], dtype=np.uint8)
        for n in ORDERS:
            first = min(max(a, n) - a, b - a)
            slot, tag = slot_and_tag(hashes[n][first:], mem.log2_slots)
            rec = mem.tables[n][slot]
            ok = (rec["conf"] == 0) | (rec["tag"] == tag)
            if ok.any():
                new, _ = next_state(rec[ok], tag[ok], nxt[first:][ok])
                mem.tables[n][slot[ok]] = new


written = []                                            # (file index or None, lo, hi)
# phase 1a: the legacy validation region, up to W0
pre = [i for i in tail if table[i].end > TRAIN_END and table[i].start < W0]
for i in pre:
    if i in never:
        continue
    lo, hi = max(table[i].start, TRAIN_END), min(table[i].end, W0)
    write(lo, hi); written.append((i, lo, hi))
# phase 1b: dev files entirely after the window region
for i in dev:
    if table[i].start >= W1 and i not in never:
        write(table[i].start, table[i].end); written.append((i, table[i].start, table[i].end))
n_phase1 = len(written)
print(f"phase 1: {n_phase1} files, {sum(hi - lo for _, lo, hi in written):,} bytes written in {time.time() - t0:.0f} s", flush=True)

# phase 2: the window-region files, each looked up and then written
pred = np.zeros((len(ORDERS), W1 - W0), dtype=np.uint8)
conf = np.zeros_like(pred)
seen_before = {}                                        # window file -> list of written (file, lo, hi) at the time of its lookup
for i in win_files:
    lo, hi = max(table[i].start, W0), min(table[i].end, W1)
    p, c = mem.lookup(data, lo, hi)
    pred[:, lo - W0:hi - W0] = p; conf[:, lo - W0:hi - W0] = c
    seen_before[i] = list(written)
    if i not in never:
        write(table[i].start, table[i].end); written.append((i, table[i].start, table[i].end))
print(f"phase 2: {len(win_files)} window files looked up, {len(written) - n_phase1} written; total {sum(hi - lo for _, lo, hi in written):,} bytes; "
      f"occupancy {occ0} -> {mem.occupancy()}; {time.time() - t0:.0f} s", flush=True)

# the feature directory: base arrays with [W0, W1) replaced
import os  # noqa: E402
os.makedirs(OUT, exist_ok=True)
base_hits = np.zeros((len(ORDERS), W1 - W0), dtype=bool)
for k, n in enumerate(ORDERS):
    for kind, arr in (("pred", pred), ("conf", conf)):
        a = np.load(f"{BASE}/{kind}_n{n}.npy")
        if kind == "conf":
            base_hits[k] = a[W0:W1] > 0
        a[W0:W1] = arr[k]
        np.save(f"{OUT}/{kind}_n{n}.npy", a)
info = json.load(open(f"{BASE}/features.json"))
info["note"] = (f"extend_table.py ({POLICY}): the base features with positions [{W0}, {W1}) replaced by lookups in the training tables extended with "
                f"{len(written)} held-out kernel files ({sum(hi - lo for _, lo, hi in written):,} bytes) the model never saw; see windows.json")
info.pop("feature_digest", None); info.pop("compare", None)
json.dump(info, open(f"{OUT}/features.json", "w"), indent=2)

# per-window record
ext_hits = conf > 0
rows = []
for w in range(N):
    a, b = W0 + w * T + 1, W0 + (w + 1) * T + 1                  # the feature positions of window w
    mid = W0 + w * T + T // 2
    fi = next(i for i in win_files if table[i].start <= mid < table[i].end)
    d = table[fi].dirname
    before = seen_before[fi]
    same = [(j, lo, hi) for j, lo, hi in before if j is not None and table[j].dirname == d]
    rows.append({"window": w, "file": table[fi].path, "dir": d,
                 "base_hits": [int(base_hits[k, a - W0:b - W0].sum()) for k in range(len(ORDERS))],
                 "ext_hits": [int(ext_hits[k, a - W0:b - W0].sum()) for k in range(len(ORDERS))],
                 "new_hits": [int((ext_hits[k] & ~base_hits[k])[a - W0:b - W0].sum()) for k in range(len(ORDERS))],
                 "lost_hits": [int((base_hits[k] & ~ext_hits[k])[a - W0:b - W0].sum()) for k in range(len(ORDERS))],
                 "material_files": len(before), "same_dir_files": len(same), "same_dir_bytes": sum(hi - lo for _, lo, hi in same)})
json.dump({"orders": list(ORDERS), "W0": W0, "W1": W1, "written": [(table[i].path if i is not None else None, lo, hi) for i, lo, hi in written],
           "never_written": sorted(table[i].path for i in never), "windows": rows}, open(f"{OUT}/windows.json", "w"))
tot = len(rows) * T
for k, n in enumerate(ORDERS):
    print(f"order {n}: hit rate base {sum(r['base_hits'][k] for r in rows) / tot:.3f} -> extended {sum(r['ext_hits'][k] for r in rows) / tot:.3f}; "
          f"new {sum(r['new_hits'][k] for r in rows) / tot:.3f}, lost {sum(r['lost_hits'][k] for r in rows) / tot:.4f}")
print(f"windows with same-directory material: {sum(r['same_dir_files'] > 0 for r in rows)} of {N}; done in {time.time() - t0:.0f} s")
