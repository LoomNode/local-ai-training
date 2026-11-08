"""PAQ's contexts as keys for the same table (idea 4 of 2026-10-07; Kyle: "4 sounds the most interesting"). The classical context-mixing
compressors are this recipe with a tiny mixer, and their context list is thirty years of what repeats in text and code. Seven of them as
pseudo-orders, written to results/mem/feat_paq next to the combined arm's eleven lookup orders (linked from feat_all; the counters dropped):
  401, 402  indirect: the current 1-byte / 2-byte context together with the byte that followed its previous occurrence
  501       sparse: bytes -2, -3, -4 (skipping -1)
  502       sparse: bytes -3 .. -6 (skipping -1 and -2)
  503       sparse: bytes -2, -4, -6, -8
  601       record: the byte above (same column of the previous line), the byte above-left, and the left byte
  602       record: the column, the byte above, and the left byte
Every key hashes bytes before the position only. One leak-free pass per key with the byte tables' rule (a chunk's lookups before its
writes, chunk boundaries cut at train_end, positions past train_end read the frozen table). CPU only, some 10 GB of RAM, about half an hour."""
import glob
import json
import os
import sys
import time

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
sys.path.insert(0, SPEED)
from flybrain.corpus_memory import CorpusMemory, _finalise  # noqa: E402

CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
MEM = SPEED + "/results/mem"
BASE, OUT = MEM + "/feat_all", MEM + "/feat_paq"
TRAIN_END, CHUNK = 200_000_000, 1 << 20
LOOKUP_ORDERS = (4, 8, 12, 16, 32, 64, 116, 132, 202, 203, 301)
KEY_IDS = (401, 402, 501, 502, 503, 601, 602)
M = np.uint64(0x9E3779B97F4A7C15 | 1)
NONE = np.uint64(256)                                    # the sentinel for "no such byte"


def refuse_if_in_use(directory):
    """A feature file that a running process has memory-mapped must not be rewritten (SIGBUS in that process, 2026-10-07)."""
    real = {os.path.realpath(f) for f in glob.glob(f"{directory}/*.npy")}
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


t0 = time.time()
refuse_if_in_use(OUT)
os.makedirs(OUT, exist_ok=True)
data = np.fromfile(CORPUS, dtype=np.uint8)
N = len(data)
pos = np.arange(N, dtype=np.int64)


def shifted(k):
    """data[p - k] as uint64, the sentinel where p - k < 0."""
    out = np.full(N, NONE, dtype=np.uint64)
    out[k:] = data[:-k]
    return out


def h(parts, kid):
    acc = np.zeros(N, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for x in parts:
            acc = acc * M + (np.asarray(x, dtype=np.uint64) + np.uint64(1))
        return _finalise(acc * np.uint64(0xC2B2AE3D27D4EB4F) + np.uint64(kid))


def prev_occurrence(values):
    """For each position p, the previous position p' < p with values[p'] == values[p], or -1."""
    order = np.argsort(values, kind="stable")
    sv = values[order]
    prev = np.full(N, -1, dtype=np.int64)
    same = sv[1:] == sv[:-1]
    prev[order[1:][same]] = order[:-1][same]
    return prev


def at(idx):
    """data[idx] as uint64, the sentinel where idx < 0."""
    out = np.full(N, NONE, dtype=np.uint64)
    ok = idx >= 0
    out[ok] = data[idx[ok]]
    return out


keys = {}
# indirect: the context ending at p and the byte that followed that context's previous occurrence (at index p' <= p - 1: before p)
v1 = np.full(N, 256, dtype=np.uint16); v1[1:] = data[:-1]
prev = prev_occurrence(v1)
keys[401] = h([v1, at(prev)], 401)
print(f"key 401 ready ({time.time() - t0:.0f} s)", flush=True)
v2 = np.full(N, 65536, dtype=np.uint32); v2[2:] = data[:-2].astype(np.uint32) * 256 + data[1:-1]
prev = prev_occurrence(v2)
keys[402] = h([v2, at(prev)], 402)
del v1, v2, prev
print(f"key 402 ready ({time.time() - t0:.0f} s)", flush=True)
# sparse
keys[501] = h([shifted(2), shifted(3), shifted(4)], 501)
keys[502] = h([shifted(3), shifted(4), shifted(5), shifted(6)], 502)
keys[503] = h([shifted(2), shifted(4), shifted(6), shifted(8)], 503)
print(f"sparse keys ready ({time.time() - t0:.0f} s)", flush=True)
# record: the previous line's byte in the same column
start_mask = np.concatenate([[True], data[:-1] == 10])
line_start = np.maximum.accumulate(np.where(start_mask, pos, 0))
starts_arr = pos[start_mask]
line_idx = np.cumsum(start_mask) - 1
prev_start = np.where(line_idx > 0, starts_arr[np.maximum(line_idx - 1, 0)], -1)
col = pos - line_start
above = prev_start + col
valid = (prev_start >= 0) & (above < line_start - 1)                      # inside the previous line, before its newline
above_idx = np.where(valid, above, -1)
above_left_idx = np.where(valid & (col > 0), above - 1, -1)
left = shifted(1)
keys[601] = h([at(above_left_idx), at(above_idx), left], 601)
keys[602] = h([col, at(above_idx), left], 602)
del start_mask, line_start, starts_arr, line_idx, prev_start, col, above, valid, above_idx, above_left_idx, left, pos
print(f"record keys ready ({time.time() - t0:.0f} s)", flush=True)

for n in LOOKUP_ORDERS:
    for kind in ("pred", "conf"):
        dst = f"{OUT}/{kind}_n{n}.npy"
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(f"{BASE}/{kind}_n{n}.npy"), dst)
bounds = sorted(set(range(0, N, CHUNK)) | {TRAIN_END, N})
stats = {}
for kid in KEY_IDS:
    H = keys[kid]
    t1 = time.time()
    mem = CorpusMemory(orders=(kid,))
    pred = np.zeros(N, dtype=np.uint8); conf = np.zeros(N, dtype=np.uint8)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        hs = {kid: H[lo:hi]}
        p, c = mem.lookup(data, lo, hi, hashes=hs)
        pred[lo:hi], conf[lo:hi] = p[0], c[0]
        if hi <= TRAIN_END:
            mem.update(data, lo, hi, hashes=hs)
    np.save(f"{OUT}/pred_n{kid}.npy", pred); np.save(f"{OUT}/conf_n{kid}.npy", conf)
    held = slice(TRAIN_END, N)
    hit = conf[held] > 0
    stats[kid] = {"occupancy": mem.occupancy()[kid], "heldout_hit_rate": float(hit.mean()),
                  "heldout_right_given_hit": float((pred[held][hit] == data[held][hit]).mean()) if hit.any() else 0.0}
    print(f"key {kid}: occupancy {stats[kid]['occupancy']:.2f}, held-out hits {stats[kid]['heldout_hit_rate']:.3f} (right {stats[kid]['heldout_right_given_hit']:.3f}); {time.time() - t1:.0f} s", flush=True)
    del mem, pred, conf, keys[kid]
info = json.load(open(f"{MEM}/feat/features.json"))
info.update(orders=[*LOOKUP_ORDERS, *KEY_IDS], key_stats=stats,
            note="paq_keys.py: the combined arm's eleven lookup orders (linked from feat_all) plus PAQ-style keys 401/402 (indirect), 501/502/503 (sparse), 601/602 (record); the same leak-free one-pass build")
info.pop("feature_digest", None); info.pop("compare", None)
json.dump(info, open(f"{OUT}/features.json", "w"), indent=2)
print(f"done in {time.time() - t0:.0f} s")
