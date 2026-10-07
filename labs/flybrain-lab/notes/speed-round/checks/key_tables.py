"""Tables keyed by things that repeat other than exact bytes, computed once over the corpus (Kyle: "is there anything else that
repeats that we could try to compute once?"). Writes results/mem/feat_keys: the base byte-table features (orders 8, 16, 32, linked
from results/mem/feat) plus one pseudo-order per new key, so one --mem-feat directory feeds all of them:
  116, 132  template: the last 16 / 32 symbols of the byte stream with every completed identifier run ([A-Za-z0-9_]+) replaced by one
            placeholder symbol, whitespace kept; plus the current partial identifier (raw, last 32 chars) when the context ends inside one;
  202, 203  words: the last 2 / 3 tokens (identifier runs by content, other non-whitespace bytes one token each; whitespace dropped),
            plus the current partial identifier;
  301       lines: the previous full line and the current line so far (each clipped to its last 256 bytes).
Every key is a 64-bit hash handed to CorpusMemory as the precomputed hash of that pseudo-order; the tables are written in one pass with
the same leak-free rule as the byte tables (a chunk's lookups are made before the chunk is written; positions past train_end read the
frozen table; chunk boundaries are cut at train_end). Hashes are polynomial in wrapping uint64 arithmetic with an odd multiplier, so
spans come from prefix sums and the multiplier's inverse. The key pass (about 26 minutes) is saved to key_n<id>.npy so a failed table
pass restarts from it; the table pass deletes those files when done unless --keep-keys. CPU only; some 10 GB of RAM."""
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
BASE, OUT = MEM + "/feat", MEM + "/feat_keys"
TRAIN_END, CHUNK = 200_000_000, 1 << 20
BASE_ORDERS = (8, 16, 32)
KEY_IDS = (116, 132, 202, 203, 301)
M_INT = 0x9E3779B97F4A7C15 | 1
M = np.uint64(M_INT)
M_INV = np.uint64(pow(M_INT, -1, 1 << 64))
PLACEHOLDER = np.uint64(0x1D)

t0 = time.time()
data = np.fromfile(CORPUS, dtype=np.uint8)
N = len(data)


def compute_keys():
    pos = np.arange(N, dtype=np.int64)
    isid = ((data >= 48) & (data <= 57)) | ((data >= 65) & (data <= 90)) | ((data >= 97) & (data <= 122)) | (data == 95)
    ws = (data == 32) | (data == 9) | (data == 10) | (data == 13)
    prev_id = np.concatenate([[False], isid[:-1]])            # byte q-1 is an identifier char
    run_end = ~isid & prev_id                                  # q is the first non-identifier byte after a run
    run_start = np.where(isid, np.maximum.accumulate(np.where(isid & ~prev_id, pos, 0)), -1)   # start of the run holding q (-1 outside runs)
    with np.errstate(over="ignore"):
        Mp = np.cumprod(np.full(N + 1, M, dtype=np.uint64), dtype=np.uint64); Mp[1:] = Mp[:-1]; Mp[0] = 1     # M^q
        Mi = np.cumprod(np.full(N + 1, M_INV, dtype=np.uint64), dtype=np.uint64)                              # M^-(q+1)
        W = np.zeros(N + 1, dtype=np.uint64)
        np.cumsum((data.astype(np.uint64) + np.uint64(1)) * Mi[:N], dtype=np.uint64, out=W[1:])              # sum_{j<q} (b_j+1) M^-(j+1)
    del Mi

    def span_hash(a, b):
        """Hash of bytes [a, b) (arrays, a <= b): sum_{j} (b_j + 1) M^(b-1-j) = M^b (W[b] - W[a]); 0 for an empty span."""
        with np.errstate(over="ignore"):
            return Mp[b] * (W[b] - W[a])

    print(f"prefix hashes of {N:,} bytes in {time.time() - t0:.0f} s", flush=True)
    rs_prev = np.concatenate([[0], run_start[:-1]])                                 # run_start[p-1]
    h_partial = span_hash(np.where(prev_id, np.maximum(rs_prev, pos - 32), pos), pos)   # the partial identifier before p (empty = 0)

    def stream_key(run_sym, byte_sym, emits, k_list, names):
        """Symbols emitted at positions q, completed for p > q: at a run end the run's symbol first, then the byte's own symbol where
        emits[q]. Keys: the last k completed symbols before p, combined with the partial identifier."""
        cnt = run_end.astype(np.int64) + emits.astype(np.int64)
        off = np.zeros(N + 1, dtype=np.int64); np.cumsum(cnt, out=off[1:])
        S = np.empty(off[-1], dtype=np.uint64)
        S[off[:-1][run_end]] = run_sym[run_end]
        S[(off[:-1] + run_end)[emits]] = byte_sym[emits]
        n_before = off[:-1]                                   # events at positions < p
        out = {}
        with np.errstate(over="ignore"):
            for k, name in zip(k_list, names):
                h = np.zeros(N, dtype=np.uint64)
                for j in range(k):
                    idx = n_before - 1 - j
                    ok = idx >= 0
                    h[ok] = h[ok] * M + (S[idx[ok]] + np.uint64(1))
                out[name] = _finalise(h * np.uint64(0xC2B2AE3D27D4EB4F) + h_partial + np.uint64(k))
                print(f"  key {name}: ready ({time.time() - t0:.0f} s)", flush=True)
        return out

    byte_sym = data.astype(np.uint64)
    print("template keys", flush=True)
    keys = stream_key(np.full(N, PLACEHOLDER, dtype=np.uint64), byte_sym, ~isid, (16, 32), (116, 132))
    print("word keys", flush=True)
    run_sym = np.zeros(N, dtype=np.uint64)
    run_sym[run_end] = span_hash(rs_prev[run_end], pos[run_end])                    # the completed run [run_start, q) at its end q
    keys.update(stream_key(run_sym, byte_sym, ~isid & ~ws, (2, 3), (202, 203)))
    del run_sym, byte_sym, rs_prev
    print("line key", flush=True)
    start_mask = np.concatenate([[True], data[:-1] == 10])
    line_start = np.maximum.accumulate(np.where(start_mask, pos, 0))               # first byte of the line holding p
    starts_arr = pos[start_mask]
    line_idx = np.cumsum(start_mask) - 1
    prev_start = starts_arr[np.maximum(line_idx - 1, 0)]
    prev_start = np.where(line_idx > 0, prev_start, 0)
    pl_hi = np.where(line_idx > 0, line_start - 1, 0)                                # the previous line without its newline
    pl_lo = np.maximum(prev_start, pl_hi - 256)
    with np.errstate(over="ignore"):
        keys[301] = _finalise(span_hash(pl_lo, pl_hi) * np.uint64(0x9E3779B97F4A7C15) + span_hash(np.maximum(line_start, pos - 256), pos) * np.uint64(0xBF58476D1CE4E5B9) + np.uint64(301))
    print(f"keys ready ({time.time() - t0:.0f} s)", flush=True)
    return keys


def refuse_if_in_use(directory):
    """A feature file that a running process has memory-mapped must not be rewritten (np.save truncates it: the process dies of SIGBUS,
    as a training run did on 2026-10-07). Checks every process's maps for files of the directory, by their real paths."""
    import glob
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


refuse_if_in_use(OUT)
os.makedirs(OUT, exist_ok=True)
if all(os.path.exists(f"{OUT}/key_n{kid}.npy") for kid in KEY_IDS):
    keys = {kid: np.load(f"{OUT}/key_n{kid}.npy") for kid in KEY_IDS}
    print("keys loaded from disk", flush=True)
else:
    keys = compute_keys()
    for kid, H in keys.items():
        np.save(f"{OUT}/key_n{kid}.npy", H)
    print(f"keys saved ({time.time() - t0:.0f} s)", flush=True)

for n in BASE_ORDERS:
    for kind in ("pred", "conf"):
        dst = f"{OUT}/{kind}_n{n}.npy"
        if not os.path.exists(dst):
            os.symlink(f"{BASE}/{kind}_n{n}.npy", dst)
bounds = sorted(set(range(0, N, CHUNK)) | {TRAIN_END, N})                        # chunk boundaries, cut at train_end
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
    del mem, pred, conf
info = json.load(open(f"{BASE}/features.json"))
info.update(orders=[*BASE_ORDERS, *KEY_IDS], key_stats=stats,
            note="key_tables.py: the byte-table features of orders 8, 16, 32 (linked from feat/) plus pseudo-orders 116/132 (template 16/32 symbols), 202/203 (last 2/3 words), 301 (previous line + current line so far); the same leak-free one-pass build")
info.pop("feature_digest", None); info.pop("compare", None)
json.dump(info, open(f"{OUT}/features.json", "w"), indent=2)
if "--keep-keys" not in sys.argv:                 # --keep-keys: leave the 8.8 GB of hashes for other table builds (assoc_all.py)
    for kid in KEY_IDS:
        os.remove(f"{OUT}/key_n{kid}.npy")
print(f"done in {time.time() - t0:.0f} s")
