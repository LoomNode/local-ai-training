"""Counters the model otherwise has to learn to keep, computed once per corpus position and served through the table channel
(item 5 of the "what else repeats" list). Writes results/mem/feat_counters: the byte-table features of orders 8, 16, 32 (linked from
results/mem/feat) plus one pseudo-order per counter, each a function of the bytes before the position (nothing from the byte itself):
  1001  column: bytes since the last newline
  1002  indentation of the current line so far (leading tabs and spaces, one each)
  1003  indentation of the previous line
  1004  brace depth { } inside the file, counted outside strings and comments
  1005  parenthesis depth ( ) inside the file, likewise
  1006  inside a string or character literal (0 / 1)
  1007  inside a comment, block or line (0 / 1)
Strings and comments come from one state machine over the quote, slash-star, star-slash, slash-slash and newline events (unescaped
quotes only; a string ends at its quote or at the end of its line). Depths reset at the corpus's file headers ("// ==== path ====").
Encoding for the channel: pred = 128 + min(value, 127), conf = 1 everywhere. The channel's byte embedding then learns one vector per
value; its output bonus lands on the logits of bytes 128..255, which are the right answer at 0.002 % of positions, so that pathway is
dead weight rather than a leak. CPU only, a few minutes."""
import json
import os
import sys
import time

import numpy as np

SPEED = "/games/ailab/lab/flybrain-lab/.worktrees/speed"
CORPUS = "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes"
MEM = SPEED + "/results/mem"
BASE, OUT = MEM + "/feat", MEM + "/feat_counters"
TRAIN_END = 200_000_000
BASE_ORDERS = (8, 16, 32)
IDS = (1001, 1002, 1003, 1004, 1005, 1006, 1007)

t0 = time.time()
data = np.fromfile(CORPUS, dtype=np.uint8)
N = len(data)
pos = np.arange(N, dtype=np.int32)                                              # positions fit int32: keeps the pass near 12 GB
nl = data == 10
start_mask = np.concatenate([[True], nl[:-1]])
line_start = np.maximum.accumulate(np.where(start_mask, pos, 0))                  # first byte of the line holding p
starts_arr = pos[start_mask]
line_idx = np.cumsum(start_mask, dtype=np.int32) - 1
prev_start = np.where(line_idx > 0, starts_arr[np.maximum(line_idx - 1, 0)], 0)   # first byte of the previous line (0 on the first line)

# unescaped quotes: an odd run of backslashes before the quote escapes it
bs = data == 92
last_nonbs = np.maximum.accumulate(np.where(~bs, pos, -1))
bsrun = pos - 1 - np.concatenate([[-1], last_nonbs[:-1]])                        # backslashes immediately before p
escaped = (bsrun & 1) == 1
del bs, last_nonbs, bsrun
dq = (data == 34) & ~escaped
sq = (data == 39) & ~escaped
del escaped
nxt = np.concatenate([data[1:], [0]])
block_start = (data == 47) & (nxt == 42)                                           # "/*" at p, p + 1: the state changes after byte p + 1
block_end = (data == 42) & (nxt == 47)
line_cmt = (data == 47) & (nxt == 47)

# events, in corpus order: (position of the byte after which the state changes, kind)
NL, DQ, SQ, BS, BE, LC = range(6)
ev_pos = np.concatenate([pos[nl], pos[dq], pos[sq], pos[block_start] + 1, pos[block_end] + 1, pos[line_cmt] + 1])
ev_kind = np.concatenate([np.full(int(m.sum()), k, dtype=np.int8) for m, k in ((nl, NL), (dq, DQ), (sq, SQ), (block_start, BS), (block_end, BE), (line_cmt, LC))])
order = np.argsort(ev_pos, kind="stable")
ev_pos, ev_kind = ev_pos[order].tolist(), ev_kind[order].tolist()
del order, nxt, block_start, block_end, line_cmt, dq, sq
print(f"{len(ev_pos):,} events ({time.time() - t0:.0f} s)", flush=True)

CODE, STRING, CHAR, BLOCK, LINE = range(5)
state, since = CODE, 0                       # state holds for positions p > since (p >= since + 1), since = the byte that set it
str_lo, str_hi, cmt_lo, cmt_hi = [], [], [], []       # intervals [lo, hi) of positions whose context ends inside a string / comment
last_used = -2                               # the last byte consumed by a two-byte token, so "/*/" does not also end the comment


def leave(p, lo, hi):
    """Close the current non-code state at event byte p: positions since + 1 .. p are inside it."""
    lo.append(since + 1); hi.append(p + 1)


for p, k in zip(ev_pos, ev_kind):
    if k in (BS, BE, LC):
        if p - 1 <= last_used:                # this two-byte token shares a byte with the previous one
            continue
    if state == CODE:
        if k == DQ:
            state, since = STRING, p
        elif k == SQ:
            state, since = CHAR, p
        elif k == BS:
            state, since, last_used = BLOCK, p, p
        elif k == LC:
            state, since, last_used = LINE, p, p
    elif state == STRING:
        if k in (DQ, NL):
            leave(p, str_lo, str_hi); state = CODE
    elif state == CHAR:
        if k in (SQ, NL):
            leave(p, str_lo, str_hi); state = CODE
    elif state == BLOCK:
        if k == BE:
            leave(p, cmt_lo, cmt_hi); state = CODE; last_used = p
    elif state == LINE:
        if k == NL:
            leave(p, cmt_lo, cmt_hi); state = CODE
if state != CODE:
    (str_lo if state in (STRING, CHAR) else cmt_lo).append(since + 1); (str_hi if state in (STRING, CHAR) else cmt_hi).append(N)
print(f"state machine: {len(str_lo):,} strings, {len(cmt_lo):,} comments ({time.time() - t0:.0f} s)", flush=True)


def fill(lo, hi):
    d = np.zeros(N + 1, dtype=np.int32)
    np.add.at(d, np.asarray(lo, dtype=np.int64), 1); np.add.at(d, np.asarray(hi, dtype=np.int64), -1)
    return np.cumsum(d[:N], dtype=np.int32) > 0


in_string = fill(str_lo, str_hi)
in_comment = fill(cmt_lo, cmt_hi)
del str_lo, str_hi, cmt_lo, cmt_hi, ev_pos, ev_kind
code = ~in_string & ~in_comment                                                    # the byte at p is read in code state

# file headers: lines starting with "// ==== "
hdr = start_mask.copy()
for j, ch in enumerate(b"// ==== "):
    hdr &= np.concatenate([data[j:], np.zeros(j, dtype=np.uint8)]) == ch
file_start = np.maximum.accumulate(np.where(hdr, pos, 0))
print(f"{int(hdr.sum()):,} file headers ({time.time() - t0:.0f} s)", flush=True)


def depth(open_byte, close_byte):
    """Count of opens minus closes among the code bytes before p, since the file's header line; clipped to 0 .. 127."""
    d = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(((data == open_byte) & code).astype(np.int32) - ((data == close_byte) & code).astype(np.int32), out=d[1:])
    return np.clip(d[:N] - d[file_start], 0, 127)


hws = (data == 32) | (data == 9)
C = np.zeros(N + 1, dtype=np.int32); np.cumsum(~hws & ~nl, out=C[1:])            # non-whitespace bytes in [0, q)
lead = hws & (C[pos + 1] - C[line_start] == 0)                                     # p is in its line's leading whitespace
L = np.zeros(N + 1, dtype=np.int32); np.cumsum(lead, out=L[1:])
values = {
    1001: pos - line_start,
    1002: L[pos] - L[line_start],
    1003: np.where(line_idx > 0, L[line_start] - L[prev_start], 0),
    1004: depth(123, 125),
    1005: depth(40, 41),
    1006: in_string.astype(np.int8),
    1007: in_comment.astype(np.int8),
}
del C, L, lead, hws, code, file_start, hdr, line_idx, prev_start, starts_arr
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
for n in BASE_ORDERS:
    for kind in ("pred", "conf"):
        dst = f"{OUT}/{kind}_n{n}.npy"
        if not os.path.exists(dst):
            os.symlink(f"{BASE}/{kind}_n{n}.npy", dst)
stats = {}
held = slice(TRAIN_END, N)
for kid in IDS:
    v = np.asarray(values[kid])
    code_v = (128 + np.minimum(v, 127).astype(np.int16)).astype(np.uint8)
    np.save(f"{OUT}/pred_n{kid}.npy", code_v)
    np.save(f"{OUT}/conf_n{kid}.npy", np.ones(N, dtype=np.uint8))
    stats[kid] = {"mean": float(v.mean()), "max": int(v.max()), "share_nonzero": float((v > 0).mean()), "share_clipped": float((v > 127).mean()),
                  "heldout_mean": float(v[held].mean())}
    print(f"counter {kid}: mean {stats[kid]['mean']:.2f}, max {stats[kid]['max']}, nonzero {stats[kid]['share_nonzero']:.3f}, clipped {stats[kid]['share_clipped']:.5f}", flush=True)
# a spot check a reader can repeat: the first held-out line with a brace, printed with its counters
q = TRAIN_END + int(np.argmax((data[TRAIN_END:TRAIN_END + 100_000] == 123)))
ls = int(line_start[q]); le = int(q + np.argmax(nl[q:q + 300]))
print("sample line:", data[ls:le].tobytes().decode("latin1"))
print("  column   ", " ".join(str(int(values[1001][i])) for i in range(ls, min(le, ls + 40))))
print("  brace    ", " ".join(str(int(values[1004][i])) for i in range(ls, min(le, ls + 40))))
print("  paren    ", " ".join(str(int(values[1005][i])) for i in range(ls, min(le, ls + 40))))
print("  string   ", " ".join(str(int(values[1006][i])) for i in range(ls, min(le, ls + 40))))
print("  comment  ", " ".join(str(int(values[1007][i])) for i in range(ls, min(le, ls + 40))))
info = json.load(open(f"{BASE}/features.json"))
info.update(orders=[*BASE_ORDERS, *IDS], counter_stats=stats,
            note="counter_tables.py: the byte-table features of orders 8, 16, 32 (linked from feat/) plus per-position counters as pseudo-orders 1001-1007 (column, indent, previous indent, brace depth, paren depth, in-string, in-comment), pred = 128 + min(value, 127), conf = 1")
info.pop("feature_digest", None); info.pop("compare", None)
json.dump(info, open(f"{OUT}/features.json", "w"), indent=2)
print(f"done in {time.time() - t0:.0f} s")
