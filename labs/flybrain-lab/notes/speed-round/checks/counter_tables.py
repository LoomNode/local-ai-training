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
dead weight rather than a leak. CPU only.

Parameters (environment; the defaults are the original constants): CORPUS, TRAIN_END, MEM (holds feat/), OUT (default
MEM/feat_counters), CHUNK (default 2^25: the size whole files are grouped into), JOBS (worker processes, default 4). Without MEM/feat (a
new corpus) nothing is linked and features.json is written from scratch.

Chunked, so RAM is bounded at any corpus size (about 65 bytes per chunk byte per worker: 4.3 GB at 64 MB, so 2.2 GB at the default 32 MB and some 9 GB for 4 workers), with the same bits as the original
whole-corpus arrays. Chunks are whole files, cut at file headers (table_common.py's cut points: the two bytes before the header's "/"
are newlines), and each chunk is computed as a corpus of its own: every counter reads at most the current and previous line and the
file since its header, except the string/comment state. The original state machine never resets at a header, so an unterminated block
comment carries over into the next file; strings and line comments cannot (they end at a newline, and a cut point follows two). Every
chunk is first computed in parallel starting in code state; then, in corpus order, a chunk whose predecessor actually ended inside a
block comment is computed again starting in that state (and its own exit state is carried on). The result is exact; the number of
positions the restart changed (where a reset at each chunk would have differed) is printed and stored in features.json as
"state_carry_positions". Workers write disjoint slices of the output memmaps (opened r+)."""
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_common import SPEED, base_info, chunk_bounds, env_int, env_str, file_cuts, link_base, refuse_if_in_use  # noqa: E402

CORPUS = env_str("CORPUS", "/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes")
MEM = env_str("MEM", SPEED + "/results/mem")
BASE, OUT = MEM + "/feat", env_str("OUT", MEM + "/feat_counters")
TRAIN_END = env_int("TRAIN_END", 200_000_000)
CHUNK, JOBS = env_int("CHUNK", 1 << 25), env_int("JOBS", 4)
BASE_ORDERS = (8, 16, 32)
IDS = (1001, 1002, 1003, 1004, 1005, 1006, 1007)
CODE, STRING, CHAR, BLOCK, LINE = range(5)
NL, DQ, SQ, BS, BE, LC = range(6)

t0 = time.time()


def counters(data, entry=CODE):
    """The seven counters of every position of `data` read as a corpus of its own, the state machine starting in state `entry` (held
    since before the first byte). Returns (values, exit state)."""
    N = len(data)
    pos = np.arange(N, dtype=np.int32)                                              # positions fit int32
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
    ev_pos = np.concatenate([pos[nl], pos[dq], pos[sq], pos[block_start] + 1, pos[block_end] + 1, pos[line_cmt] + 1])
    ev_kind = np.concatenate([np.full(int(m.sum()), k, dtype=np.int8) for m, k in ((nl, NL), (dq, DQ), (sq, SQ), (block_start, BS), (block_end, BE), (line_cmt, LC))])
    order = np.argsort(ev_pos, kind="stable")
    ev_pos, ev_kind = ev_pos[order].tolist(), ev_kind[order].tolist()
    del order, nxt, block_start, block_end, line_cmt, dq, sq

    state, since = entry, -1                     # state holds for positions p > since (p >= since + 1), since = the byte that set it
    str_lo, str_hi, cmt_lo, cmt_hi = [], [], [], []       # intervals [lo, hi) of positions whose context ends inside a string / comment
    last_used = -2                               # the last byte consumed by a two-byte token, so "/*/" does not also end the comment

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
                str_lo.append(since + 1); str_hi.append(p + 1); state = CODE
        elif state == CHAR:
            if k in (SQ, NL):
                str_lo.append(since + 1); str_hi.append(p + 1); state = CODE
        elif state == BLOCK:
            if k == BE:
                cmt_lo.append(since + 1); cmt_hi.append(p + 1); state = CODE; last_used = p
        elif state == LINE:
            if k == NL:
                cmt_lo.append(since + 1); cmt_hi.append(p + 1); state = CODE
    if state != CODE:
        (str_lo if state in (STRING, CHAR) else cmt_lo).append(since + 1); (str_hi if state in (STRING, CHAR) else cmt_hi).append(N)

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
    return values, state


def counter_job(job):
    """One chunk [lo, hi) entered in state `entry`: its codes into the slices of the output memmaps. Returns (index, exit state, the
    chunk's sums for the stats, how many 1006/1007 positions this write changed)."""
    i, lo, hi, entry = job
    data = np.asarray(np.memmap(CORPUS, dtype=np.uint8, mode="r")[lo:hi])
    values, state = counters(data, entry)
    sums, changed = {}, 0
    h0 = max(TRAIN_END, lo) - lo
    for kid in IDS:
        v = np.asarray(values[kid])
        code_v = (128 + np.minimum(v, 127).astype(np.int16)).astype(np.uint8)
        a = np.load(f"{OUT}/pred_n{kid}.npy", mmap_mode="r+")
        if entry != CODE and kid in (1006, 1007):
            changed += int((a[lo:hi] != code_v).sum())
        a[lo:hi] = code_v; a.flush(); del a
        c = np.load(f"{OUT}/conf_n{kid}.npy", mmap_mode="r+")
        c[lo:hi] = 1; c.flush(); del c
        sums[kid] = (int(v.sum(dtype=np.int64)), int(v.max()), int((v > 0).sum()), int((v > 127).sum()), int(v[h0:].sum(dtype=np.int64)) if h0 < hi - lo else 0)
    return i, state, sums, changed


def main():
    N = os.path.getsize(CORPUS)
    refuse_if_in_use(OUT)
    os.makedirs(OUT, exist_ok=True)
    info, base_ok = base_info(BASE, CORPUS, N, TRAIN_END, CHUNK)
    linked = link_base(BASE, OUT, BASE_ORDERS, base_ok)
    cuts = file_cuts(CORPUS)
    bounds = chunk_bounds(cuts, N, CHUNK)
    chunks = list(zip(bounds[:-1], bounds[1:]))
    print(f"{len(cuts):,} file headers, {len(chunks):,} chunks, {JOBS} workers ({time.time() - t0:.0f} s)", flush=True)
    for kid in IDS:
        for kind in ("pred", "conf"):
            a = np.lib.format.open_memmap(f"{OUT}/{kind}_n{kid}.npy", mode="w+", dtype=np.uint8, shape=(N,))
            del a
    results = {}
    with get_context("fork").Pool(JOBS) as pool:
        for i, state, sums, _ in pool.imap_unordered(counter_job, [(i, lo, hi, CODE) for i, (lo, hi) in enumerate(chunks)]):
            results[i] = (state, sums)
            if len(results) % 20 == 0:
                print(f"  {len(results):,} / {len(chunks):,} chunks ({time.time() - t0:.0f} s)", flush=True)
    # carry the string/comment state across chunks: redo, in order, every chunk entered in another state than code
    carried, reruns, prev = 0, 0, CODE
    for i, (lo, hi) in enumerate(chunks):
        if prev != CODE:
            _, state, sums, changed = counter_job((i, lo, hi, prev))
            results[i] = (state, sums); carried += changed; reruns += 1
        prev = results[i][0]
    print(f"state carried into {reruns} chunk(s): {carried:,} string/comment positions changed by it ({time.time() - t0:.0f} s)", flush=True)

    stats = {}
    n_held = N - TRAIN_END
    for kid in IDS:
        s = [results[i][1][kid] for i in range(len(chunks))]
        stats[kid] = {"mean": sum(x[0] for x in s) / N, "max": max(x[1] for x in s), "share_nonzero": sum(x[2] for x in s) / N,
                      "share_clipped": sum(x[3] for x in s) / N, "heldout_mean": sum(x[4] for x in s) / n_held if n_held > 0 else 0.0}
        print(f"counter {kid}: mean {stats[kid]['mean']:.2f}, max {stats[kid]['max']}, nonzero {stats[kid]['share_nonzero']:.3f}, clipped {stats[kid]['share_clipped']:.5f}", flush=True)
    # a spot check a reader can repeat: the first held-out line with a brace, printed with its counters (codes - 128: clipped at 127)
    data = np.memmap(CORPUS, dtype=np.uint8, mode="r")
    if TRAIN_END < N:
        q = TRAIN_END + int(np.argmax((data[TRAIN_END:TRAIN_END + 100_000] == 123)))
        ls = q - int(np.argmax(data[max(0, q - 100_000):q][::-1] == 10)) if (data[max(0, q - 100_000):q] == 10).any() else max(0, q - 100_000)
        le = int(q + np.argmax(data[q:q + 300] == 10))
        print("sample line:", data[ls:le].tobytes().decode("latin1"))
        for name, kid in (("column", 1001), ("brace", 1004), ("paren", 1005), ("string", 1006), ("comment", 1007)):
            v = np.load(f"{OUT}/pred_n{kid}.npy", mmap_mode="r")
            print(f"  {name:9s}", " ".join(str(int(v[i]) - 128) for i in range(ls, min(le, ls + 40))))
    info.update(orders=[*linked, *IDS], counter_stats=stats, state_carry_positions=carried, state_carry_chunks=reruns,
                note="counter_tables.py: the byte-table features of orders 8, 16, 32 (linked from feat/) plus per-position counters as pseudo-orders 1001-1007 (column, indent, previous indent, brace depth, paren depth, in-string, in-comment), pred = 128 + min(value, 127), conf = 1")
    info.pop("feature_digest", None); info.pop("compare", None)
    json.dump(info, open(f"{OUT}/features.json", "w"), indent=2)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
