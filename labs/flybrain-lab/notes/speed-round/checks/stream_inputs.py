"""Per-byte inputs for the streaming reader (the use layer's input caches, 2026-10-14): the 18 table inputs and the match features
of the next byte from a state carried byte to byte, instead of recomputed over the whole text at every byte (stream_reader.py's first
version: about 4 ms a byte at a few KB of text and growing with it; the network itself is 6 ms on six cores). TextState holds what the
build's functions hold at a position: the last 64 bytes (the byte orders' hashes), the last 32 template symbols and 3 word tokens, the
current identifier run, the current and previous line, and the counters' machine (strings, comments, the depths since the file
header, indentation). push(byte) updates it and looks the keys up in the saved two-way tables: row n + 1 of the inputs in
text_features's format and exactly its values (tests/test_stream_inputs.py checks every position of a text with headers, strings,
comments, unbalanced files and long runs against use_features.text_features, which runs the build's own code). One lookahead exists
in the build: a file header line ("// ==== path ====") resets the brace and paren depths from its first byte, which is known only
eight bytes later; push() then corrects the rows of those eight positions and returns the first corrected row (the header's line
start), and the reader recomputes those positions (Reader.feed does; a header met at depth 0, the usual case, changes nothing).

A long append (extend: a prompt, a file's prefix) takes the vectorised rows and warms the state by replaying only the tail: from the
latest line start that leaves at least 32 symbols, 3 word tokens and one newline before the end (the key pass's halo argument,
key_tables.py: a line start is a cut point, and runs, lines and the symbol stream from there on are the global ones); the counters'
state at that line start comes from the rows and the text (in a block comment or in code, the unclipped depths since the file header,
the previous line's indentation). With no such line the replay starts at the text's start. The replay updates the state only (no
lookups): a few microseconds a byte.

MatchState is the in-window match lookup (flybrain.model.inputs.match_features: for orders 8, 16 and 32, the byte after the latest
earlier index of the window with the same context, and a class by how many there were) as one dictionary per order keyed by the
context bytes themselves (match_features keys by a 62-bit hash; the two agree unless the hash collides, about 5 in 10^13 windows).
start(pos0) opens a window; push(t) gives the features of window index t; rows(pos0, hi) replays a whole window.

usage (the check on a real text): stream_inputs.py --tables DIR --check FILE [--bytes N]: every row of a cold replay against
text_features, then warm starts at several cuts, with the bytes per second of each path."""
import argparse
import os
import sys
import time
from collections import deque
from itertools import islice

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_common import SPEED  # noqa: E402

sys.path.insert(0, SPEED)
from flybrain.corpus_memory import _FNV_OFFSET, _FNV_PRIME, _MIX1, _MIX2, _SALT  # noqa: E402
from counter_tables import IDS as COUNTER_IDS  # noqa: E402
from key_tables import KEY_IDS, M_INT, M_INV as _M_INV, PLACEHOLDER as _PH  # noqa: E402
from use_features import BYTE_ORDERS, HEADER, Tables, text_features  # noqa: E402

MASK = (1 << 64) - 1
M, M_INV, PH = M_INT, int(_M_INV), int(_PH)
K_TMPL, K_L1, K_L2 = 0xC2B2AE3D27D4EB4F, 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9      # key_tables.py's mixing constants
CODE, STRING, CHAR, BLOCK, LINE = range(5)
MARK = HEADER[1:]                                        # b"// ==== ": a header line's first eight bytes
ISID = [48 <= b <= 57 or 65 <= b <= 90 or 97 <= b <= 122 or b == 95 for b in range(256)]
ISWS = [b in (32, 9, 10, 13) for b in range(256)]
ISID_ARR, ISWS_ARR = np.array(ISID), np.array(ISWS)
ALL18 = (*BYTE_ORDERS, *KEY_IDS, *COUNTER_IDS)


def finalise(h: int) -> int:
    """corpus_memory._finalise on one value."""
    h ^= h >> 33; h = (h * _MIX1) & MASK; h ^= h >> 33; h = (h * _MIX2) & MASK
    return h ^ (h >> 33)


class TextState:
    """The text so far, the rows of its inputs (pred, conf: uint8 [n + 1, len(orders)], row p the inputs of position p), and the state
    that gives the next row."""

    def __init__(self, tables: Tables, orders, cap: int = 1 << 16):
        self.tables, self.orders = tables, [int(o) for o in orders]
        unknown = [n for n in self.orders if n not in BYTE_ORDERS and n not in KEY_IDS and n not in COUNTER_IDS]
        missing = [n for n in self.orders if (n in BYTE_ORDERS or n in KEY_IDS) and n not in tables.t]
        if unknown or missing:
            raise ValueError(f"orders {unknown} are unknown and {missing} have no saved table in {tables.dir}")
        self.bo = sorted(n for n in self.orders if n in BYTE_ORDERS)
        self.keys = [n for n in self.orders if n in KEY_IDS]
        self.cnt = [n for n in self.orders if n in COUNTER_IDS]
        self.col = {o: j for j, o in enumerate(self.orders)}
        self.shift = 64 - tables.log2
        self.text = bytearray()
        self.pred = np.zeros((cap, len(self.orders)), np.uint8); self.conf = np.zeros_like(self.pred)
        self._reset(0, CODE, 0, 0, 0)
        self._write(0)

    # --- state
    def _reset(self, base: int, state: int, dbrace: int, dparen: int, prev_indent: int) -> None:
        """The state at text position `base`, a line start: the machine's state, the unclipped depths since the file header and the
        previous line's indentation carried in; everything else is local from here (span hashes are position-independent)."""
        self.base = self.n = base
        self.W, self.Mp, self.mi = [0], [1], M_INV                         # W[i] = sum_{j<i} (b_j+1) M^-(j+1), Mp[i] = M^i, local i = p - base
        self.run_start, self.prev_id = -1, False
        self.sym, self.wrd = deque(maxlen=32), deque(maxlen=3)             # the last template symbols and word tokens, oldest first
        self.line_start, self.has_prev, self.prev_hash = base, base > 0, 0
        self.column, self.indent, self.in_lead, self.prev_indent = 0, 0, True, prev_indent
        self.db, self.dp, self.state = dbrace, dparen, state
        self.last_used, self.bsrun, self.hdr = base - 2, 0, 0

    def _span(self, a: int, b: int) -> int:
        """key_tables.py's span_hash of text [a, b): sum_j (b_j + 1) M^(b-1-j) mod 2^64; 0 for an empty span."""
        ia, ib = a - self.base, b - self.base
        return (self.Mp[ib] * ((self.W[ib] - self.W[ia]) & MASK)) & MASK

    def _advance(self, b: int, record: bool):
        """The byte `b` at text position n (already in self.text): the state after it, and (record) row n + 1. Returns the first row
        corrected by a header's depth reset, or None."""
        p = self.n; i = p - self.base; fix = None
        st = self.state
        if st == CODE:                                                     # the byte is read in the state in effect at its position
            if b == 123: self.db += 1
            elif b == 125: self.db -= 1
            elif b == 40: self.dp += 1
            elif b == 41: self.dp -= 1
        if b == 10:                                                        # the events at p, in counter_tables.py's order
            if st in (STRING, CHAR, LINE): st = CODE
        elif b == 34:
            if not (self.bsrun & 1):
                if st == CODE: st = STRING
                elif st == STRING: st = CODE
        elif b == 39:
            if not (self.bsrun & 1):
                if st == CODE: st = CHAR
                elif st == CHAR: st = CODE
        if p > 0 and p - 1 > self.last_used:                               # a two-byte token not sharing a byte with the last one used
            q = self.text[p - 1]
            if q == 47 and b == 42:
                if st == CODE: st = BLOCK; self.last_used = p
            elif q == 42 and b == 47:
                if st == BLOCK: st = CODE; self.last_used = p
            elif q == 47 and b == 47:
                if st == CODE: st = LINE; self.last_used = p
        self.state = st
        self.bsrun = self.bsrun + 1 if b == 92 else 0
        if b == 10:                                                        # the line [line_start, p) is complete
            self.prev_hash = self._span(max(self.line_start, p - 256), p)
            self.prev_indent, self.has_prev = self.indent, True
            self.line_start, self.column, self.indent, self.in_lead, self.hdr = p + 1, 0, 0, True, 0
        else:
            self.column += 1
            if b == 32 or b == 9:
                if self.in_lead: self.indent += 1
            else:
                self.in_lead = False
            if self.hdr >= 0:
                if b == MARK[self.hdr]:
                    self.hdr += 1
                    if self.hdr == len(MARK):
                        self.hdr = -1; fix = self._header(p)
                else:
                    self.hdr = -1
        if ISID[b]:                                                        # identifier runs; the symbol and token streams
            if not self.prev_id: self.run_start, self.prev_id = p, True
        else:
            if self.prev_id:
                self.sym.append(PH); self.wrd.append(self._span(self.run_start, p)); self.prev_id = False
            self.sym.append(b)
            if not ISWS[b]: self.wrd.append(b)
        self.W.append((self.W[i] + (b + 1) * self.mi) & MASK)
        self.mi = (self.mi * M_INV) & MASK
        self.Mp.append((self.Mp[i] * M) & MASK)
        self.n = p + 1
        if record:
            self._write(self.n)
        return fix

    def _header(self, p: int):
        """A header line recognised at its eighth byte p: the depths count from its line start (counter_tables.py's file_start), so the
        rows of the line start and the seven bytes after it, written with the old depths, are corrected. Returns the first changed row."""
        self.db = self.dp = 0
        q0, first = p - len(MARK) + 1, None
        for o in (1004, 1005):
            j = self.col.get(o)
            if j is not None and q0 + len(MARK) <= len(self.pred) and (self.pred[q0:q0 + len(MARK), j] != 128).any():
                self.pred[q0:q0 + len(MARK), j] = 128; first = q0
        return first

    # --- rows
    def _grow(self, need: int) -> None:
        if need > len(self.pred):
            extra = max(need, 2 * len(self.pred)) - len(self.pred)
            self.pred = np.concatenate([self.pred, np.zeros((extra, self.pred.shape[1]), np.uint8)])
            self.conf = np.concatenate([self.conf, np.zeros((extra, self.conf.shape[1]), np.uint8)])

    def _lookup(self, table, h: int):
        """check_tables.lookup for one key: the matching buddy of the two slots, else (0, 0)."""
        a = (h >> self.shift) & ~1; tag = h & 0xFFFF
        r = table[a]
        if r["conf"] > 0 and r["tag"] == tag:
            return int(r["byte"]), int(r["conf"])
        r = table[a + 1]
        if r["conf"] > 0 and r["tag"] == tag:
            return int(r["byte"]), int(r["conf"])
        return 0, 0

    def _write(self, q: int) -> None:
        """Row q, the inputs of position q = self.n from the state: the byte orders' lookups (orders n <= q only), the keys' lookups,
        the counters' codes (128 + min(value, 127), conf 1)."""
        self._grow(q + 1)
        pred, conf, text = self.pred[q], self.conf[q], self.text
        col = self.col
        if self.bo:
            h, j = _FNV_OFFSET, 0
            for n in self.bo:
                if n > q:
                    break
                while j < n:
                    j += 1; h = ((h ^ text[q - j]) * _FNV_PRIME) & MASK
                pred[col[n]], conf[col[n]] = self._lookup(self.tables.t[n], finalise(h ^ ((n * _SALT) & MASK)))
        if self.keys:
            hp = self._span(max(self.run_start, q - 32), q) if self.prev_id else 0
            for n in self.keys:
                if n == 116 or n == 132:
                    k, src = (16, self.sym) if n == 116 else (32, self.sym)
                elif n == 202 or n == 203:
                    k, src = (2, self.wrd) if n == 202 else (3, self.wrd)
                else:
                    cur = self._span(max(self.line_start, q - 256), q)
                    key = finalise(((self.prev_hash if self.has_prev else 0) * K_L1 + cur * K_L2 + 301) & MASK)
                    pred[col[n]], conf[col[n]] = self._lookup(self.tables.t[n], key); continue
                h = 0
                for s in islice(reversed(src), k):                     # most recent first, as stream_key's loop
                    h = (h * M + s + 1) & MASK
                pred[col[n]], conf[col[n]] = self._lookup(self.tables.t[n], finalise((h * K_TMPL + hp + k) & MASK))
        for n in self.cnt:
            if n == 1001: v = self.column
            elif n == 1002: v = self.indent
            elif n == 1003: v = self.prev_indent if self.has_prev else 0
            elif n == 1004: v = self.db
            elif n == 1005: v = self.dp
            elif n == 1006: v = 1 if self.state in (STRING, CHAR) else 0
            else: v = 1 if self.state in (BLOCK, LINE) else 0
            pred[col[n]] = 128 + (127 if v > 127 else (0 if v < 0 else v)); conf[col[n]] = 1

    # --- the interface
    def push(self, b: int):
        """One byte: the state and row n + 1. Returns the first row a header's depth reset corrected, or None."""
        self.text.append(b)
        return self._advance(b, True)

    def extend(self, data) -> None:
        """Many bytes: the rows from text_features (the vectorised build functions over the whole text), the state by a replay of the
        tail from a cut line start."""
        if not data:
            return
        if len(self.text) + len(data) < len(MARK):                        # the build's counters need eight bytes; push the few
            for b in data:
                self.push(b)
            return
        self.text += data
        N = len(self.text)
        pred, conf = text_features(self.tables, self.text, self.orders)
        self._grow(N + 1)
        self.pred[:N + 1] = pred; self.conf[:N + 1] = conf
        c, carried = self._cut(N, pred)
        self._reset(c, *carried)
        text = self.text
        for p in range(c, N):
            self._advance(text[p], False)
        assert self.n == N

    def _cut(self, N: int, pred):
        """The latest line start c with at least 32 template symbols, 3 word tokens and a newline in [c, N) (so the deques, the run,
        the current and previous line are local to the replay), and the counters' state carried to it; (0, the initial state) when
        there is none or the rows lack the counters it needs."""
        start = (0, (CODE, 0, 0, 0))
        if N == 0 or not all(o in self.col for o in (1003, 1004, 1005, 1006, 1007)):
            return start
        t = np.frombuffer(bytes(self.text), np.uint8)
        isid = ISID_ARR[t]; ws = ISWS_ARR[t]
        prev_id = np.concatenate([[False], isid[:-1]])
        run_end = ~isid & prev_id
        sym = np.cumsum((run_end.astype(np.int64) + ~isid)[::-1])[::-1]            # symbols emitted at positions >= q
        tok = np.cumsum((run_end.astype(np.int64) + (~isid & ~ws))[::-1])[::-1]
        nls = np.flatnonzero(t == 10)
        if len(nls) == 0:
            return start
        last_nl = int(nls[-1])
        starts = nls[nls < last_nl] + 1                                             # line starts c <= last_nl (after a newline)
        ok = starts[(sym[starts] >= 32) & (tok[starts] >= 3)]
        if len(ok) == 0:
            return start
        c = int(ok[-1])
        j3, j6, j7 = self.col[1003], self.col[1006], self.col[1007]
        state = BLOCK if pred[c, j7] == 129 else CODE
        prev_indent = int(pred[c, j3]) - 128
        idx = self.text.rfind(b"\n" + MARK, 0, c + len(MARK))                       # the last header line start at or before c
        fs = idx + 1 if idx >= 0 else 0
        code = (pred[fs:c, j6] == 128) & (pred[fs:c, j7] == 128)
        seg = t[fs:c]
        db = int(((seg == 123) & code).sum() - ((seg == 125) & code).sum())
        dp = int(((seg == 40) & code).sum() - ((seg == 41) & code).sum())
        return c, (state, db, dp, prev_indent)


class MatchState:
    """match_features over a growing window: start(pos0), then push(t) for every text index t from pos0 on, in order."""

    def __init__(self, orders, text: bytearray):
        self.orders, self.text = [int(n) for n in orders], text
        self.pos0, self.d = 0, [{} for _ in self.orders]

    def start(self, pos0: int) -> None:
        self.pos0, self.d = pos0, [{} for _ in self.orders]

    def push(self, t: int):
        """The features of window index t - pos0 (the byte text[t] is in): pred and cls per order, for predicting text[t + 1]."""
        a = t - self.pos0; text = self.text
        P = []; C = []
        for n, d in zip(self.orders, self.d):
            if a < n - 1:
                P.append(0); C.append(0); continue
            key = bytes(text[t - n + 1:t + 1])
            e = d.get(key)
            if e is None:
                d[key] = (t, 1); P.append(0); C.append(0)
            else:
                s, cnt = e; d[key] = (t, cnt + 1)
                P.append(text[s + 1]); C.append(1 + (cnt >= 2) + (cnt >= 4))
        return P, C

    def rows(self, pos0: int, hi: int):
        """start(pos0) and push every t in [pos0, hi): int64 arrays [hi - pos0, len(orders)] (pred, cls)."""
        self.start(pos0)
        P = np.zeros((hi - pos0, len(self.orders)), np.int64); C = np.zeros_like(P)
        for t in range(pos0, hi):
            P[t - pos0], C[t - pos0] = self.push(t)
        return P, C


def check(tables: Tables, text: bytes, orders=ALL18, cuts=(1 / 3, 1 / 2, 0.9)) -> int:
    """A cold replay of `text` and warm starts at the given fractions against text_features; returns the number of differing rows."""
    data = np.frombuffer(text, np.uint8)
    t0 = time.time(); ref_p, ref_c = text_features(tables, data, orders); t_vec = time.time() - t0
    N = len(text); bad = 0
    st = TextState(tables, orders)
    t0 = time.time(); fixes = [f for f in map(st.push, text) if f is not None]; t_cold = time.time() - t0
    d = int((st.pred[:N + 1] != ref_p).any(1).sum() + (st.conf[:N + 1] != ref_c).any(1).sum())
    print(f"cold replay of {N:,} bytes: {d} rows differ, {len(fixes)} header corrections; {N / t_cold:,.0f} bytes/s "
          f"(text_features over the whole text: {t_vec:.2f} s)", flush=True)
    bad += d
    for f in cuts:
        a = int(N * f)
        st = TextState(tables, orders)
        t0 = time.time(); st.extend(text[:a]); t_warm = time.time() - t0
        for b in text[a:]:
            st.push(b)
        d = int((st.pred[:N + 1] != ref_p).any(1).sum() + (st.conf[:N + 1] != ref_c).any(1).sum())
        print(f"warm start at {a:,} (replay from {st.base:,}, {t_warm * 1e3:.0f} ms): {d} rows differ", flush=True)
        bad += d
    print("exact" if bad == 0 else f"MISMATCH: {bad} rows")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", required=True); ap.add_argument("--check", required=True); ap.add_argument("--bytes", type=int, default=100_000)
    a = ap.parse_args()
    text = open(a.check, "rb").read()[:a.bytes]
    if not text.startswith(HEADER):
        text = HEADER + os.path.basename(a.check).encode() + b" ====\n" + text
    sys.exit(1 if check(Tables(a.tables), text) else 0)


if __name__ == "__main__":
    main()
