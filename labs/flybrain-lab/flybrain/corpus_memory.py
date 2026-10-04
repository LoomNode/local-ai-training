"""Corpus memory: a direct-mapped next-byte table per context order, written in one pass over the training bytes, no gradient.

The fly's recipe at corpus scale: a fixed address (a 64-bit hash of the n bytes before a position, the stand-in for the frozen
sparse random expansion) and one-shot writes (a counter that moves by one per observation, no error signal) into one layer of
slots. A slot holds a 16-bit tag, the byte that followed the context and a confidence; confidence 0 marks an empty slot.

The update of one (context, next byte) pair on its slot:
    empty, or the tag differs          -> store the tag and the byte, confidence 1
    tag matches, stored byte == next   -> confidence + 1, saturating at 255
    tag matches, stored byte != next   -> confidence - 1; when that would reach 0, store the next byte with confidence 1

Vectorised in chunks of positions. Every slot update in a chunk starts from the table as it was when the chunk began, and a slot
ends the chunk with the value computed at the *last* position that hit it (positions apply in order, the last one wins). With
chunk = 1 that is the sequential rule exactly. With a larger chunk a context that repeats inside one chunk moves its slot by
one step for the chunk, not one per repeat, so confidence counts chunks that confirmed a byte, not occurrences.

A position p predicts data[p] from the n bytes data[p - n : p]. Positions p < n have no full context for order n: no write, no hit.
`boundary` is the first byte that must never be written: an update that would read data[p] at p >= boundary raises before any
write, and a frozen memory (read-only tables) takes lookups only.

Features for every position (scripts/build_corpus_memory.py --features): the lookup of a chunk is made BEFORE the chunk is written, so
a position sees only what earlier chunks wrote. `update` and `lookup` take the chunk's `hashes` (context_hashes of [lo, hi)) when the
caller already has them, to hash each chunk once for both."""

from __future__ import annotations

import numpy as np

ORDERS = (4, 8, 16, 32)
LOG2_SLOTS = 27            # slots per order = 2 ** LOG2_SLOTS, 4 bytes each
CHUNK = 1 << 20            # positions per vectorised update
CONF_MAX = 255
TAG_BITS = 16

SLOT = np.dtype([("tag", "<u2"), ("byte", "u1"), ("conf", "u1")])

_MASK64 = (1 << 64) - 1
_FNV_OFFSET, _FNV_PRIME = 0xCBF29CE484222325, 0x100000001B3
_MIX1, _MIX2, _SALT = 0xFF51AFD7ED558CCD, 0xC4CEB9FE1A85EC53, 0x9E3779B97F4A7C15
_U64 = np.uint64
_S33 = _U64(33)


def _salt(n: int) -> int:
    return (n * _SALT) & _MASK64


def _finalise(h: np.ndarray) -> np.ndarray:
    """MurmurHash3's 64-bit finaliser: a bijection that spreads every input bit over every output bit, so any bits of the result
    can serve as slot index and tag. Returns a new array."""
    h = h ^ (h >> _S33)
    h *= _U64(_MIX1)
    h ^= h >> _S33
    h *= _U64(_MIX2)
    h ^= h >> _S33
    return h


def hash_context(context: bytes) -> int:
    """Scalar reference of the vectorised hash: FNV-1a over the context read backwards (the byte before the position first), xor the
    order's salt, then the finaliser. Python integers, so it is the definition the numpy code is checked against."""
    h = _FNV_OFFSET
    for b in reversed(context):
        h = ((h ^ b) * _FNV_PRIME) & _MASK64
    h ^= _salt(len(context))
    h ^= h >> 33
    h = (h * _MIX1) & _MASK64
    h ^= h >> 33
    h = (h * _MIX2) & _MASK64
    h ^= h >> 33
    return h


def context_hashes(data: np.ndarray, lo: int, hi: int, orders) -> dict[int, np.ndarray]:
    """hash_context(data[p - n : p]) for every p in [lo, hi) and n in orders, as uint64 arrays of length hi - lo. Where p < n there is
    no full context and the value is arbitrary (the callers mask those positions). Reads data[max(0, lo - max(orders)) : hi - 1]
    only: nothing at or after p goes into the hash of p."""
    K, m = max(orders), hi - lo
    left = max(0, lo - K)
    buf = np.zeros(K + m, dtype=np.uint64)             # buf[K + i] = data[lo + i], zero before the start of the data
    buf[K - (lo - left):] = data[left:hi]
    h = np.full(m, _FNV_OFFSET, dtype=np.uint64)
    out: dict[int, np.ndarray] = {}
    for j in range(1, K + 1):                          # j-th byte back from p: buf[K + i - j]
        h ^= buf[K - j: K - j + m]
        h *= _U64(_FNV_PRIME)
        if j in orders:
            out[j] = _finalise(h ^ _U64(_salt(j)))
    return out


def slot_and_tag(h: np.ndarray, log2_slots: int) -> tuple[np.ndarray, np.ndarray]:
    """Slot index from the top log2_slots bits of the hash, 16-bit tag from the lowest 16: disjoint bits."""
    if not 1 <= log2_slots <= 64 - TAG_BITS:
        raise ValueError(f"log2_slots must be in [1, {64 - TAG_BITS}], got {log2_slots}")
    return (h >> _U64(64 - log2_slots)).astype(np.int64), (h & _U64((1 << TAG_BITS) - 1)).astype(np.uint16)


def next_state(old: np.ndarray, tag: np.ndarray, byte: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """The update rule on arrays: slot records `old` (SLOT) after one (tag, byte) observation each. Also the counts of each case."""
    c = old["conf"]
    empty = c == 0
    same = (old["tag"] == tag) & ~empty
    agree = same & (old["byte"] == byte)
    miss = same & ~agree
    decay = miss & (c > 1)
    new = np.empty(old.shape, dtype=SLOT)
    new["tag"] = tag
    new["byte"] = np.where(agree | decay, old["byte"], byte)          # the stored byte stays unless the slot is (re)claimed
    new["conf"] = np.where(agree, np.minimum(c, CONF_MAX - 1) + 1, np.where(decay, c - 1, 1))
    counts = {"new": int(empty.sum()), "evict": int((~same & ~empty).sum()), "confirm": int(agree.sum()),
              "decay": int(decay.sum()), "replace": int((miss & ~decay).sum())}
    return new, counts


class CorpusMemory:
    """One table of 2 ** log2_slots slots per order. `boundary` (None = unbounded) is the first byte that must not be written."""

    def __init__(self, orders=ORDERS, log2_slots: int = LOG2_SLOTS, boundary: int | None = None):
        self.orders = tuple(sorted({int(n) for n in orders}))
        if not self.orders or self.orders[0] < 1:
            raise ValueError(f"orders must be positive integers, got {orders}")
        slot_and_tag(np.zeros(1, dtype=np.uint64), log2_slots)            # validates log2_slots
        self.log2_slots, self.boundary = int(log2_slots), boundary
        self.tables = {n: np.zeros(1 << self.log2_slots, dtype=SLOT) for n in self.orders}
        self.next_pos = 0                                                  # positions < next_pos have been written; update starts here
        self.frozen = False
        self.stats = {n: dict.fromkeys(("positions", "slots_written", "new", "evict", "confirm", "decay", "replace"), 0) for n in self.orders}

    def freeze(self) -> "CorpusMemory":
        """Make every table read-only: any later write raises."""
        for t in self.tables.values():
            t.flags.writeable = False
        self.frozen = True
        return self

    def update(self, data: np.ndarray, lo: int, hi: int, hashes: dict | None = None) -> None:
        """Write positions [lo, hi) (one chunk): for each order, the pair (data[p-n:p], data[p]). Positions must come in order, once.
        `hashes`, if given, is context_hashes(data, lo, hi, orders) already computed (the same numbers, hashed once)."""
        if self.frozen:
            raise RuntimeError("the memory is frozen: lookups only")
        if self.boundary is not None and hi > self.boundary:
            raise AssertionError(f"update [{lo}, {hi}) reaches the boundary {self.boundary}: bytes at or after it are never written")
        if not self.next_pos <= lo <= hi <= len(data):
            raise ValueError(f"update [{lo}, {hi}) must start at or after {self.next_pos} and lie inside the data ({len(data)} bytes)")
        m = hi - lo
        if m:
            nxt = np.asarray(data[lo:hi], dtype=np.uint8)
            hashes = context_hashes(data, lo, hi, self.orders) if hashes is None else hashes
            rel = np.arange(m, dtype=np.int64)
            posbits = max(1, (m - 1).bit_length())
            if self.log2_slots + posbits > 62:
                raise ValueError("chunk too large for the sort key")
            for n in self.orders:
                first = min(max(lo, n) - lo, m)                            # positions p < n have no full context
                if first == m:
                    continue
                slot, tag = slot_and_tag(hashes[n][first:], self.log2_slots)
                key = (slot << posbits) | rel[first:]                      # sorting by (slot, position) puts the last position of a slot last
                key.sort()
                s = key >> posbits
                last = np.ones(len(key), dtype=bool)
                np.not_equal(s[1:], s[:-1], out=last[:-1])
                kslot = s[last]
                kpos = (key[last] & ((1 << posbits) - 1)) - first          # index into this order's tag / next-byte arrays
                table = self.tables[n]
                new, counts = next_state(table[kslot], tag[kpos], nxt[first:][kpos])
                table[kslot] = new
                st = self.stats[n]
                st["positions"] += m - first
                st["slots_written"] += len(kslot)
                for k, v in counts.items():
                    st[k] += v
        self.next_pos = hi

    def fit(self, data: np.ndarray, start: int = 0, end: int | None = None, chunk: int = CHUNK, progress=None) -> None:
        """Write positions [start, end) in chunks of `chunk`; end defaults to the boundary. No chunk straddles the boundary: the whole
        range is checked first, so a range that would touch it writes nothing. progress(hi, end), if given, is called after each chunk."""
        end = self.boundary if end is None else end
        if end is None:
            raise ValueError("fit needs an end or a boundary")
        if self.boundary is not None and end > self.boundary:
            raise AssertionError(f"fit [{start}, {end}) reaches the boundary {self.boundary}: bytes at or after it are never written")
        if chunk < 1:
            raise ValueError("chunk must be positive")
        for lo in range(start, end, chunk):
            hi = min(lo + chunk, end)
            self.update(data, lo, hi)
            if progress is not None:
                progress(hi, end)

    def lookup(self, data: np.ndarray, lo: int, hi: int, hashes: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(pred, conf), both uint8 [len(orders), hi - lo]: the stored byte and confidence of each position's context at each order,
        conf 0 (and pred 0) for a miss: an empty slot, a tag that differs, or p < n. Reads data[p-n:p] only, never data[p] or later.
        Read-only: works on a frozen memory. `hashes`, if given, is context_hashes(data, lo, hi, orders) already computed."""
        m = hi - lo
        pred = np.zeros((len(self.orders), m), dtype=np.uint8)
        conf = np.zeros_like(pred)
        if m == 0:
            return pred, conf
        hashes = context_hashes(data, lo, hi, self.orders) if hashes is None else hashes
        for k, n in enumerate(self.orders):
            first = min(max(lo, n) - lo, m)
            if first == m:
                continue
            slot, tag = slot_and_tag(hashes[n][first:], self.log2_slots)
            rec = self.tables[n][slot]
            ok = (rec["conf"] > 0) & (rec["tag"] == tag)
            pred[k, first:] = np.where(ok, rec["byte"], 0)
            conf[k, first:] = np.where(ok, rec["conf"], 0)
        return pred, conf

    def peek(self, context: bytes) -> tuple[int, int]:
        """(byte, confidence) the table of order len(context) holds for this context; confidence 0 = a miss."""
        n = len(context)
        pred, conf = self.lookup(np.frombuffer(context + b"\0", dtype=np.uint8), n, n + 1)
        k = self.orders.index(n)
        return int(pred[k, 0]), int(conf[k, 0])

    def occupancy(self) -> dict[int, float]:
        """Share of non-empty slots per order."""
        return {n: float(np.count_nonzero(t["conf"])) / len(t) for n, t in self.tables.items()}

    def conf_histogram(self) -> dict[int, list[int]]:
        """Per order, how many slots hold each confidence 0..255 (index 0 = empty)."""
        return {n: np.bincount(t["conf"], minlength=CONF_MAX + 1).tolist() for n, t in self.tables.items()}

    def state_equal(self, other: "CorpusMemory") -> bool:
        return self.orders == other.orders and self.log2_slots == other.log2_slots and all(np.array_equal(self.tables[n], other.tables[n]) for n in self.orders)
