"""The corpus memory (flybrain.corpus_memory), its build script and the memory-mixing probe. Tiny synthetic data, CPU, no checkpoint
from disk: the tables are 2 ** 6 to 2 ** 16 slots and the streams a few hundred to a few thousand bytes."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest

from flybrain.corpus_memory import CONF_MAX, SLOT, CorpusMemory, context_hashes, hash_context, slot_and_tag

LOG2 = 16


def stream(seed: int, n: int, alphabet: int = 256) -> np.ndarray:
    raw = np.frombuffer(random.Random(seed).randbytes(n), dtype=np.uint8).copy()
    return raw if alphabet >= 256 else raw % np.uint8(alphabet)


def observe(mem: CorpusMemory, pairs) -> None:
    """Write exactly these (context, next byte) observations, in this order, each through its own one-position update. The stream is
    padded up to the memory's next position, so observe can be called again on the same memory."""
    base = mem.next_pos
    pos, parts = [], []
    for ctx, nxt in pairs:
        parts.append(bytes(ctx) + bytes([nxt]))
        pos.append(base + sum(len(p) for p in parts) - 1)
    data = np.frombuffer(bytes(base) + b"".join(parts), dtype=np.uint8)
    for p in pos:
        mem.update(data, p, p + 1)


# ---- a pure-Python model of the rule, written from its statement, to check the vectorised table against -----------------------

def ref_step(rec: tuple, tag: int, byte: int) -> tuple:
    t, b, c = rec
    if c == 0 or t != tag:
        return (tag, byte, 1)
    if b == byte:
        return (tag, b, min(c + 1, 255))
    if c - 1 == 0:
        return (tag, byte, 1)
    return (tag, b, c - 1)


def ref_tables(data: np.ndarray, orders, log2: int, chunk: int, end: int) -> dict:
    """Every lookup in a chunk sees the table as it was when the chunk began; positions are assigned in order, the last one wins."""
    tables = {n: {} for n in orders}
    raw = bytes(data)
    for lo in range(0, end, chunk):
        hi = min(lo + chunk, end)
        for n in orders:
            before = dict(tables[n])
            for p in range(max(lo, n), hi):
                h = hash_context(raw[p - n:p])
                slot, tag = h >> (64 - log2), h & 0xFFFF
                tables[n][slot] = ref_step(before.get(slot, (0, 0, 0)), tag, raw[p])
    return tables


def as_table(ref: dict, log2: int) -> np.ndarray:
    t = np.zeros(1 << log2, dtype=SLOT)
    for slot, (tag, byte, conf) in ref.items():
        t[slot] = (tag, byte, conf)
    return t


# ---- address ------------------------------------------------------------------------------------------------------------------

def test_the_vectorised_hash_is_the_scalar_definition_at_every_position():
    data = stream(0, 300)
    orders = (1, 3, 8, 32)
    for lo, hi in ((0, 300), (2, 300), (40, 200), (100, 101)):         # starts before, near and far past the longest context
        out = context_hashes(data, lo, hi, orders)
        for n in orders:
            assert out[n].dtype == np.uint64 and len(out[n]) == hi - lo
            for p in range(max(lo, n), hi):
                assert int(out[n][p - lo]) == hash_context(bytes(data[p - n:p])), (n, p, lo)


def test_the_slot_comes_from_the_top_bits_and_the_tag_from_the_lowest_sixteen():
    h = np.array([0xFEDCBA9876543210, 0x0000000000000001, 0xFFFFFFFFFFFFFFFF, 0x8000000000000000], dtype=np.uint64)
    slot, tag = slot_and_tag(h, 27)
    assert slot.tolist() == [0xFEDCBA9876543210 >> 37, 0, (1 << 27) - 1, 1 << 26]
    assert tag.tolist() == [0x3210, 1, 0xFFFF, 0] and tag.dtype == np.uint16
    with pytest.raises(ValueError):
        slot_and_tag(h, 49)                                              # slot and tag bits would overlap


def test_contexts_that_differ_get_different_addresses_and_the_order_is_part_of_the_hash():
    assert hash_context(b"abcd") != hash_context(b"abce") != hash_context(b"bbcd")
    assert hash_context(b"\0\0ab") != hash_context(b"ab")               # zero padding is not the same context as a shorter one
    assert hash_context(b"") != hash_context(b"\0")
    ctxs = {bytes(random.Random(i).randbytes(8)) for i in range(2000)}
    assert len({hash_context(c) for c in ctxs}) == len(ctxs)
    slots = [hash_context(c) >> 49 for c in ctxs]                        # 15-bit slots: the 2000 contexts spread over thousands of them
    assert len(set(slots)) > 1500


# ---- one-shot writes and exact reads ----------------------------------------------------------------------------------------

def test_a_context_seen_once_is_a_hit_with_the_right_byte():
    data = np.frombuffer(b"the cat sat on the mat", dtype=np.uint8)      # "the" is followed by a space at 3 and again at 18
    mem = CorpusMemory(orders=(3,), log2_slots=LOG2)
    mem.fit(data, 0, 15)                                                 # the first "the" is written, the second is not
    assert mem.peek(b"the") == (ord(" "), 1)
    pred, conf = mem.lookup(data, 18, 19)                                # the context of position 18 is data[15:18] = "the"
    assert pred.shape == conf.shape == (1, 1) and pred.dtype == conf.dtype == np.uint8
    assert (int(pred[0, 0]), int(conf[0, 0])) == (ord(" "), 1)


def test_an_unseen_context_is_a_miss():
    mem = CorpusMemory(orders=(2, 4), log2_slots=LOG2)
    mem.fit(np.frombuffer(b"abcabcabcabc", dtype=np.uint8), 0, 12)
    assert mem.peek(b"zz") == (0, 0) and mem.peek(b"zzzz") == (0, 0)
    other = np.frombuffer(b"qrstuvwxyz0123456789", dtype=np.uint8)
    pred, conf = mem.lookup(other, 0, len(other))
    assert not pred.any() and not conf.any()                             # nothing in this text was ever written


def test_orders_are_separate_tables_and_the_longer_context_can_miss_where_the_shorter_hits():
    mem = CorpusMemory(orders=(2, 4), log2_slots=LOG2)
    observe(mem, [(b"xxab", ord("P")), (b"yyab", ord("Q"))])             # both contexts end in "ab": one slot of order 2, two slots of order 4
    assert mem.peek(b"xxab") == (ord("P"), 1) and mem.peek(b"yyab") == (ord("Q"), 1) and mem.peek(b"zzab") == (0, 0)
    assert mem.peek(b"ab") == (ord("Q"), 1)                              # order 2 saw P then Q at confidence 1: replaced
    pred, conf = mem.lookup(np.frombuffer(b"zzab?", dtype=np.uint8), 4, 5)
    assert conf.tolist() == [[1], [0]] and pred.tolist() == [[ord("Q")], [0]]   # order 2 hits, order 4 ("zzab") misses


def test_the_first_positions_have_no_full_context_so_no_write_and_no_hit():
    mem = CorpusMemory(orders=(4,), log2_slots=LOG2)
    mem.fit(np.frombuffer(b"\0\0ab" + b"Z", dtype=np.uint8), 0, 5)       # position 4 writes the context "\0\0ab"; positions 0-3 write nothing
    assert mem.peek(b"\0\0ab") == (ord("Z"), 1) and mem.stats[4]["positions"] == 1
    text = np.frombuffer(b"abXY", dtype=np.uint8)                        # at p = 2 only "ab" lies before: zero padding would read as "\0\0ab"
    pred, conf = mem.lookup(text, 2, 3)
    assert conf.tolist() == [[0]]


# ---- the confidence rule ----------------------------------------------------------------------------------------------------

AB = b"ab"


def after(nexts: list[int], orders=(2,)) -> tuple[int, int]:
    mem = CorpusMemory(orders=orders, log2_slots=LOG2)
    observe(mem, [(AB, b) for b in nexts])
    return mem.peek(AB)


def test_confidence_counts_confirmations_and_saturates_at_255():
    X = ord("X")
    assert after([X]) == (X, 1)
    assert after([X, X, X, X]) == (X, 4)
    assert after([X] * 254) == (X, 254)
    assert after([X] * 255) == (X, 255)
    assert after([X] * 300) == (X, CONF_MAX) and CONF_MAX == 255        # it stays at 255, it does not wrap to 0


def test_a_different_byte_lowers_confidence_by_one_and_the_stored_byte_stays():
    X, Y = ord("X"), ord("Y")
    assert after([X, X, X, Y]) == (X, 2)
    assert after([X, X, X, Y, Y]) == (X, 1)
    assert after([X, X, X, Y, X]) == (X, 3)                              # a confirmation after a miss counts again


def test_the_new_byte_replaces_the_stored_one_when_confidence_would_reach_zero():
    X, Y, Z = ord("X"), ord("Y"), ord("Z")
    assert after([X, Y]) == (Y, 1)                                       # 1 - 1 = 0: replaced, confidence 1
    assert after([X, X, Y, Y]) == (Y, 1)                                 # 2 -> 1 (still X) -> would be 0: Y, 1
    assert after([X, X, Y, Y, Y]) == (Y, 2)
    assert after([X, Y, Z]) == (Z, 1) and after([X, Y, X]) == (X, 1)     # and it can be replaced again at once


def test_a_slot_taken_by_another_context_is_overwritten_whatever_its_confidence():
    for log2 in (3, 4):
        mem = CorpusMemory(orders=(2,), log2_slots=log2)                 # 8 or 16 slots: collisions are easy to find
        by_slot: dict[int, list] = {}
        for i in range(2000):
            ctx = bytes(random.Random(i).randbytes(2))
            h = hash_context(ctx)
            by_slot.setdefault(h >> (64 - log2), []).append((ctx, h & 0xFFFF))
        a, b = next((v[0], v[1]) for v in by_slot.values() if len(v) > 1 and v[0][1] != v[1][1])   # same slot, different tags
        observe(mem, [(a[0], 7)] * 5)
        assert mem.peek(a[0]) == (7, 5) and mem.peek(b[0]) == (0, 0)     # the slot is a's; b's tag differs: a miss, not a's byte
        observe(mem, [(b[0], 9)])
        assert mem.peek(b[0]) == (9, 1) and mem.peek(a[0]) == (0, 0)     # b took the slot at confidence 1 although a held 5


def test_a_slot_holds_the_tag_so_two_different_contexts_in_one_slot_never_read_each_others_byte():
    mem = CorpusMemory(orders=(2,), log2_slots=1)                        # two slots: nearly everything collides
    seen = {}
    for i in range(300):
        ctx = bytes(random.Random(i).randbytes(2))
        observe(mem, [(ctx, i % 251)])
        seen[ctx] = i % 251
        pred, conf = mem.peek(ctx)
        assert (pred, conf) == (i % 251, 1)                               # just written: a hit with its own byte
    hits = sum(1 for ctx, b in seen.items() if mem.peek(ctx)[1] > 0)
    assert hits <= 2                                                      # only the last context written to each slot is still there
    assert all(mem.peek(ctx) == (b, 1) for ctx, b in seen.items() if mem.peek(ctx)[1] > 0)


# ---- chunks: sequential at 1, snapshot above ------------------------------------------------------------------------------

def test_a_chunk_of_one_is_the_sequential_rule_and_a_larger_chunk_the_snapshot_rule():
    data = stream(1, 1500, alphabet=3)                                   # three symbols over 64 slots: contexts and slots repeat constantly
    orders, log2 = (2, 3, 5), 6
    for chunk in (1, 7, 64, 500, 5000):
        mem = CorpusMemory(orders, log2)
        mem.fit(data, 0, len(data), chunk=chunk)
        ref = ref_tables(data, orders, log2, chunk, len(data))
        for n in orders:
            assert np.array_equal(mem.tables[n], as_table(ref[n], log2)), (n, chunk)
    seq, snap = CorpusMemory(orders, log2), CorpusMemory(orders, log2)
    seq.fit(data, 0, len(data), chunk=1)
    snap.fit(data, 0, len(data), chunk=500)
    assert not seq.state_equal(snap)                                     # the two rules really differ on repeated contexts


def test_a_context_repeated_inside_one_chunk_moves_its_slot_once_and_the_last_position_wins():
    X, Y = ord("X"), ord("Y")
    data = np.frombuffer(AB + bytes([X]) + AB + bytes([X]) + AB + bytes([X]) + AB + bytes([Y]), dtype=np.uint8)
    pos = [2, 5, 8, 11]                                                  # "ab" is followed by X, X, X, Y here
    one = CorpusMemory((2,), LOG2)
    one.update(data, 2, 12)                                              # one chunk holding all four: computed from the empty table, last = Y
    assert one.peek(AB) == (Y, 1)
    two = CorpusMemory((2,), LOG2)
    two.update(data, 2, 9)                                               # X, X, X in one chunk: confidence 1, not 3
    assert two.peek(AB) == (X, 1)
    two.update(data, 9, 12)                                              # the next chunk starts from that: a Y lowers X from 1 and replaces it
    assert two.peek(AB) == (Y, 1)
    seq = CorpusMemory((2,), LOG2)
    for p in pos:
        seq.update(data, p, p + 1)
    assert seq.peek(AB) == (X, 2)                                        # X X X Y sequentially: 3, then 2


def test_the_stats_account_for_every_slot_write():
    data = stream(2, 3000, alphabet=4)
    mem = CorpusMemory((2, 4), 8)
    mem.fit(data, 0, len(data), chunk=256)
    for n in (2, 4):
        s = mem.stats[n]
        assert s["slots_written"] == s["new"] + s["evict"] + s["confirm"] + s["decay"] + s["replace"] and s["slots_written"] <= s["positions"]
        assert s["positions"] == len(data) - n
    occ = mem.occupancy()
    assert all(0 < v <= 1 for v in occ.values())
    assert sum(mem.conf_histogram()[2][1:]) == round(occ[2] * (1 << 8))


def test_the_build_is_deterministic():
    data = stream(3, 2000)
    a, b = CorpusMemory((3, 6), 10), CorpusMemory((3, 6), 10)
    a.fit(data, 0, len(data), chunk=300)
    b.fit(data, 0, len(data), chunk=300)
    assert a.state_equal(b) and a.stats == b.stats


# ---- the boundary: held-out bytes are never written ---------------------------------------------------------------------------

def test_bytes_at_or_after_the_boundary_never_change_the_tables():
    B = 1200
    a = stream(4, 2000)
    b = a.copy()
    b[B:] = stream(5, 2000 - B)                                          # the same bytes before B, different bytes from B on
    assert not np.array_equal(a[B:], b[B:])
    ma, mb = CorpusMemory((2, 5, 9), 10, boundary=B), CorpusMemory((2, 5, 9), 10, boundary=B)
    ma.fit(a, 0, B, chunk=500)                                           # chunks [0, 500), [500, 1000), [1000, 1200): the last ends at the boundary
    mb.fit(b, 0, B, chunk=500)
    assert ma.state_equal(mb) and ma.stats == mb.stats and ma.next_pos == B
    mc = CorpusMemory((2, 5, 9), 10, boundary=B)
    mc.fit(b, 0, None, chunk=500)                                        # end defaults to the boundary
    assert mc.state_equal(ma)


def test_an_update_or_fit_that_reaches_the_boundary_raises_before_it_writes_anything():
    B = 1000
    data = stream(6, 2000)
    mem = CorpusMemory((3,), 10, boundary=B)
    mem.fit(data, 0, 900)
    before = mem.tables[3].copy()
    for lo, hi in ((900, 1001), (990, 1500), (1000, 1001), (1000, 1500)):
        with pytest.raises(AssertionError, match="boundary"):
            mem.update(data, lo, hi)                                     # a chunk straddling the boundary, and chunks past it
    with pytest.raises(AssertionError, match="boundary"):
        mem.fit(data, 900, 1001, chunk=50)                               # the range is refused as a whole: not even the chunks below B are written
    assert np.array_equal(mem.tables[3], before) and mem.next_pos == 900
    mem.update(data, 900, 1000)                                          # up to B is fine
    assert mem.next_pos == 1000
    with pytest.raises(AssertionError, match="boundary"):
        mem.update(data, 1000, 1001)


def test_a_frozen_memory_answers_lookups_and_refuses_writes():
    data = stream(7, 800)
    mem = CorpusMemory((2, 6), 10, boundary=500)
    mem.fit(data, 0, 500)
    before = {n: t.copy() for n, t in mem.tables.items()}
    mem.freeze()
    for t in mem.tables.values():
        assert not t.flags.writeable
    with pytest.raises(RuntimeError, match="frozen"):
        mem.update(data, 500, 501)
    pred, conf = mem.lookup(data, 500, 800)                              # held-out positions: read-only
    assert pred.shape == (2, 300)
    assert all(np.array_equal(mem.tables[n], before[n]) for n in (2, 6))
    with pytest.raises(ValueError):
        mem.tables[2]["conf"][0] = 9


def test_updates_must_come_in_order_and_inside_the_data():
    data = stream(8, 100)
    mem = CorpusMemory((2,), 8)
    mem.update(data, 0, 50)
    with pytest.raises(ValueError):
        mem.update(data, 40, 60)                                         # position 40 was already written
    with pytest.raises(ValueError):
        mem.update(data, 50, 101)                                        # beyond the data
    mem.update(data, 50, 100)


def test_the_lookup_at_p_never_reads_the_byte_at_p_or_after_it():
    data = stream(9, 600, alphabet=5)
    mem = CorpusMemory((2, 4, 8), 12)
    mem.fit(data, 0, 400)
    base_pred, base_conf = mem.lookup(data, 400, 600)
    assert base_conf.any()                                               # the test is not vacuous: held-out positions do hit
    for p in (400, 431, 500, 599):
        garbled = data.copy()
        garbled[p:] = stream(10, 600 - p)                                # change the byte at p and everything after it
        pred, conf = mem.lookup(garbled, p, p + 1)
        assert pred[:, 0].tolist() == base_pred[:, p - 400].tolist() and conf[:, 0].tolist() == base_conf[:, p - 400].tolist(), p
    last = data.copy()
    last[599] ^= 0x55                                                    # the last byte of the range is the context of nothing in it
    pred, conf = mem.lookup(last, 400, 600)
    assert np.array_equal(pred, base_pred) and np.array_equal(conf, base_conf)


# ---- the build script ---------------------------------------------------------------------------------------------------------

def test_the_build_script_writes_features_by_absolute_position_from_the_training_bytes_alone(tmp_path):
    from scripts.build_corpus_memory import build
    rng = random.Random(11)
    unit = rng.randbytes(40)
    head = unit * 20 + rng.randbytes(100)                                # 900 bytes of training text: a 40-byte unit repeated, then noise
    tails = [unit * 5 + rng.randbytes(80), rng.randbytes(280)]           # two held-out tails: one repeats the unit, one is noise
    out = []
    for i, tail in enumerate(tails):
        corpus = tmp_path / f"c{i}.bytes"
        corpus.write_bytes(head + tail)
        out.append((build(corpus, tmp_path / f"mem{i}", orders=(4, 8), log2_slots=12, chunk=256, train_end=len(head), save_tables=True, log=lambda s: None),
                    np.frombuffer(corpus.read_bytes(), dtype=np.uint8)))
    (rec0, data0), (rec1, data1) = out
    assert rec0["occupancy"] == rec1["occupancy"] and rec0["update_counts"] == rec1["update_counts"]   # the tail did not touch the tables
    assert rec0["confidence_histogram"] == rec1["confidence_histogram"]
    t0 = np.load(tmp_path / "mem0" / "table_n4.npy")
    assert np.array_equal(t0, np.load(tmp_path / "mem1" / "table_n4.npy"))
    assert rec0["build_seconds"] > 0 and rec0["valid_from"] == len(head) and rec0["heldout_positions"] == len(tails[0])
    mem = CorpusMemory((4, 8), 12, boundary=len(head))
    mem.fit(data0, 0, len(head), chunk=256)
    pred, conf = mem.lookup(data0, len(head), len(data0))
    for k, n in enumerate((4, 8)):
        f_pred, f_conf = np.load(tmp_path / "mem0" / f"pred_n{n}.npy"), np.load(tmp_path / "mem0" / f"conf_n{n}.npy")
        assert len(f_pred) == len(f_conf) == len(data0)                  # indexed by absolute position
        assert np.array_equal(f_pred[len(head):], pred[k]) and np.array_equal(f_conf[len(head):], conf[k])
        assert not f_pred[:len(head)].any() and not f_conf[:len(head)].any()   # training positions are not features
    f_pred, f_conf = np.load(tmp_path / "mem0" / "pred_n8.npy"), np.load(tmp_path / "mem0" / "conf_n8.npy")
    p = len(head) + 60                                                   # inside the repeated unit of tail 0: its 8-byte context was seen, the byte is right
    assert f_conf[p] > 0 and f_pred[p] == data0[p]
    g_conf = np.load(tmp_path / "mem1" / "conf_n8.npy")                  # the noise tail: (almost) nothing hits
    assert np.count_nonzero(g_conf[len(head):]) <= 2
    assert json.loads((tmp_path / "mem0" / "build.json").read_text())["orders"] == [4, 8]


def test_the_build_script_refuses_a_train_end_past_the_file(tmp_path):
    from scripts.build_corpus_memory import build
    (tmp_path / "c.bytes").write_bytes(b"x" * 100)
    with pytest.raises(ValueError, match="train_end"):
        build(tmp_path / "c.bytes", tmp_path / "m", orders=(2,), log2_slots=8, train_end=101, log=lambda s: None)


# ---- the probe: the window rule, buckets, the weights and their fit, the mixture, the intervals -----------------------------------

from scripts import probe_memory_mix as pm  # noqa: E402  (after the memory tests: it needs nothing from them)

ORD = (4, 8, 16, 32)


def test_a_hit_counts_only_if_the_whole_context_lies_inside_the_models_window():
    T = 64
    conf = np.ones((4, 2, T), dtype=np.uint8)                            # a hit at every order at every byte
    out = pm.in_window(conf, ORD)
    assert conf.all()                                                    # the raw lookups are not modified
    for k, n in enumerate(ORD):
        t = np.arange(T)
        assert (out[k] > 0).tolist() == [(t + 1 >= n).tolist()] * 2      # byte t is predicted at p = start + t + 1, its context starts at p - n >= start
    assert out[1, 0, 6] == 0 and out[1, 0, 7] == 1                       # order 8: the context of byte 6 starts one byte before the window, byte 7's at its first byte
    assert out[0, 0, 2] == 0 and out[0, 0, 3] == 1


def test_the_bucket_is_the_longest_order_with_a_hit_inside_the_window_and_its_confidence_class():
    orders = (4, 8)
    pred = np.zeros((2, 1, 16), dtype=np.uint8)
    conf = np.zeros_like(pred)
    pred[:, 0, 5], conf[:, 0, 5] = [ord("a"), ord("b")], [3, 20]         # t = 5: order 8 reaches before the window (6 < 8): only order 4 counts
    pred[:, 0, 9], conf[:, 0, 9] = [ord("a"), ord("b")], [3, 20]         # t = 9: both inside, the longer one decides
    pred[:, 0, 12], conf[:, 0, 12] = [ord("a"), 0], [1, 0]               # order 4 only
    y = np.full((1, 16), ord("b"), dtype=np.uint8)
    bucket, right = pm.assign_buckets(pred, pm.in_window(conf, orders), y)
    assert (bucket[0, 5], right[0, 5]) == (0 * 4 + 1, False)             # n=4, confidence 3 -> class 2-3; byte 'a' is not 'b'
    assert (bucket[0, 9], right[0, 9]) == (1 * 4 + 3, True)              # n=8, confidence 20 -> class 16+; byte 'b' is right
    assert (bucket[0, 12], right[0, 12]) == (0 * 4 + 0, False)           # n=4, confidence 1
    assert bucket[0, 0] == 2 * 4 and not right[0, 0]                     # no hit: the last bucket
    assert pm.bucket_labels(orders)[:5] == ["n=4 conf 1", "n=4 conf 2-3", "n=4 conf 4-15", "n=4 conf 16+", "n=8 conf 1"] and pm.bucket_labels(orders)[-1] == "no hit"


def test_confidence_classes_are_1_2to3_4to15_16plus():
    conf = np.array([[[1, 2, 3, 4, 15, 16, 17, 255]]], dtype=np.uint8)
    pred = np.full_like(conf, 7)
    bucket, right = pm.assign_buckets(pred, conf, np.full((1, 8), 7, dtype=np.uint8))
    assert bucket[0].tolist() == [0, 1, 1, 2, 2, 3, 3, 3] and right.all()


def test_the_fit_gives_lambda_near_one_when_the_memory_is_always_right_and_zero_when_it_is_always_wrong():
    a = np.random.default_rng(0).uniform(0.01, 0.9, 500)
    assert pm.fit_lambda(a, np.ones(500, dtype=bool)) == pm.LAM_MAX >= 0.99
    assert pm.fit_lambda(a, np.zeros(500, dtype=bool)) == 0.0
    assert pm.fit_lambda(np.array([]), np.array([], dtype=bool)) == 0.0


def test_the_fit_finds_the_closed_form_optimum_and_a_grid_agrees():
    a, n, n_right = 0.3, 1000, 800                                       # the model gives 0.3 to the true byte, the memory is right 80 % of the time
    right = np.arange(n) < n_right
    assert pm.fit_lambda(np.full(n, a), right) == pytest.approx((0.8 - a) / (1 - a), abs=1e-9)    # lambda* = (r - a) / (1 - a) when r > a
    assert pm.fit_lambda(np.full(n, 0.9), right) == pytest.approx(0.0, abs=1e-12)                 # the model is already better than the memory
    rng = np.random.default_rng(1)
    for _ in range(5):
        a, right = rng.uniform(0.001, 0.99, 300), rng.random(300) < rng.uniform(0.1, 0.9)
        lam = pm.fit_lambda(a, right)
        grid = np.linspace(0, pm.LAM_MAX, 4001)
        ll = lambda l: np.log((1 - l) * a + l * right).mean()
        best = grid[int(np.argmax([ll(g) for g in grid]))]
        assert abs(lam - best) < 1e-3 and ll(lam) >= max(ll(g) for g in grid) - 1e-12


def test_lambda_zero_reproduces_the_model_alone_loss_exactly():
    rng = np.random.default_rng(2)
    ce = rng.exponential(1.0, (4, 32)).astype(np.float32)
    right, bucket = rng.random((4, 32)) < 0.5, rng.integers(0, 17, (4, 32)).astype(np.int8)
    alone = ce.astype(np.float64) / pm.LN2
    assert np.array_equal(pm.mix_bits(ce, right, bucket, np.zeros(17)), alone)                      # bit for bit
    lam = np.zeros(17)
    lam[3] = 0.6
    mixed = pm.mix_bits(ce, right, bucket, lam)
    assert np.array_equal(mixed[bucket != 3], alone[bucket != 3]) and not np.array_equal(mixed[bucket == 3], alone[bucket == 3])


def test_the_mixture_is_the_stated_probability():
    ce = np.log(4.0) * np.ones((1, 2))                                   # the model gives 0.25 to the true byte
    bucket = np.zeros((1, 2), dtype=np.int8)
    lam = np.zeros(17)
    lam[0] = 0.5
    bits = pm.mix_bits(ce, np.array([[True, False]]), bucket, lam)
    assert bits[0, 0] == pytest.approx(-math.log2(0.5 * 0.25 + 0.5))     # memory right: 0.625
    assert bits[0, 1] == pytest.approx(-math.log2(0.5 * 0.25))           # memory wrong: 0.125 = 3 bits


def test_the_no_hit_bucket_is_never_mixed_and_a_dev_fit_never_loses_on_dev():
    rng = np.random.default_rng(3)
    ce = rng.exponential(1.5, (16, 64)).astype(np.float32)
    bucket = rng.integers(0, 17, (16, 64)).astype(np.int8)
    right = rng.random((16, 64)) < 0.7                                   # even where the "no hit" bucket claims the memory was right, it is not mixed
    lam = pm.fit_lambdas(ce, right, bucket, 17)
    assert lam[16] == 0.0 and lam.max() > 0
    r = pm.evaluate(ce, right, bucket, lam)
    assert r["cold"]["gain"] > 0 and r["warm"]["gain"] > 0               # the weights are the maximiser on these very bytes
    assert r["cold"]["gain"] == pytest.approx(r["cold"]["alone"] - r["cold"]["mixed"])
    assert r["cold"]["gain_ci95"][0] <= r["cold"]["gain"] <= r["cold"]["gain_ci95"][1]
    assert pm.evaluate(ce, right, bucket, np.zeros(17))["cold"]["gain"] == 0.0


def test_cold_is_every_byte_and_warm_the_second_half_of_each_window():
    ce = np.concatenate([np.full((3, 4), math.log(2)), np.full((3, 4), 3 * math.log(2))], axis=1)      # 1 bit in the first half, 3 in the second
    bucket, right = np.zeros((3, 8), dtype=np.int8), np.zeros((3, 8), dtype=bool)
    r = pm.evaluate(ce, right, bucket, np.zeros(17))
    assert r["cold"]["alone"] == pytest.approx(2.0) and r["warm"]["alone"] == pytest.approx(3.0)


def test_the_bucket_table_adds_up_to_the_gain_and_the_shares_to_one():
    rng = np.random.default_rng(4)
    ce = rng.exponential(1.5, (8, 64)).astype(np.float32)
    bucket = rng.integers(0, 17, (8, 64)).astype(np.int8)
    right = (rng.random((8, 64)) < 0.6) & (bucket != 16)
    lam = pm.fit_lambdas(ce, right, bucket, 17)
    for regime in ("cold", "warm"):
        rows = pm.bucket_table(ce, right, bucket, lam, regime)
        r = pm.evaluate(ce, right, bucket, lam)[regime]
        assert len(rows) == 17 and sum(b["share"] for b in rows) == pytest.approx(1.0)
        assert sum(b["gain_contribution"] for b in rows) == pytest.approx(r["gain"])
        assert rows[16]["gain_contribution"] == 0.0 and rows[16]["lambda"] == 0.0 and rows[16]["memory_right"] == 0.0


def test_the_paired_bootstrap_is_deterministic_and_tight_when_every_window_gains_alike():
    g = np.random.default_rng(5).normal(1.0, 0.5, 200)
    assert pm.bootstrap_gain(g, 10) == pm.bootstrap_gain(g, 10) != pm.bootstrap_gain(g, 10, seed=1)
    idx = np.random.default_rng(0).integers(0, 200, size=(1000, 200))   # 1,000 resamples of the 200 windows, seed 0, percentile interval
    assert pm.bootstrap_gain(g, 10) == pytest.approx(np.percentile(g[idx].sum(1) / (200 * 10), [2.5, 97.5]), abs=1e-15) and pm.N_BOOT == 1000
    lo, hi = pm.bootstrap_gain(g, 10)
    assert lo < g.mean() / 10 < hi and hi - lo == pytest.approx(2 * 1.96 * g.std() / 10 / math.sqrt(200), rel=0.2)
    assert pm.bootstrap_gain(np.full(50, 2.0), 10) == [0.2, 0.2]


def test_the_per_byte_scorer_is_what_evaluate_fixed_averages(tmp_path):
    import torch
    from flybrain.data.text import fixed_windows, load_bytes
    from flybrain.model.hybrid import GRUHybrid
    from scripts.cross_eval import evaluate_fixed
    torch.manual_seed(1)
    m = GRUHybrid(d_model=16, hidden=16, backbone="mingru", where="none", n_layers=1, d_ff=32, max_len=1024).eval()
    corpus = tmp_path / "c.bytes"
    corpus.write_bytes(random.Random(0).randbytes(700) + (b"int main(void) { return 0; }\n" * 60))
    T, n, offset = 32, 9, 100
    x, y = pm.load_windows(corpus, offset, n, T)
    data = load_bytes(corpus, max_bytes=n * T + 1, offset=offset)
    x2, y2 = fixed_windows(data, T, n)
    assert torch.equal(x, x2) and torch.equal(y, y2)                     # the windows cross_eval cuts
    for batch in (4, 2):                                                 # a short last batch (9 = 4 + 4 + 1) changes nothing
        ce = pm.score_per_byte(m, x, y, batch=batch)
        row = evaluate_fixed(m, data, T, batch, n, torch.device("cpu"))
        assert ce.shape == (n, T) and ce.dtype == np.float32
        assert float(ce.astype(np.float64).mean() / pm.LN2) == pytest.approx(row["bpb"], abs=1e-6)
        assert float(ce[:, T // 2:].astype(np.float64).mean() / pm.LN2) == pytest.approx(row["bpb_warm"], abs=1e-6)


def _held_out_world(corpus):
    """Training files a, b, c; then d (a twin of a), e (new), f (a twin of b), g (new): the training range ends at the start of d."""
    bodies = {k: random.Random(k).randbytes(300 + 10 * k).replace(b"\n", b" ") for k in range(7)}
    path = corpus([("a.c", bodies[0]), ("b.c", bodies[1]), ("c.c", bodies[2]), ("d.c", bodies[0]), ("e.c", bodies[3]), ("f.c", bodies[1]), ("g.c", bodies[4])])
    return path


def test_a_window_is_removed_when_it_overlaps_a_held_out_file_whose_body_also_occurs_in_training(corpus):
    from flybrain.data.files import file_table
    path = _held_out_world(corpus)
    table, hashes = pm.corpus_files(path)
    ix = {e.path: e for e in table}
    train_end = ix["d.c"].start
    T, offset = 64, train_end
    n = (table[-1].end - offset - 1) // T
    got = pm.duplicate_windows(table, hashes, offset, n, T, train_end)
    twins = [ix["d.c"], ix["f.c"]]
    expect = np.array([any(offset + i * T < e.end and offset + i * T + T + 1 > e.start for e in twins) for i in range(n)])
    assert np.array_equal(got["overlaps"], expect) and 0 < expect.sum() < n
    assert (got["training_files"], got["heldout_files"], got["heldout_files_with_a_training_twin"]) == (3, 4, 2)
    only_e = [i for i in range(n) if ix["e.c"].start <= offset + i * T and offset + i * T + T + 1 <= ix["e.c"].end]
    assert only_e and not got["overlaps"][only_e].any()                  # a window wholly inside a file without a twin stays


def test_the_file_straddling_the_training_end_is_a_held_out_file(corpus):
    body = random.Random(9).randbytes(400)
    path = corpus([("a.c", body), ("s.c", random.Random(8).randbytes(500)), ("t.c", body)])   # s straddles: its start is inside training, its end is not
    table, hashes = pm.corpus_files(path)
    train_end = table[1].start + 100
    got = pm.duplicate_windows(table, hashes, table[2].start, 4, 32, train_end)
    assert got["training_files"] == 1 and got["heldout_files"] == 2 and got["heldout_files_with_a_training_twin"] == 1   # t.c repeats a.c; s.c is held out


def test_the_stored_lookups_line_up_with_the_scored_bytes(tmp_path):
    from scripts.build_corpus_memory import build
    rng = random.Random(21)
    unit = rng.randbytes(50)
    head = unit * 30 + rng.randbytes(200)
    tail = unit * 20 + rng.randbytes(150) + unit * 5
    corpus = tmp_path / "c.bytes"
    corpus.write_bytes(head + tail)
    build(corpus, tmp_path / "mem", orders=(4, 8), log2_slots=14, chunk=300, train_end=len(head), log=lambda s: None)
    data = np.frombuffer(corpus.read_bytes(), dtype=np.uint8)
    mem = CorpusMemory((4, 8), 14, boundary=len(head))
    mem.fit(data, 0, len(head), chunk=300)
    T, n, offset = 16, 20, len(head) + 7
    pred, conf = pm.memory_features(tmp_path / "mem", offset, n, (4, 8), T)
    assert pred.shape == conf.shape == (2, n, T)
    for i in (0, 3, 19):
        for t in (0, 5, 15):
            p = offset + i * T + t + 1                                   # the byte window i scores at t is data[p]
            want_pred, want_conf = mem.lookup(data, p, p + 1)
            assert pred[:, i, t].tolist() == want_pred[:, 0].tolist() and conf[:, i, t].tolist() == want_conf[:, 0].tolist(), (i, t)
    assert conf.any()
    with pytest.raises(ValueError, match="training bytes"):
        pm.memory_features(tmp_path / "mem", len(head) - 2, 1, (4, 8), T)   # a window whose first predicted byte is in training


def test_blocks_are_claimed_once_saved_on_completion_and_read_back_in_order(tmp_path, monkeypatch, tiny_checkpoint):
    import shutil
    ck = tiny_checkpoint("mingru_sparse_post")
    ckdir = tmp_path / "ck"
    ckdir.mkdir()
    shutil.copy(ck, ckdir / "tiny.pt")
    corpus = tmp_path / "c.bytes"
    corpus.write_bytes(random.Random(0).randbytes(400) + (b"int main(void) { return 0; }\n" * 40))
    monkeypatch.setattr(pm, "CKPTS", {"tiny": "tiny"})
    monkeypatch.setattr(pm, "SPLITS", {"dev": 10, "test": 500})
    monkeypatch.setattr(pm, "BLOCK", 2)
    monkeypatch.setattr(pm, "T", 32)
    args = type("A", (), dict(threads=1, loss_dir=str(tmp_path / "loss"), ckpts="tiny", splits="dev,test", windows=4, corpus=str(corpus), ckpt_dir=str(ckdir)))()
    claims = tmp_path / "loss" / "claims"
    claims.mkdir(parents=True)
    (claims / "tiny_dev_b001.lock").write_text("")                       # another worker has dev block 1
    pm.run_worker(args)
    got = sorted(p.name for p in (tmp_path / "loss").glob("*.npy"))
    assert got == ["tiny_dev_b000.npy", "tiny_test_b000.npy", "tiny_test_b001.npy"]
    assert not list((tmp_path / "loss").glob("*.tmp.npy"))
    assert json.loads((tmp_path / "loss" / "tiny_dev_b000.json").read_text())["windows"] == 2
    with pytest.raises(FileNotFoundError, match="blocks \\[1\\]"):
        pm.load_loss(tmp_path / "loss", "tiny", "dev", 4, block=2)
    assert pm.complete_windows(tmp_path / "loss", ["tiny"], ["dev", "test"], 2) == 2
    (claims / "tiny_dev_b001.lock").unlink()
    pm.run_worker(args)                                                  # the missing block is now free; the finished ones are left alone
    full = pm.load_loss(tmp_path / "loss", "tiny", "dev", 4, block=2)
    x, y = pm.load_windows(corpus, 10, 4, 32)
    ref = pm.score_per_byte(pm.load_model(ckdir, "tiny"), x, y)
    assert full.shape == (4, 32) and np.allclose(full, ref, atol=1e-5)
    assert pm.complete_windows(tmp_path / "loss", ["tiny"], ["dev", "test"], 2) == 4


def test_the_strata_cut_the_windows_by_the_longest_orders_hit_share_and_the_gains_add_up():
    rng = np.random.default_rng(6)
    W, T = 16, 32
    conf = np.zeros((2, W, T), dtype=np.uint8)
    conf[1] = (rng.random((W, T)) < (np.arange(W)[:, None] / W)).astype(np.uint8)       # window w has a hit at order 32 with probability w / 16
    pred = np.full((2, W, T), 5, dtype=np.uint8)
    y = np.full((W, T), 5, dtype=np.uint8)
    f = pm.SplitFeatures("dev", y, pred, conf, (4, 32))
    ce = rng.exponential(1.0, (W, T)).astype(np.float32)
    lam = pm.fit_lambdas(ce, f.right, f.bucket, 9)
    rows = pm.strata(ce, f, lam, 4)
    assert [r["windows"] for r in rows] == [4, 4, 4, 4]
    assert [r["longest_order_hit_share"] for r in rows] == sorted(r["longest_order_hit_share"] for r in rows)
    assert sum(r["share_of_total_gain"] for r in rows) == pytest.approx(1.0)
    assert sum(r["gain_cold"] * 4 for r in rows) / 16 == pytest.approx(pm.evaluate(ce, f.right, f.bucket, lam)["cold"]["gain"])


def test_the_analysis_runs_end_to_end_and_writes_one_json_per_split(tmp_path, monkeypatch, corpus, capsys):
    from flybrain.data.files import file_table
    from scripts.build_corpus_memory import build

    def body(seed: int, n: int) -> bytes:
        return random.Random(seed).randbytes(n).replace(b"\n", b" ")

    shared = body(100, 300)
    files = [(f"t{i}.c", shared + body(i, 500)) for i in range(8)]                        # training files share a 300-byte prefix
    files += [("h_twin.c", shared + body(0, 500))] + [(f"h{i}.c", shared[:150] + body(60 + i, 700)) for i in range(5)]   # held out: a twin of t0, then new files
    path = corpus(files)
    table = file_table(path)
    train_end = next(e.start for e in table if e.path == "h_twin.c")
    build(path, tmp_path / "mem", orders=(4, 8, 16, 32), log2_slots=12, chunk=256, train_end=train_end, log=lambda s: None)
    T, n = 32, 8
    monkeypatch.setattr(pm, "T", T)
    monkeypatch.setattr(pm, "N_WINDOWS", 16)
    monkeypatch.setattr(pm, "BLOCK", 4)
    twin_end = next(e.end for e in table if e.path == "h_twin.c")
    monkeypatch.setattr(pm, "SPLITS", {"dev": twin_end - 100, "test": twin_end + 1500})      # dev's first windows lie in the twin, test's are all in new files
    monkeypatch.setattr(pm, "CKPTS", {"ck1": "x1", "ck2": "x2"})
    rng = np.random.default_rng(0)
    loss = tmp_path / "loss"
    loss.mkdir()
    for name in ("ck1", "ck2"):
        for split in ("dev", "test"):
            for b in range(n // 4):
                np.save(loss / f"{name}_{split}_b{b:03d}.npy", rng.exponential(1.2, (4, T)).astype(np.float32))
    args = type("A", (), dict(mem_dir=str(tmp_path / "mem"), loss_dir=str(loss), out_dir=str(tmp_path), corpus=str(path), ckpts="ck1,ck2", windows=0, why="a test"))()
    res = pm.run_analysis(args)
    assert "### test (weights fitted on dev)" in capsys.readouterr().out
    twin = next(e for e in table if e.path == "h_twin.c")
    for split, offset in pm.SPLITS.items():
        rec = json.loads((tmp_path / f"mem_mix_{split}.json").read_text())
        assert rec["split"] == split and rec["yardstick"]["windows_scored"] == n and rec["yardstick"]["full_yardstick"] is False
        assert rec["yardstick"]["why_this_many_windows"] == "a test" and rec["memory"]["occupancy"]["4"] > 0
        gone = [i for i in range(n) if offset + i * T < twin.end and offset + i * T + T + 1 > twin.start]
        assert rec["duplicates"]["windows_removed"] == len(gone) and rec["duplicates"]["windows_kept"] == n - len(gone)
        assert (len(gone) > 0) == (split == "dev")                                            # the test is not vacuous: dev loses windows, test none
        assert rec["duplicates"]["heldout_files_with_a_training_twin"] == 1
        assert set(rec["checkpoints"]) == {"ck1", "ck2"} and rec["mixture"]["bootstrap"]["resamples"] == 1000
        for name, c in rec["checkpoints"].items():
            assert len(c["lambda"]) == 17 and c["lambda"][16] == 0.0 and len(c["buckets_cold"]) == 17 and c["score_check"] is None
            assert c["all"]["n_windows"] == n and c["excl_twins"]["n_windows"] == n - len(gone)
            assert all(np.isfinite(c["all"][k]["gain"]) for k in ("cold", "warm"))
            assert sum(b["gain_contribution"] for b in c["buckets_cold"]) == pytest.approx(c["all"]["cold"]["gain"])
    dev = json.loads((tmp_path / "mem_mix_dev.json").read_text())
    assert all(c["all"]["cold"]["gain"] >= 0 for c in dev["checkpoints"].values())             # fitted in sample
    assert set(res) == {"ck1", "ck2"}


def test_a_split_whose_windows_all_overlap_twins_has_no_clean_row_and_nothing_breaks():
    rng = np.random.default_rng(7)
    W, T = 6, 16
    conf = (rng.random((4, W, T)) < 0.3).astype(np.uint8)
    f = pm.SplitFeatures("dev", rng.integers(0, 4, (W, T)).astype(np.uint8), rng.integers(0, 4, (4, W, T)).astype(np.uint8), conf)
    ce = {"dev": rng.exponential(1.0, (W, T)).astype(np.float32), "test": rng.exponential(1.0, (W, T)).astype(np.float32)}
    keep = {"dev": np.zeros(W, dtype=bool), "test": np.ones(W, dtype=bool)}
    out = pm.analyse_checkpoint(ce, {"dev": f, "test": f}, keep)
    assert out["splits"]["dev"]["excl_twins"] == {"n_windows": 0, "fixed_lambda": None, "refit_lambda": None, "buckets_cold_refit": None}
    assert out["splits"]["test"]["excl_twins"]["refit_lambda"]["lambda"] == [0.0] * 17        # no clean dev window to fit on: no mixing


def test_the_cluster_bootstrap_resamples_runs_of_neighbouring_windows_and_is_wider_when_they_move_together():
    g = np.repeat(np.random.default_rng(8).normal(1.0, 0.5, 25), 8)      # 25 clusters of 8 identical windows: 200 windows, 25 independent draws
    lo_w, hi_w = pm.bootstrap_gain(g, 10)
    lo_c, hi_c = pm.bootstrap_gain(g, 10, block=8)
    assert hi_c - lo_c > 2.0 * (hi_w - lo_w)                              # the window bootstrap pretends there are 200 independent windows
    assert pm.bootstrap_gain(g, 10, block=1) == [lo_w, hi_w]
    assert pm.bootstrap_gain(np.full(64, 2.0), 10, block=8) == [0.2, 0.2]
    with pytest.raises(ValueError, match="do not divide"):
        pm.bootstrap_gain(g[:-1], 10, block=8)
    rng = np.random.default_rng(9)
    ce, bucket = rng.exponential(1.0, (16, 32)).astype(np.float32), rng.integers(0, 17, (16, 32)).astype(np.int8)
    r = pm.evaluate(ce, rng.random((16, 32)) < 0.5, bucket, np.zeros(17))
    assert set(r["cold"]) == {"alone", "mixed", "gain", "gain_ci95", "gain_ci95_clusters"}
    assert "gain_ci95_clusters" not in pm.evaluate(ce[:12], rng.random((12, 32)) < 0.5, bucket[:12], np.zeros(17))["cold"]   # 12 windows: not whole clusters
