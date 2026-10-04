"""Speed round, row 2, tests b and c: the corpus table's prediction as a network input (b) and hashed n-gram embedding tables (c).
CPU only, tiny models, tiny synthetic corpora written here, no checkpoint from disk.

Part 1 (this section): the table's prediction for EVERY position of a corpus, for the training positions without a leak."""

from __future__ import annotations

import json
import random
import sys

import numpy as np
import pytest

from flybrain.corpus_memory import CorpusMemory


def noise(seed: int, n: int) -> bytes:
    return random.Random(seed).randbytes(n)


def features(tmp_path, data: bytes, train_end: int, *, orders=(4, 8), log2=16, chunk=64, name="f", **kw):
    """build_features on `data` -> (record, {order: pred array}, {order: conf array}) read back from the files, indexed by absolute position."""
    from scripts.build_corpus_memory import build_features
    corpus = tmp_path / f"{name}.bytes"
    corpus.write_bytes(data)
    rec = build_features(corpus, tmp_path / name, orders=orders, log2_slots=log2, chunk=chunk, train_end=train_end, log=lambda s: None, **kw)
    out = tmp_path / name
    return rec, {n: np.load(out / f"pred_n{n}.npy") for n in orders}, {n: np.load(out / f"conf_n{n}.npy") for n in orders}


# ---- Part 1: features for every position, no leak ------------------------------------------------------------------------------

def test_a_context_whose_only_occurrence_is_the_position_itself_is_a_miss(tmp_path):
    data = noise(0, 600)                                                 # no context of 4 or 8 bytes occurs twice in 600 random bytes
    rec, pred, conf = features(tmp_path, data, train_end=600)
    for n in (4, 8):
        assert len(conf[n]) == len(data) and conf[n].dtype == pred[n].dtype == np.uint8   # one entry per absolute position
        assert not conf[n].any() and not pred[n].any(), n                # not a single hit: no position sees its own continuation
    mem = CorpusMemory((4, 8), 16)                                       # the leak this rules out: look up after the bytes were written
    mem.fit(np.frombuffer(data, dtype=np.uint8), 0, 600, chunk=64)
    _, leaked = mem.lookup(np.frombuffer(data, dtype=np.uint8), 8, 600)
    assert (leaked > 0).mean() > 0.99                                    # nearly every position would be a hit on itself: the test is not vacuous


def test_a_context_first_seen_in_an_earlier_chunk_is_a_hit_with_the_byte_that_followed_it(tmp_path):
    unit = noise(1, 20)
    raw = bytearray(noise(2, 400))
    raw[10:30] = unit                                                    # first occurrence, chunk 0 (chunk = 64)
    raw[150:170] = unit                                                  # second occurrence, chunk 2
    rec, pred, conf = features(tmp_path, bytes(raw), train_end=400)
    for n in (4, 8):
        first = [p for p in range(10 + n, 30)]                           # the unit's first pass: nothing earlier holds these contexts
        assert not conf[n][first].any(), n
        second = [p for p in range(150 + n, 170)]
        assert (conf[n][second] > 0).all() and (pred[n][second] == np.frombuffer(bytes(raw), dtype=np.uint8)[second]).all(), n
        assert conf[n][second].tolist() == [1] * len(second)             # seen once so far: confidence 1


def test_a_context_repeated_inside_its_own_chunk_is_not_seen_until_the_chunk_has_been_written(tmp_path):
    unit = noise(3, 20)
    raw = bytearray(noise(4, 128))
    raw[10:30] = unit                                                    # both occurrences in chunk 0 when chunk = 64 ...
    raw[40:60] = unit
    p = 40 + 8                                                           # an order-8 context inside the second one
    _, _, conf = features(tmp_path, bytes(raw), train_end=128, chunk=64, name="same")
    assert conf[8][p] == 0 and conf[4][p] == 0                           # ... so the second is a miss: nothing from its own chunk
    _, pred2, conf2 = features(tmp_path, bytes(raw), train_end=128, chunk=16, name="split")
    assert conf2[8][p] > 0 and pred2[8][p] == raw[p]                     # chunk = 16 puts the first occurrence in an earlier chunk: a hit


def test_training_features_are_the_lookup_made_before_the_chunk_is_written(tmp_path):
    rng = np.random.default_rng(5)
    data = rng.integers(0, 3, 2000).astype(np.uint8)                     # three symbols: contexts, slots and bytes repeat all the time
    orders, log2, chunk, train_end = (3, 6), 8, 100, 1500
    rec, pred, conf = features(tmp_path, data.tobytes(), train_end, orders=orders, log2=log2, chunk=chunk)
    ref = CorpusMemory(orders, log2, boundary=train_end)                 # the definition, written out: lookup, then write, chunk by chunk
    want_pred = np.zeros((2, 2000), dtype=np.uint8)
    want_conf = np.zeros_like(want_pred)
    for lo in range(0, train_end, chunk):
        hi = min(lo + chunk, train_end)
        want_pred[:, lo:hi], want_conf[:, lo:hi] = ref.lookup(data, lo, hi)
        ref.update(data, lo, hi)
    ref.freeze()
    want_pred[:, train_end:], want_conf[:, train_end:] = ref.lookup(data, train_end, 2000)
    for k, n in enumerate(orders):
        assert np.array_equal(pred[n], want_pred[k]) and np.array_equal(conf[n], want_conf[k]), n
    assert conf[3].any() and (pred[3][conf[3] > 0] != data[conf[3] > 0]).any() and (pred[3][conf[3] > 0] == data[conf[3] > 0]).any()   # hits, some right some wrong
    assert rec["table_digest"] == {str(n): __import__("hashlib").blake2b(ref.tables[n].view(np.uint8), digest_size=16).hexdigest() for n in orders}


def test_held_out_features_are_the_lookups_in_the_frozen_table(tmp_path):
    unit = noise(6, 40)
    head = unit * 15 + noise(7, 100)                                     # 700 bytes of training text
    tail = unit * 4 + noise(8, 100) + unit * 2                           # a held-out tail that repeats the unit, then noise, then the unit
    data = np.frombuffer(head + tail, dtype=np.uint8)
    rec, pred, conf = features(tmp_path, head + tail, train_end=len(head), orders=(4, 8), log2=14, chunk=256)
    mem = CorpusMemory((4, 8), 14, boundary=len(head))
    mem.fit(data, 0, len(head), chunk=256)                               # the table of test a: training bytes only, frozen, then lookups
    mem.freeze()
    want_pred, want_conf = mem.lookup(data, len(head), len(data))
    for k, n in enumerate((4, 8)):
        assert np.array_equal(pred[n][len(head):], want_pred[k]) and np.array_equal(conf[n][len(head):], want_conf[k]), n
    assert (conf[8][len(head):] > 0).sum() > 100                         # the repeated unit is found; the tail's bytes were never written
    assert rec["heldout_positions"] == len(tail) and rec["train_end"] == len(head)


def test_the_feature_at_a_position_never_depends_on_the_byte_there_or_after_it(tmp_path):
    data = noise(9, 300)
    unit = noise(10, 30)
    raw = bytearray(data)
    raw[20:50] = unit
    raw[100:130] = unit
    raw[200:230] = unit                                                  # repeats across chunks, and into the held-out part (train_end = 250)
    base, base_pred, base_conf = features(tmp_path, bytes(raw), train_end=250, name="base")
    assert base_conf[8].any()
    for i, p in enumerate((0, 7, 8, 63, 64, 108, 129, 200, 249, 250, 280)):   # training positions on both sides of chunk edges, and held-out ones
        garbled = bytearray(raw)
        garbled[p:] = noise(100 + p, 300 - p)                            # the byte at p and everything after it, replaced
        _, pred, conf = features(tmp_path, bytes(garbled), train_end=250, name=f"g{i}")
        for n in (4, 8):
            assert np.array_equal(pred[n][:p + 1], base_pred[n][:p + 1]) and np.array_equal(conf[n][:p + 1], base_conf[n][:p + 1]), (n, p)


def test_the_hit_rates_by_tenth_are_the_counts_of_the_arrays(tmp_path):
    rng = np.random.default_rng(11)
    data = rng.integers(0, 4, 3000).astype(np.uint8)
    rec, pred, conf = features(tmp_path, data.tobytes(), train_end=2400, orders=(2, 5), log2=10, chunk=128)
    edges = [2400 * i // 10 for i in range(11)]
    for n in (2, 5):
        by_tenth = [float(np.count_nonzero(conf[n][a:b])) / (b - a) for a, b in zip(edges[:-1], edges[1:])]
        assert rec["train_hit_rate_by_tenth"][str(n)] == pytest.approx(by_tenth, abs=1e-12) and len(by_tenth) == 10
        right = [float(np.count_nonzero((conf[n][a:b] > 0) & (pred[n][a:b] == data[a:b]))) / max(np.count_nonzero(conf[n][a:b]), 1) for a, b in zip(edges[:-1], edges[1:])]
        assert rec["train_right_given_hit_by_tenth"][str(n)] == pytest.approx(right, abs=1e-12)
        assert rec["heldout_hit_rate"][str(n)] == pytest.approx(np.count_nonzero(conf[n][2400:]) / 600, abs=1e-12)
    assert rec["train_hit_rate_by_tenth"]["2"][0] < rec["train_hit_rate_by_tenth"]["2"][-1]      # the table fills as the pass goes on


def test_the_record_holds_orders_slots_chunk_seconds_and_digests_and_two_builds_agree(tmp_path):
    data = noise(12, 500) * 3
    a, *_ = features(tmp_path, data, train_end=1200, orders=(8, 4), log2=11, chunk=100, name="a")
    b, *_ = features(tmp_path, data, train_end=1200, orders=(4, 8), log2=11, chunk=100, name="b")
    on_disk = json.loads((tmp_path / "a" / "features.json").read_text())
    assert on_disk["orders"] == [4, 8] and on_disk["log2_slots"] == 11 and on_disk["slots_per_order"] == 2048 and on_disk["chunk"] == 100
    assert on_disk["build_seconds"] > 0 and on_disk["train_end"] == 1200 and on_disk["corpus_bytes"] == 1500
    assert set(on_disk["table_digest"]) == {"4", "8"} and set(on_disk["feature_digest"]) == {"pred_n4", "conf_n4", "pred_n8", "conf_n8"}
    assert a["table_digest"] == b["table_digest"] and a["feature_digest"] == b["feature_digest"]    # the order the orders are given in changes nothing


def test_compare_checks_the_held_out_part_against_the_plain_builds_arrays(tmp_path):
    from scripts.build_corpus_memory import build
    unit = noise(13, 40)
    head, tail = unit * 20 + noise(14, 60), unit * 5 + noise(15, 80)
    corpus = tmp_path / "c.bytes"
    corpus.write_bytes(head + tail)
    build(corpus, tmp_path / "plain", orders=(4, 8, 16), log2_slots=13, chunk=128, train_end=len(head), log=lambda s: None)
    rec, *_ = features(tmp_path, head + tail, train_end=len(head), orders=(8, 16), log2=13, chunk=128, compare=tmp_path / "plain")
    assert rec["compare"]["dir"] == str(tmp_path / "plain") and rec["compare"]["all_equal"] is True
    assert rec["compare"]["heldout_arrays_equal"] == {"8": True, "16": True} and rec["compare"]["table_digest_equal"] == {"8": True, "16": True}
    pred = np.load(tmp_path / "plain" / "pred_n8.npy")
    pred[len(head) + 50] ^= 1                                            # one held-out byte of the plain build differs
    np.save(tmp_path / "plain" / "pred_n8.npy", pred)
    rec, *_ = features(tmp_path, head + tail, train_end=len(head), orders=(8, 16), log2=13, chunk=128, name="g", compare=tmp_path / "plain")
    assert rec["compare"]["all_equal"] is False and rec["compare"]["heldout_arrays_equal"] == {"8": False, "16": True}
    with pytest.raises(ValueError, match="log2_slots"):
        features(tmp_path, head + tail, train_end=len(head), orders=(8,), log2=12, chunk=128, name="h", compare=tmp_path / "plain")   # another table size: not comparable


def test_the_features_mode_of_the_script_defaults_to_orders_8_16_32_and_writes_under_mem_feat(tmp_path, monkeypatch):
    import scripts.build_corpus_memory as bcm
    corpus = tmp_path / "c.bytes"
    corpus.write_bytes(noise(16, 3000) * 2)
    monkeypatch.setattr(bcm, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(sys, "argv", ["build_corpus_memory.py", "--features", "--corpus", str(corpus), "--log2-slots", "12", "--chunk", "512", "--train-end", "4000"])
    bcm.main()
    out = tmp_path / "results" / "mem" / "feat"
    info = json.loads((out / "features.json").read_text())
    assert info["orders"] == [8, 16, 32] and sorted(p.name for p in out.glob("*.npy")) == sorted(f"{k}_n{n}.npy" for k in ("pred", "conf") for n in (8, 16, 32))
    assert bcm.ORDERS == (4, 8, 16, 32) and bcm.FEATURE_ORDERS == (8, 16, 32)       # the plain mode keeps its orders
