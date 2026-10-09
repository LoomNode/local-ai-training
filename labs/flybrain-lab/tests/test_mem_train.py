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


# ---- Part 2 (test b): the table's prediction as a network input -----------------------------------------------------------------

import math  # noqa: E402  (the first part needs none of this)

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from flybrain.data.text import batch_starts, batch_windows, fixed_windows, load_bytes  # noqa: E402
from flybrain.model.hybrid import GRUHybrid  # noqa: E402

PERIOD, CORPUS_BYTES, TABLE_END, FIXED = 997, 200_000, 130_000, 150_000
SMALL = ["--task", "code", "--train-bytes", "3000", "--val-bytes", "4000", "--d-model", "16", "--n-layers", "2", "--d-ff", "32",
         "--n-key", "64", "--top-k", "4", "--key-conv", "2", "--normalize", "--chunk", "8", "--batch", "2", "--seq-len", "32",
         "--steps", "6", "--lr", "0.003", "--save", "--seed", "0", "--device", "cpu", "--run-name", "tiny"]
GOLDEN_TOP_KEYS = ["run_name", "task", "corpus_file", "n_spans", "save", "learned_key_dim", "arms", "device", "steps", "batch", "seq_len", "span", "lr", "seed",
                   "d_model", "n_key", "learned_key", "claws", "top_k", "theta", "n_layers", "d_ff", "decay", "gru_hidden", "key_conv", "freeze_conv",
                   "normalize", "signed", "shuffle_keys", "gate", "rope", "chunk", "amp", "state_bits", "mem_every", "key_convs", "count_head",
                   "learn_decay", "local_attn", "local_at", "decays", "warmup", "cosine", "wd", "train_bytes", "val_bytes", "val_fixed", "val_offset",
                   "eval_every", "meter", "ckpt_every", "unigram_bpb", "bigram_bpb"]       # results of the unmodified runner, a yardstick run
GOLDEN_ARM_KEYS = ["val_bpb", "val_bpb_first_q", "val_bpb_last_q", "val_bpb_copyable", "copyable_frac", "val_bpb_noncopyable", "arm", "seconds",
                   "trainable_scalars", "curve", "flops_per_token_recurrent", "train_seconds", "eval_seconds", "bytes_seen", "peak_vram_bytes",
                   "curve_fixed", "val_fixed"]
GOLDEN_STATE = {"mingru_sparse_post": (43, "6d9d2bf20ef90268"), "transformer": (29, "00e760820dac191c"), "mingru": (33, "e6f3d8933dbb86df")}   # tensors, hash of sorted (name, shape)
ARM_FLAGS = {"mingru_sparse_post": [], "transformer": ["--rope"], "mingru": ["--local-attn", "8"]}
TRAINED = ["--arms", "mingru", "--steps", "60", "--lr", "0.02", "--batch", "4", "--train-bytes", "40000"]     # a few CPU steps over windows from 40 periods


@pytest.fixture(scope="module")
def repeat_world(tmp_path_factory):
    """A 200,000-byte corpus that is one random 997-byte block repeated, and the table features of orders 4 and 8 built on its first 130,000
    bytes (chunks of 512, 2 ** 22 slots so that no two of the 997 contexts share a slot). Every context is followed by the same byte, so
    the table is right wherever it hits, and a 32-byte window of the random block predicts nothing. The confidence counter of a context
    is the number of periods seen, so the training windows reach the classes 1, 2-3, 4-15 and 16+ (from byte 16,000 on) as eval windows do."""
    from scripts.build_corpus_memory import build_features
    root = tmp_path_factory.mktemp("repeat_world")
    corpus = root / "rep.bytes"
    corpus.write_bytes((noise(21, PERIOD) * (CORPUS_BYTES // PERIOD + 1))[:CORPUS_BYTES])
    build_features(corpus, root / "feat", orders=(4, 8), log2_slots=22, chunk=512, train_end=TABLE_END, log=lambda s: None)
    return type("World", (), dict(corpus=corpus, feat=root / "feat", root=root))


def tiny(arm: str, **kw) -> GRUHybrid:
    """The three arms of the spine at a CPU size: the transformer, the minGRU backbone with local attention, the hybrid with memory heads."""
    torch.manual_seed(3)
    if arm == "transformer":
        return GRUHybrid(d_model=16, hidden=16, key="sparse", where="none", backbone="transformer", n_layers=2, d_ff=32, max_len=64, rope=True, **kw)
    if arm == "mingru":
        return GRUHybrid(d_model=16, hidden=16, key="sparse", where="none", backbone="mingru", n_layers=2, d_ff=32, local_attn=8, **kw)
    return GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=2, d_ff=32, n_key=64, top_k=4, key_conv=2, normalize=True,
                     gate=True, chunk=8, mem_every=1, local_attn=8, **kw)


ARMS = ("transformer", "mingru", "hybrid")


def kw(mem) -> dict:
    """The keyword tensors forward takes for a (pred, conf) pair of feature tensors."""
    return {"mem_pred": mem[0], "mem_conf": mem[1]}


def run_main(monkeypatch, tmp_path, tag, corpus, *flags):
    """The runner's main() in-process, results into tmp_path/tag: (the run's JSON, the results dir)."""
    import scripts.run_sparsela as sra
    res = tmp_path / tag
    res.mkdir()
    monkeypatch.setattr(sra, "RESULTS", res)
    monkeypatch.setattr(sra, "_CMASK", {})
    monkeypatch.setattr(sys, "argv", ["run_sparsela.py", "--corpus-file", str(corpus), *SMALL, *flags])
    sra.main()
    return json.loads((res / "tiny.json").read_text()), res


def old_forward(m: GRUHybrid, tokens):
    """GRUHybrid.forward as it was before the extra input channels, copied line by line: with no switch on the new forward must give
    the same numbers bit for bit."""
    e = m.embed(tokens)
    if m.where == "pre":
        z = m.mem.ln1(e)
        e = e + m.mem.memory(z, z.roll(1, dims=0) if m.shuffle_keys else None)
    if m.backbone == "gru":
        out, hstate = m.gru(e, None)
    else:
        out = m._run_core(e)
        hstate = out[:, -1]
    if m.where == "post":
        z = m.mem.ln1(out)
        out = out + m.mem.memory(z, z.roll(1, dims=0) if m.shuffle_keys else None)
    logits = m.decoder(out)
    if m.count is not None:
        log_pmem, g = m.count(out, tokens)
        logits = torch.logaddexp(torch.log1p(-g) + F.log_softmax(logits, dim=-1), g.log() + log_pmem)
    return logits, hstate


# -- nothing changes with no switch -----

@pytest.mark.parametrize("arm", ARMS)
def test_with_no_switch_logits_loss_and_gradients_are_the_old_forwards_bit_for_bit(arm):
    m = tiny(arm)
    assert m.table is None
    assert not [n for n, _ in m.named_parameters() if n.split(".")[0] in ("table", "ngram", "match")]
    g = torch.Generator().manual_seed(1)
    x, y = torch.randint(0, 256, (3, 32), generator=g), torch.randint(0, 256, (3, 32), generator=g)
    got = {}
    for name, fwd in (("new", lambda: m(x)), ("old", lambda: old_forward(m, x))):
        m.zero_grad(set_to_none=True)
        logits, h = fwd()
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        got[name] = (logits.detach().clone(), h.detach().clone(), loss.detach().clone(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None})
    for a, b in zip(got["new"][:3], got["old"][:3]):
        assert torch.equal(a, b)
    assert got["new"][3].keys() == got["old"][3].keys() and all(torch.equal(got["new"][3][k], got["old"][3][k]) for k in got["new"][3])


@pytest.mark.parametrize("arm", ["mingru_sparse_post", "transformer", "mingru"])
def test_a_tiny_run_with_no_switch_writes_the_same_json_fields_and_checkpoint_tensors(monkeypatch, tmp_path, repeat_world, arm):
    import hashlib
    out, res = run_main(monkeypatch, tmp_path, "a", repeat_world.corpus, "--arms", arm, *ARM_FLAGS[arm], "--val-fixed", "4", "--val-offset", "110000")
    assert list(out) == GOLDEN_TOP_KEYS and list(out["arms"][0]) == GOLDEN_ARM_KEYS            # same fields, same order, nothing added
    ck = torch.load(res / "ckpt" / f"tiny_{arm}.pt", map_location="cpu", weights_only=False)
    n, sig = GOLDEN_STATE[arm]
    assert sorted(ck) == ["args", "arm", "state"] and len(ck["state"]) == n
    assert hashlib.blake2b(repr(sorted((k, tuple(v.shape)) for k, v in ck["state"].items())).encode(), digest_size=8).hexdigest() == sig
    assert not {"mem_feat", "mem_orders", "ngram_emb", "ngram_rows", "ngram_dim", "ngram_sparse", "match_feat"} & set(ck["args"])


# -- the window carries the features of the byte it predicts -----

def test_batch_starts_are_the_starts_batch_windows_cuts():
    data = torch.randint(0, 256, (500,))
    for seed in (0, 5, 123):
        x, y = batch_windows(data, 32, 6, seed)
        starts = batch_starts(500, 32, 6, seed)
        assert starts.shape == (6,) and starts.min() >= 0 and starts.max() <= 500 - 33
        for b, s in enumerate(starts.tolist()):
            assert torch.equal(x[b], data[s:s + 32]) and torch.equal(y[b], data[s + 1:s + 33])
    with pytest.raises(ValueError, match="shorter than"):
        batch_starts(32, 32, 2, 0)


def test_the_store_serves_the_features_of_position_p_plus_one_for_the_input_byte_at_p(repeat_world):
    from flybrain.data.memfeat import FeatureStore
    store = FeatureStore(repeat_world.feat, (8, 4))                      # the order the caller gives is the order of the last axis
    assert store.orders == (8, 4) and store.train_end == TABLE_END and store.size == CORPUS_BYTES
    raw = {(k, n): np.load(repeat_world.feat / f"{k}_n{n}.npy") for k in ("pred", "conf") for n in (4, 8)}
    starts = torch.tensor([5, 1500, 2222])
    pred, conf = store.windows(starts, 32)
    assert pred.shape == conf.shape == (3, 32, 2) and pred.dtype == conf.dtype == torch.uint8
    for b, s in enumerate(starts.tolist()):
        for t in (0, 7, 31):
            for k, n in enumerate((8, 4)):
                assert pred[b, t, k] == raw[("pred", n)][s + t + 1] and conf[b, t, k] == raw[("conf", n)][s + t + 1]
    assert conf[1].any()
    at_p, _ = store.at(starts, 32)                                       # the raw read: the position itself, one byte early
    assert at_p[1, 5, 0] == raw[("pred", 8)][1505]
    pred2, conf2 = store.fixed(FIXED, 4, 32)
    assert pred2.shape == (4, 32, 2)
    for i in (0, 3):
        for t in (0, 31):
            assert pred2[i, t, 1] == raw[("pred", 4)][FIXED + i * 32 + t + 1]


def test_the_store_refuses_orders_it_does_not_hold_windows_past_the_file_and_another_corpus(repeat_world):
    from flybrain.data.memfeat import FeatureStore
    with pytest.raises(ValueError, match="orders .*16"):
        FeatureStore(repeat_world.feat, (4, 16))
    store = FeatureStore(repeat_world.feat, (4,))
    with pytest.raises(ValueError, match="past the end"):
        store.windows(torch.tensor([CORPUS_BYTES - 20]), 32)
    with pytest.raises(FileNotFoundError):
        FeatureStore(repeat_world.root, (4,))


def test_features_of_training_and_fixed_windows_predict_the_target_byte_and_shifted_by_one_they_do_not(repeat_world):
    from flybrain.data.memfeat import FeatureStore
    store = FeatureStore(repeat_world.feat, (4, 8))
    train = load_bytes(repeat_world.corpus, max_bytes=40_000)
    T = 32
    for seed in range(4):                                                # the training path: batch_windows + batch_starts
        x, y = batch_windows(train, T, 16, seed)
        starts = batch_starts(train.numel(), T, 16, seed)
        pred, conf = store.windows(starts, T)
        hit = conf > 0
        assert hit.float().mean() > 0.5                                  # windows from the first period or two have nothing earlier to hit
        target = y[:, :, None].expand_as(pred)
        assert (pred[hit].long() == target[hit]).all()                   # the table is always right here: its byte is the byte the window predicts
        late, late_conf = store.at(starts, T)                            # the same read one byte early: the byte the model is GIVEN
        h2 = late_conf > 0
        assert (late[h2].long() == x[:, :, None].expand_as(late)[h2]).all() and (late[h2].long() == target[h2]).float().mean() < 0.02
    data = load_bytes(repeat_world.corpus, max_bytes=8 * T + 1, offset=FIXED)
    xf, yf = fixed_windows(data, T, 8)                                   # the fixed path
    pred, conf = store.fixed(FIXED, 8, T)
    hit = conf > 0
    assert hit.float().mean() > 0.99 and (pred[hit].long() == yf[:, :, None].expand_as(pred)[hit]).all()


def _large_bonus(m: GRUHybrid, value: float = 30.0):
    for lin in m.table.bonus:
        lin.bias.data.fill_(value)
    return m


@pytest.mark.parametrize("arm", ARMS)
def test_a_model_with_a_large_output_bonus_predicts_every_byte_with_a_hit_and_not_with_features_shifted_by_one(repeat_world, arm):
    from flybrain.data.memfeat import FeatureStore
    store = FeatureStore(repeat_world.feat, (4, 8))
    m = _large_bonus(tiny(arm, mem_orders=(4, 8))).eval()
    T = 32
    data = load_bytes(repeat_world.corpus, max_bytes=16 * T + 1, offset=FIXED)
    x, y = fixed_windows(data, T, 16)
    mem = store.fixed(FIXED, 16, T)
    hit = mem[1].any(-1).bool()
    assert hit.float().mean() > 0.99
    with torch.no_grad():
        right = m(x, **kw(mem))[0].argmax(-1) == y
        assert right[hit].all()                                          # every byte with a hit is predicted: the features are aligned
        late = store.at(FIXED + T * torch.arange(16), T)                  # shifted by one: each window's feature for the byte it is given
        wrong = m(x, **kw(late))[0].argmax(-1) == y
    assert wrong.float().mean() < 0.02                                   # and the model does not predict them with the features shifted
    train = load_bytes(repeat_world.corpus, max_bytes=40_000)            # the training path as well
    for seed in range(3):
        xt, yt = batch_windows(train, T, 16, seed)
        starts = batch_starts(train.numel(), T, 16, seed)
        mem = store.windows(starts, T)
        h = mem[1].any(-1).bool()
        with torch.no_grad():
            ok = m(xt, **kw(mem))[0].argmax(-1) == yt
        assert h.any() and ok[h].all()


# -- alignment through the runner's own code path, on a corpus whose answer is known by construction -----

UNIT_LEN, UNIT_EVERY, UNIT_FIRST = 24, 150, 40
PLANTED_HELD = (3100, 3600, 103_100, 103_400)                            # units in the fixed validation windows and in the legacy validation bytes


@pytest.fixture(scope="module")
def plant_world(tmp_path_factory):
    """110,000 bytes of random noise with one 24-byte unit U planted every 150 bytes from byte 40 over the first 3,000 (the table's
    training bytes) and again at 3,100 and 3,600 (the fixed validation windows) and at 103,100 and 103,400 (the legacy validation
    bytes; the byte before and after each copy is unique to it). The table (orders 4 and 8, chunks of 64) is built on the first 3,000 bytes. By construction: the input byte U[j] of a
    repeated unit (not the first) has, for order n, a hit that predicts U[j + 1] when n - 1 <= j <= 22, a hit (with whatever followed the
    unit) at j = 23, and a miss at j < n - 1; every other position (noise, and the first unit) is a miss."""
    from scripts.build_corpus_memory import build_features
    root = tmp_path_factory.mktemp("plant_world")
    unit = noise(31, UNIT_LEN)
    raw = bytearray(noise(32, 110_000))
    starts = [UNIT_FIRST + UNIT_EVERY * k for k in range(20)] + list(PLANTED_HELD)
    for i, o in enumerate(starts):
        raw[o:o + UNIT_LEN] = unit
        raw[o - 1], raw[o + UNIT_LEN] = 200 + i, 100 + i                 # the byte before and the byte after each copy differ from copy to copy: no context that reaches
                                                                         # them is ever seen twice by chance (a repeated noise byte would be a legitimate hit)
    corpus = root / "plant.bytes"
    corpus.write_bytes(bytes(raw))
    build_features(corpus, root / "feat", orders=(4, 8), log2_slots=22, chunk=64, train_end=3000, log=lambda s: None)
    return type("World", (), dict(corpus=corpus, feat=root / "feat", raw=bytes(raw), unit=unit, starts=starts))


def planted_expectation(world, p: int, n: int):
    """(hit, pred or None) of order n for the feature of the byte AFTER the input byte at absolute position p, from the planting alone."""
    for o in world.starts:
        if o <= p < o + UNIT_LEN:
            j = p - o
            if o == UNIT_FIRST or j < n - 1:
                return False, None                                       # the first unit has no earlier copy; the context reaches back into noise
            return True, (world.unit[j + 1] if j <= UNIT_LEN - 2 else None)
    return False, None


def planted_mismatches(world, calls, select=lambda training, s: True) -> tuple[list, int, int]:
    """Every (kind, window start, index, order, what) where a recorded model input does not carry the planted feature, how many positions
    were required to be hits, and how many windows were examined. calls: (training, tokens, mem_pred, mem_conf) as the model received them;
    select(training, window start) picks the windows to examine."""
    bad, required, windows = [], 0, 0
    for training, tokens, mem_pred, mem_conf in calls:
        for b in range(tokens.size(0)):
            w = bytes(tokens[b].tolist())
            s = world.raw.find(w)
            assert s >= 0 and world.raw.find(w, s + 1) < 0, "a window is not a unique piece of the corpus"
            if not select(training, s):
                continue
            windows += 1
            for t in range(tokens.size(1)):
                for k, n in enumerate((4, 8)):
                    hit, pred = planted_expectation(world, s + t, n)
                    got_hit, got_pred = bool(mem_conf[b, t, k] > 0), int(mem_pred[b, t, k])
                    required += hit
                    if got_hit != hit or (pred is not None and got_pred != pred):
                        bad.append(("train" if training else "eval", s, t, n, (hit, pred, got_hit, got_pred)))
    return bad, required, windows


def record_model_inputs(monkeypatch):
    """Wrap GRUHybrid.forward to keep what the runner hands it: (training mode, the bytes, mem_pred, mem_conf)."""
    calls = []
    orig = GRUHybrid.forward

    def spy(self, tokens, state=None, *, mem_pred=None, mem_conf=None):
        if mem_pred is not None:
            calls.append((self.training, tokens.clone(), mem_pred.clone(), mem_conf.clone()))
        return orig(self, tokens, state, mem_pred=mem_pred, mem_conf=mem_conf)

    monkeypatch.setattr(GRUHybrid, "forward", spy)
    return calls


def _planted_run(monkeypatch, tmp_path, world, tag="run"):
    calls = record_model_inputs(monkeypatch)
    run_main(monkeypatch, tmp_path, tag, world.corpus, "--arms", "mingru", "--steps", "30", "--batch", "4", "--train-bytes", "3000", "--val-bytes", "600",
             "--val-fixed", "20", "--val-offset", "3000", "--mem-feat", str(world.feat), "--mem-orders", "4,8")
    return calls


def test_the_feature_the_model_receives_at_index_t_is_the_lookup_of_the_bytes_ending_at_t_for_training_and_fixed_batches(monkeypatch, tmp_path, plant_world):
    calls = _planted_run(monkeypatch, tmp_path, plant_world)
    assert sum(c[0] for c in calls) == 30                                  # one call per training step, the rest are evaluations
    bad, required, n = planted_mismatches(plant_world, calls, lambda training, s: training)
    assert n == 30 * 4 and required > 300 and not bad, bad[:5]            # the training path: the bytes ending at t give the feature for the byte at t + 1
    bad, required, n = planted_mismatches(plant_world, calls, lambda training, s: not training and 3000 <= s < 3000 + 20 * 32)
    assert n >= 20 and required >= 2 * 2 * 15 * (n // 20) and not bad, bad[:5]    # the fixed-val path: units at 3,100 and 3,600, features of the frozen table
    bad, required, n = planted_mismatches(plant_world, calls, lambda training, s: not training and s >= 103_000)
    assert n >= 4 and required > 0 and not bad, bad[:5]                    # and the legacy validation windows (units at 103,100 and 103,400)


@pytest.mark.parametrize("shift", [0, 1, 2])
def test_the_planted_check_has_teeth_the_features_one_byte_early_or_late_are_caught(monkeypatch, tmp_path, plant_world, shift):
    from flybrain.data.memfeat import FeatureStore
    if shift:                                                            # the mistake it must catch: the window reads positions s + t + 1 - 1 or s + t + 1 + 1
        monkeypatch.setattr(FeatureStore, "windows", lambda self, starts, seq_len: self.at(torch.as_tensor(starts) + 1 + (1 if shift == 2 else -1), seq_len))
    calls = _planted_run(monkeypatch, tmp_path, plant_world, tag=f"s{shift}")
    bad, required, n = planted_mismatches(plant_world, calls, lambda training, s: training)
    assert required > 300
    assert (len(bad) > 100) == (shift != 0), len(bad)


# -- the channel: input side and output side -----

def test_confidence_classes_are_miss_1_2to3_4to15_16plus():
    from flybrain.model.inputs import conf_class
    c = torch.tensor([0, 1, 2, 3, 4, 15, 16, 17, 255], dtype=torch.uint8)
    assert conf_class(c).tolist() == [0, 1, 2, 2, 3, 3, 4, 4, 4]


def test_the_input_side_adds_the_byte_and_class_embeddings_of_a_hit_and_only_the_miss_class_for_a_miss():
    from flybrain.model.inputs import FeatureChannel
    ch = FeatureChannel(d_model=4, d_out=4, n_orders=2, n_classes=5)
    assert all(not p.any() for p in ch.parameters())                     # zero-initialised: it adds nothing at the start
    pred = torch.tensor([[[7, 9], [7, 9], [200, 0]]])
    cls = torch.tensor([[[0, 3], [2, 0], [4, 1]]])
    assert not ch.embed(pred, cls).any()
    g = torch.Generator().manual_seed(0)
    for p in ch.parameters():
        if p.dim() == 2 and p.size(1) == 4:
            p.data.copy_(torch.randn(p.shape, generator=g))
    e = ch.embed(pred, cls)
    assert e.shape == (1, 3, 4)
    byte, klass = ch.byte_emb, ch.class_emb
    want0 = klass[0].weight[0] + (byte[1].weight[9] + klass[1].weight[3])        # order 0 missed (class 0 only); order 1 hit byte 9 at class 3
    want1 = (byte[0].weight[7] + klass[0].weight[2]) + klass[1].weight[0]        # order 0 hit byte 7 at class 2; order 1 missed
    want2 = (byte[0].weight[200] + klass[0].weight[4]) + (byte[1].weight[0] + klass[1].weight[1])   # two hits; the byte embedding of byte 0 counts for a hit
    assert torch.allclose(e[0, 0], want0, atol=1e-6) and torch.allclose(e[0, 1], want1, atol=1e-6) and torch.allclose(e[0, 2], want2, atol=1e-6)
    assert torch.equal(ch.embed(torch.tensor([[[7, 9]]]), torch.tensor([[[0, 0]]])), ch.embed(torch.tensor([[[8, 8]]]), torch.tensor([[[0, 0]]])))   # a miss ignores the predicted byte


def test_the_output_side_adds_a_class_selected_linear_bonus_to_the_logit_of_each_hit_only():
    from flybrain.model.inputs import FeatureChannel
    d = 3
    ch = FeatureChannel(d_model=4, d_out=d, n_orders=2, n_classes=5)
    logits = torch.zeros(1, 4, 256)
    h = torch.tensor([[[1.0, 2.0, 3.0]] * 4])
    assert torch.equal(ch.add_bonus(logits, h, torch.zeros(1, 4, 2, dtype=torch.long), torch.zeros(1, 4, 2, dtype=torch.long)), logits)   # zeros at the start
    for o, lin in enumerate(ch.bonus):                                   # one output per hit class: order o, class c -> weights (o + 1, c, 0), bias 10 * c
        for c in range(4):
            lin.weight.data[c] = torch.tensor([o + 1.0, float(c), 0.0])
            lin.bias.data[c] = 10.0 * (c + 1)
    pred = torch.tensor([[[5, 5], [6, 7], [8, 9], [11, 11]]])
    cls = torch.tensor([[[1, 1], [2, 0], [0, 0], [4, 3]]])
    out = ch.add_bonus(logits, h, pred, cls)
    bonus = lambda o, c: (o + 1.0) * 1.0 + float(c - 1) * 2.0 + 10.0 * c   # linear in h = (1, 2, 3); the class c picks row c - 1: weights (o + 1, c - 1, 0), bias 10 c
    assert out[0, 0, 5].item() == pytest.approx(bonus(0, 1) + bonus(1, 1)) and (out[0, 0] != 0).sum() == 1      # two orders, one byte: the bonuses add
    assert out[0, 1, 6].item() == pytest.approx(bonus(0, 2)) and (out[0, 1] != 0).sum() == 1                    # the second order missed: no bonus on byte 7
    assert not out[0, 2].any()                                                                                # no hit: untouched
    assert out[0, 3, 11].item() == pytest.approx(bonus(0, 4) + bonus(1, 3)) and (out[0, 3] != 0).sum() == 1
    assert out.shape == logits.shape and not logits.any()               # the input logits are not modified in place


# -- zero at the start, learning, plumbing -----

@pytest.mark.parametrize("arm", ARMS)
def test_with_table_features_on_the_model_at_initialisation_gives_exactly_the_plain_models_logits(repeat_world, arm):
    from flybrain.data.memfeat import FeatureStore
    store = FeatureStore(repeat_world.feat, (4, 8))
    plain, on = tiny(arm).eval(), tiny(arm, mem_orders=(4, 8)).eval()
    assert set(plain.state_dict()) < set(on.state_dict()) and all(torch.equal(v, on.state_dict()[k]) for k, v in plain.state_dict().items())
    x, _ = fixed_windows(load_bytes(repeat_world.corpus, max_bytes=4 * 32 + 1, offset=FIXED), 32, 4)
    mem = store.fixed(FIXED, 4, 32)
    assert mem[1].any()
    with torch.no_grad():
        assert torch.equal(plain(x)[0], on(x, **kw(mem))[0])
    assert sum(p.numel() for p in on.table.parameters()) == on.table_params() == 2 * (256 * 16 + 5 * 16) + 2 * (16 * 4 + 4)


def test_a_model_with_table_features_refuses_to_run_without_them_and_a_plain_one_refuses_them():
    plain, on = tiny("mingru").eval(), tiny("mingru", mem_orders=(4, 8)).eval()
    x = torch.randint(0, 256, (2, 32))
    mem = (torch.zeros(2, 32, 2, dtype=torch.uint8), torch.zeros(2, 32, 2, dtype=torch.uint8))
    with pytest.raises(ValueError, match="table features"):
        on(x)
    with pytest.raises(ValueError, match="table features"):
        on(x, mem_pred=mem[0])                                           # one of the two tensors is not enough
    with pytest.raises(ValueError, match="no table"):
        plain(x, **kw(mem))
    with pytest.raises(ValueError, match="one entry per order"):
        on(x, mem_pred=mem[0][..., :1], mem_conf=mem[1][..., :1])        # one order given, two built
    with pytest.raises(ValueError, match="table features"):
        on.init_state(2)                                                 # streaming has no way to feed them


def test_mem_dropout_withdraws_the_table_at_the_set_fraction_of_positions_while_training_and_never_at_evaluation():
    """--mem-dropout (consolidation test 3): at mem_drop_p = 1 the training forward equals the forward on all-miss inputs, at 0 the full
    forward; evaluation ignores it; a fraction in between lands in between on the loss the bonus makes; the compiled forward agrees."""
    m = _large_bonus(tiny("transformer", mem_orders=(4, 8)))
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    pred = y[..., None].expand(2, 32, 2).to(torch.uint8).clone()                      # the table always right
    conf = torch.full((2, 32, 2), 20, dtype=torch.uint8)
    miss = (torch.zeros_like(pred), torch.zeros_like(conf))
    loss = lambda mod, mem: F.cross_entropy(mod(x, **kw(mem))[0].reshape(-1, 256), y.reshape(-1)).item()
    with torch.no_grad():
        m.eval(); full_eval, miss_eval = loss(m, (pred, conf)), loss(m, miss)
        assert full_eval < miss_eval - 1.0                                              # the table is worth a lot on this data
        m.train(); m.mem_dropout = 1.0
        m.mem_drop_p.fill_(1.0); assert abs(loss(m, (pred, conf)) - miss_eval) < 1e-5
        m.mem_drop_p.fill_(0.0); assert abs(loss(m, (pred, conf)) - full_eval) < 1e-5
        torch.manual_seed(0); m.mem_drop_p.fill_(0.5); half = loss(m, (pred, conf))
        assert full_eval + 0.2 < half < miss_eval - 0.2
        m.eval(); assert abs(loss(m, (pred, conf)) - full_eval) < 1e-5                   # evaluation never drops
        m.train(); m.mem_dropout = 0.0; m.mem_drop_p.fill_(1.0); assert abs(loss(m, (pred, conf)) - full_eval) < 1e-5   # the switch off
        m.mem_dropout = 1.0; torch.manual_seed(0); m.mem_drop_p.fill_(0.5)
        got = torch.compile(m, backend="eager", fullgraph=True)(x, **kw((pred, conf)))[0]
        torch.manual_seed(0); want = m(x, **kw((pred, conf)))[0]
    assert torch.equal(want, got)


def test_the_new_forward_has_no_control_flow_that_depends_on_feature_values():
    """torch.compile will trace this forward (the speed branch compiles the training step): with fullgraph dynamo refuses any Python branch
    on a tensor's value, and the compiled forward must give the eager numbers."""
    m = tiny("transformer", mem_orders=(4, 8)).eval()
    for lin in m.table.bonus:
        lin.bias.data.fill_(0.5)
    x = torch.randint(0, 256, (2, 32))
    mem = (torch.randint(0, 256, (2, 32, 2), dtype=torch.uint8), torch.randint(0, 40, (2, 32, 2), dtype=torch.uint8))
    with torch.no_grad():
        want = m(x, **kw(mem))[0]
        got = torch.compile(m, backend="eager", fullgraph=True)(x, **kw(mem))[0]
    assert torch.equal(want, got)


def test_with_table_features_a_few_cpu_steps_give_a_lower_loss_than_the_plain_model_on_text_the_table_predicts(monkeypatch, tmp_path, repeat_world):
    """The windows trained on must carry the confidence classes the held-out windows do: a class that no training window has is an input
    embedding and a logit bonus that never moved from zero, and the model then reads worse than the plain one where it meets it (measured
    while writing this: 7.79 against 4.73 bits when training windows came from the first 3 periods and eval windows sat at 100)."""
    from flybrain.data.memfeat import FeatureStore
    from flybrain.model.inputs import conf_class
    store = FeatureStore(repeat_world.feat, (4, 8))
    seen = torch.cat([conf_class(store.windows(batch_starts(40_000, 32, 4, 100_000 * 0 + step), 32)[1]).reshape(-1) for step in range(60)])
    evl = conf_class(store.fixed(FIXED, 16, 32)[1]).reshape(-1)
    assert set(evl.unique().tolist()) <= set(seen.unique().tolist()) and (seen == 4).float().mean() > 0.2
    flags = [*TRAINED, "--val-fixed", "16", "--val-offset", str(FIXED)]
    plain, _ = run_main(monkeypatch, tmp_path, "plain", repeat_world.corpus, *flags)
    fed, res = run_main(monkeypatch, tmp_path, "fed", repeat_world.corpus, *flags, "--mem-feat", str(repeat_world.feat), "--mem-orders", "4,8")
    a, b = plain["arms"][0]["val_fixed"]["bpb"], fed["arms"][0]["val_fixed"]["bpb"]
    assert math.isfinite(a) and math.isfinite(b)
    assert b < a - 0.5, (a, b)                                           # a 32-byte window of random bytes predicts nothing; the table does
    assert all(math.isfinite(r["bpb"]) for r in fed["arms"][0]["curve_fixed"])
    assert fed["arms"][0]["val_bpb"] < plain["arms"][0]["val_bpb"] - 0.5  # the legacy validation bytes are fed too


def test_the_run_records_the_feature_directory_the_orders_and_the_added_parameters(monkeypatch, tmp_path, repeat_world):
    out, res = run_main(monkeypatch, tmp_path, "r", repeat_world.corpus, "--arms", "mingru", "--mem-feat", str(repeat_world.feat), "--mem-orders", "4,8")
    assert out["mem_feat"] == str(repeat_world.feat) and out["mem_orders"] == "4,8"
    arm = out["arms"][0]
    added = 2 * (256 * 16 + 5 * 16) + 2 * (16 * 4 + 4)
    assert arm["mem_feat"]["dir"] == str(repeat_world.feat) and arm["mem_feat"]["orders"] == [4, 8] and arm["mem_feat"]["added_params"] == added
    assert arm["mem_feat"]["table_digest"] == json.loads((repeat_world.feat / "features.json").read_text())["table_digest"]
    plain, _ = run_main(monkeypatch, tmp_path, "p", repeat_world.corpus, "--arms", "mingru")
    assert arm["trainable_scalars"] == plain["arms"][0]["trainable_scalars"] + added
    ck = torch.load(res / "ckpt" / "tiny_mingru.pt", map_location="cpu", weights_only=False)
    assert ck["args"]["mem_feat"] == str(repeat_world.feat) and ck["args"]["mem_orders"] == "4,8"
    assert [k for k in ck["state"] if k.startswith("table.")]            # the new tensors are in the checkpoint


def test_a_checkpoint_trained_with_table_features_loads_through_rebuild(monkeypatch, tmp_path, repeat_world):
    from scripts.cross_eval import rebuild
    for arm, extra in (("mingru", []), ("transformer", ["--rope"]), ("mingru_sparse_post", [])):
        _, res = run_main(monkeypatch, tmp_path, arm, repeat_world.corpus, "--arms", arm, *extra, "--mem-feat", str(repeat_world.feat), "--mem-orders", "4,8")
        ck = torch.load(res / "ckpt" / f"tiny_{arm}.pt", map_location="cpu", weights_only=False)
        m = rebuild(ck)
        assert m.table is not None and m.table_orders == (4, 8)
        assert all(torch.equal(v, ck["state"][k]) for k, v in m.state_dict().items()) and m.state_dict().keys() == ck["state"].keys()


def _cross_eval(monkeypatch, tmp_path, ckpt, corpus, *flags):
    import scripts.cross_eval as ce
    monkeypatch.setattr(ce, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["cross_eval.py", "--ckpts", str(ckpt), "--corpus-file", str(corpus), "--device", "cpu", "--run-name", "ce", *flags])
    ce.main()
    return json.loads((tmp_path / "ce.json").read_text())


def test_cross_eval_fixed_with_mem_feat_reproduces_the_in_run_number_and_without_it_refuses(monkeypatch, tmp_path, repeat_world):
    out, res = run_main(monkeypatch, tmp_path, "t", repeat_world.corpus, "--arms", "mingru", "--steps", "40", "--lr", "0.02", "--train-bytes", "40000",
                        "--val-fixed", "8", "--val-offset", str(FIXED), "--mem-feat", str(repeat_world.feat), "--mem-orders", "4,8")
    ck = res / "ckpt" / "tiny_mingru.pt"
    in_run = out["arms"][0]["val_fixed"]
    rows = _cross_eval(monkeypatch, tmp_path, ck, repeat_world.corpus, "--offset", str(FIXED), "--fixed", "8", "--lens", "32", "--mem-feat", str(repeat_world.feat))["rows"]
    assert rows[0]["bpb"] == pytest.approx(in_run["bpb"], abs=1e-5) and rows[0]["bpb_warm"] == pytest.approx(in_run["bpb_warm"], abs=1e-5)
    assert in_run["bpb"] < 7.0                                           # the table is doing work in the number being reproduced
    with pytest.raises(SystemExit, match="trained with table features"):
        _cross_eval(monkeypatch, tmp_path, ck, repeat_world.corpus, "--offset", str(FIXED), "--fixed", "8", "--lens", "32")
    with pytest.raises(SystemExit, match="--fixed"):
        _cross_eval(monkeypatch, tmp_path, ck, repeat_world.corpus, "--offset", str(FIXED), "--lens", "32", "--mem-feat", str(repeat_world.feat))
    plain, pres = run_main(monkeypatch, tmp_path, "plain", repeat_world.corpus, "--arms", "mingru")
    with pytest.raises(SystemExit, match="without table features"):
        _cross_eval(monkeypatch, tmp_path, pres / "ckpt" / "tiny_mingru.pt", repeat_world.corpus, "--offset", str(FIXED), "--fixed", "8", "--lens", "32", "--mem-feat", str(repeat_world.feat))
    rows = _cross_eval(monkeypatch, tmp_path, pres / "ckpt" / "tiny_mingru.pt", repeat_world.corpus, "--offset", str(FIXED), "--fixed", "8", "--lens", "32")["rows"]   # a plain checkpoint scores as before
    assert math.isfinite(rows[0]["bpb"])


def test_the_runner_refuses_inconsistent_table_feature_switches(monkeypatch, tmp_path, repeat_world, capsys):
    from scripts.run_sparsela import parse_args
    for flags, msg in ((["--mem-orders", "4,8"], "--mem-feat"), (["--mem-feat", "x", "--task", "copy"], "text task"),
                       (["--task", "code", "--mem-feat", "x", "--mem-orders", "8,8"], "distinct"), (["--task", "code", "--mem-feat", "x", "--mem-orders", "0,8"], "positive")):
        with pytest.raises(SystemExit):
            parse_args(flags)
        assert msg in capsys.readouterr().err
    args = parse_args(["--task", "code", "--mem-feat", "d"])
    assert args.mem_feat == "d" and args.mem_orders == "8,16,32"          # the default orders
    assert parse_args([]).mem_feat == ""                                   # off by default
    with pytest.raises(ValueError, match="orders .*16"):
        run_main(monkeypatch, tmp_path, "o", repeat_world.corpus, "--arms", "mingru", "--mem-feat", str(repeat_world.feat))   # the default 8,16,32 against a dir that holds 4 and 8
    with pytest.raises(ValueError, match="training bytes"):
        run_main(monkeypatch, tmp_path, "b", repeat_world.corpus, "--arms", "mingru", "--mem-feat", str(repeat_world.feat), "--mem-orders", "4,8", "--train-bytes", "140000")
    other = tmp_path / "other.bytes"
    other.write_bytes(b"x" * 9000)
    with pytest.raises(ValueError, match="another corpus"):
        run_main(monkeypatch, tmp_path, "c", other, "--arms", "mingru", "--mem-feat", str(repeat_world.feat), "--mem-orders", "4,8")
    with pytest.raises(ValueError, match="GRUHybrid"):
        run_main(monkeypatch, tmp_path, "g", repeat_world.corpus, "--arms", "gru", "--mem-feat", str(repeat_world.feat), "--mem-orders", "4,8")


# ---- Part 3 (test c): hashed n-gram embedding tables learned by gradient --------------------------------------------------------

NG_ORDERS, NG_ROWS, NG_DIM = (2, 3, 4, 6, 8), 512, 8


def ngram_row_ref(history: list[int], n: int, size: int) -> int:
    """The row of the n bytes ending a history, from the hash's statement with Python integers (the definition the tensor code is checked against):
    v = byte + 1; acc = v_t, then acc = (acc * MULT + v_(t-k+1)) mod P for k = 2..n; h = (acc + salt_n) mod P; h = (h * A1 + B1) mod P;
    h ^= h >> 15; h = (h * A2) mod P; row = h mod size. Orders whose every n-gram fits in `size` rows (256 ** n <= size) are indexed directly."""
    from flybrain.model import inputs as I
    last = history[-n:]
    if 256 ** n <= size:
        return sum(b * 256 ** j for j, b in enumerate(reversed(last)))
    acc = last[-1] + 1
    for b in reversed(last[:-1]):
        acc = (acc * I.NGRAM_MULT + b + 1) % I.NGRAM_P
    h = (acc + n * I.NGRAM_SALT) % I.NGRAM_P
    h = (h * I.NGRAM_A1 + I.NGRAM_B1) % I.NGRAM_P
    h ^= h >> 15
    h = (h * I.NGRAM_A2) % I.NGRAM_P
    return h % size


def _windows(kind: str, B: int = 3, T: int = 40) -> torch.Tensor:
    g = torch.Generator().manual_seed(5)
    if kind == "random":
        return torch.randint(0, 256, (B, T), generator=g)
    if kind == "repetitive":                                              # one byte, a short cycle, and a long run broken by one byte
        x = torch.zeros(B, T, dtype=torch.long)
        x[1] = torch.arange(T) % 3
        x[2, T // 2] = 255
        return x
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["random", "repetitive"])
def test_the_ngram_rows_are_the_scalar_hash_of_the_last_n_bytes_at_every_position_with_enough_history(kind):
    from flybrain.model.inputs import ngram_index, ngram_sizes
    x = _windows(kind)
    for rows in (NG_ROWS, 70_000):                                        # 70,000 rows: order 2 is direct (65,536 pairs), the others are hashed
        idx = ngram_index(x, NG_ORDERS, rows)
        sizes = ngram_sizes(NG_ORDERS, rows)
        assert idx.shape == (3, 40, len(NG_ORDERS)) and idx.dtype == torch.long
        assert sizes == ((65_536 if rows == 70_000 else rows),) + (rows,) * 4
        for b in range(3):
            for t in range(40):
                for k, n in enumerate(NG_ORDERS):
                    if t >= n - 1:
                        assert int(idx[b, t, k]) == ngram_row_ref(x[b, :t + 1].tolist(), n, sizes[k]), (kind, rows, b, t, n)


def test_the_ngram_hash_is_deterministic_depends_on_the_bytes_only_and_differs_between_orders():
    from flybrain.model.inputs import ngram_index
    x = _windows("random", B=8, T=64)
    a = ngram_index(x, NG_ORDERS, 262_144)
    assert torch.equal(a, ngram_index(x, NG_ORDERS, 262_144)) and torch.equal(a, ngram_index(x.clone(), NG_ORDERS, 262_144))
    assert torch.equal(a[..., 1:3], ngram_index(x, (3, 4), 262_144))      # an order's rows do not depend on which other orders are asked for
    assert torch.equal(ngram_index(x[2:5], NG_ORDERS, 262_144), a[2:5])  # nor on the other windows of the batch
    for k in range(1, 5):                                                  # the same bytes at two orders: different rows (a chance hit is 1 in 262,144)
        for j in range(k):
            both = slice(NG_ORDERS[k] - 1, None)
            assert (a[..., k][:, both] == a[..., j][:, both]).float().mean() < 0.01
    y = x.clone()                                                          # the same last-n bytes after different earlier ones: the same row
    y[:, :20] = torch.randint(0, 256, (8, 20))
    b = ngram_index(y, NG_ORDERS, 262_144)
    assert torch.equal(a[:, 40:], b[:, 40:])                               # positions 40+ have 8 identical bytes behind them at every order
    flat = a[:, 8:, 4].reshape(-1).tolist()                                # 448 distinct random 8-grams: almost all get their own row
    assert len(set(flat)) >= len(flat) - 3


def test_order_two_is_direct_indexed_when_the_rows_allow_it_so_no_two_pairs_share_a_row():
    from flybrain.model.inputs import ngram_index
    a, b = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    x = torch.stack([a.reshape(-1), b.reshape(-1)], dim=1)                 # every ordered pair of bytes, as a window of two
    idx = ngram_index(x, (1, 2), 262_144)
    assert torch.equal(idx[:, 1, 0], x[:, 1]) and torch.equal(idx[:, 1, 1], x[:, 0] * 256 + x[:, 1])
    assert idx[:, 1, 1].unique().numel() == 65_536 and int(idx[:, 1, 1].max()) == 65_535


def test_orders_past_eight_index_without_overflow_and_leave_the_direct_orders_alone():
    from flybrain.model.inputs import ngram_index, ngram_sizes
    x = _windows("random", B=4, T=48)
    orders, rows = (2, 4, 8, 12, 16, 32), 2_097_152                      # 256 ** 8 is 2 ** 64: the direct row of order 9+ does not fit int64
    idx = ngram_index(x, orders, rows)
    sizes = ngram_sizes(orders, rows)
    assert idx.shape == (4, 48, 6) and sizes == (65_536,) + (rows,) * 5
    for k, size in enumerate(sizes):
        assert int(idx[..., k].min()) >= 0 and int(idx[..., k].max()) <= size
    assert torch.equal(idx[..., :3], ngram_index(x, (2, 4, 8), rows))     # the orders row 2c used give the same rows as before
    assert torch.equal(idx[:, 1:, 0], x[:, :-1] * 256 + x[:, 1:])          # order 2 is still direct


def test_the_reserved_row_is_used_exactly_where_fewer_than_n_bytes_of_history_lie_in_the_window():
    from flybrain.model.inputs import ngram_index, ngram_sizes
    x = _windows("random", B=4, T=24)
    for rows in (NG_ROWS, 70_000):
        sizes = ngram_sizes(NG_ORDERS, rows)
        idx = ngram_index(x, NG_ORDERS, rows)
        for k, n in enumerate(NG_ORDERS):
            short = torch.arange(24) < n - 1                               # index t has t + 1 bytes of history: fewer than n for t < n - 1
            assert (idx[:, short, k] == sizes[k]).all(), (rows, n)         # the reserved row, one past the hashed rows
            assert (idx[:, ~short, k] < sizes[k]).all() and (idx[:, ~short, k] >= 0).all()
    one = ngram_index(x, (1, 2), 262_144)
    assert (one[:, :, 0] < 256).all() and (one[:, 0, 1] == 65_536).all() and (one[:, 1:, 1] < 65_536).all()   # order 1 never lacks history; order 2 lacks it at index 0


def test_the_ngram_row_at_t_never_depends_on_bytes_after_t():
    from flybrain.model.inputs import ngram_index
    x = _windows("random", B=2, T=48)
    base = ngram_index(x, NG_ORDERS, NG_ROWS)
    for t in (0, 5, 17, 30, 47):
        y = x.clone()
        y[:, t + 1:] = torch.randint(0, 256, (2, 47 - t))
        assert torch.equal(ngram_index(y, NG_ORDERS, NG_ROWS)[:, :t + 1], base[:, :t + 1]), t


def test_the_table_has_one_block_per_order_with_a_reserved_row_and_a_zero_projection():
    from flybrain.model.inputs import NgramEmbedding, ngram_sizes
    emb = NgramEmbedding((2, 3), rows=NG_ROWS, dim=NG_DIM, d_model=16)
    assert emb.table.weight.shape == (2 * (NG_ROWS + 1), NG_DIM) and emb.proj.weight.shape == (16, NG_DIM)
    assert not emb.proj.weight.any() and emb.table.weight.std() > 0         # the projection starts at zero, the rows do not: the sum is exactly zero
    assert sum(p.numel() for p in emb.parameters()) == emb.n_params() == 2 * (NG_ROWS + 1) * NG_DIM + NG_DIM * 16
    big = NgramEmbedding((1, 2, 3), rows=262_144, dim=4, d_model=8)         # orders 1 and 2 direct: 256 and 65,536 rows (+ the reserved row each)
    assert big.table.weight.shape[0] == (256 + 1) + (65_536 + 1) + (262_144 + 1) and ngram_sizes((1, 2, 3), 262_144) == (256, 65_536, 262_144)
    x = _windows("random")
    assert not emb(x).any() and emb(x).shape == (3, 40, 16)
    state = torch.random.get_rng_state()
    NgramEmbedding((2, 3), rows=NG_ROWS, dim=NG_DIM, d_model=16)
    assert torch.equal(state, torch.random.get_rng_state())                # building the table does not touch the global generator


@pytest.mark.parametrize("arm", ARMS)
def test_with_ngram_embeddings_on_the_model_at_initialisation_gives_exactly_the_plain_models_logits(arm):
    plain, on = tiny(arm).eval(), tiny(arm, ngram_orders=NG_ORDERS, ngram_rows=NG_ROWS, ngram_dim=NG_DIM).eval()
    assert set(plain.state_dict()) < set(on.state_dict()) and all(torch.equal(v, on.state_dict()[k]) for k, v in plain.state_dict().items())
    assert {k for k in on.state_dict() if k not in plain.state_dict()} == {"ngram.table.weight", "ngram.proj.weight"}
    x = _windows("random", B=3, T=32)
    with torch.no_grad():
        assert torch.equal(plain(x)[0], on(x)[0])
    assert on.ngram_params() == sum(p.numel() for p in on.ngram.parameters()) == len(NG_ORDERS) * (NG_ROWS + 1) * NG_DIM + NG_DIM * 16
    assert plain.ngram is None and plain.ngram_params() == 0


def test_the_projection_learns_first_then_the_rows_which_start_with_no_gradient():
    m = tiny("mingru", ngram_orders=(2, 3), ngram_rows=NG_ROWS, ngram_dim=NG_DIM)
    x, y = _windows("random", B=3, T=32), _windows("random", B=3, T=32)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for step in range(2):
        opt.zero_grad()
        F.cross_entropy(m(x)[0].reshape(-1, 256), y.reshape(-1)).backward()
        if step == 0:
            assert m.ngram.proj.weight.grad.abs().sum() > 0 and not m.ngram.table.weight.grad.any()      # zero projection: no signal reaches the rows yet
        else:
            assert m.ngram.table.weight.grad.abs().sum() > 0                                              # after one step the projection is not zero
        opt.step()


def test_the_new_forward_with_ngram_embeddings_traces_with_fullgraph_and_matches_eager():
    m = tiny("transformer", ngram_orders=NG_ORDERS, ngram_rows=NG_ROWS, ngram_dim=NG_DIM, mem_orders=(4, 8)).eval()
    m.ngram.proj.weight.data.normal_(0, 0.1)
    x = torch.randint(0, 256, (2, 32))
    mem = (torch.randint(0, 256, (2, 32, 2), dtype=torch.uint8), torch.randint(0, 40, (2, 32, 2), dtype=torch.uint8))
    with torch.no_grad():
        want = m(x, **kw(mem))[0]
        got = torch.compile(m, backend="eager", fullgraph=True)(x, **kw(mem))[0]
    assert torch.equal(want, got) and not torch.equal(want, tiny("transformer", mem_orders=(4, 8)).eval()(x, **kw(mem))[0])


def test_streaming_refuses_ngram_embeddings():
    m = tiny("mingru", ngram_orders=(2,), ngram_rows=NG_ROWS, ngram_dim=NG_DIM).eval()
    with pytest.raises(ValueError, match="n-gram"):
        m.init_state(2)


NG = ["--ngram-emb", "2,3,4", "--ngram-rows", "4096", "--ngram-dim", "16"]


def test_with_ngram_embeddings_a_few_cpu_steps_give_a_lower_loss_than_the_plain_model_on_text_with_repeated_contexts(monkeypatch, tmp_path, repeat_world):
    flags = [*TRAINED, "--val-fixed", "16", "--val-offset", str(FIXED)]
    plain, _ = run_main(monkeypatch, tmp_path, "plain", repeat_world.corpus, *flags)
    fed, res = run_main(monkeypatch, tmp_path, "ngram", repeat_world.corpus, *flags, *NG)
    a, b = plain["arms"][0]["val_fixed"]["bpb"], fed["arms"][0]["val_fixed"]["bpb"]
    assert math.isfinite(a) and math.isfinite(b)
    assert b < a - 0.3, (a, b)                                           # a 32-byte window of random bytes predicts nothing; the table of contexts learns the block
    assert all(math.isfinite(r["bpb"]) for r in fed["arms"][0]["curve_fixed"]) and fed["arms"][0]["val_bpb"] < plain["arms"][0]["val_bpb"]


def test_the_run_records_orders_rows_dim_and_the_added_parameters_and_the_checkpoint_holds_the_tables(monkeypatch, tmp_path, repeat_world):
    out, res = run_main(monkeypatch, tmp_path, "r", repeat_world.corpus, "--arms", "mingru", *NG)
    assert out["ngram_emb"] == "2,3,4" and out["ngram_rows"] == 4096 and out["ngram_dim"] == 16 and out["ngram_sparse"] is False
    added = 3 * (4096 + 1) * 16 + 16 * 16
    arm = out["arms"][0]
    assert arm["ngram_emb"] == {"orders": [2, 3, 4], "rows": 4096, "dim": 16, "sparse": False, "added_params": added}
    plain, _ = run_main(monkeypatch, tmp_path, "p", repeat_world.corpus, "--arms", "mingru")
    assert arm["trainable_scalars"] == plain["arms"][0]["trainable_scalars"] + added
    assert "mem_feat" not in out and "mem_feat" not in arm
    ck = torch.load(res / "ckpt" / "tiny_mingru.pt", map_location="cpu", weights_only=False)
    assert ck["args"]["ngram_emb"] == "2,3,4" and ck["state"]["ngram.table.weight"].shape == (3 * 4097, 16) and ck["state"]["ngram.proj.weight"].any()


def test_a_checkpoint_with_ngram_tables_loads_through_rebuild_and_cross_eval_needs_no_flag_to_reproduce_the_in_run_number(monkeypatch, tmp_path, repeat_world):
    from scripts.cross_eval import rebuild
    for arm, extra in (("mingru", []), ("transformer", ["--rope"]), ("mingru_sparse_post", [])):
        _, res = run_main(monkeypatch, tmp_path, arm, repeat_world.corpus, "--arms", arm, *extra, *NG, "--ngram-sparse")
        ck = torch.load(res / "ckpt" / f"tiny_{arm}.pt", map_location="cpu", weights_only=False)
        m = rebuild(ck)
        assert m.ngram is not None and m.ngram.orders == (2, 3, 4) and m.ngram.rows == 4096 and m.ngram.dim == 16
        assert m.state_dict().keys() == ck["state"].keys() and all(torch.equal(v, ck["state"][k]) for k, v in m.state_dict().items())
    out, res = run_main(monkeypatch, tmp_path, "ce", repeat_world.corpus, *TRAINED, "--steps", "30", *NG, "--val-fixed", "8", "--val-offset", str(FIXED))
    rows = _cross_eval(monkeypatch, tmp_path, res / "ckpt" / "tiny_mingru.pt", repeat_world.corpus, "--offset", str(FIXED), "--fixed", "8", "--lens", "32")["rows"]
    assert rows[0]["bpb"] == pytest.approx(out["arms"][0]["val_fixed"]["bpb"], abs=1e-5)


def test_the_tables_are_in_the_main_optimizer_without_weight_decay_or_in_a_sparse_adam_of_their_own():
    import scripts.run_sparsela as sra
    args = sra.parse_args(["--task", "code", "--wd", "0.1", "--lr", "0.01", "--arms", "mingru", "--d-model", "16", "--n-layers", "2", "--d-ff", "32", *NG])
    model = sra.build("mingru", args)
    table = model.ngram.table.weight
    opt = sra.make_optimizer(model, args)
    assert sra.make_sparse_optimizer(model, args) is None and not model.ngram.table.sparse
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    free = {id(p) for g in opt.param_groups if g["weight_decay"] == 0 for p in g["params"]}
    assert id(table) in free and id(table) not in decayed                     # an ordinary parameter, but not decayed
    assert id(model.ngram.proj.weight) in decayed and id(model.embed.weight) in decayed       # the other matrices still are
    sparse_args = sra.parse_args(["--task", "code", "--wd", "0.1", "--lr", "0.01", "--arms", "mingru", "--d-model", "16", "--n-layers", "2", "--d-ff", "32", *NG, "--ngram-sparse"])
    model = sra.build("mingru", sparse_args)
    opt, sopt = sra.make_optimizer(model, sparse_args), sra.make_sparse_optimizer(model, sparse_args)
    assert model.ngram.table.sparse and isinstance(sopt, torch.optim.SparseAdam) and [p for g in sopt.param_groups for p in g["params"]] == [model.ngram.table.weight]
    assert id(model.ngram.table.weight) not in {id(p) for g in opt.param_groups for p in g["params"]}      # one optimizer per parameter
    nowd = sra.parse_args(["--task", "code", "--arms", "mingru", *NG, "--ngram-sparse"])                  # no weight decay: plain Adam without the table
    assert id(model.ngram.table.weight) not in {id(p) for g in sra.make_optimizer(sra.build("mingru", nowd), nowd).param_groups for p in g["params"]}


def test_sparse_gradients_train_with_a_sparse_adam_clip_like_dense_ones_and_the_run_saves_both_optimizers(monkeypatch, tmp_path, repeat_world):
    flags = [*TRAINED, "--steps", "20", "--wd", "0.1", *NG, "--ckpt-every", "10", "--val-fixed", "8", "--val-offset", str(FIXED)]
    dense, dres = run_main(monkeypatch, tmp_path, "dense", repeat_world.corpus, *flags)
    sparse, sres = run_main(monkeypatch, tmp_path, "sparse", repeat_world.corpus, *flags, "--ngram-sparse")
    assert sparse["arms"][0]["ngram_emb"]["sparse"] is True and sparse["arms"][0]["ngram_emb"]["added_params"] == dense["arms"][0]["ngram_emb"]["added_params"]
    for out in (dense, sparse):
        assert math.isfinite(out["arms"][0]["val_fixed"]["bpb"]) and out["arms"][0]["val_fixed"]["bpb"] < 8.0
    ck = torch.load(sres / "ckpt" / "tiny_mingru.step10.pt", map_location="cpu", weights_only=False)
    assert "opt_sparse" in ck and "opt_sparse" not in torch.load(dres / "ckpt" / "tiny_mingru.step10.pt", map_location="cpu", weights_only=False)
    moved = (ck["state"]["ngram.table.weight"] != torch.load(sres / "ckpt" / "tiny_mingru.pt", map_location="cpu", weights_only=False)["state"]["ngram.table.weight"]).any(1)
    assert moved.sum() > 100                                               # the rows the windows touched moved between step 10 and step 20


def test_the_runner_refuses_inconsistent_ngram_switches(capsys):
    from scripts.run_sparsela import parse_args
    for flags, msg in ((["--ngram-rows", "100"], "--ngram-emb"), (["--ngram-dim", "8"], "--ngram-emb"), (["--ngram-sparse"], "--ngram-emb"),
                       (["--ngram-emb", "2,2"], "distinct"), (["--ngram-emb", "0,3"], "positive"), (["--ngram-emb", "2,x"], "integers"),
                       (["--ngram-emb", "2", "--ngram-rows", "0"], "positive"), (["--ngram-emb", "2", "--ngram-dim", "0"], "positive")):
        with pytest.raises(SystemExit):
            parse_args(["--task", "code", *flags])
        assert msg in capsys.readouterr().err
    a = parse_args(["--task", "code", "--ngram-emb", "2,3,4,6,8"])
    assert (a.ngram_emb, a.ngram_rows, a.ngram_dim, a.ngram_sparse) == ("2,3,4,6,8", 262_144, 64, False)
    assert parse_args([]).ngram_emb == ""                                   # off by default


def test_table_features_and_ngram_embeddings_work_together_each_alone_and_zero_at_the_start(monkeypatch, tmp_path, repeat_world):
    from flybrain.data.memfeat import FeatureStore
    store = FeatureStore(repeat_world.feat, (4, 8))
    plain = tiny("mingru").eval()
    both = tiny("mingru", mem_orders=(4, 8), ngram_orders=NG_ORDERS, ngram_rows=NG_ROWS, ngram_dim=NG_DIM).eval()
    x, _ = fixed_windows(load_bytes(repeat_world.corpus, max_bytes=4 * 32 + 1, offset=FIXED), 32, 4)
    with torch.no_grad():
        assert torch.equal(plain(x)[0], both(x, **kw(store.fixed(FIXED, 4, 32)))[0])
    assert both.table_params() > 0 and both.ngram_params() > 0
    flags = [*TRAINED, "--val-fixed", "16", "--val-offset", str(FIXED)]
    fed, res = run_main(monkeypatch, tmp_path, "both", repeat_world.corpus, *flags, "--mem-feat", str(repeat_world.feat), "--mem-orders", "4,8", *NG)
    arm = fed["arms"][0]
    assert arm["mem_feat"]["added_params"] == 2 * (256 * 16 + 5 * 16) + 2 * (16 * 4 + 4)               # the same counts as each switch alone
    assert arm["ngram_emb"]["added_params"] == 3 * (4096 + 1) * 16 + 16 * 16 and fed["mem_feat"] and fed["ngram_emb"] == "2,3,4"
    plain_run, _ = run_main(monkeypatch, tmp_path, "plain", repeat_world.corpus, *flags)
    assert arm["val_fixed"]["bpb"] < plain_run["arms"][0]["val_fixed"]["bpb"] - 0.5
    rows = _cross_eval(monkeypatch, tmp_path, res / "ckpt" / "tiny_mingru.pt", repeat_world.corpus, "--offset", str(FIXED), "--fixed", "16", "--lens", "32", "--mem-feat", str(repeat_world.feat))["rows"]
    assert rows[0]["bpb"] == pytest.approx(arm["val_fixed"]["bpb"], abs=1e-5)
    with pytest.raises(SystemExit, match="trained with table features"):
        _cross_eval(monkeypatch, tmp_path, res / "ckpt" / "tiny_mingru.pt", repeat_world.corpus, "--offset", str(FIXED), "--fixed", "16", "--lens", "32")


# ---- Part 4: in-window match features (--match-feat) -----------------------------------------------------------------------------

MATCH_ORDERS = (1, 2, 3, 4, 8, 16, 32)
MF = ["--match-feat", "3,4"]


def match_ref(x: list[int], n: int) -> list[tuple[int, int]]:
    """(predicted byte, class) at each index t, by brute force from the statement: the n bytes ending at t (t >= n - 1, inside the window);
    the most recent earlier index s < t whose n bytes ending at s are the same gives the byte at s + 1, and the class is 1, 2 (2-3) or 3
    (4 or more) by the number of earlier indices with that context; otherwise (0, 0)."""
    last: dict = {}
    count: dict = {}
    out = []
    for t in range(len(x)):
        if t < n - 1:
            out.append((0, 0))
            continue
        ctx = tuple(x[t - n + 1:t + 1])
        if ctx in last:
            c = count[ctx]
            out.append((x[last[ctx] + 1], 1 if c == 1 else 2 if c < 4 else 3))
        else:
            out.append((0, 0))
        last[ctx] = t
        count[ctx] = count.get(ctx, 0) + 1
    return out


def match_windows(kind: str, T: int = 200) -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    if kind == "random":
        return torch.randint(0, 256, (3, T), generator=g)
    if kind == "small alphabet":                                          # many repeats at every order up to 4, a few at 8
        return torch.randint(0, 3, (4, T), generator=g)
    text = (b"static int probe_device(struct device *dev) { return -ENODEV; }\n" * 4)[:T]
    flipped = bytearray(text)
    for i in (37, 90, 151):
        flipped[i] ^= 0x20
    x = torch.zeros(6, T, dtype=torch.long)
    x[1] = torch.arange(T) % 3                                            # a cycle of 3
    x[2] = torch.arange(T) % 7                                            # a cycle of 7
    x[3, T // 2] = 255                                                    # a run of one byte broken once
    x[4] = torch.tensor(list(text))                                       # a line of C repeated
    x[5] = torch.tensor(list(flipped))                                    # the same with three bytes changed
    return x


@pytest.mark.parametrize("kind", ["random", "small alphabet", "repetitive"])
def test_the_match_features_are_the_brute_force_previous_occurrence_at_every_order(kind):
    from flybrain.model.inputs import match_features
    x = match_windows(kind)
    pred, cls = match_features(x, MATCH_ORDERS)
    assert pred.shape == cls.shape == (x.size(0), 200, len(MATCH_ORDERS)) and pred.dtype == cls.dtype == torch.long
    seen = set()
    for b in range(x.size(0)):
        for k, n in enumerate(MATCH_ORDERS):
            want = match_ref(x[b].tolist(), n)
            assert pred[b, :, k].tolist() == [w[0] for w in want] and cls[b, :, k].tolist() == [w[1] for w in want], (kind, b, n)
            seen |= {w[1] for w in want}
    if kind == "random":
        assert 0 in seen                                                    # random bytes: mostly misses (and a few chance repeats at orders 1 and 2)
    else:
        assert seen == {0, 1, 2, 3}                                         # the test is not vacuous: misses and all three hit classes occur


def test_match_features_are_causal_a_miss_where_there_is_no_history_and_no_false_match_among_random_bytes():
    from flybrain.model.inputs import match_features
    x = match_windows("small alphabet")
    base = match_features(x, (2, 4, 8))
    for t in (0, 5, 17, 60, 199):
        y = x.clone()
        y[:, t + 1:] = torch.randint(0, 256, (x.size(0), 199 - t))        # everything after t replaced
        got = match_features(y, (2, 4, 8))
        assert torch.equal(got[0][:, :t + 1], base[0][:, :t + 1]) and torch.equal(got[1][:, :t + 1], base[1][:, :t + 1]), t
    for k, n in enumerate((2, 4, 8)):
        assert not base[1][:, :n - 1, k].any()                              # fewer than n bytes in the window: a miss
    g = torch.Generator().manual_seed(2)
    r = torch.randint(0, 256, (16, 2048), generator=g)                    # a spine window of random bytes: no 8-, 16- or 32-gram occurs twice,
    pred, cls = match_features(r, (8, 16, 32))                            # so a single hit would be a collision of the 62-bit keys
    assert not cls.any() and not pred.any()


def test_the_match_keys_keep_62_bits_and_stay_below_two_to_the_62():
    from flybrain.model import inputs as I
    assert I.MATCH_KEY_BITS == 62 and I.MATCH_P == (1 << 31) - 1 and I.MATCH_M1 != I.MATCH_M2 and max(I.MATCH_M1, I.MATCH_M2) < I.MATCH_P
    assert (I.MATCH_P - 1) * I.MATCH_M1 + 256 < (1 << 62) and ((I.MATCH_P - 1) << 31) + I.MATCH_P - 1 < (1 << 62)       # every int64 product and key fits


def test_the_match_feature_at_t_predicts_the_byte_at_t_plus_one_known_by_construction():
    from flybrain.model.inputs import match_features
    block = noise(77, 13)                                                  # a random block repeated: the byte after any 4 bytes is the same each period
    x = torch.tensor(list((block * 4)[:40]))[None]
    y = torch.tensor(list((block * 4)[1:41]))[None]
    for order in (1, 4):
        pred, cls = match_features(x, (order,))
        for t in range(40):
            if t >= 13 + order - 1:                                       # the context ending at t occurred one period earlier: its byte is the target
                assert int(cls[0, t, 0]) > 0 and int(pred[0, t, 0]) == int(y[0, t]) != int(x[0, t]), (order, t)
    pred, cls = match_features(x, (4,))
    assert not cls[0, :13 + 3, 0].any()                                    # nothing earlier in the window: misses
    assert cls[0, 16:29, 0].tolist() == [1] * 13 and cls[0, 29:39, 0].tolist() == [2] * 10 and cls[0, 39, 0] == 2     # 1, then 2 earlier occurrences, then 3: classes 1, 2, 2


@pytest.mark.parametrize("arm", ARMS)
def test_with_match_features_on_the_model_at_initialisation_gives_exactly_the_plain_models_logits(arm):
    plain, on = tiny(arm).eval(), tiny(arm, match_orders=(4, 8)).eval()
    assert set(plain.state_dict()) < set(on.state_dict()) and all(torch.equal(v, on.state_dict()[k]) for k, v in plain.state_dict().items())
    assert all(k.startswith("match.") for k in on.state_dict() if k not in plain.state_dict()) and on.table is None and on.ngram is None
    x = match_windows("small alphabet", 32)
    with torch.no_grad():
        assert torch.equal(plain(x)[0], on(x)[0])
    assert on.match_params() == sum(p.numel() for p in on.match.parameters()) == 2 * (256 * 16 + 4 * 16) + 2 * (16 * 3 + 3)
    assert plain.match is None and plain.match_params() == 0


@pytest.mark.parametrize("arm", ARMS)
def test_a_model_with_a_large_match_bonus_predicts_the_matched_byte_in_a_repeated_block(arm):
    m = tiny(arm, match_orders=(4,)).eval()
    for lin in m.match.bonus:
        lin.bias.data.fill_(30.0)
    block = noise(78, 13)
    x = torch.tensor(list((block * 4)[:40]))[None].repeat(3, 1)
    y = torch.tensor(list((block * 4)[1:41]))[None].repeat(3, 1)
    with torch.no_grad():
        right = m(x)[0].argmax(-1) == y
    assert right[:, 16:].all()                                             # from the second period on every byte is the matched one


def test_the_match_channel_is_separate_from_the_table_channel_and_both_add_their_bonuses():
    m = tiny("mingru", mem_orders=(4, 8), match_orders=(4,), ngram_orders=(2, 3), ngram_rows=NG_ROWS, ngram_dim=NG_DIM).eval()
    assert {k.split(".")[0] for k in m.state_dict() if k.startswith(("table.", "match.", "ngram."))} == {"table", "match", "ngram"}
    assert m.table is not m.match and not {id(p) for p in m.table.parameters()} & {id(p) for p in m.match.parameters()}     # separate parameters
    assert m.table_params() > 0 and m.match_params() > 0 and m.ngram_params() > 0
    plain = tiny("mingru").eval()
    x = match_windows("small alphabet", 32)
    mem = (torch.randint(0, 256, (4, 32, 2), dtype=torch.uint8), torch.randint(0, 40, (4, 32, 2), dtype=torch.uint8))
    with torch.no_grad():
        assert torch.equal(plain(x)[0], m(x, **kw(mem))[0])                 # all three zero at the start: the plain logits exactly


def test_the_new_forward_with_match_features_traces_with_fullgraph_and_matches_eager():
    m = tiny("transformer", match_orders=(2, 4)).eval()
    for lin in m.match.bonus:
        lin.bias.data.fill_(0.5)
    m.match.byte_emb[0].weight.data.normal_(0, 0.1)
    x = match_windows("small alphabet", 32)
    with torch.no_grad():
        want = m(x)[0]
        got = torch.compile(m, backend="eager", fullgraph=True)(x)[0]
    assert torch.equal(want, got) and not torch.equal(want, tiny("transformer").eval()(x)[0])


def test_streaming_refuses_match_features():
    with pytest.raises(ValueError, match="match"):
        tiny("mingru", match_orders=(4,)).eval().init_state(2)


@pytest.fixture(scope="module")
def match_world(tmp_path_factory):
    """250,000 bytes made of segments: a fresh random 11-byte block over 64 symbols, repeated 6 times, then the next block. Nothing is
    learnable across segments (a fixed block would be memorised by the weights, which the plain model does within a few steps): the plain
    model's floor is log2(64) = 6 bits. But inside a 32-byte window the second period copies the first: from the 14th byte of a block on, the
    byte after the last 3 or 4 bytes is the byte that followed their earlier occurrence."""
    rng = random.Random(41)
    corpus = tmp_path_factory.mktemp("match_world") / "m.bytes"
    corpus.write_bytes(b"".join(bytes(rng.randrange(64) + 32 for _ in range(11)) * 6 for _ in range(3900))[:250_000])
    return corpus


def test_with_match_features_a_few_cpu_steps_give_a_lower_loss_than_the_plain_model_on_repeated_text(monkeypatch, tmp_path, match_world):
    flags = ["--arms", "mingru", "--steps", "100", "--lr", "0.005", "--batch", "4", "--train-bytes", "40000", "--val-fixed", "16", "--val-offset", "200000"]
    plain, _ = run_main(monkeypatch, tmp_path, "plain", match_world, *flags)
    fed, res = run_main(monkeypatch, tmp_path, "match", match_world, *flags, *MF)
    a, b = plain["arms"][0]["val_fixed"]["bpb"], fed["arms"][0]["val_fixed"]["bpb"]
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a - 6.0) < 0.5, a                                           # the plain model has learned all there is without copying: 64 equally likely symbols
    assert b < a - 1.5, (a, b)                                             # and the match features copy the repeat
    assert all(math.isfinite(r["bpb"]) for r in fed["arms"][0]["curve_fixed"]) and fed["arms"][0]["val_bpb"] < plain["arms"][0]["val_bpb"] - 1.5


def test_the_run_records_the_match_orders_and_the_added_parameters_and_the_checkpoint_holds_the_channel(monkeypatch, tmp_path, match_world):
    out, res = run_main(monkeypatch, tmp_path, "r", match_world, "--arms", "mingru", *MF)
    assert out["match_feat"] == "3,4" and "mem_feat" not in out and "ngram_emb" not in out
    added = 2 * (256 * 16 + 4 * 16) + 2 * (16 * 3 + 3)
    arm = out["arms"][0]
    assert arm["match_feat"] == {"orders": [3, 4], "added_params": added}
    plain, _ = run_main(monkeypatch, tmp_path, "p", match_world, "--arms", "mingru")
    assert arm["trainable_scalars"] == plain["arms"][0]["trainable_scalars"] + added
    ck = torch.load(res / "ckpt" / "tiny_mingru.pt", map_location="cpu", weights_only=False)
    assert ck["args"]["match_feat"] == "3,4" and [k for k in ck["state"] if k.startswith("match.")]


def test_a_checkpoint_with_match_features_loads_through_rebuild_and_cross_eval_needs_no_flag_to_reproduce_the_in_run_number(monkeypatch, tmp_path, match_world):
    from scripts.cross_eval import rebuild
    for arm, extra in (("mingru", []), ("transformer", ["--rope"]), ("mingru_sparse_post", [])):
        _, res = run_main(monkeypatch, tmp_path, arm, match_world, "--arms", arm, *extra, *MF)
        ck = torch.load(res / "ckpt" / f"tiny_{arm}.pt", map_location="cpu", weights_only=False)
        m = rebuild(ck)
        assert m.match is not None and m.match_orders == (3, 4) and m.table is None
        assert m.state_dict().keys() == ck["state"].keys() and all(torch.equal(v, ck["state"][k]) for k, v in m.state_dict().items())
    out, res = run_main(monkeypatch, tmp_path, "ce", match_world, "--arms", "mingru", "--steps", "30", "--lr", "0.02", "--train-bytes", "1000", *MF, "--val-fixed", "8", "--val-offset", "50000")
    rows = _cross_eval(monkeypatch, tmp_path, res / "ckpt" / "tiny_mingru.pt", match_world, "--offset", "50000", "--fixed", "8", "--lens", "32")["rows"]
    assert rows[0]["bpb"] == pytest.approx(out["arms"][0]["val_fixed"]["bpb"], abs=1e-5)


def test_the_runner_refuses_inconsistent_match_switches(capsys):
    from scripts.run_sparsela import parse_args
    for flags, msg in ((["--match-feat", "4,4"], "distinct"), (["--match-feat", "0,4"], "positive"), (["--match-feat", "a"], "integers")):
        with pytest.raises(SystemExit):
            parse_args(["--task", "code", *flags])
        assert msg in capsys.readouterr().err
    a = parse_args(["--task", "code", "--match-feat", "8,16,32"])
    assert a.match_feat == "8,16,32" and parse_args([]).match_feat == ""   # off by default


def test_match_features_work_with_the_table_features_and_the_ngram_tables_in_one_run(monkeypatch, tmp_path, repeat_world):
    flags = [*TRAINED, "--val-fixed", "16", "--val-offset", str(FIXED)]
    out, res = run_main(monkeypatch, tmp_path, "all", repeat_world.corpus, *flags, "--mem-feat", str(repeat_world.feat), "--mem-orders", "4,8", *NG, "--match-feat", "4,8")
    arm = out["arms"][0]
    assert set(arm) >= {"mem_feat", "ngram_emb", "match_feat"} and arm["match_feat"]["orders"] == [4, 8] and math.isfinite(arm["val_fixed"]["bpb"])
    plain, _ = run_main(monkeypatch, tmp_path, "plain", repeat_world.corpus, *flags)
    assert arm["val_fixed"]["bpb"] < plain["arms"][0]["val_fixed"]["bpb"] - 0.5
    only, ores = run_main(monkeypatch, tmp_path, "match_only", repeat_world.corpus, *flags, "--match-feat", "4,8")
    assert "mem_feat" not in only["arms"][0] and "ngram_emb" not in only["arms"][0]
    rows = _cross_eval(monkeypatch, tmp_path, res / "ckpt" / "tiny_mingru.pt", repeat_world.corpus, "--offset", str(FIXED), "--fixed", "16", "--lens", "32", "--mem-feat", str(repeat_world.feat))["rows"]
    assert rows[0]["bpb"] == pytest.approx(arm["val_fixed"]["bpb"], abs=1e-5)
