"""flybrain/data/text.py: the corpus is held as uint8 (one byte per byte, so a 5 GB corpus is 5 GB in memory), windows come out as
int64, and the chunked n-gram baselines equal the plain loops they replaced."""
import math

import numpy as np
import torch

from flybrain.data.text import batch_windows, bytes_tensor, fixed_windows, load_bytes, ngram_bpb


def reference_ngram_bpb(train, val, order):
    """The loops the chunked ngram_bpb replaced (verbatim from the old text.py)."""
    train_list, val_list = train.tolist(), val.tolist()
    if order == 1:
        counts = [1] * 256
        for b in train_list:
            counts[int(b)] += 1
        total = float(sum(counts))
        return sum(-math.log2(counts[int(b)] / total) for b in val_list) / max(len(val_list), 1)
    counts = {}
    for i in range(len(train_list) - 1):
        prev, nxt = int(train_list[i]), int(train_list[i + 1])
        if prev not in counts:
            counts[prev] = [1] * 256
        counts[prev][nxt] += 1
    uni = [1] * 256
    for b in train_list:
        uni[int(b)] += 1
    uni_tot = float(sum(uni))
    nll, n = 0.0, 0
    for i in range(len(val_list) - 1):
        prev, nxt = int(val_list[i]), int(val_list[i + 1])
        row = counts.get(prev)
        nll += -math.log2(uni[nxt] / uni_tot) if row is None else -math.log2(row[nxt] / float(sum(row)))
        n += 1
    return nll / max(n, 1)


def test_load_bytes_is_uint8_and_windows_are_int64(tmp_path):
    raw = bytes(range(256)) * 40 + b"tail"
    p = tmp_path / "c.bytes"
    p.write_bytes(raw)
    data = load_bytes(p)
    assert data.dtype == torch.uint8 and data.numel() == len(raw) and bytes(data.numpy().tobytes()) == raw
    part = load_bytes(p, max_bytes=100, offset=300)
    assert part.dtype == torch.uint8 and part.numel() == 100 and part.numpy().tobytes() == raw[300:400]
    assert load_bytes(p, max_bytes=10 ** 9, offset=len(raw) - 5).numel() == 5           # past the end: what is there, as a slice would
    x, y = batch_windows(data, seq_len=32, batch=4, seed=3)
    assert x.dtype == y.dtype == torch.long and x.shape == (4, 32) and torch.equal(x[:, 1:], y[:, :-1])
    xf, yf = fixed_windows(data, 16, 8)
    assert xf.dtype == yf.dtype == torch.long and xf.shape == (8, 16) and torch.equal(xf.reshape(-1)[1:], yf.reshape(-1)[:-1])
    assert torch.equal(xf, fixed_windows(data.to(torch.long), 16, 8)[0])                 # the same windows from an int64 corpus


def test_ngram_bpb_matches_the_plain_loops_across_chunks():
    g = np.random.default_rng(0)
    train = torch.from_numpy(g.choice(np.arange(256, dtype=np.uint8), size=5_000, p=np.r_[np.full(40, 0.02), np.full(216, 0.2 / 216)]))
    val = torch.from_numpy(g.choice(np.arange(256, dtype=np.uint8), size=700))             # some previous bytes unseen in train: the unigram path
    for order in (1, 2):
        ref = reference_ngram_bpb(train, val, order)
        assert math.isclose(ngram_bpb(train, val, order), ref, rel_tol=1e-9)
        assert math.isclose(ngram_bpb(train, val, order, chunk=333), ref, rel_tol=1e-9)   # chunk boundaries fall mid-pair
        assert math.isclose(ngram_bpb(train.to(torch.long), val.to(torch.long), order, chunk=1000), ref, rel_tol=1e-9)
    text = bytes_tensor("abcabcabd")
    assert math.isclose(ngram_bpb(text, text, 2), reference_ngram_bpb(text, text, 2), rel_tol=1e-9)
    assert ngram_bpb(train, val[:0], 1) == 0.0 and ngram_bpb(train, val[:1], 2) == 0.0
