"""Byte corpus helpers: 256-symbol UTF-8, no tokenizer."""

from __future__ import annotations

import torch

from flybrain.data.text import batch_windows, bytes_tensor, load_bytes, ngram_bpb, split_bytes


def test_bytes_tensor_is_utf8_values():
    t = bytes_tensor("Ab\n")
    assert t.dtype == torch.long
    assert t.tolist() == [65, 98, 10]


def test_batch_windows_are_shifted_pairs():
    data = bytes_tensor("abcdefghij")
    x, y = batch_windows(data, seq_len=4, batch=2, seed=0)
    assert x.shape == (2, 4)
    assert y.shape == (2, 4)
    for b in range(2):
        matched = False
        for i in range(int(data.numel()) - 4):
            if torch.equal(x[b], data[i : i + 4]) and torch.equal(y[b], data[i + 1 : i + 5]):
                matched = True
                break
        assert matched, "window is not a contiguous next-byte pair from the corpus"


def test_split_bytes_is_prefix_suffix_with_no_overlap():
    data = bytes_tensor("0123456789abcdef")
    train, val = split_bytes(data, val_frac=0.25)
    assert train.numel() + val.numel() == data.numel()
    assert torch.equal(train, data[: train.numel()])
    assert torch.equal(val, data[train.numel() :])
    assert train.numel() == 12 and val.numel() == 4


def test_unigram_constant_stream_beats_mixed_stream():
    const = bytes_tensor("a" * 40)
    mixed = bytes_tensor("abcdefghijklmnopqrstuvwxyz012345")
    c_train, c_val = split_bytes(const, val_frac=0.25)
    m_train, m_val = split_bytes(mixed, val_frac=0.25)
    assert ngram_bpb(c_train, c_val, order=1) < ngram_bpb(m_train, m_val, order=1)
    assert ngram_bpb(c_train, c_val, order=1) < 8.0


def test_load_bytes_reads_raw_file(tmp_path):
    path = tmp_path / "chunk.bin"
    path.write_bytes(b"Hi\n")
    t = load_bytes(path)
    assert t.tolist() == [72, 105, 10]
    clipped = load_bytes(path, max_bytes=2)
    assert clipped.tolist() == [72, 105]
