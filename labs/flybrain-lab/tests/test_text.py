"""Byte corpus helpers: 256-symbol UTF-8, no tokenizer."""

from __future__ import annotations

import pytest
import torch

from flybrain.data.text import batch_windows, bytes_tensor, fixed_windows, load_bytes, ngram_bpb, split_bytes


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


def test_fixed_windows_are_exact_slices_and_use_no_rng():
    data = bytes_tensor("abcdefghijklmnopqrstuvwxyz")
    x, y = fixed_windows(data, seq_len=5, n=4)
    assert x.shape == y.shape == (4, 5) and x.dtype == data.dtype
    for i in range(4):
        assert torch.equal(x[i], data[i * 5 : (i + 1) * 5]) and torch.equal(y[i], data[i * 5 + 1 : (i + 1) * 5 + 1])
    state = torch.get_rng_state()
    again = fixed_windows(data, 5, 4)
    assert torch.equal(state, torch.get_rng_state())                 # the global generator was not touched
    assert torch.equal(again[0], x) and torch.equal(again[1], y)     # and there is nothing random to differ
    x[0, 0] = 99
    assert int(data[0]) == ord("a")                                  # windows are copies, not views of the corpus


def test_fixed_windows_need_n_times_seq_len_plus_one_bytes():
    data = bytes_tensor("abcdefghij")                                # 10 bytes
    x, y = fixed_windows(data, seq_len=3, n=3)                       # needs exactly 3 * 3 + 1 = 10
    assert y[2].tolist() == data[7:10].tolist()
    with pytest.raises(ValueError, match="need 13 bytes for 4 windows of 3"):
        fixed_windows(data, seq_len=3, n=4)
    with pytest.raises(ValueError, match="data has 0"):
        fixed_windows(data[:0], seq_len=3, n=1)                      # a --val-offset past the end of the file reads nothing
    with pytest.raises(ValueError, match="n >= 1"):
        fixed_windows(data, seq_len=3, n=0)


def test_aligned_starts_are_multiples_of_the_window_and_leave_room_for_the_target():
    import torch
    from flybrain.data.text import batch_starts, batch_windows
    n, T = 10 * 2048 + 1, 2048                  # ten whole blocks and the one byte past the last target
    s = batch_starts(n, T, 64, 7, aligned=True)
    assert (s % T == 0).all() and int(s.max()) <= n - T - 1 and int(s.min()) >= 0
    assert set(int(v) // T for v in s) <= set(range(10)) and len(set(int(v) for v in s)) > 3
    data = torch.arange(n) % 251
    x, y = batch_windows(data, T, 4, 7, aligned=True)
    assert (x[:, 0] % 251 == (batch_starts(n, T, 4, 7, aligned=True) % 251)).all() and (y[:, 0] == (x[:, 0] + 1) % 251).all()
    assert not (batch_starts(n, T, 64, 7) % T == 0).all()   # the default is unaligned, as before
