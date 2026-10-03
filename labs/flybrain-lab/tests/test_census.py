"""Census bins: self-copyable (A), prefix-only (B) and unmatched (C) bytes."""

from __future__ import annotations


def test_ngram_next_maps_each_window_to_its_continuations():
    from flybrain.census import ngram_next
    assert ngram_next(b"abab", 2) == {b"ab": {ord("a")}, b"ba": {ord("b")}}
    assert ngram_next(b"abc", 3) == {} and ngram_next(b"", 2) == {}


def test_bins_on_a_hand_built_stream():
    from flybrain.census import A, B, C, bins
    stream = b"xxabcdabcd"                      # header "xx", body "abcdabcd"; n = 3
    prefix = b"bcdaQcdababcd"                   # holds bcd->a, cda->{Q, b}, dab->a, abc->d
    #            q:   2  3  4  5  6  7  8  9      a  b  c  d  a  b  c  d
    assert bins(stream, 2, 8, prefix, 3) == [C, C, C, B, B, B, C, A]
    # q=2 has only two bytes before it; q=3 'xxa', q=4 'xab' occur nowhere; q=5 abc->d and q=6 bcd->a come from the prefix;
    # q=7 cda->b is one of cda's two continuations in the prefix; q=8 dab->c is not (dab->a only); q=9 abc->d was seen at q=5 (A beats B)
    assert bins(stream, 2, 8, b"", 3) == [C, C, C, C, C, C, C, A]            # no prefix: only self-copies
    assert bins(stream, 6, 4, prefix, 3) == [B, B, C, A]                       # a shorter scored region keeps the header and earlier body as context


def test_a_bin_is_the_labs_copyable_mask():
    import torch
    from flybrain.census import A, bins
    from flybrain.data.corpus import copyable_mask
    torch.manual_seed(0)
    s = torch.randint(0, 4, (1, 300))           # tiny alphabet: repeats everywhere
    stream = bytes(s[0].tolist())
    for n in (2, 4, 8):
        codes = bins(stream, 1, len(stream) - 1, b"", n)
        assert [c == A for c in codes] == copyable_mask(s[:, :-1], s[:, 1:], n=n)[0].tolist(), n


def test_a_scored_byte_with_fewer_than_n_bytes_before_it_is_c_and_never_wraps_around():
    from flybrain.census import A, B, C, bins
    assert bins(b"abc", 1, 2, b"abcabcabcabc", 8) == [C, C]                   # the whole stream is shorter than n
    assert bins(b"abababab", 0, 8, b"abababab", 4) == [C, C, C, C, B, B, A, A]  # q < n has no window (a negative slice would wrap)
    assert bins(b"hello", 5, 0, b"hello", 2) == []                             # nothing scored


def test_bins_reject_a_scored_region_outside_the_stream():
    import pytest
    from flybrain.census import bins
    with pytest.raises(ValueError):
        bins(b"abc", 2, 5, b"", 2)
    with pytest.raises(ValueError):
        bins(b"abc", -1, 2, b"", 2)
