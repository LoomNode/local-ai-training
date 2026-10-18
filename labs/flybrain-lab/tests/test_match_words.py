"""The word orders of match_features (900: the word prefix so far; 902: the last two words plus the prefix) against a pure-Python
reference, with the byte orders unchanged beside them."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flybrain.model import inputs as I
from flybrain.model.inputs import MATCH_WORD_PAIR, MATCH_WORD_PREFIX, match_features

ID_BYTES = set(range(48, 58)) | set(range(65, 91)) | set(range(97, 123)) | {95}        # [0-9A-Za-z_]
TEXT = Path("/games/ailab/corpora/pkgs1g.bytes")


def classify(c: int) -> int:
    return 1 if c == 1 else 2 if c < 4 else 3


def word_ref(x: list[int], order: int) -> list[tuple[int, int]]:
    """(predicted byte, class) at each index t from the statement. The run so far is the maximal run of identifier bytes ending at t and
    including t; a completed run is one that a non-identifier byte has closed at an index <= t. 900 keys on the run so far (a miss where
    it is empty); 902 on (the completed run before the last, the last completed run, the run so far, possibly empty), a miss with fewer
    than two completed runs. The prediction is the byte after the most recent earlier index with the same key; the class counts them."""
    words: list[tuple[int, ...]] = []
    run: list[int] = []
    seen: dict = {}
    out = []
    for t, b in enumerate(x):
        if b in ID_BYTES:
            run.append(b)
        else:
            if run:
                words.append(tuple(run))
            run = []
        if order == MATCH_WORD_PREFIX:
            key = tuple(run) if run else None
        else:
            key = (words[-2], words[-1], tuple(run)) if len(words) >= 2 else None
        if key is None:
            out.append((0, 0))
            continue
        earlier = seen.setdefault(key, [])
        out.append((x[earlier[-1] + 1], classify(len(earlier))) if earlier else (0, 0))
        earlier.append(t)
    return out


def byte_ref(x: list[int], n: int) -> list[tuple[int, int]]:
    """The original definition: the n bytes ending at t (t >= n - 1); the most recent earlier index with the same n bytes gives the byte after it."""
    last: dict = {}
    count: dict = {}
    out = []
    for t in range(len(x)):
        if t < n - 1:
            out.append((0, 0))
            continue
        ctx = tuple(x[t - n + 1:t + 1])
        out.append((x[last[ctx] + 1], classify(count[ctx])) if ctx in last else (0, 0))
        last[ctx] = t
        count[ctx] = count.get(ctx, 0) + 1
    return out


def random_windows(B: int = 3, T: int = 256, seed: int = 3) -> torch.Tensor:
    """Bytes from a small alphabet of letters, a digit, '_', spaces and punctuation, so that runs and word pairs repeat."""
    alphabet = torch.tensor(list(b"aab_1 ,.(\n"))
    g = torch.Generator().manual_seed(seed)
    return alphabet[torch.randint(0, alphabet.numel(), (B, T), generator=g)]


def text_window() -> torch.Tensor:
    if not TEXT.exists():
        pytest.skip(f"{TEXT} is not on this machine")
    with TEXT.open("rb") as f:
        f.seek(100000)
        return torch.tensor(list(f.read(2048)))[None]


def check_against(ref, x: torch.Tensor, orders: tuple[int, ...]) -> set[int]:
    pred, cls = match_features(x, orders)
    assert pred.shape == cls.shape == (x.size(0), x.size(1), len(orders)) and pred.dtype == cls.dtype == torch.long
    seen = set()
    for b in range(x.size(0)):
        for k, n in enumerate(orders):
            want = ref(x[b].tolist(), n)
            got = list(zip(pred[b, :, k].tolist(), cls[b, :, k].tolist()))
            assert got == want, (b, n, [t for t in range(len(want)) if got[t] != want[t]][:5])
            seen |= {w[1] for w in want}
    return seen


def test_the_word_orders_are_the_reference_on_random_windows_from_a_small_alphabet():
    seen = check_against(word_ref, random_windows(), (MATCH_WORD_PREFIX, MATCH_WORD_PAIR))
    assert seen == {0, 1, 2, 3}                                           # not vacuous: misses and all three hit classes occur


def test_the_word_orders_are_the_reference_on_a_window_of_real_text():
    seen = check_against(word_ref, text_window(), (MATCH_WORD_PREFIX, MATCH_WORD_PAIR))
    assert seen == {0, 1, 2, 3}


def test_the_word_pair_key_tells_apart_word_pairs_that_a_shared_multiplier_would_merge():
    # with one multiplier M for the words and their mix, ("a", "bc") and ("b", "ac") would share (v_a + v_b) M^2 + v_c M: the mix must not
    line = b"a bc x b ac x ab c x a bc y b ac y ab c y a_b c1 z b_a c1 z "
    x = torch.tensor(list(line * 2))[None]
    check_against(word_ref, x, (MATCH_WORD_PREFIX, MATCH_WORD_PAIR))


def test_the_byte_orders_are_unchanged_the_original_definition_beside_the_word_orders():
    x = torch.cat([random_windows(2, 256, 7), torch.randint(0, 4, (2, 256), generator=torch.Generator().manual_seed(1))])
    check_against(byte_ref, x, (4, 8))
    pred, cls = match_features(x, (4, 8, MATCH_WORD_PREFIX, MATCH_WORD_PAIR))
    pb, cb = match_features(x, (4, 8))
    assert torch.equal(pred[..., :2], pb) and torch.equal(cls[..., :2], cb)      # the byte columns do not depend on the word orders being asked
    pw, cw = match_features(x, (MATCH_WORD_PAIR, MATCH_WORD_PREFIX))
    assert torch.equal(pred[..., 3], pw[..., 0]) and torch.equal(cls[..., 2], cw[..., 1])    # and the columns follow the order given


def test_a_mixed_call_has_the_right_shapes_and_the_prefix_order_misses_on_every_non_identifier_byte():
    x = torch.cat([random_windows(2, 300, 5), text_window()[:, :300]])
    pred, cls = match_features(x, (8, MATCH_WORD_PREFIX, MATCH_WORD_PAIR))
    assert pred.shape == cls.shape == (3, 300, 3) and pred.dtype == cls.dtype == torch.long
    isid = torch.tensor([[int(b) in ID_BYTES for b in row] for row in x.tolist()])
    assert not cls[..., 1][~isid].any() and not pred[..., 1][~isid].any()
    assert cls[..., 1][isid].any()                                         # and hits on identifier bytes
    assert cls[..., 2][~isid].any()                                        # the pair order does predict at separators (the next word's first byte)
    assert (pred >= 0).all() and (pred <= 255).all() and (cls >= 0).all() and (cls < I.MATCH_CLASSES).all()
    pb, cb = match_features(x, (8,))
    assert torch.equal(pred[..., :1], pb) and torch.equal(cls[..., :1], cb)


def test_the_word_orders_are_causal_and_integer_valued_with_no_gradient():
    x = random_windows(3, 256, 9)
    base = match_features(x, (MATCH_WORD_PREFIX, MATCH_WORD_PAIR))
    for t in (0, 4, 31, 100, 254):
        y = x.clone()
        y[:, t + 1:] = torch.randint(0, 256, (3, 255 - t), generator=torch.Generator().manual_seed(t))
        got = match_features(y, (MATCH_WORD_PREFIX, MATCH_WORD_PAIR))
        assert torch.equal(got[0][:, :t + 1], base[0][:, :t + 1]) and torch.equal(got[1][:, :t + 1], base[1][:, :t + 1]), t
    assert not base[0].requires_grad and not base[1].requires_grad
    assert torch.equal(match_features(x, (MATCH_WORD_PREFIX, MATCH_WORD_PAIR))[0], base[0])    # deterministic


def test_the_word_mix_constants_are_distinct_odd_below_p_with_no_small_linear_relation():
    P = I.MATCH_P
    for ks in I.MATCH_WORD_MIX:
        assert len(set(ks)) == 3 and all(k % 2 == 1 and 0 < k < P for k in ks)
        k2, k1, k0 = ks
        inv0 = pow(k0, P - 2, P)
        for d2 in range(-256, 257):                                       # d2 k2 + d1 k1 + d0 k0 = 0 mod P only at d = 0 for byte-sized d:
            for d1 in range(-256, 257):                                   # two different (word, word, prefix) triples never collide by structure
                d0 = (-(d2 * k2 + d1 * k1)) % P * inv0 % P
                d0 = d0 - P if d0 > P // 2 else d0
                assert abs(d0) > 256 or (d2, d1, d0) == (0, 0, 0)
    assert all(k < P for k in (I.MATCH_M1, I.MATCH_M2)) and (P - 1) * max(max(ks) for ks in I.MATCH_WORD_MIX) < (1 << 62)   # products fit int64


def test_an_order_that_is_neither_a_byte_order_nor_a_word_order_is_refused():
    x = random_windows(1, 64)
    for n in (0, 65, 901):
        with pytest.raises(ValueError, match="match orders"):
            match_features(x, (8, n))
    assert I.MATCH_WORD_ORDERS == (900, 902) and I.parse_orders("8,900,902", "--match-feat") == (8, 900, 902)


def test_the_word_orders_trace_with_fullgraph_and_match_eager():
    x = random_windows(2, 128, 4)
    want = match_features(x, (8, MATCH_WORD_PREFIX, MATCH_WORD_PAIR))
    got = torch.compile(match_features, backend="eager", fullgraph=True)(x, (8, MATCH_WORD_PREFIX, MATCH_WORD_PAIR))
    assert torch.equal(want[0], got[0]) and torch.equal(want[1], got[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_the_word_orders_give_the_same_answer_on_cuda():
    x = random_windows(3, 256, 6)
    want = match_features(x, (8, MATCH_WORD_PREFIX, MATCH_WORD_PAIR))
    got = match_features(x.cuda(), (8, MATCH_WORD_PREFIX, MATCH_WORD_PAIR))
    assert torch.equal(want[0], got[0].cpu()) and torch.equal(want[1], got[1].cpu())
