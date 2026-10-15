"""The per-byte inputs of the streaming reader (notes/speed-round/checks/stream_inputs.py) against the build's own functions
(use_features.text_features: context_hashes, compute_keys, counters, the two-way lookup) on a text made of the awkward cases: file
headers at depth 0 and not, strings with escaped quotes and backslash runs, comments ("/*/", "*/*", "//" in strings), unbalanced
files, negative and deep nesting, long identifiers, long lines, tabs, CR LF, high bytes, a header marker mid-line; and MatchState
against flybrain.model.inputs.match_features."""
import os
import sys

import numpy as np
import pytest

CHECKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notes", "speed-round", "checks")
sys.path.insert(0, CHECKS)

from stream_inputs import ALL18, MatchState, TextState  # noqa: E402
from use_features import text_features  # noqa: E402
from flybrain.corpus_memory import SLOT, context_hashes, slot_and_tag  # noqa: E402
from key_tables import KEY_IDS, compute_keys  # noqa: E402


class FakeTables:
    """Small two-way tables (2^10 slots) holding every third context of `text`, so lookups hit, miss and meet their buddy."""

    def __init__(self, text: bytes, log2: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dir, self.log2, self.meta = "fake", log2, {"log2_slots": log2}
        data = np.concatenate([np.frombuffer(text, np.uint8), np.zeros(1, np.uint8)])
        hashes = context_hashes(data, 0, len(data), (4, 8, 12, 16, 32, 64))
        hashes.update(compute_keys(data))
        self.t = {}
        for n, h in hashes.items():
            tab = np.zeros(1 << log2, dtype=SLOT)
            slot, tag = slot_and_tag(np.asarray(h[::3]), log2)
            which = np.where(np.arange(len(slot)) % 2 == 0, slot & ~1, slot | 1)
            tab["tag"][which] = tag; tab["byte"][which] = rng.integers(0, 256, len(slot)); tab["conf"][which] = rng.integers(1, 256, len(slot))
            self.t[n] = tab


def awkward_text() -> bytes:
    long_id = b"a_very_long_identifier_name_that_runs_past_thirty_two_bytes_" * 2
    long_line = b"x = [" + b", ".join(b"%d" % i for i in range(120)) + b"]\n"
    f1 = (b"\n// ==== src/one.c ====\n"
          b"#include <stdio.h>\n"
          b"/* a block comment with \"quotes\" and { braces ( */\n"
          b"int f(int a) {\n\tif (a) { return \"str\\\"ing\\\\\"; } // trailing { comment\n"
          b"    char c = '\\''; char d = '}'; /*/ odd */ x = 1; /* a *//* b */\n"
          b"\t\tunterminated \"string { ( runs to the line end\n"
          b"  " + long_id + b" = 3;\r\n" + long_line +
          b"  // ==== not a header: mid-line\n"
          b"// === almost a header\n"
          b"{ { {\n")                                                       # ends at depth +3 with an open paren inside the string gone
    f2 = (b"\n// ==== src/two.py ====\n"
          b"def g(x):\n    return x  # ) ) ) }\n" + b"}" * 5 + b"\n" + b"(" * 140 + b"\n" + b")" * 140 + b"\n"
          b"s = '''a\nb'''\n" + bytes(range(128, 256)) + b"\n/* open block at file end\n")
    f3 = b"\n// ==== src/three.c ====\n still in the block */ int y = 0;\n\"\\\\\\\"\" z;\n\n\n   \t  mixed indent\n"
    body = f1 + f2 + f3
    return body * 2 + b"\n// ==== tail.c ====\n" + b"q(" * 50 + b"\nend"


@pytest.fixture(scope="module")
def world():
    text = awkward_text()
    tables = FakeTables(text)
    ref = text_features(tables, np.frombuffer(text, np.uint8), ALL18)
    return text, tables, ref


def rows_equal(st, ref, N):
    return bool((st.pred[:N + 1] == ref[0][:N + 1]).all() and (st.conf[:N + 1] == ref[1][:N + 1]).all())


def test_cold_replay_matches_text_features(world):
    text, tables, ref = world
    st = TextState(tables, ALL18)
    fixes = [f for f in map(st.push, text) if f is not None]
    assert st.n == len(text)
    assert rows_equal(st, ref, len(text))
    assert len(fixes) >= 2                                   # headers met at depth != 0 corrected their eight rows
    for f in fixes:
        assert text[f:f + 8] == b"// ==== " and (f == 0 or text[f - 1] == 10)


def test_warm_start_then_stream(world):
    text, tables, ref = world
    N = len(text)
    for a in (1, 40, 200, N // 3, N // 2, N - 300, N - 1):
        st = TextState(tables, ALL18)
        st.extend(text[:a])
        assert st.n == a and rows_equal(st, ref, a)
        for b in text[a:]:
            st.push(b)
        assert rows_equal(st, ref, N), a
    st = TextState(tables, ALL18)                             # two appends, then bytes
    st.extend(text[:500]); st.extend(text[500:1200])
    for b in text[1200:]:
        st.push(b)
    assert rows_equal(st, ref, N)


def test_warm_start_replays_only_a_tail(world):
    text, tables, _ = world
    st = TextState(tables, ALL18)
    st.extend(text)
    assert 0 < st.base < len(text) and len(text) - st.base < 2000


def test_match_state_matches_match_features(world):
    import torch
    from flybrain.model.inputs import match_features
    text = world[0]
    orders = (8, 16, 32)
    for pos0, T in ((0, 700), (321, 1500), (len(text) - 100, 100)):
        hi = pos0 + T
        ref_p, ref_c = match_features(torch.from_numpy(np.frombuffer(text[pos0:hi], np.uint8).astype(np.int64))[None], orders)
        ms = MatchState(orders, bytearray(text))
        P, C = ms.rows(pos0, hi)
        assert np.array_equal(P, ref_p[0].numpy()) and np.array_equal(C, ref_c[0].numpy()), pos0
        ms2 = MatchState(orders, bytearray(text)); ms2.start(pos0)        # push by push gives the same
        for t in range(pos0, hi):
            p, c = ms2.push(t)
            assert p == list(P[t - pos0]) and c == list(C[t - pos0])
