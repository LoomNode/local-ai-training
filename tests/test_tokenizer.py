"""Tests for byte-level BPE tokenizer (stdlib only, no third-party deps)."""
from __future__ import annotations

from local_ai_training.tokenizer import BpeTokenizer

CORPUS = (
    "low lower newest widest\n" * 20
    + "the cat sat on the mat\n" * 10
)


def _trained() -> BpeTokenizer:
    return BpeTokenizer.train(CORPUS, vocab_size=300)


# ---------------------------------------------------------------------------
# 1. smoke – train without error, vocab_size property matches request
# ---------------------------------------------------------------------------
def test_train_smoke():
    tok = _trained()
    assert tok.vocab_size == 300


# ---------------------------------------------------------------------------
# 2. round-trip: decode(encode(text)) == text  (critical correctness gate)
# ---------------------------------------------------------------------------
def test_roundtrip():
    tok = _trained()
    for text in [
        "low",
        "lower",
        "newest",
        "hello world",
        "the cat sat on the mat",
        "xyz 123 !@#",
    ]:
        assert tok.decode(tok.encode(text)) == text, f"round-trip failed for {text!r}"


# ---------------------------------------------------------------------------
# 3. encode produces ints in [0, vocab_size)
# ---------------------------------------------------------------------------
def test_encode_range():
    tok = _trained()
    ids = tok.encode(CORPUS[:200])
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < tok.vocab_size for i in ids)


# ---------------------------------------------------------------------------
# 4. serialisation round-trip
# ---------------------------------------------------------------------------
def test_serialise_roundtrip():
    tok = _trained()
    tok2 = BpeTokenizer.from_json(tok.to_json())
    assert tok2.vocab_size == tok.vocab_size
    sample = "low lower newest"
    assert tok2.encode(sample) == tok.encode(sample)


# ---------------------------------------------------------------------------
# 5. determinism – two trains on identical text yield identical merges
# ---------------------------------------------------------------------------
def test_determinism():
    tok_a = BpeTokenizer.train(CORPUS, vocab_size=300)
    tok_b = BpeTokenizer.train(CORPUS, vocab_size=300)
    assert tok_a.encode("low lower newest") == tok_b.encode("low lower newest")


# ---------------------------------------------------------------------------
# 6. merges never cross pretoken boundaries
# ---------------------------------------------------------------------------
def test_no_cross_boundary_merge():
    # "ab" appears within tokens but "b " (token-boundary) must never merge.
    # We train on isolated words; the pair ('b', ' ') should never become
    # a single token if word-boundary splitting is respected.
    tok = BpeTokenizer.train("ab " * 200, vocab_size=260)
    ids = tok.encode("ab ")
    decoded = tok.decode(ids)
    assert decoded == "ab "
    # The space must be its own token (or part of a word-start token),
    # not merged across the logical word boundary that precedes it.
    # Concretely: encoding "ab" and "ab " must share a common prefix.
    ids_ab = tok.encode("ab")
    ids_ab_space = tok.encode("ab ")
    assert ids_ab_space[: len(ids_ab)] == ids_ab
