"""Corpus builders."""

from __future__ import annotations

from scrt.data import build_corpus, build_structured_language, build_synonym_language


def test_synonym_and_structured_shapes():
    s = build_synonym_language(vocab_size=48, length=1000, seed=0)
    assert s.data.numel() == 1000
    assert s.vocab_size == 48
    st = build_structured_language(vocab_size=96, length=2000, seed=0)
    assert st.data.numel() == 2000
    assert st.data.max() < 96


def test_build_corpus_factory():
    c = build_corpus("structured", vocab_size=64, length=500, seed=1)
    assert c.name == "structured"
    assert c.data.numel() == 500
