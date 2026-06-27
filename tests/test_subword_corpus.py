"""Tests for SubwordCorpus, train_subword_tokenizer, and build_subword_corpus."""
from __future__ import annotations

import torch

from local_ai_training.data import build_subword_corpus, train_subword_tokenizer
from local_ai_training.tokenizer import BpeTokenizer

# ---------------------------------------------------------------------------
# Shared fixture text — large enough to have distinct train / val splits
# ---------------------------------------------------------------------------
TEXT = (
    "the quick brown fox jumps over the lazy dog\n" * 500
    + "pack my box with five dozen liquor jugs\n" * 300
    + "how vexingly quick daft zebras jump\n" * 200
)


# ---------------------------------------------------------------------------
# 1. train_subword_tokenizer returns a BpeTokenizer
# ---------------------------------------------------------------------------
def test_train_subword_tokenizer_returns_bpe():
    tok = train_subword_tokenizer(TEXT, vocab_size=300)
    assert isinstance(tok, BpeTokenizer)
    assert tok.vocab_size <= 300


# ---------------------------------------------------------------------------
# 2. train_subword_tokenizer never sees the validation tail
# ---------------------------------------------------------------------------
def test_train_uses_train_split_only():
    # Craft a text where only the LAST 10% contains a unique marker.
    # The tokenizer should NOT have merged the marker into a single token,
    # confirming it never trained on the validation portion.
    validation_fraction = 0.1
    marker_char = "\x01"  # latin-1 safe, won't appear naturally in TEXT
    train_part = "a b c d e f g h i j\n" * 2000
    val_part = marker_char * (len(train_part) // 9)  # ~10% of total
    combined = train_part + val_part

    tok = train_subword_tokenizer(
        combined, vocab_size=300, validation_fraction=validation_fraction
    )
    # The marker byte is in the base vocab (id = ord(marker_char) = 1),
    # so encoding works — but no merge that STARTS from that byte should exist.
    marker_id = ord(marker_char)
    for pair in tok._merges:
        assert pair[0] != marker_id, (
            "tokenizer trained on a pair starting with the validation-only marker byte"
        )


# ---------------------------------------------------------------------------
# 3. build_subword_corpus splits correctly
# ---------------------------------------------------------------------------
def test_corpus_split_sizes():
    tok = train_subword_tokenizer(TEXT, vocab_size=300)
    corpus = build_subword_corpus(TEXT, tok)

    val_len = int(len(TEXT) * 0.1)
    assert corpus.train_text == TEXT[:-val_len]
    assert corpus.validation_text == TEXT[-val_len:]


# ---------------------------------------------------------------------------
# 4. train_ids and validation_ids are long tensors, shapes consistent with text
# ---------------------------------------------------------------------------
def test_corpus_tensor_properties():
    tok = train_subword_tokenizer(TEXT, vocab_size=300)
    corpus = build_subword_corpus(TEXT, tok)

    assert corpus.train_ids.dtype == torch.long
    assert corpus.validation_ids.dtype == torch.long
    assert corpus.train_ids.ndim == 1
    assert corpus.validation_ids.ndim == 1
    # Subword tokens compress text: fewer ids than chars
    assert corpus.train_ids.numel() <= len(corpus.train_text)
    assert corpus.validation_ids.numel() <= len(corpus.validation_text)
    # But must have at least one token
    assert corpus.train_ids.numel() > 0
    assert corpus.validation_ids.numel() > 0


# ---------------------------------------------------------------------------
# 5. decode round-trip via corpus.decode
# ---------------------------------------------------------------------------
def test_corpus_split_and_decode_roundtrip():
    tok = train_subword_tokenizer(TEXT, vocab_size=400)
    corpus = build_subword_corpus(TEXT, tok)

    # vocab_size may be less than 400 if corpus exhausts merges
    assert corpus.vocab_size <= 400

    # decode(train_ids) must reproduce train_text exactly
    assert corpus.decode(corpus.train_ids) == corpus.train_text
    assert corpus.decode(corpus.validation_ids) == corpus.validation_text


# ---------------------------------------------------------------------------
# 6. SubwordCorpus.vocab_size matches tokenizer.vocab_size
# ---------------------------------------------------------------------------
def test_vocab_size_field():
    tok = train_subword_tokenizer(TEXT, vocab_size=300)
    corpus = build_subword_corpus(TEXT, tok)
    assert corpus.vocab_size == tok.vocab_size


# ---------------------------------------------------------------------------
# 7. train_chars parameter limits how much text is used for tokenizer training
# ---------------------------------------------------------------------------
def test_train_chars_limits_training_data():
    # With train_chars=100 only the first 100 chars of the train split are used.
    # The tokenizer should still be valid and encode/decode correctly.
    tok = train_subword_tokenizer(TEXT, vocab_size=300, train_chars=100)
    assert isinstance(tok, BpeTokenizer)
    # Should still round-trip a short sample
    sample = "the quick"
    assert tok.decode(tok.encode(sample)) == sample


# ---------------------------------------------------------------------------
# 8. custom validation_fraction is respected
# ---------------------------------------------------------------------------
def test_custom_validation_fraction():
    tok = train_subword_tokenizer(TEXT, vocab_size=300, validation_fraction=0.2)
    corpus = build_subword_corpus(TEXT, tok, validation_fraction=0.2)

    val_len = int(len(TEXT) * 0.2)
    assert corpus.validation_text == TEXT[-val_len:]
    assert corpus.train_text == TEXT[:-val_len:]


# ---------------------------------------------------------------------------
# 9. SubwordCorpus is frozen (immutable)
# ---------------------------------------------------------------------------
def test_corpus_frozen():
    import dataclasses

    tok = train_subword_tokenizer(TEXT, vocab_size=300)
    corpus = build_subword_corpus(TEXT, tok)
    assert dataclasses.is_dataclass(corpus)
    try:
        corpus.vocab_size = 999  # type: ignore[misc]
        raise AssertionError("should have raised FrozenInstanceError")
    except dataclasses.FrozenInstanceError:
        pass


# ---------------------------------------------------------------------------
# 10. CharCorpus unchanged (regression guard)
# ---------------------------------------------------------------------------
def test_char_corpus_unchanged():
    from local_ai_training.data import build_char_corpus

    text = "abcdefghij" * 100
    cc = build_char_corpus(text)
    assert cc.train_text == text[: -int(len(text) * 0.1)]
    assert cc.decode(cc.train_ids) == cc.train_text
