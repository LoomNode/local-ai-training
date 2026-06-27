import hashlib
import zipfile
from pathlib import Path

import pytest
import torch

from local_ai_training import data
from local_ai_training.data import batch_from_starts, build_char_corpus, make_batch_schedule


def _fake_text8_zip(zip_path: Path, content: bytes) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("text8", content)


def test_download_text8_verifies_checksum_and_extracts(tmp_path: Path, monkeypatch) -> None:
    content = b"abc abc abc"
    source = tmp_path / "source.zip"
    _fake_text8_zip(source, content)
    monkeypatch.setattr(data, "TEXT8_ZIP_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(data, "TEXT8_EXPECTED_CHARS", len(content))
    monkeypatch.setattr(
        data, "_download_file", lambda url, dest: dest.write_bytes(source.read_bytes())
    )

    text_path = data.download_text8(tmp_path / "cache")

    assert text_path.read_bytes() == content


def test_download_text8_rejects_a_tampered_archive(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.zip"
    _fake_text8_zip(source, b"not the real corpus")
    # Leave the pinned real checksum in place; the fake must be rejected.
    monkeypatch.setattr(
        data, "_download_file", lambda url, dest: dest.write_bytes(source.read_bytes())
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        data.download_text8(tmp_path / "cache")


def _fake_enwik8_zip(zip_path: Path, content: bytes) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("enwik8", content)


def test_download_enwik8_verifies_checksum_and_extracts(tmp_path: Path, monkeypatch) -> None:
    content = b"<page>Abc 123.</page> " * 4
    source = tmp_path / "source.zip"
    _fake_enwik8_zip(source, content)
    monkeypatch.setattr(data, "ENWIK8_ZIP_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(data, "ENWIK8_EXPECTED_CHARS", len(content))
    monkeypatch.setattr(
        data, "_download_file", lambda url, dest: dest.write_bytes(source.read_bytes())
    )

    text_path = data.download_enwik8(tmp_path / "cache")

    assert text_path.read_bytes() == content


def test_download_enwik8_rejects_a_tampered_archive(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.zip"
    _fake_enwik8_zip(source, b"not the real corpus")
    monkeypatch.setattr(
        data, "_download_file", lambda url, dest: dest.write_bytes(source.read_bytes())
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        data.download_enwik8(tmp_path / "cache")


def test_build_char_corpus_uses_deterministic_tail_validation_split() -> None:
    corpus = build_char_corpus("abcde" * 20, validation_fraction=0.1)

    assert corpus.train_text == "abcde" * 18
    assert corpus.validation_text == "abcde" * 2
    assert corpus.vocabulary == tuple("abcde")
    assert corpus.decode(corpus.train_ids[:5]) == "abcde"


def test_batch_schedule_is_seeded_and_stays_inside_next_token_range() -> None:
    first = make_batch_schedule(
        data_length=100, steps=4, batch_size=3, block_size=8, seed=17
    )
    second = make_batch_schedule(
        data_length=100, steps=4, batch_size=3, block_size=8, seed=17
    )

    assert torch.equal(first, second)
    assert first.shape == (4, 3)
    assert first.min().item() >= 0
    assert first.max().item() <= 100 - 8 - 1


def test_batch_from_starts_returns_shifted_character_targets() -> None:
    data = torch.arange(20)

    inputs, targets = batch_from_starts(data, torch.tensor([0, 5]), block_size=4)

    assert inputs.tolist() == [[0, 1, 2, 3], [5, 6, 7, 8]]
    assert targets.tolist() == [[1, 2, 3, 4], [6, 7, 8, 9]]

