"""Deterministic character corpus and batching utilities."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import torch
from huggingface_hub import hf_hub_download
from torch import Tensor

from local_ai_training.tokenizer import BpeTokenizer

TINY_SHAKESPEARE_REPO = "SamPIngram/tinyshakespeare"
TINY_SHAKESPEARE_REVISION = "6d8bc3fdfca13bf8a128bb0e0914cead1e2d208c"

# Canonical 100MB char-level benchmark (cleaned Wikipedia, 27-char vocab). The zip is pinned
# by SHA-256 rather than a hosting revision, so the corpus is reproducible and verified
# without executing any remote code.
TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
TEXT8_ZIP_SHA256 = "a6640522afe85d1963ad56c05b0ede0a0c000dddc9671758a6cc09b7a38e5232"
TEXT8_EXPECTED_CHARS = 100_000_000

# enwik8 is the same 100MB source as text8 but un-stripped: text8 == enwik8 lowercased and
# reduced to a-z + space, whereas enwik8 keeps capitals, digits, punctuation, and markup. Read
# byte-level (latin-1) it has a ~205-value vocab — rich characters without the embedding bloat a
# subword vocab would bring. Same pinning discipline as text8.
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
ENWIK8_ZIP_SHA256 = "547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc"
ENWIK8_EXPECTED_CHARS = 100_000_000


@dataclass(frozen=True)
class CharCorpus:
    train_text: str
    validation_text: str
    vocabulary: tuple[str, ...]
    train_ids: Tensor
    validation_ids: Tensor

    def decode(self, token_ids: Tensor) -> str:
        return "".join(self.vocabulary[int(index)] for index in token_ids.flatten())


def build_char_corpus(text: str, *, validation_fraction: float = 0.1) -> CharCorpus:
    if not text:
        raise ValueError("corpus text must not be empty")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    validation_length = int(len(text) * validation_fraction)
    if validation_length < 2 or len(text) - validation_length < 2:
        raise ValueError("corpus is too short for train and validation next-token splits")

    vocabulary = tuple(sorted(set(text)))
    char_to_id = {character: index for index, character in enumerate(vocabulary)}
    train_text = text[:-validation_length]
    validation_text = text[-validation_length:]

    def encode(partition: str) -> Tensor:
        return torch.tensor([char_to_id[character] for character in partition], dtype=torch.long)

    return CharCorpus(
        train_text=train_text,
        validation_text=validation_text,
        vocabulary=vocabulary,
        train_ids=encode(train_text),
        validation_ids=encode(validation_text),
    )


@dataclass(frozen=True)
class SubwordCorpus:
    train_text: str
    validation_text: str
    tokenizer: BpeTokenizer
    train_ids: Tensor
    validation_ids: Tensor
    vocab_size: int

    def decode(self, token_ids: Tensor) -> str:
        return self.tokenizer.decode(token_ids.flatten().tolist())


def _split_text(text: str, validation_fraction: float) -> tuple[str, str]:
    """Return (train_text, validation_text) using the canonical final-N% split."""
    validation_length = int(len(text) * validation_fraction)
    return text[:-validation_length], text[-validation_length:]


def train_subword_tokenizer(
    text: str,
    *,
    vocab_size: int,
    train_chars: int = 2_000_000,
    validation_fraction: float = 0.1,
) -> BpeTokenizer:
    """Train a BPE tokenizer on the train split of *text* only (never the validation tail).

    Only the first *train_chars* characters of the train split are used for training,
    keeping the operation tractable on large corpora. BPE training uses incremental
    pair-count updates, but it is still pure Python; a ~2MB slice builds an 8K vocab
    quickly and — because merges are frequency-driven and common subwords dominate —
    yields a vocabulary nearly identical to one trained on far more text.
    """
    if not text:
        raise ValueError("corpus text must not be empty")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    train_text, _ = _split_text(text, validation_fraction)
    training_sample = train_text[:train_chars]
    return BpeTokenizer.train(training_sample, vocab_size)


def build_subword_corpus(
    text: str,
    tokenizer: BpeTokenizer,
    *,
    validation_fraction: float = 0.1,
) -> SubwordCorpus:
    """Encode *text* with *tokenizer* and return a :class:`SubwordCorpus`."""
    if not text:
        raise ValueError("corpus text must not be empty")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    train_text, validation_text = _split_text(text, validation_fraction)

    def encode(partition: str) -> Tensor:
        return torch.tensor(tokenizer.encode(partition), dtype=torch.long)

    return SubwordCorpus(
        train_text=train_text,
        validation_text=validation_text,
        tokenizer=tokenizer,
        train_ids=encode(train_text),
        validation_ids=encode(validation_text),
        vocab_size=tokenizer.vocab_size,
    )


def make_batch_schedule(
    *, data_length: int, steps: int, batch_size: int, block_size: int, seed: int
) -> Tensor:
    if min(data_length, steps, batch_size, block_size) <= 0:
        raise ValueError("schedule dimensions must be positive")
    high = data_length - block_size
    if high <= 0:
        raise ValueError("data_length must exceed block_size")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, high, (steps, batch_size), generator=generator)


def batch_from_starts(data: Tensor, starts: Tensor, *, block_size: int) -> tuple[Tensor, Tensor]:
    if data.ndim != 1 or starts.ndim != 1:
        raise ValueError("data and starts must be one-dimensional")
    offsets = torch.arange(block_size, device=starts.device)
    indices = starts[:, None] + offsets[None, :]
    if indices.numel() and indices.max().item() + 1 >= data.numel():
        raise ValueError("batch start exceeds next-token data range")
    source = data.to(device=starts.device)
    return source[indices], source[indices + 1]


def _download_file(url: str, dest: Path) -> None:
    urlretrieve(url, dest)  # noqa: S310 - pinned http(s) URL, verified by checksum below


def download_text8(cache_dir: str | Path) -> Path:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "text8.zip"
    text_path = cache / "text8"

    if not text_path.is_file():
        if not zip_path.is_file():
            _download_file(TEXT8_URL, zip_path)
        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        if digest != TEXT8_ZIP_SHA256:
            raise ValueError(
                f"text8.zip checksum mismatch: expected {TEXT8_ZIP_SHA256}, got {digest}"
            )
        with zipfile.ZipFile(zip_path) as archive:
            if archive.namelist() != ["text8"]:
                raise ValueError(f"unexpected text8 archive contents: {archive.namelist()}")
            archive.extract("text8", cache)

    if text_path.stat().st_size != TEXT8_EXPECTED_CHARS:
        raise ValueError("extracted text8 is missing or has an unexpected size")
    return text_path


def download_enwik8(cache_dir: str | Path) -> Path:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "enwik8.zip"
    text_path = cache / "enwik8"

    if not text_path.is_file():
        if not zip_path.is_file():
            _download_file(ENWIK8_URL, zip_path)
        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        if digest != ENWIK8_ZIP_SHA256:
            raise ValueError(
                f"enwik8.zip checksum mismatch: expected {ENWIK8_ZIP_SHA256}, got {digest}"
            )
        with zipfile.ZipFile(zip_path) as archive:
            if archive.namelist() != ["enwik8"]:
                raise ValueError(f"unexpected enwik8 archive contents: {archive.namelist()}")
            archive.extract("enwik8", cache)

    if text_path.stat().st_size != ENWIK8_EXPECTED_CHARS:
        raise ValueError("extracted enwik8 is missing or has an unexpected size")
    return text_path


def download_tiny_shakespeare(cache_dir: str | Path) -> Path:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=TINY_SHAKESPEARE_REPO,
        repo_type="dataset",
        filename="input.txt",
        revision=TINY_SHAKESPEARE_REVISION,
        cache_dir=str(cache),
    )
    path = Path(downloaded)
    if not path.is_file() or path.stat().st_size < 1_000:
        raise ValueError("downloaded Tiny Shakespeare file is missing or unexpectedly small")
    return path
