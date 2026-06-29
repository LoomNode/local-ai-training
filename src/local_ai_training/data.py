"""Deterministic character corpus and batching utilities."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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


@dataclass(frozen=True)
class TokenShardResult:
    tokens_path: Path
    metadata_path: Path
    token_count: int
    rows_seen: int
    rows_used: int


@dataclass(frozen=True)
class TokenShardCorpus:
    train_ids: Tensor
    validation_ids: Tensor
    tokenizer: BpeTokenizer
    vocab_size: int

    def decode(self, token_ids: Tensor) -> str:
        return self.tokenizer.decode(token_ids.flatten().tolist())


def _split_text(text: str, validation_fraction: float) -> tuple[str, str]:
    """Return (train_text, validation_text) using the canonical final-N% split."""
    validation_length = int(len(text) * validation_fraction)
    return text[:-validation_length], text[-validation_length:]


def byte_text(text: str) -> str:
    """Map Unicode text to the latin-1 byte-string format used by the byte BPE."""
    return text.encode("utf-8").decode("latin-1")


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


def build_token_shard(
    rows: Iterable[dict[str, Any]],
    *,
    tokenizer: BpeTokenizer,
    output_dir: str | Path,
    target_tokens: int,
    dataset_name: str,
    subset: str,
    revision: str,
    text_field: str = "text",
) -> TokenShardResult:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if tokenizer.vocab_size > 65_536:
        raise ValueError("tokenizer vocab_size must fit in uint16")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokens_path = output / "tokens.uint16"
    metadata_path = output / "metadata.json"
    tokenizer_json = tokenizer.to_json()
    tokenizer_hash = hashlib.sha256(tokenizer_json.encode("utf-8")).hexdigest()

    token_count = 0
    rows_seen = 0
    rows_used = 0
    with tokens_path.open("wb") as handle:
        for row in rows:
            rows_seen += 1
            text = row.get(text_field)
            if not isinstance(text, str) or not text:
                continue
            ids = tokenizer.encode(byte_text(text))
            if not ids:
                continue
            tensor = torch.tensor(ids, dtype=torch.uint16)
            handle.write(tensor.numpy().tobytes())
            token_count += len(ids)
            rows_used += 1
            if token_count >= target_tokens:
                break
    if token_count < 2:
        raise ValueError("token shard is too small")
    metadata = {
        "dataset": dataset_name,
        "subset": subset,
        "revision": revision,
        "text_field": text_field,
        "target_tokens": target_tokens,
        "actual_tokens": token_count,
        "rows_seen": rows_seen,
        "rows_used": rows_used,
        "token_dtype": "uint16",
        "tokens_file": tokens_path.name,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_json": tokenizer_json,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return TokenShardResult(
        tokens_path=tokens_path,
        metadata_path=metadata_path,
        token_count=token_count,
        rows_seen=rows_seen,
        rows_used=rows_used,
    )


def load_token_shard(
    metadata_path: str | Path, *, validation_fraction: float = 0.1
) -> TokenShardCorpus:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    metadata_file = Path(metadata_path)
    metadata = json.loads(metadata_file.read_text())
    if metadata.get("token_dtype") != "uint16":
        raise ValueError("unsupported token shard dtype")
    tokenizer_json = metadata["tokenizer_json"]
    expected_hash = metadata["tokenizer_sha256"]
    actual_hash = hashlib.sha256(tokenizer_json.encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("tokenizer hash mismatch")
    tokenizer = BpeTokenizer.from_json(tokenizer_json)
    tokens_file = metadata_file.with_name(metadata["tokens_file"])
    raw = torch.from_file(str(tokens_file), dtype=torch.uint16, size=int(metadata["actual_tokens"]))
    ids = raw.to(dtype=torch.long)
    validation_length = int(ids.numel() * validation_fraction)
    if validation_length < 2 or ids.numel() - validation_length < 2:
        raise ValueError("token shard is too short for train and validation splits")
    return TokenShardCorpus(
        train_ids=ids[:-validation_length].clone(),
        validation_ids=ids[-validation_length:].clone(),
        tokenizer=tokenizer,
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
