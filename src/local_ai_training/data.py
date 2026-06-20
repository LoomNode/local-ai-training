"""Deterministic character corpus and batching utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


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

