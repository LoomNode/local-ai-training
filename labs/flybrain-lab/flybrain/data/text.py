"""Byte-level corpus windows. Vocab is UTF-8 bytes, not BPE."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor


def bytes_tensor(text: str) -> Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def load_bytes(path: str | Path, max_bytes: int | None = None, offset: int = 0) -> Tensor:
    raw = Path(path).read_bytes()
    end = offset + max_bytes if max_bytes is not None else None
    raw = raw[offset:end]
    return torch.tensor(list(raw), dtype=torch.long)


def split_bytes(data: Tensor, val_frac: float = 0.1) -> tuple[Tensor, Tensor]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1)")
    cut = int(data.numel() * (1.0 - val_frac))
    if cut < 1 or cut >= int(data.numel()):
        raise ValueError("split produced an empty train or val split")
    return data[:cut].contiguous(), data[cut:].contiguous()


def ngram_bpb(train: Tensor, val: Tensor, order: int = 1) -> float:
    """Add-one MLE bits/byte. order=1 unigram, order=2 bigram."""
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    train_list = train.tolist()
    val_list = val.tolist()
    if order == 1:
        counts = [1] * 256
        for b in train_list:
            counts[int(b)] += 1
        total = float(sum(counts))
        nll = 0.0
        for b in val_list:
            nll += -math.log2(counts[int(b)] / total)
        return nll / max(len(val_list), 1)
    counts: dict[int, list[int]] = {}
    for i in range(len(train_list) - 1):
        prev = int(train_list[i])
        nxt = int(train_list[i + 1])
        if prev not in counts:
            counts[prev] = [1] * 256
        counts[prev][nxt] += 1
    nll = 0.0
    n = 0
    uni = [1] * 256
    for b in train_list:
        uni[int(b)] += 1
    uni_tot = float(sum(uni))
    for i in range(len(val_list) - 1):
        prev = int(val_list[i])
        nxt = int(val_list[i + 1])
        row = counts.get(prev)
        if row is None:
            nll += -math.log2(uni[nxt] / uni_tot)
        else:
            nll += -math.log2(row[nxt] / float(sum(row)))
        n += 1
    return nll / max(n, 1)


def batch_windows(data: Tensor, seq_len: int, batch: int, seed: int) -> tuple[Tensor, Tensor]:
    n = int(data.numel())
    if n < seq_len + 1:
        raise ValueError(f"corpus length {n} shorter than seq_len+1={seq_len + 1}")
    max_start = n - seq_len - 1
    rng = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, max_start + 1, (batch,), generator=rng)
    x = torch.stack([data[int(i) : int(i) + seq_len] for i in starts])
    y = torch.stack([data[int(i) + 1 : int(i) + 1 + seq_len] for i in starts])
    return x, y
