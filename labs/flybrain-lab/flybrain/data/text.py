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


def batch_starts(n: int, seq_len: int, batch: int, seed: int) -> Tensor:
    """The window starts batch_windows draws from n bytes for this seed: the same generator, the same call. x[b] = data[s : s + seq_len]
    for s = starts[b]; the byte data[s + t] is the input at window index t, so a per-position feature of the target byte of index t
    lives at absolute position s + t + 1 (flybrain.data.memfeat.FeatureStore.windows)."""
    if n < seq_len + 1:
        raise ValueError(f"corpus length {n} shorter than seq_len+1={seq_len + 1}")
    max_start = n - seq_len - 1
    rng = torch.Generator().manual_seed(seed)
    return torch.randint(0, max_start + 1, (batch,), generator=rng)


def batch_starts_ranges(ranges, seq_len: int, batch: int, seed: int) -> Tensor:
    """Window starts drawn from several ranges of the corpus (the distillation plan's mixed arm, 2026-10-20): ranges = [(lo, hi, weight),
    ...]; each window picks a range with probability proportional to its weight, then a uniform start in [lo, hi - seq_len - 1], so a
    window (and the target byte after it) never crosses a range end. The generator and seed convention of batch_starts."""
    rng = torch.Generator().manual_seed(seed)
    w = torch.tensor([float(r[2]) for r in ranges]); w = w / w.sum()
    which = torch.multinomial(w, batch, replacement=True, generator=rng)
    starts = torch.empty(batch, dtype=torch.long)
    for k, (lo, hi, _) in enumerate(ranges):
        m = which == k
        if m.any():
            max_start = int(hi) - seq_len - 1
            if max_start < int(lo):
                raise ValueError(f"range [{lo}, {hi}) is shorter than seq_len+1={seq_len + 1}")
            starts[m] = torch.randint(int(lo), max_start + 1, (int(m.sum()),), generator=rng)
    return starts


def windows_at(data: Tensor, starts, seq_len: int) -> tuple[Tensor, Tensor]:
    """x, y for given window starts (batch_windows with the starts chosen by the caller)."""
    x = torch.stack([data[int(i) : int(i) + seq_len] for i in starts])
    y = torch.stack([data[int(i) + 1 : int(i) + 1 + seq_len] for i in starts])
    return x, y


def batch_windows(data: Tensor, seq_len: int, batch: int, seed: int) -> tuple[Tensor, Tensor]:
    starts = batch_starts(int(data.numel()), seq_len, batch, seed)
    x = torch.stack([data[int(i) : int(i) + seq_len] for i in starts])
    y = torch.stack([data[int(i) + 1 : int(i) + 1 + seq_len] for i in starts])
    return x, y


def fixed_windows(data: Tensor, seq_len: int, n: int) -> tuple[Tensor, Tensor]:
    """n non-overlapping windows from the start of data: x[i] = data[i*seq_len:(i+1)*seq_len], y[i] one byte ahead. No RNG."""
    if n < 1 or seq_len < 1:
        raise ValueError(f"need n >= 1 and seq_len >= 1, got n={n}, seq_len={seq_len}")
    need, have = n * seq_len + 1, int(data.numel())
    if have < need:
        raise ValueError(f"need {need} bytes for {n} windows of {seq_len} (n*seq_len+1), data has {have}")
    x = data[: n * seq_len].reshape(n, seq_len).clone()
    y = data[1:need].reshape(n, seq_len).clone()
    return x, y
