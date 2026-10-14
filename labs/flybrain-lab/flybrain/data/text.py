"""Byte-level corpus windows. Vocab is UTF-8 bytes, not BPE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def bytes_tensor(text: str) -> Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def load_bytes(path: str | Path, max_bytes: int | None = None, offset: int = 0) -> Tensor:
    """The file's bytes from `offset` (at most `max_bytes`) as a uint8 tensor, read straight from the file: one byte of memory per byte
    of corpus. Windows cut from it (batch_windows, fixed_windows) come out as int64. The old form, a Python list of ints turned into an
    int64 tensor, cost 16 bytes per byte while loading and 8 after, which a 5 GB corpus turns into 90 GB."""
    count = -1 if max_bytes is None else int(max_bytes)
    return torch.from_numpy(np.fromfile(str(path), dtype=np.uint8, count=count, offset=int(offset)))


def split_bytes(data: Tensor, val_frac: float = 0.1) -> tuple[Tensor, Tensor]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1)")
    cut = int(data.numel() * (1.0 - val_frac))
    if cut < 1 or cut >= int(data.numel()):
        raise ValueError("split produced an empty train or val split")
    return data[:cut].contiguous(), data[cut:].contiguous()


def ngram_bpb(train: Tensor, val: Tensor, order: int = 1, chunk: int = 1 << 26) -> float:
    """Add-one MLE bits/byte. order=1 unigram, order=2 bigram. Counts are gathered over `train` in chunks (any dtype, any size: a 5 GB
    corpus is never materialised as int64), as the plain loops did: a bigram row exists only for a previous byte seen in train, else
    the unigram model answers; add-one on every row and on the unigram counts."""
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    n_train, n_val = int(train.numel()), int(val.numel())
    uni = torch.zeros(256, dtype=torch.int64)
    big = torch.zeros(256 * 256, dtype=torch.int64)
    for i in range(0, n_train, chunk):
        c = train[i : i + chunk + 1].to(torch.long)              # one byte of overlap, for the pair across the chunk boundary
        uni += torch.bincount(c[: min(chunk, n_train - i)], minlength=256)
        if order == 2 and c.numel() >= 2:
            big += torch.bincount(c[:-1] * 256 + c[1:], minlength=256 * 256)
    uni_add = (uni + 1).to(torch.float64)
    uni_tot = float(uni_add.sum())
    v = val.to(torch.long)
    if order == 1:
        nll = float(-torch.log2(uni_add[v] / uni_tot).sum()) if n_val else 0.0
        return nll / max(n_val, 1)
    if n_val < 2:
        return 0.0
    rows = big.reshape(256, 256)
    seen = rows.sum(1) > 0                                        # a row exists for a previous byte that occurred in train (as a pair)
    rows_add = (rows + 1).to(torch.float64)
    row_tot = rows_add.sum(1)
    prev, nxt = v[:-1], v[1:]
    p_row = rows_add[prev, nxt] / row_tot[prev]
    p_uni = uni_add[nxt] / uni_tot
    p = torch.where(seen[prev], p_row, p_uni)
    return float(-torch.log2(p).sum()) / (n_val - 1)


def batch_starts(n: int, seq_len: int, batch: int, seed: int, aligned: bool = False) -> Tensor:
    """The window starts batch_windows draws from n bytes for this seed: the same generator, the same call. x[b] = data[s : s + seq_len]
    for s = starts[b]; the byte data[s + t] is the input at window index t, so a per-position feature of the target byte of index t
    lives at absolute position s + t + 1 (flybrain.data.memfeat.FeatureStore.windows). aligned: starts are multiples of seq_len, so a
    window is one block of a block-built corpus (checks/term_corpus.py: a preamble and content per block)."""
    if n < seq_len + 1:
        raise ValueError(f"corpus length {n} shorter than seq_len+1={seq_len + 1}")
    max_start = n - seq_len - 1
    rng = torch.Generator().manual_seed(seed)
    if aligned:
        return torch.randint(0, max_start // seq_len + 1, (batch,), generator=rng) * seq_len
    return torch.randint(0, max_start + 1, (batch,), generator=rng)


def batch_windows(data: Tensor, seq_len: int, batch: int, seed: int, aligned: bool = False) -> tuple[Tensor, Tensor]:
    starts = batch_starts(int(data.numel()), seq_len, batch, seed, aligned)
    x = torch.stack([data[int(i) : int(i) + seq_len] for i in starts]).to(torch.long)
    y = torch.stack([data[int(i) + 1 : int(i) + 1 + seq_len] for i in starts]).to(torch.long)
    return x, y


def fixed_windows(data: Tensor, seq_len: int, n: int) -> tuple[Tensor, Tensor]:
    """n non-overlapping windows from the start of data: x[i] = data[i*seq_len:(i+1)*seq_len], y[i] one byte ahead. No RNG."""
    if n < 1 or seq_len < 1:
        raise ValueError(f"need n >= 1 and seq_len >= 1, got n={n}, seq_len={seq_len}")
    need, have = n * seq_len + 1, int(data.numel())
    if have < need:
        raise ValueError(f"need {need} bytes for {n} windows of {seq_len} (n*seq_len+1), data has {have}")
    x = data[: n * seq_len].reshape(n, seq_len).clone().to(torch.long)
    y = data[1:need].reshape(n, seq_len).clone().to(torch.long)
    return x, y
