"""Per-position table features of a corpus, as written by scripts/build_corpus_memory.py --features, served to windows.

A directory holds features.json and, for each order n, pred_n<n>.npy and conf_n<n>.npy: uint8 arrays of the corpus's length indexed by
absolute byte position, where feature[p] is the table's prediction for the byte at p from the n bytes before it (conf 0 = a miss).

The alignment, in one place. A window whose input starts at absolute byte s has the byte data[s + t] as the input at index t; that
input is used to predict the byte data[s + t + 1]. The feature the network gets at index t is therefore the one of position
s + t + 1: the table's prediction for the target of that index, made from bytes that are all at or before s + t (the input itself and
earlier ones). `windows` and `fixed` apply the + 1; `at` is the raw read by position, and is what one would get by forgetting it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class FeatureStore:
    """The features of `orders` (the order of the last axis is the order given) from `directory`, memory-mapped read-only."""

    def __init__(self, directory: str | Path, orders):
        self.dir = Path(directory)
        self.info = json.loads((self.dir / "features.json").read_text())
        self.orders = tuple(int(n) for n in orders)
        have = {int(n) for n in self.info["orders"]}
        if not self.orders or len(set(self.orders)) != len(self.orders) or not set(self.orders) <= have:
            raise ValueError(f"orders {list(self.orders)} are not among the orders {sorted(have)} of the features in {self.dir}")
        # model #2 on tokens (notes/model2): a feature directory built over a token stream has n_tokens positions, uint16 predictions and
        # value orders (counters stored as values, not predictions); a byte directory has corpus_bytes positions and uint8 everything
        self.unit = "token" if "n_tokens" in self.info else "byte"
        self.size = int(self.info["n_tokens"]) if self.unit == "token" else int(self.info["corpus_bytes"])
        self.train_end = int(self.info["train_end_tok"]) if self.unit == "token" else int(self.info["train_end"])
        self.value_orders = tuple(int(n) for n in self.info.get("value_orders", ()) if int(n) in self.orders)
        self.pred = [np.load(self.dir / f"pred_n{n}.npy", mmap_mode="r") for n in self.orders]
        self.conf = [np.load(self.dir / f"conf_n{n}.npy", mmap_mode="r") for n in self.orders]
        self.pred_dtype = np.uint16 if any(a.dtype == np.uint16 for a in self.pred) else np.uint8
        for a in self.pred + self.conf:
            if a.dtype not in (np.uint8, np.uint16) or a.shape != (self.size,):
                raise ValueError(f"a feature array in {self.dir} is {a.dtype}{a.shape}, features.json says uint8/uint16 ({self.size},)")

    def at(self, first, seq_len: int) -> tuple[Tensor, Tensor]:
        """(pred, conf), uint8 [B, seq_len, len(orders)]: the features of positions first[b] .. first[b] + seq_len - 1, by absolute position."""
        first = [int(f) for f in torch.as_tensor(first).reshape(-1).tolist()]
        if first and (min(first) < 0 or max(first) + seq_len > self.size):
            raise ValueError(f"a window of {seq_len} features from position {max(first)} runs past the end of the corpus ({self.size} bytes)")
        pred = np.empty((len(first), seq_len, len(self.orders)), dtype=self.pred_dtype)
        conf = np.empty((len(first), seq_len, len(self.orders)), dtype=np.uint8)
        for k in range(len(self.orders)):
            for b, f in enumerate(first):
                pred[b, :, k] = self.pred[k][f:f + seq_len]
                conf[b, :, k] = self.conf[k][f:f + seq_len]
        return torch.from_numpy(pred), torch.from_numpy(conf)

    def windows(self, starts, seq_len: int) -> tuple[Tensor, Tensor]:
        """The features for windows whose input bytes start at the absolute positions `starts` (batch_starts, or offset + i * seq_len):
        index t carries the features of position starts[b] + t + 1, the prediction for the byte the input at t is used to predict."""
        return self.at(torch.as_tensor(starts) + 1, seq_len)

    def fixed(self, offset: int, n_windows: int, seq_len: int) -> tuple[Tensor, Tensor]:
        """The features of the n_windows fixed windows from byte `offset` (flybrain.data.text.fixed_windows over the bytes from there)."""
        return self.windows(offset + seq_len * torch.arange(n_windows), seq_len)

    def check_corpus(self, corpus: str | Path) -> None:
        """Refuse a corpus of another size than the one these features were built on."""
        size = Path(corpus).stat().st_size
        want = self.size * (2 if self.unit == "token" else 1)        # a token stream file holds 2 bytes per position
        if size != want:
            raise ValueError(f"the features in {self.dir} are for another corpus: {self.size} {self.unit}s ({want} bytes), {corpus} has {size} bytes")
