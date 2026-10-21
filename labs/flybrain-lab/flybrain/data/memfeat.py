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


class StackedFeatureStore:
    """Several FeatureStores read piecewise by absolute position (the distillation plan's mixed arm, 2026-10-20: a corpus made of mixB's
    train region, the slice, then mixB's dev and test regions, whose features live in two directories). segments: [(directory, lo, hi,
    base), ...] in order, contiguous from 0: position p in [lo, hi) reads that directory at p - lo + base. The FeatureStore interface:
    orders, info, size, train_end (the training positions from 0, as far as every segment's own train_end allows), at, windows, fixed,
    check_corpus. A window must lie inside one segment (the range sampler keeps it so)."""

    def __init__(self, segments, orders):
        self.segments = []
        pos = 0
        for d, lo, hi, base in segments:
            lo, hi, base = int(lo), int(hi), int(base)
            if lo != pos or hi <= lo:
                raise ValueError(f"segments must be contiguous from 0: got [{lo}, {hi}) after {pos}")
            st = FeatureStore(d, orders)
            if base + (hi - lo) > st.size:
                raise ValueError(f"segment [{lo}, {hi}) at base {base} runs past the features in {d} ({st.size} positions)")
            self.segments.append((st, lo, hi, base)); pos = hi
        s0 = self.segments[0][0]
        self.orders = s0.orders; self.info = dict(s0.info); self.info["stacked"] = [(str(s.dir), lo, hi, base) for s, lo, hi, base in self.segments]
        self.unit, self.value_orders, self.pred_dtype = s0.unit, s0.value_orders, s0.pred_dtype
        if any(s.unit != self.unit for s, *_ in self.segments):
            raise ValueError("stacked features must all be byte features or all token features")
        self.size = pos
        self.train_end = 0
        for st, lo, hi, base in self.segments:
            n = min(hi - lo, max(0, st.train_end - base))
            self.train_end += n
            if n < hi - lo: break

    def at(self, first, seq_len: int) -> tuple[Tensor, Tensor]:
        first = [int(f) for f in torch.as_tensor(first).reshape(-1).tolist()]
        pred = np.empty((len(first), seq_len, len(self.orders)), dtype=self.pred_dtype); conf = np.empty((len(first), seq_len, len(self.orders)), dtype=np.uint8)
        for b, f in enumerate(first):
            for st, lo, hi, base in self.segments:
                if lo <= f < hi:
                    if f + seq_len > hi:
                        raise ValueError(f"a window of {seq_len} features from {f} crosses the segment end {hi}")
                    pb, cb = st.at([f - lo + base], seq_len); pred[b] = pb[0].numpy(); conf[b] = cb[0].numpy(); break
            else:
                raise ValueError(f"position {f} is outside the stacked features (0..{self.size})")
        return torch.from_numpy(pred), torch.from_numpy(conf)

    def windows(self, starts, seq_len: int) -> tuple[Tensor, Tensor]:
        return self.at(torch.as_tensor(starts) + 1, seq_len)

    def fixed(self, offset: int, n_windows: int, seq_len: int) -> tuple[Tensor, Tensor]:
        return self.windows(offset + seq_len * torch.arange(n_windows), seq_len)

    def check_corpus(self, corpus: str | Path) -> None:
        size = Path(corpus).stat().st_size; want = self.size * (2 if self.unit == "token" else 1)
        if size != want:
            raise ValueError(f"{corpus} has {size} bytes; the stacked features cover {self.size} {self.unit}s ({want} bytes)")


def parse_stack(spec: str):
    """'DIR:lo:hi:base,DIR:lo:hi:base' -> [(DIR, lo, hi, base), ...] for StackedFeatureStore."""
    out = []
    for part in spec.split(","):
        d, lo, hi, base = part.rsplit(":", 3); out.append((d, int(lo), int(hi), int(base)))
    return out


class TeacherStore:
    """The teacher's next-byte targets (the distillation plan, 2026-10-20; distill_teacher.py): teacher_ids.npy [rows, top] uint8 and
    teacher_p.npy [rows, top] float16 for the slice's positions, memory-mapped; absolute position p reads row p - offset, and positions
    outside [offset, offset + rows) get zeros (no teacher: the loss falls back to the text's byte there). The alignment of
    FeatureStore.windows: index t of a window starting at s carries the targets of position s + t + 1."""

    def __init__(self, directory: str | Path, offset: int = 0):
        self.dir = Path(directory); self.offset = int(offset)
        self.info = json.loads((self.dir / "teacher.json").read_text())
        self.ids = np.load(self.dir / "teacher_ids.npy", mmap_mode="r"); self.p = np.load(self.dir / "teacher_p.npy", mmap_mode="r")
        if self.ids.dtype != np.uint8 or self.p.dtype != np.float16 or self.ids.shape != self.p.shape or self.ids.ndim != 2:
            raise ValueError(f"teacher arrays in {self.dir} are {self.ids.dtype}{self.ids.shape} / {self.p.dtype}{self.p.shape}")
        self.rows, self.top = self.ids.shape

    def at(self, first, seq_len: int) -> tuple[Tensor, Tensor]:
        first = [int(f) for f in torch.as_tensor(first).reshape(-1).tolist()]
        ids = np.zeros((len(first), seq_len, self.top), dtype=np.uint8); p = np.zeros((len(first), seq_len, self.top), dtype=np.float16)
        for b, f in enumerate(first):
            lo, hi = f - self.offset, f - self.offset + seq_len
            a0, a1 = max(0, lo), min(self.rows, hi)
            if a1 > a0:
                ids[b, a0 - lo:a1 - lo] = self.ids[a0:a1]; p[b, a0 - lo:a1 - lo] = self.p[a0:a1]
        return torch.from_numpy(ids), torch.from_numpy(p)

    def windows(self, starts, seq_len: int) -> tuple[Tensor, Tensor]:
        return self.at(torch.as_tensor(starts) + 1, seq_len)
