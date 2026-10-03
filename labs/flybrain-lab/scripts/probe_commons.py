#!/usr/bin/env python3
"""Do related files have anything to share? Scores the held-out targets of the file-ordered Linux corpus.

--mode context: plain forward, batch 1, one sequence per condition per target, scoring the target's first bytes after
  alone (header + target), related (its same-directory training files up to P bytes, then the target), random (an equal-size
  draw from other directories) and never-trained (related files from the target's own split). The alone per-byte loss is binned by
  the census (flybrain.census.bins, n = 8) to give the headroom H(X) = share(X) * (loss_alone(X) - loss_alone(A)) for X = B and B'.
--mode handoff: reader agents (fresh state, stream API) read the related files; their pooled tables are preloaded into a fresh agent that
  scores the target. Needs flybrain.group and GRUHybrid.stream (the agent-state plan); the context mode does not."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.census import bins
from flybrain.data.files import DEV_FILES, MIN_NEVER, TAIL_START, TRAIN_END, file_table, hash_table, read_stream, related, split_range, splits, targets
from flybrain.data.prefixes import prefix_pairs, reader_chunks
from scripts.cross_eval import rebuild

RESULTS = ROOT / "results"
BIN_N = 8          # the census n of the headroom numbers: what copyable_mask has always used
FIRST = 2048       # every model also reports its first FIRST scored bytes, so rows with shorter scoring compare
GATE_P = 32_768    # the gate number is read at this prefix size
GATE_THRESHOLD = 0.02  # Gate 1: H(B) - H(B') at or above this is Stage 2
N_BOOT = 1000      # resamples of the targets for every 95 % interval
SCOPES = ("full", "first2048")
PARTS = ("related", "random", "never")             # which prefix defined the census bins
CONDS = ("alone", "related", "random", "never")    # which prefix the model read
PART_IX = {p: k for k, p in enumerate(PARTS)}
COND_IX = {c: k for k, c in enumerate(CONDS)}
FILL = {"related": ("alone", "related", "random"), "random": ("alone", "related", "random"), "never": ("alone", "never")}


def apply_chunk(model, chunk: int) -> None:
    """Set the chunk length of every memory head (0 keeps the checkpoint's own): long sequences need the chunked form."""
    if chunk > 0:
        for h in getattr(model, "all_heads", lambda: [])():
            h.chunk = int(chunk)


def arm_settings(arm: str, ps, n_scored: int) -> tuple[list[int], int]:
    """The transformer reference reads 2,048 prefix bytes and scores the first 2,048 target bytes; every other arm takes the flags."""
    return ([2048], 2048) if arm == "transformer" else (list(ps), n_scored)


def to_tokens(data: bytes, device) -> Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device).long()[None]


def bits_of(logits: Tensor, x: Tensor) -> np.ndarray:
    """Bits of every byte of x[0] given the bytes before it, from logits [1, T, 256]. Entry 0 has no context (NaN)."""
    ce = F.cross_entropy(logits[0, :-1].float(), x[0, 1:], reduction="none") / math.log(2)
    return np.concatenate([[np.nan], ce.cpu().numpy()])


@torch.no_grad()
def byte_bits(model, data: bytes, device, rope_train_len: int = 0) -> np.ndarray:
    """One plain forward over `data`. rope_train_len > 0 (the rotary transformer arm) NTK-scales the base to this sequence's length
    (1.0 when it is not longer than the training length), set afresh for every sequence so nothing leaks from the previous one."""
    if rope_train_len:
        model.core.set_rope_scale(max(1.0, len(data) / rope_train_len))
    x = to_tokens(data, device)
    logits, _ = model(x)
    return bits_of(logits, x)


def score_with_prefix(model, prefix: bytes, stream: bytes, body_start: int, n_scored: int, device, rope_train_len: int = 0) -> np.ndarray:
    """Bits of the scored bytes stream[body_start : body_start + n_scored] when `stream` is read after `prefix` (b"" = alone)."""
    bits = byte_bits(model, prefix + stream, device, rope_train_len)
    a = len(prefix) + body_start
    return bits[a : a + n_scored]
