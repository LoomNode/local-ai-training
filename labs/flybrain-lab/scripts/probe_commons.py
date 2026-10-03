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


def new_cube(n_p: int) -> np.ndarray:
    """[scope, P, partition, condition, code, (count, summed bits)]: what one target adds to every pooled number."""
    return np.zeros((len(SCOPES), n_p, len(PARTS), len(CONDS), 3, 2))


def fill(cube: np.ndarray, pi: int, part: str, codes, bits_by_cond: dict[str, np.ndarray]) -> None:
    """Set the cells of one (P, partition) on one target's own cube from that partition's census codes and the per-byte bits of its
    conditions, for the full scored region and its first FIRST bytes. It assigns, it does not add: call it once per target, P and
    partition, then stack the per-target cubes."""
    codes = np.asarray(codes)
    for s, cap in enumerate((len(codes), min(FIRST, len(codes)))):
        for cond in FILL[part]:
            for k in range(3):
                m = codes[:cap] == k
                cube[s, pi, PART_IX[part], COND_IX[cond], k] = (m.sum(), bits_by_cond[cond][:cap][m].sum())


def _bpb(cell: np.ndarray) -> float | None:
    """Bits per byte of a [code, (count, bits)] cell pooled over its codes; None when it holds no bytes."""
    n = cell[:, 0].sum()
    return float(cell[:, 1].sum() / n) if n else None


def _by_code(cell: np.ndarray, field: int) -> dict[str, float | None]:
    """count (field 0) or bits per byte (field 1) of each code A, B, C."""
    if field == 0:
        return {c: int(cell[k, 0]) for k, c in enumerate("ABC")}
    return {c: (float(cell[k, 1] / cell[k, 0]) if cell[k, 0] else None) for k, c in enumerate("ABC")}


def headroom(cs_p: np.ndarray) -> dict:
    """Headroom numbers of one P from a pooled cube slice [partition, condition, code, 2], in bits/byte of the alone condition:
    H(X) = share(X) * (loss(X) - loss(A)), None when A or the partition holds no bytes; gate = H(B) - H(B')."""
    out = {}
    for part, tag in (("related", "B"), ("random", "Bp"), ("never", "Bn")):
        cell = cs_p[PART_IX[part], COND_IX["alone"]]
        n, s = cell[:, 0], cell[:, 1]
        total = n.sum()
        loss = [float(s[k] / n[k]) if n[k] else None for k in range(3)]
        out[f"share_{tag}"] = float(n[1] / total) if total else None
        out[f"loss_{tag}"] = loss[1]
        out[f"H_{tag}"] = None if not n[0] or not total else (float(n[1] / total * (loss[1] - loss[0])) if n[1] else 0.0)
        if part == "related":
            out["loss_A"] = loss[0]
    out["gate"] = None if out["H_B"] is None or out["H_Bp"] is None else out["H_B"] - out["H_Bp"]
    return out


def ci95(values: list) -> list[float] | None:
    """2.5 % and 97.5 % percentiles of the resampled values; None when more than a tenth of the resamples had no value."""
    got = [v for v in values if v is not None]
    return [float(np.percentile(got, 2.5)), float(np.percentile(got, 97.5))] if got and len(got) >= 0.9 * len(values) else None


def effects_of(cs_p: np.ndarray) -> dict[str, float | None]:
    """Every gain and headroom number of one (scope, P) slice [partition, condition, code, 2] pooled over targets, in bits/byte.
    A gain is alone minus the condition on the same bytes (positive = fewer bits); gain_<partition> is read in the bins of that
    partition and bin_<partition>_<code> is the share-weighted part of it that falls on the code, so the three add up to the gain."""
    h = headroom(cs_p)
    out = {k: h[k] for k in ("H_B", "H_Bp", "H_Bn", "gate")}
    for part in PARTS:
        alone, other = cs_p[PART_IX[part], COND_IX["alone"]], cs_p[PART_IX[part], COND_IX[part]]
        n = alone[:, 0].sum()
        share_gain = (alone[:, 1] - other[:, 1]) / n if n else None
        out[f"gain_{part}"] = None if share_gain is None else float(share_gain.sum())
        for k, c in enumerate("ABC"):
            out[f"bin_{part}_{c}"] = None if share_gain is None else float(share_gain[k])
    out["gain_related_vs_random"] = None if out["gain_related"] is None or out["gain_random"] is None else out["gain_related"] - out["gain_random"]
    return out


def bootstrap_effects(cubes: np.ndarray, ps, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """95 % intervals of effects_of over resamples of the targets (files) with replacement: {scope: {P: {name: [lo, hi] | None}}}."""
    rng = np.random.default_rng(seed)
    t = len(cubes)
    runs = {(s, pi): [] for s in range(len(SCOPES)) for pi in range(len(ps))}
    for _ in range(n_boot):
        pooled = cubes[rng.integers(0, t, t)].sum(0)
        for (s, pi), acc in runs.items():
            acc.append(effects_of(pooled[s, pi]))
    return {scope: {str(ps[pi]): {k: ci95([e[k] for e in acc]) for k in acc[0]} for (s2, pi), acc in runs.items() if s2 == s} for s, scope in enumerate(SCOPES)}


def summarize(cubes: np.ndarray, ps, with_never: bool, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """The nested result for stacked per-target cubes [target, scope, P, partition, condition, code, 2]: pooled numbers, the gains and
    headroom of effects_of, and their bootstrap intervals over targets. The never-trained numbers (bpb[scope]["never"], the never entries
    of bins, H_Bn and the never gains) pool only the targets that filled the never partition, so they are not comparable with the
    all-target bpb[scope]["alone"] and must not be subtracted from it: the paired number is gains[scope][P]["gain_never"] (alone minus
    never on the same bytes). A caller who wants related and never numbers on matched targets passes only the never-subset cubes."""
    pooled = cubes.sum(0)
    out = {"n_scored_bytes": {}, "bpb": {}, "bins": {}, "headroom": {}, "gains": {}, "ci95": bootstrap_effects(cubes, ps, n_boot, seed)}
    conds = ("related", "random", "never") if with_never else ("related", "random")
    for s, scope in enumerate(SCOPES):
        cs = pooled[s]
        out["n_scored_bytes"][scope] = int(cs[0, PART_IX["related"], COND_IX["alone"], :, 0].sum())
        out["bpb"][scope] = {"alone": _bpb(cs[0, PART_IX["related"], COND_IX["alone"]])}
        for cond in conds:
            # cond == "never" pools only the targets that filled the never partition: do not subtract it from the all-target "alone" above;
            # the paired number is gains[scope][P]["gain_never"], and matched related and never numbers need only the never-subset cubes
            out["bpb"][scope][cond] = {str(p): _bpb(cs[pi, PART_IX[cond], COND_IX[cond]]) for pi, p in enumerate(ps)}
        out["bins"][scope] = {str(p): {part: {"count": _by_code(cs[pi, PART_IX[part], COND_IX["alone"]], 0),
                                              "bpb": {c: _by_code(cs[pi, PART_IX[part], COND_IX[c]], 1) for c in FILL[part]}}
                                       for part in PARTS if part != "never" or with_never} for pi, p in enumerate(ps)}
        out["headroom"][scope] = {str(p): headroom(cs[pi]) for pi, p in enumerate(ps)}
        out["gains"][scope] = {str(p): effects_of(cs[pi]) for pi, p in enumerate(ps)}
    return out
