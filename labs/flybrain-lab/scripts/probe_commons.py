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


def probe_target(model, device, path, table, hashes, i: int, ps, seed: int, n_scored: int, rope_train_len: int = 0,
                 never_range: tuple[int, int] | None = None, train_end: int = TRAIN_END):
    """(cube, record) for target i, or None when its body is empty."""
    stream, body_start, scored = read_stream(path, table[i], n_scored)
    if scored == 0:
        return None
    score = lambda prefix: score_with_prefix(model, prefix, stream, body_start, scored, device, rope_train_len)
    alone = score(b"")
    pairs = prefix_pairs(path, table, i, hashes, ps, seed, 0, train_end, train_end)
    npairs = None
    if never_range is not None and len(related(i, table, *never_range, hashes)) >= MIN_NEVER:
        npairs = prefix_pairs(path, table, i, hashes, ps, seed, *never_range, train_end)
    cube = new_cube(len(ps))
    bpb = {"alone": float(alone.mean()), "related": {}, "random": {}, "never": {} if npairs else None}
    for pi, p in enumerate(ps):
        rel, rnd = pairs[p]
        b = {"alone": alone, "related": score(rel), "random": score(rnd)}
        fill(cube, pi, "related", bins(stream, body_start, scored, rel, BIN_N), b)
        fill(cube, pi, "random", bins(stream, body_start, scored, rnd, BIN_N), b)
        bpb["related"][str(p)], bpb["random"][str(p)] = float(b["related"].mean()), float(b["random"].mean())
        if npairs:
            nt = npairs[p][0]
            b["never"] = score(nt)
            fill(cube, pi, "never", bins(stream, body_start, scored, nt, BIN_N), b)
            bpb["never"][str(p)] = float(b["never"].mean())
    rec = {"index": i, "path": table[i].path, "n_scored": scored, "never": npairs is not None, "bpb": bpb,
           "prefix_bytes": {str(p): [len(pairs[p][0]), len(npairs[p][0]) if npairs else None] for p in ps}}
    return cube, rec


def probe_context(model, device, path, split: str, ps, seed: int = 0, n_scored: int = 4096, rope_train_len: int = 0, never: bool = True,
                  train_end: int = TRAIN_END, tail_start: int = TAIL_START, dev_files: int = DEV_FILES, limit: int = 0) -> dict:
    table = file_table(path)
    hashes = hash_table(path, table)
    idxs = splits(table, tail_start, dev_files)[split]
    if not idxs:
        raise ValueError(f"split {split!r} holds no files")
    never_range = split_range(table, idxs) if never else None
    cubes, recs = [], []
    for i in targets(table, idxs, hashes, 4, train_end)[: limit or None]:
        got = probe_target(model, device, path, table, hashes, i, ps, seed, n_scored, rope_train_len, never_range, train_end)
        if got is not None:
            cubes.append(got[0]); recs.append(got[1])
            if len(recs) % 25 == 0:
                print(f"  {len(recs)} targets", flush=True)
    if not cubes:
        raise ValueError("no target to score")
    cubes = np.stack(cubes)
    sub = np.array([r["never"] for r in recs])
    res = {"split": split, "ps": list(ps), "n_scored_cap": n_scored, "rope_train_len": rope_train_len, "n_targets": len(recs),
           "n_never_targets": int(sub.sum()), "gate_p": GATE_P, "gate_number": None, "gate_ci95": None,
           "all": summarize(cubes, ps, False, N_BOOT, seed), "never_subset": summarize(cubes[sub], ps, True, N_BOOT, seed) if sub.any() else None,
           "targets": recs}
    if GATE_P in ps:
        res["gate_number"] = res["all"]["gains"]["full"][str(GATE_P)]["gate"]
        res["gate_ci95"] = res["all"]["ci95"]["full"][str(GATE_P)]["gate"]
    return res


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _ci(v, ci, fmt: str = "+.3f") -> str:
    """A value with its 95 % interval, '+0.034 (+0.020, +0.048)'; n/a without a value."""
    return "n/a" if v is None else format(v, fmt) + ("" if ci is None else f" ({format(ci[0], fmt)}, {format(ci[1], fmt)})")


def _g(block: dict, scope: str, p, key: str) -> str:
    """A gain or headroom number of a context-probe block with its interval."""
    return _ci(block["gains"][scope][str(p)][key], block["ci95"][scope][str(p)][key])


def render_context(res: dict) -> str:
    """Markdown tables: bits/byte per condition, gains over alone with 95 % intervals, bits/byte and gain per census bin (n = 8),
    the headroom numbers and the gate number (four decimals, with the side of the 0.02 threshold the unrounded number falls on).
    The all-targets block prints no never-trained gain: that gain pools only the never subset, so it reads n/a there and is printed
    in the never-trained subset block, where it sits beside gains on the same targets."""
    ps = res["ps"]
    lines = [f"{res.get('run_name', '')} {res['split']}: {res['n_targets']} targets ({res['n_never_targets']} with never-trained files), "
             f"{res['all']['n_scored_bytes']['full']} scored bytes, P = {ps}"]
    for block, title in (("all", "all targets"), ("never_subset", "never-trained subset")):
        b = res.get(block)
        if not b:
            continue
        for scope in SCOPES:
            lines += [f"\n{title}, {scope} ({b['n_scored_bytes'][scope]} bytes): bits/byte, alone = {_f(b['bpb'][scope]['alone'])}",
                      "| P | related | random | never trained |", "|---:|---:|---:|---:|"]
            lines += [f"| {p} | {_f(b['bpb'][scope]['related'][str(p)])} | {_f(b['bpb'][scope]['random'][str(p)])} | "
                      f"{_f(b['bpb'][scope].get('never', {}).get(str(p)))} |" for p in ps]
            lines += [f"\n{title}, {scope}: gains over alone in bits/byte (positive = fewer bits), 95 % interval over targets",
                      "| P | gain related | gain random | related minus random | gain never trained |", "|---:|---:|---:|---:|---:|"]
            lines += [f"| {p} | " + " | ".join(_g(b, scope, p, k) for k in ("gain_related", "gain_random", "gain_related_vs_random")) + " | "
                      + (_g(b, scope, p, "gain_never") if block == "never_subset" else "n/a (never subset only)") + " |" for p in ps]
        for p in ps:
            for part, tag in (("related", "B"), ("random", "B'")) + ((("never", "Bn"),) if block == "never_subset" else ()):
                cell = b["bins"]["full"][str(p)][part]
                lines += [f"\n{title}, P = {p}, census bins from the {part} prefix (full scope)",
                          "| bin | share | " + " | ".join(FILL[part]) + f" | {part} gain, share-weighted |", "|---|---:|" + "---:|" * (len(FILL[part]) + 1)]
                n = sum(cell["count"].values())
                for code, name in zip("ABC", ("A", tag, "C")):
                    lines.append(f"| {name} | {_f(cell['count'][code] / n if n else None)} | " + " | ".join(_f(cell["bpb"][c][code]) for c in FILL[part])
                                 + f" | {_g(b, 'full', p, f'bin_{part}_{code}')} |")
        lines += [f"\n{title}: headroom in bits/byte (alone-condition loss), 95 % interval over targets",
                  "| P | scope | share B | share B' | H(B) | H(B') | H(B) - H(B') |", "|---:|---|---:|---:|---:|---:|---:|"]
        for scope in SCOPES:
            for p in ps:
                h = b["headroom"][scope][str(p)]
                lines.append(f"| {p} | {scope} | {_f(h['share_B'])} | {_f(h['share_Bp'])} | " + " | ".join(_g(b, scope, p, k) for k in ("H_B", "H_Bp", "gate")) + " |")
    ci, gate = res["gate_ci95"], res["gate_number"]
    straddle = " The interval straddles the 0.02 threshold." if ci is not None and ci[0] < GATE_THRESHOLD < ci[1] else ""
    side = "" if gate is None else f" ({'above' if gate >= GATE_THRESHOLD else 'below'} the {GATE_THRESHOLD} threshold)"   # the unrounded number decides; 0.02 itself is above
    lines.append(f"\ngate number H(B) - H(B') at n = {BIN_N}, P = {res['gate_p']}: {'n/a' if gate is None else format(gate, '.4f')}{side}, 95 % interval over targets "
                 f"{None if ci is None else [round(c, 4) for c in ci]}.{straddle}")
    return "\n".join(lines)


ALPHAS = (0.03, 0.1, 0.3, 1.0)
MIN_GAIN = 0.01  # the transfer ratio is reported when alone - context is at least this many bits/byte


def add_commons(a, b):
    """Per-head sum of two Commons, so readers can be merged one at a time: the pooling needs the running sum and the newest reader's tables,
    not every reader's. It does not bound handoff_target's footprint, which also holds the sequential agent's tables and the state being scored."""
    from flybrain.group import Commons
    return Commons([x + y for x, y in zip(a.S, b.S)], [x + y for x, y in zip(a.Ksum, b.Ksum)])


def head_diff(commons, state) -> list[tuple[float, float]]:
    """Per head (all_heads() order): relative difference of S and of Ksum between the merged reader tables and the tables of one agent
    that read the same files in sequence. Only the lowest head's keys do not depend on earlier reads, so only its difference is ~0."""
    rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-12))
    return [(rel(S, h.S[0]), rel(K, h.Ksum[0])) for S, K, h in zip(commons.S, commons.Ksum, state.heads)]


@torch.no_grad()
def handoff_target(model, device, stream: bytes, body_start: int, n_scored: int, readers: list[bytes], alphas, seg: int = 2048) -> dict:
    """Bits of the scored bytes (arrays of length n_scored, shorter when the stream ends before body_start + n_scored, exactly as
    score_with_prefix's are) with no memory ('alone', through the stream API), with the merged tables of one fresh reader agent per entry
    of `readers` preloaded at each alpha, and after one agent read them in order with new_document() between files; plus
    head_diff(merged readers, sequential agent)."""
    if not readers:
        raise ValueError("handoff needs at least one reader")
    from flybrain.group import merge, preload, read
    x = to_tokens(stream, device)
    score = lambda state: bits_of(model.stream(x, state)[0], x)[body_start : body_start + n_scored]
    commons, seq = None, model.init_state(1, device)
    for r in readers:
        t = to_tokens(r, device)
        c = merge(read(model, t, None, seg)[1])
        commons = c if commons is None else add_commons(commons, c)
        seq = read(model, t, seq, seg)[1].new_document()
    del c  # the last reader's merged tables: only the pooled commons and the sequential agent stay alive while the target is scored
    return {"alone": score(model.init_state(1, device)),
            "handoff": {a: score(preload(model.init_state(1, device), commons, a)) for a in alphas},
            "sequential": score(seq), "head_diff": head_diff(commons, seq)}


def transfer_ratio(alone: float, other: float, context: float) -> float | None:
    """(alone - other) / (alone - context), reported only when the denominator is at least MIN_GAIN."""
    return (alone - other) / (alone - context) if alone - context >= MIN_GAIN else None


def best_alpha(bpb_by_alpha: dict[float, float]) -> float:
    """The alpha with the lowest handoff bits/byte; ties go to the smaller alpha."""
    return min(bpb_by_alpha, key=lambda a: (bpb_by_alpha[a], a))


def handoff_effects(arr: dict[str, np.ndarray], alphas, idx=None) -> dict[str, float | None]:
    """Gains over alone (positive = fewer bits) and transfer ratios from per-target [sum of bits, count] arrays, one array per row
    ('alone', 'context', 'sequential', 'handoff:<alpha>'); all targets, or the rows idx of a resample."""
    pick = (lambda a: a) if idx is None else (lambda a: a[idx])
    bpb = {k: pick(a)[:, 0].sum() / pick(a)[:, 1].sum() for k, a in arr.items()}
    out = {"gain_context": bpb["alone"] - bpb["context"], "gain_sequential": bpb["alone"] - bpb["sequential"],
           "transfer_sequential": transfer_ratio(bpb["alone"], bpb["sequential"], bpb["context"])}
    for a in alphas:
        out[f"gain_handoff:{a}"] = bpb["alone"] - bpb[f"handoff:{a}"]
        out[f"transfer:{a}"] = transfer_ratio(bpb["alone"], bpb[f"handoff:{a}"], bpb["context"])
    return out


def bootstrap_handoff(arr: dict[str, np.ndarray], alphas, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, list[float] | None]:
    """95 % intervals of handoff_effects over resamples of the targets (files) with replacement; a ratio whose denominator falls
    below MIN_GAIN in more than a tenth of the resamples gets None."""
    rng = np.random.default_rng(seed)
    t = len(next(iter(arr.values())))
    runs = [handoff_effects(arr, alphas, rng.integers(0, t, t)) for _ in range(n_boot)]
    return {k: ci95([r[k] for r in runs]) for k in runs[0]}


def probe_handoff(model, device, path, split: str, alphas=ALPHAS, seed: int = 0, n_scored: int = 4096, budget: int = 32_768, per_file: int = 8192,
                  seg: int = 2048, train_end: int = TRAIN_END, tail_start: int = TAIL_START, dev_files: int = DEV_FILES, limit: int = 0) -> dict:
    """Readers are the target's related files in path order, the first per_file bytes of each, budget bytes in all. The context row is a
    plain forward over exactly those bytes as a prefix, so the transfer ratio compares the same material."""
    table = file_table(path)
    hashes = hash_table(path, table)
    idxs = splits(table, tail_start, dev_files)[split]
    if not idxs:
        raise ValueError(f"split {split!r} holds no files")
    per: dict[str, list[tuple[float, int]]] = {}
    recs, diffs = [], []
    for i in targets(table, idxs, hashes, 4, train_end)[: limit or None]:
        stream, body_start, scored = read_stream(path, table[i], n_scored)
        if scored == 0:
            continue
        readers = [b for _, b in reader_chunks(path, table, i, hashes, budget, per_file, 0, train_end)]
        out = handoff_target(model, device, stream, body_start, scored, readers, alphas, seg)
        rows = {"alone": out["alone"], "context": score_with_prefix(model, b"".join(readers), stream, body_start, scored, device),
                "alone_forward": score_with_prefix(model, b"", stream, body_start, scored, device), "sequential": out["sequential"]}
        for name, bits in [*rows.items(), *((f"handoff:{a}", v) for a, v in out["handoff"].items())]:
            per.setdefault(name, []).append((float(bits.sum()), len(bits)))
        diffs.append(out["head_diff"])
        recs.append({"index": i, "path": table[i].path, "n_scored": scored, "readers": len(readers), "reader_bytes": sum(map(len, readers)),
                     "bpb": {**{k: float(v.mean()) for k, v in rows.items()}, "handoff": {str(a): float(v.mean()) for a, v in out["handoff"].items()}}})
        if len(recs) % 25 == 0:
            print(f"  {len(recs)} targets", flush=True)
    if not recs:
        raise ValueError("no target to score")
    arr = {k: np.array(v) for k, v in per.items()}  # [target, (sum of bits, count)]
    pooled = {k: a[:, 0].sum() / a[:, 1].sum() for k, a in arr.items()}
    handoff = {a: pooled[f"handoff:{a}"] for a in alphas}
    best = best_alpha(handoff)
    eff, ci = handoff_effects(arr, alphas), bootstrap_handoff(arr, alphas, N_BOOT, seed)
    d = np.array(diffs)  # [target, head, (S, Ksum)]
    return {"split": split, "seed": seed, "alphas": list(alphas), "budget": budget, "per_file": per_file, "n_targets": len(recs),
            "n_scored_bytes": sum(r["n_scored"] for r in recs),
            "bpb": {"alone": pooled["alone"], "alone_forward": pooled["alone_forward"], "context": pooled["context"], "sequential": pooled["sequential"],
                    "handoff": {str(a): v for a, v in handoff.items()}},
            "best_alpha": best, "gain_context": eff["gain_context"], "gain_sequential": eff["gain_sequential"],
            "gain_handoff": {str(a): eff[f"gain_handoff:{a}"] for a in alphas},
            "transfer": {str(a): eff[f"transfer:{a}"] for a in alphas}, "transfer_best": eff[f"transfer:{best}"], "transfer_sequential": eff["transfer_sequential"],
            "ci95": {"gain_context": ci["gain_context"], "gain_sequential": ci["gain_sequential"], "gain_handoff": {str(a): ci[f"gain_handoff:{a}"] for a in alphas},
                     "transfer": {str(a): ci[f"transfer:{a}"] for a in alphas}, "transfer_best": ci[f"transfer:{best}"], "transfer_sequential": ci["transfer_sequential"]},
            "head_diff": [{"head": h, "S_rel": float(d[:, h, 0].mean()), "Ksum_rel": float(d[:, h, 1].mean())} for h in range(d.shape[1])],
            "readers": {"mean_files": float(np.mean([r["readers"] for r in recs])), "mean_bytes": float(np.mean([r["reader_bytes"] for r in recs]))},
            "targets": recs}


def render_handoff(res: dict) -> str:
    """Markdown tables: bits/byte, gain over alone and transfer ratio (95 % intervals over targets) per memory condition, and the per-head table difference."""
    b, ci = res["bpb"], res["ci95"]
    lines = [f"{res.get('run_name', '')} {res['split']}: {res['n_targets']} targets, {res['n_scored_bytes']} scored bytes, readers {res['readers']['mean_files']:.1f} files / "
             f"{res['readers']['mean_bytes']:.0f} bytes on average, best alpha {res['best_alpha']}",
             "| condition | bits/byte | gain vs alone (95 % interval) | transfer ratio (95 % interval) |", "|---|---:|---:|---:|",
             f"| alone | {b['alone']:.4f} | | |",
             f"| context (same bytes as a plain prefix) | {b['context']:.4f} | {_ci(res['gain_context'], ci['gain_context'])} | 1.00 |"]
    lines += [f"| handoff alpha {a} | {v:.4f} | {_ci(res['gain_handoff'][a], ci['gain_handoff'][a])} | {_ci(res['transfer'][a], ci['transfer'][a], '.2f')} |" for a, v in b["handoff"].items()]
    lines += [f"| sequential reader | {b['sequential']:.4f} | {_ci(res['gain_sequential'], ci['gain_sequential'])} | {_ci(res['transfer_sequential'], ci['transfer_sequential'], '.2f')} |",
              f"\nalone through the plain forward: {b['alone_forward']:.4f}", "\n| head (all_heads() order) | S relative difference | Ksum relative difference |", "|---:|---:|---:|"]
    lines += [f"| {h['head']} | {h['S_rel']:.2e} | {h['Ksum_rel']:.2e} |" for h in res["head_diff"]]
    return "\n".join(lines)


def load_checkpoint(path: str, device, chunk: int):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = rebuild(ck).to(device).eval()
    apply_chunk(model, chunk)
    return model, ck


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("context", "handoff"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True, choices=("dev", "test"))
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--corpus", default=str(ROOT / "corpora" / "linux_full.bytes"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ps", default="2048,4096", help="context: prefix sizes in bytes (the transformer arm always uses 2048)")
    ap.add_argument("--n-scored", type=int, default=4096, help="scored bytes per target (the transformer arm always scores 2048)")
    ap.add_argument("--chunk", type=int, default=None, help="head chunk length; 0 keeps the checkpoint's own. Default 4096 in context mode, 0 in handoff mode")
    ap.add_argument("--no-never", action="store_true", help="context: skip the never-trained condition")
    ap.add_argument("--alpha", type=float, default=None, help="handoff: score only this alpha (the one chosen on dev); default scores every alpha")
    ap.add_argument("--budget", type=int, default=32_768, help="handoff: total reader bytes per target")
    ap.add_argument("--per-file", type=int, default=8192, help="handoff: bytes each reader reads")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N targets only")
    ap.add_argument("--train-end", type=int, default=TRAIN_END, help="tiny-corpus tests only")
    ap.add_argument("--tail-start", type=int, default=TAIL_START, help="tiny-corpus tests only")
    ap.add_argument("--dev-files", type=int, default=DEV_FILES, help="tiny-corpus tests only")
    a = ap.parse_args(argv)
    out = RESULTS / f"{a.run_name}.json"
    if out.exists():
        sys.exit(f"{out} exists: new run names only")
    chunk = a.chunk if a.chunk is not None else (4096 if a.mode == "context" else 0)
    device = torch.device(a.device)
    model, ck = load_checkpoint(a.ckpt, device, chunk)
    if a.mode == "context":
        ps, n_scored = arm_settings(ck["arm"], [int(x) for x in a.ps.split(",")], a.n_scored)
        rope_train_len = int(ck["args"]["seq_len"]) if ck["arm"] == "transformer" else 0
        print(f"arm {ck['arm']}: P = {ps}, {n_scored} scored bytes per target, head chunk {chunk or 'as trained'}", flush=True)
        res = probe_context(model, device, a.corpus, a.split, ps, a.seed, n_scored, rope_train_len, not a.no_never, a.train_end, a.tail_start, a.dev_files, a.limit)
        text = render_context
    else:
        alphas = ALPHAS if a.alpha is None else (a.alpha,)
        res = probe_handoff(model, device, a.corpus, a.split, alphas, a.seed, a.n_scored, a.budget, a.per_file, 2048, a.train_end, a.tail_start, a.dev_files, a.limit)
        text = render_handoff
    res = {"run_name": a.run_name, "mode": a.mode, "ckpt": Path(a.ckpt).name, "arm": ck["arm"], "chunk": chunk, "args": vars(a), **res}
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(text(res))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
