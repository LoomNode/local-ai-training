"""Context material for a held-out target: its same-directory files or a seeded random draw from other directories, cut at a byte budget.

Every function here applies the leakage rules: the target and any file with its content never appear, random material never comes
from the target's directory, and only files entirely inside the training range are drawn at random."""

from __future__ import annotations

import random
from pathlib import Path

from flybrain.data.files import TRAIN_END, FileEntry, read_region, related


def chunks(path: str | Path, table: list[FileEntry], order: list[int], limit: int, per_file: int | None = None) -> list[tuple[int, bytes]]:
    """(file index, bytes) for the files of `order`: header + body, each cut at per_file bytes, the whole list cut at `limit` bytes."""
    out: list[tuple[int, bytes]] = []
    have = 0
    for j in order:
        if have >= limit:
            break
        e = table[j]
        n = min(limit - have, e.end - e.start if per_file is None else min(per_file, e.end - e.start))
        if n > 0:
            out.append((j, read_region(path, e.start, e.start + n)))
            have += n
    return out


def joined(cs: list[tuple[int, bytes]]) -> bytes:
    return b"".join(b for _, b in cs)


def random_order(table: list[FileEntry], i: int, hashes: list[str], seed: int, train_end: int = TRAIN_END) -> list[int]:
    """Seeded permutation of the training-range files outside table[i]'s directory whose content differs from table[i]'s."""
    d = table[i].dirname
    pool = [j for j, e in enumerate(table) if e.end <= train_end and e.dirname != d and hashes[j] != hashes[i]]
    random.Random(seed * 1_000_003 + i).shuffle(pool)
    return pool


def prefix_pairs(path: str | Path, table: list[FileEntry], i: int, hashes: list[str], ps: list[int] | tuple[int, ...], seed: int,
                 lo: int = 0, hi: int = TRAIN_END, train_end: int = TRAIN_END) -> dict[int, tuple[bytes, bytes]]:
    """For each P: (related prefix, random prefix). Related = table[i]'s same-directory files inside [lo, hi) in path order, cut at P bytes.
    Random = the seeded draw of `random_order`, cut at the related prefix's length so both sets have equal size."""
    rel = related(i, table, lo, hi, hashes)
    order = random_order(table, i, hashes, seed, train_end)
    out = {}
    for p in ps:
        r = joined(chunks(path, table, rel, p))
        out[p] = (r, joined(chunks(path, table, order, len(r))))
    return out


def reader_chunks(path: str | Path, table: list[FileEntry], i: int, hashes: list[str], budget: int = 32_768, per_file: int = 8192,
                  lo: int = 0, hi: int = TRAIN_END) -> list[tuple[int, bytes]]:
    """Reader material: the first per_file bytes (header included) of each related file in path order, `budget` bytes in all."""
    return chunks(path, table, related(i, table, lo, hi, hashes), budget, per_file)
