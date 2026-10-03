"""File table of a header-delimited byte corpus (the format of scripts/build_corpus.py): offsets, content hashes,
same-directory lookups and the dev/test split of the held-out tail."""

from __future__ import annotations

import hashlib
import mmap
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

HEADER = re.compile(rb"\n// ==== ([^\n]+?) ====\n")
TRAIN_END = 200_000_000   # training reads bytes [0, TRAIN_END)
TAIL_START = 202_100_000  # held-out tail: files whose header starts here or later (past the legacy validation window)
DEV_FILES = 500           # the first DEV_FILES tail files are dev, the rest are test
MIN_NEVER = 2             # never-trained pass: a target's own split must hold this many other same-directory files


@dataclass(frozen=True)
class FileEntry:
    path: str
    start: int  # offset of the header's leading newline
    body: int   # first content byte
    end: int    # start of the next header, or EOF

    @property
    def dirname(self) -> str:
        return posixpath.dirname(self.path)


def file_table(path: str | Path) -> list[FileEntry]:
    """One entry per header, in byte order. A header cut by EOF is not a file (its bytes stay in the previous body)."""
    with open(path, "rb") as f:
        if f.seek(0, 2) == 0:
            return []
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            hits = [(m.group(1).decode("utf-8"), m.start(), m.end()) for m in HEADER.finditer(mm)]
            size = len(mm)
    return [FileEntry(p, s, b, hits[k + 1][1] if k + 1 < len(hits) else size) for k, (p, s, b) in enumerate(hits)]


def read_region(path: str | Path, start: int, end: int) -> bytes:
    """Bytes [start, end) by seek and read; never the whole file."""
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(max(0, end - start))


def content_hash(path: str | Path, entry: FileEntry) -> str:
    """Hash of the body only, so the same content under two paths hashes alike."""
    return hashlib.blake2b(read_region(path, entry.body, entry.end), digest_size=16).hexdigest()


def hash_table(path: str | Path, table: list[FileEntry]) -> list[str]:
    return [content_hash(path, e) for e in table]


def related(i: int, table: list[FileEntry], lo: int, hi: int, hashes: list[str]) -> list[int]:
    """Files in table[i]'s directory that lie entirely inside bytes [lo, hi), minus i and any file with i's content, sorted by path."""
    d = table[i].dirname
    out = [j for j, e in enumerate(table) if j != i and e.dirname == d and e.start >= lo and e.end <= hi and hashes[j] != hashes[i]]
    return sorted(out, key=lambda j: (table[j].path, table[j].start))


def splits(table: list[FileEntry], tail_start: int = TAIL_START, dev_files: int = DEV_FILES) -> dict[str, list[int]]:
    """dev = the first dev_files held-out tail files (header at or after tail_start), test = the rest, both in file order."""
    tail = [i for i, e in enumerate(table) if e.start >= tail_start]
    return {"dev": tail[:dev_files], "test": tail[dev_files:]}


def split_range(table: list[FileEntry], split_indices: list[int]) -> tuple[int, int]:
    """Byte range [lo, hi) covered by a split's files (the tail files are contiguous)."""
    return table[split_indices[0]].start, table[split_indices[-1]].end


def targets(table: list[FileEntry], split_indices: list[int], hashes: list[str], min_related: int = 4, train_end: int = TRAIN_END) -> list[int]:
    """Held-out files with at least min_related same-directory files entirely inside the training range."""
    return [i for i in split_indices if len(related(i, table, 0, train_end, hashes)) >= min_related]


def read_stream(path: str | Path, entry: FileEntry, n_scored: int) -> tuple[bytes, int, int]:
    """(header + the first n_scored body bytes, body_start = header length, bytes actually scored: fewer when the body is shorter)."""
    scored = min(n_scored, entry.end - entry.body)
    return read_region(path, entry.start, entry.body + scored), entry.body - entry.start, scored
