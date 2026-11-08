#!/usr/bin/env python3
"""Build a byte corpus from a source tree: chosen extensions, per-file size window, hex-dump filter, seeded file shuffle."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

_STRIP = re.compile(rb"0x|[\s,;{}]")
_HEX = re.compile(rb"^[0-9a-fA-F]+$")


def _hex_line(line: bytes) -> bool:
    core = _STRIP.sub(b"", line)
    return len(core) >= 32 and _HEX.match(core) is not None


def looks_like_hex_dump(b: bytes, sample_lines: int = 200) -> bool:
    """More than half the sampled lines are runs of hex bytes (tables, embedded blobs, dumps)."""
    lines = [l for l in b.splitlines()[:sample_lines] if l.strip()]
    return bool(lines) and sum(1 for l in lines if _hex_line(l)) > 0.5 * len(lines)


def build(root: Path, exts: tuple[str, ...], min_size: int, max_size: int, max_bytes: int, seed: int, out: Path) -> tuple[int, int]:
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in exts and min_size <= p.stat().st_size <= max_size]
    random.Random(seed).shuffle(files)
    buf = bytearray(); n = 0
    for p in files:
        b = p.read_bytes()
        if looks_like_hex_dump(b):
            continue
        try:
            b.decode("utf-8")
        except UnicodeDecodeError:
            continue
        buf += b"\n// ==== " + str(p.relative_to(root)).encode() + b" ====\n" + b; n += 1
        if len(buf) >= max_bytes:
            break
    out.write_bytes(bytes(buf[:max_bytes]))
    return min(len(buf), max_bytes), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--exts", default=".c,.h,.py")
    ap.add_argument("--min-size", type=int, default=2_000)
    ap.add_argument("--max-size", type=int, default=60_000)
    ap.add_argument("--max-bytes", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n_bytes, n_files = build(Path(a.root), tuple(a.exts.split(",")), a.min_size, a.max_size, a.max_bytes, a.seed, Path(a.out))
    print(f"{a.out}: {n_bytes} bytes from {n_files} files")


if __name__ == "__main__":
    main()
