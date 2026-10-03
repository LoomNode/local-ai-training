"""Exact-match census of scored bytes. A byte is A if it continues an n-gram seen earlier in its own stream, B if (not A) the
n-gram and its continuation occur in a prefix, C otherwise. Pure Python: no model, no GPU."""

from __future__ import annotations

A, B, C = 0, 1, 2  # codes returned by bins()


def ngram_next(source: bytes, n: int) -> dict[bytes, set[int]]:
    """Every n-byte window of `source` -> the set of bytes that followed it somewhere in `source`."""
    out: dict[bytes, set[int]] = {}
    for q in range(n, len(source)):
        out.setdefault(source[q - n : q], set()).add(source[q])
    return out


def bins(stream: bytes, body_start: int, n_scored: int, prefix: bytes, n: int) -> list[int]:
    """One code per scored byte stream[body_start : body_start + n_scored]. The n bytes before a scored byte may reach back into
    the header; a byte with fewer than n bytes before it has no n-gram and is C. The same call with a random prefix gives B' for B."""
    end = body_start + n_scored
    if body_start < 0 or n_scored < 0 or end > len(stream):
        raise ValueError(f"scored region [{body_start}, {end}) is outside a stream of {len(stream)} bytes")
    after = ngram_next(prefix, n)
    seen: dict[bytes, set[int]] = {}
    codes = [C] * max(0, min(n, end) - body_start)
    for q in range(n, end):
        g, b = stream[q - n : q], stream[q]
        if q >= body_start:
            codes.append(A if b in seen.get(g, ()) else B if b in after.get(g, ()) else C)
        seen.setdefault(g, set()).add(b)
    return codes
