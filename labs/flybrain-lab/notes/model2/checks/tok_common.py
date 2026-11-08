"""Model #2's token stream: the pieces every token script shares (spec docs/superpowers/specs/2026-10-16-model2-tokens-design.md).

A corpus is tokenized per file unit (table_common.file_cuts: a unit starts at a header's "/" and ends before the next header's), so no
token spans two files. Every byte belongs to exactly one token: a file's bytes are decoded as UTF-8 with surrogateescape, the valid
runs go through the BPE, and each invalid byte becomes its own single-byte token (the byte-level alphabet has one for every byte), so
the stream decodes back to the corpus byte for byte. A stream directory holds tokens.u16 (one id per token), tok_len.u8 (each token's
byte length), tok_off.u64 (each token's first byte, plus one sentinel equal to the corpus length) and layout.json (the byte layout's
fields plus the token counts). TokenStream reads one; span(a, b) is the tokens whose first byte lies in [a, b): the rule that maps the
clean dev byte windows to token spans for the bits/byte bar."""
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "speed-round", "checks"))
from table_common import file_cuts  # noqa: E402

_SURR = re.compile("[\udc80-\udcff]+")


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-to-character map, the one tokenizers' ByteLevel uses: printable bytes map to themselves, the rest to U+0100 on."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def byte_token_ids(tok) -> np.ndarray:
    """The id of each single-byte token, [256] int32 (checked against the tokenizer's alphabet)."""
    from tokenizers import pre_tokenizers
    b2u = bytes_to_unicode()
    if set(b2u.values()) != set(pre_tokenizers.ByteLevel.alphabet()):
        raise ValueError("the tokenizer's byte alphabet is not GPT-2's")
    ids = np.array([tok.token_to_id(b2u[b]) for b in range(256)], dtype=np.int32)
    if (ids < 0).any() or None in ids.tolist():
        raise ValueError("a single-byte token is missing from the vocabulary")
    return ids


def id_bytes(tok) -> list[bytes]:
    """The byte string of every id (a merged token is the concatenation of its bytes)."""
    u2b = {c: b for b, c in bytes_to_unicode().items()}
    return [bytes(u2b[ch] for ch in tok.id_to_token(i)) for i in range(tok.get_vocab_size())]


def id_byte_len(tok) -> np.ndarray:
    """Each id's byte length, [V] uint8 (1..255 asserted)."""
    lens = np.array([len(b) for b in id_bytes(tok)], dtype=np.int64)
    if lens.min() < 1 or lens.max() > 255:
        raise ValueError(f"token byte lengths must lie in 1..255, got {lens.min()}..{lens.max()}")
    return lens.astype(np.uint8)


def split_valid(data: bytes) -> list[tuple[bool, object]]:
    """The file's bytes as a list of (valid, piece): a str of valid UTF-8 (to encode) or a bytes run of invalid bytes (one token each)."""
    s = data.decode("utf-8", errors="surrogateescape")
    out, last = [], 0
    for m in _SURR.finditer(s):
        if m.start() > last:
            out.append((True, s[last:m.start()]))
        out.append((False, m.group().encode("utf-8", errors="surrogateescape")))
        last = m.end()
    if last < len(s):
        out.append((True, s[last:]))
    return out


def encode_files(tok, files: list[bytes], byte_ids: np.ndarray) -> list[np.ndarray]:
    """The ids of each file (uint16 arrays), the valid runs of every file encoded in one encode_batch (the Rust pool does the work)."""
    pieces = [split_valid(f) for f in files]
    strs = [p for ps in pieces for ok, p in ps if ok]
    encs = iter(tok.encode_batch(strs, add_special_tokens=False)) if strs else iter(())
    out = []
    for ps in pieces:
        parts = []
        for ok, p in ps:
            parts.append(np.asarray(next(encs).ids, dtype=np.uint16) if ok else byte_ids[np.frombuffer(p, dtype=np.uint8)].astype(np.uint16))
        out.append(np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16))
    return out


def file_units(corpus: str) -> list[tuple[int, int]]:
    """[(lo, hi)] of every file unit: [0, cut_1), [cut_1, cut_2), ..., [cut_k, N)."""
    n = os.path.getsize(corpus)
    cuts = [c for c in file_cuts(corpus) if 0 < c < n]
    bounds = [0, *cuts, n]
    return [(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TokenStream:
    """A stream directory, memory-mapped read-only."""

    def __init__(self, directory: str):
        self.dir = directory
        self.layout = json.load(open(os.path.join(directory, "layout.json")))
        self.n = int(self.layout["n_tokens"])
        self.tokens = np.memmap(os.path.join(directory, "tokens.u16"), dtype=np.uint16, mode="r", shape=(self.n,))
        self.tok_len = np.memmap(os.path.join(directory, "tok_len.u8"), dtype=np.uint8, mode="r", shape=(self.n,))
        self.tok_off = np.memmap(os.path.join(directory, "tok_off.u64"), dtype=np.uint64, mode="r", shape=(self.n + 1,))
        self.train_end_tok = int(self.layout["train_end_tok"])

    def span(self, a: int, b: int) -> tuple[int, int]:
        """(t0, t1): the tokens whose first byte lies in [a, b)."""
        t0 = int(np.searchsorted(self.tok_off, np.uint64(a), side="left"))
        t1 = int(np.searchsorted(self.tok_off, np.uint64(b), side="left"))
        return t0, min(t1, self.n)

    def bytes_in(self, t0: int, t1: int) -> int:
        return int(np.asarray(self.tok_len[t0:t1], dtype=np.int64).sum())

    def decode(self, t0: int, t1: int, idb: list[bytes]) -> bytes:
        """The bytes of tokens [t0, t1) (idb: id_bytes(tok))."""
        return b"".join(idb[int(i)] for i in self.tokens[t0:t1])


def dev_spans(stream: TokenStream, offset: int, T: int, indices) -> list[tuple[int, int]]:
    """The token spans of the fixed byte windows i (of T bytes from `offset`) in `indices`: the bits/byte bar's windows on tokens."""
    return [stream.span(offset + int(i) * T, offset + (int(i) + 1) * T) for i in indices]


if __name__ == "__main__":  # tok_common.py --check STREAM_DIR TOKENIZER [n]: decode n random file units and compare with the corpus
    if len(sys.argv) >= 4 and sys.argv[1] == "--check":
        from tokenizers import Tokenizer
        st = TokenStream(sys.argv[2]); tok = Tokenizer.from_file(sys.argv[3]); k = int(sys.argv[4]) if len(sys.argv) > 4 else 50
        corpus = st.layout["corpus"]; units = file_units(corpus); idb = id_bytes(tok)
        rng = np.random.default_rng(0); m = np.memmap(corpus, dtype=np.uint8, mode="r")
        bad = 0
        for u in rng.choice(len(units), size=min(k, len(units)), replace=False):
            lo, hi = units[u]; t0, t1 = st.span(lo, hi)
            if st.decode(t0, t1, idb) != m[lo:hi].tobytes():
                bad += 1; print(f"unit {u} [{lo}, {hi}) does not round-trip")
        tl = st.layout
        print(f"{len(units):,} units, {k} checked, {bad} bad; {st.n:,} tokens, {int(tl['corpus_bytes']) / st.n:.3f} bytes/token; "
              f"train {st.train_end_tok:,}, dev {tl['dev_tok'][1] - tl['dev_tok'][0]:,}, test {tl['test_tok'][1] - tl['test_tok'][0]:,} tokens")
        sys.exit(1 if bad else 0)
    print(__doc__)
