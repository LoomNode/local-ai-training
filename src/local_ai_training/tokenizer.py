"""Byte-level BPE tokenizer — stdlib only (re, json, collections).

Byte vocabulary: ids 0..255 map to bytes([i]) decoded as latin-1.
Merges are learned from word-frequency-deduped pretokens so training
is tractable on large corpora. Merges never cross pretoken boundaries.
Ties in pair frequency are broken deterministically (lexicographic).
All str<->bytes conversions use latin-1 so the round-trip is lossless
for arbitrary byte sequences.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Pretokenisation
# ---------------------------------------------------------------------------
# Split on whitespace-runs; keep the whitespace attached to the *following*
# word (GPT-2 style) so that a leading space is part of the token and merges
# cannot cross the boundary between a trailing char of one word and the
# leading space of the next.
_PRETOKEN_RE = re.compile(r"\S+|\s+")


def _pretokenise(text: str) -> list[str]:
    """Split *text* into pretokens (words / whitespace chunks)."""
    return _PRETOKEN_RE.findall(text)


def _word_freqs(text: str) -> dict[tuple[int, ...], int]:
    """Return {byte-tuple-pretoken: frequency} with deduplication."""
    freq: Counter[tuple[int, ...]] = Counter()
    for tok in _pretokenise(text):
        key = tuple(tok.encode("latin-1"))
        freq[key] += 1
    return dict(freq)


# ---------------------------------------------------------------------------
# Pair statistics (over word-frequency dict)
# ---------------------------------------------------------------------------

def _get_pair_counts(
    vocab: dict[tuple[int, ...], int]
) -> dict[tuple[int, int], int]:
    counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for word, freq in vocab.items():
        for a, b in zip(word, word[1:], strict=False):
            counts[(a, b)] += freq
    return dict(counts)


def _merge_vocab(
    vocab: dict[tuple[int, ...], int],
    pair: tuple[int, int],
    new_id: int,
) -> dict[tuple[int, ...], int]:
    """Return new vocab with every occurrence of *pair* replaced by *new_id*."""
    a, b = pair
    out: dict[tuple[int, ...], int] = {}
    for word, freq in vocab.items():
        new_word: list[int] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        out[tuple(new_word)] = freq
    return out


# ---------------------------------------------------------------------------
# BpeTokenizer
# ---------------------------------------------------------------------------

class BpeTokenizer:
    """Minimal byte-level BPE tokenizer with no third-party dependencies."""

    # Each merge is stored as (token_a_id, token_b_id) -> merged_id.
    # The token strings (latin-1 decoded bytes) are stored in _id_to_str.

    def __init__(
        self,
        merges: list[tuple[int, int]],
        id_to_str: list[str],
        target_vocab_size: int | None = None,
    ) -> None:
        self._merges: list[tuple[int, int]] = merges
        # Pad id_to_str to target_vocab_size if provided (corpus may be too
        # small to learn every merge; unused ids won't appear in encoded output)
        if target_vocab_size is not None and len(id_to_str) < target_vocab_size:
            id_to_str = list(id_to_str) + [""] * (target_vocab_size - len(id_to_str))
        self._id_to_str: list[str] = id_to_str
        # Build fast lookup: pair -> merged token id (in merge order)
        self._merge_map: dict[tuple[int, int], int] = {}
        # Base vocab is 256; merged tokens start at 256
        for idx, pair in enumerate(merges):
            self._merge_map[pair] = 256 + idx

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_str)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> BpeTokenizer:
        """Train a BPE tokenizer on *text* to *vocab_size* tokens."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (byte base vocab)")

        # Initialise id->str for the 256 byte tokens
        id_to_str: list[str] = [bytes([i]).decode("latin-1") for i in range(256)]

        num_merges = vocab_size - 256
        merges: list[tuple[int, int]] = []

        # Word-frequency table (deduped for tractability)
        wf = _word_freqs(text)

        for _ in range(num_merges):
            counts = _get_pair_counts(wf)
            if not counts:
                break
            # Deterministic tie-break: highest count, then lex smallest pair
            best_pair = max(
                counts,
                key=lambda p: (counts[p], -p[0], -p[1]),
            )
            if counts[best_pair] < 1:
                break
            new_id = len(id_to_str)
            new_str = id_to_str[best_pair[0]] + id_to_str[best_pair[1]]
            id_to_str.append(new_str)
            merges.append(best_pair)
            wf = _merge_vocab(wf, best_pair, new_id)

        return cls(merges, id_to_str, target_vocab_size=vocab_size)

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of token ids."""
        ids: list[int] = []
        for pretoken in _pretokenise(text):
            token_ids = list(pretoken.encode("latin-1"))  # list[int] 0..255
            # Apply merges in order
            token_ids = self._apply_merges(token_ids)
            ids.extend(token_ids)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to a string."""
        return "".join(self._id_to_str[i] for i in ids)

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(
            {
                "merges": self._merges,
                "id_to_str": self._id_to_str,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, data: str) -> BpeTokenizer:
        """Deserialise from a JSON string produced by :meth:`to_json`.

        The serialised ``id_to_str`` already has the correct length
        (including any padding added during :meth:`train`), so no
        ``target_vocab_size`` is needed here.
        """
        obj = json.loads(data)
        merges = [tuple(p) for p in obj["merges"]]
        return cls(merges, obj["id_to_str"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_merges(self, ids: list[int]) -> list[int]:
        """Greedily apply learned merges (earliest merge wins)."""
        if len(ids) < 2:
            return ids
        # Iterate over merge_map in priority order (insertion order == merge order)
        # We do a classic "scan and replace" loop until no more merges apply.
        changed = True
        while changed and len(ids) >= 2:
            changed = False
            new_ids: list[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1:
                    pair = (ids[i], ids[i + 1])
                    if pair in self._merge_map:
                        new_ids.append(self._merge_map[pair])
                        i += 2
                        changed = True
                        continue
                new_ids.append(ids[i])
                i += 1
            ids = new_ids
        return ids
