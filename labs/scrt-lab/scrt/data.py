"""Datasets for SCRT experiments.

v1 synonym language was too easy — all arms saturated at the same loss.
v2 adds:
  - structured synthetic language (longer-range dependencies)
  - character-level real text (WikiText-2 / tinyshakespeare) when available
"""

from __future__ import annotations

import math
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class SynthCorpus:
    """1D token stream + optional metadata."""

    data: torch.Tensor  # 1D long
    vocab_size: int
    synonym_groups: list[list[int]]
    name: str = "synonym"
    # For char-level: stoi/itos
    stoi: Optional[dict[str, int]] = None
    itos: Optional[dict[int, str]] = None


def build_synonym_language(
    vocab_size: int = 48,
    n_concepts: int = 8,
    forms_per_concept: int = 3,
    length: int = 50_000,
    seed: int = 0,
) -> SynthCorpus:
    """Original easy language (kept for regression tests)."""
    g = torch.Generator().manual_seed(seed)
    n_reserved = 2
    budget = vocab_size - n_reserved
    need = n_concepts * forms_per_concept
    if need > budget:
        n_concepts = max(2, budget // forms_per_concept)
        need = n_concepts * forms_per_concept
    if need > budget:
        raise ValueError(f"need {need} surface tokens, only {budget} free")

    synonym_groups: list[list[int]] = []
    tok = n_reserved
    for _ in range(n_concepts):
        group = list(range(tok, tok + forms_per_concept))
        synonym_groups.append(group)
        tok += forms_per_concept

    trans = torch.rand(n_concepts, n_concepts, generator=g)
    for i in range(n_concepts):
        trans[i, i] += 2.0
        trans[i, (i + 1) % n_concepts] += 1.5
    trans = trans / trans.sum(dim=1, keepdim=True)

    stream = torch.empty(length, dtype=torch.long)
    concept = int(torch.randint(0, n_concepts, (1,), generator=g).item())
    for t in range(length):
        forms = synonym_groups[concept]
        form_i = int(torch.randint(0, len(forms), (1,), generator=g).item())
        stream[t] = forms[form_i]
        concept = int(torch.multinomial(trans[concept], 1, generator=g).item())

    return SynthCorpus(
        data=stream,
        vocab_size=vocab_size,
        synonym_groups=synonym_groups,
        name="synonym",
    )


def build_structured_language(
    vocab_size: int = 96,
    length: int = 500_000,
    seed: int = 0,
    copy_span: int = 8,
) -> SynthCorpus:
    """Harder synthetic: nested brackets + delayed copy + synonym concepts.

    Forces the model to keep state across depth (good for looped/SCRT depth)
    rather than only local bigrams.
    """
    g = torch.Generator().manual_seed(seed)
    # Token layout:
    # 0 PAD, 1 BOS, 2-5 brackets, 6-21 open/close tags, rest concept forms
    assert vocab_size >= 48
    OPEN_B, CLOSE_B = 2, 3
    OPEN_P, CLOSE_P = 4, 5
    # copy marker then payload tokens
    COPY_MARK = 6
    n_concepts = 12
    forms_per = 3
    base = 20
    synonym_groups: list[list[int]] = []
    tok = base
    for _ in range(n_concepts):
        grp = list(range(tok, tok + forms_per))
        if grp[-1] >= vocab_size:
            break
        synonym_groups.append(grp)
        tok += forms_per
    n_concepts = len(synonym_groups)

    stream: list[int] = []
    while len(stream) < length:
        kind = int(torch.randint(0, 3, (1,), generator=g).item())
        if kind == 0 and n_concepts > 0:
            # concept chain with mild structure
            c = int(torch.randint(0, n_concepts, (1,), generator=g).item())
            for _ in range(int(torch.randint(3, 8, (1,), generator=g).item())):
                form = synonym_groups[c][
                    int(torch.randint(0, len(synonym_groups[c]), (1,), generator=g).item())
                ]
                stream.append(form)
                # prefer next concept
                c = (c + 1) % n_concepts if torch.rand(1, generator=g).item() < 0.7 else c
        elif kind == 1:
            # nested brackets: ( [ tok* ] )
            depth = int(torch.randint(1, 4, (1,), generator=g).item())
            opens = []
            for d in range(depth):
                o = OPEN_B if d % 2 == 0 else OPEN_P
                stream.append(o)
                opens.append(CLOSE_B if o == OPEN_B else CLOSE_P)
            # payload
            for _ in range(int(torch.randint(2, 6, (1,), generator=g).item())):
                if n_concepts:
                    c = int(torch.randint(0, n_concepts, (1,), generator=g).item())
                    stream.append(
                        synonym_groups[c][
                            int(
                                torch.randint(
                                    0, len(synonym_groups[c]), (1,), generator=g
                                ).item()
                            )
                        ]
                    )
            for cl in reversed(opens):
                stream.append(cl)
        else:
            # delayed copy: COPY_MARK a b c ... then same sequence again
            n = int(torch.randint(3, copy_span + 1, (1,), generator=g).item())
            payload = [
                int(torch.randint(base, min(vocab_size, base + n_concepts * forms_per), (1,), generator=g).item())
                for _ in range(n)
            ]
            stream.append(COPY_MARK)
            stream.extend(payload)
            stream.extend(payload)

    data = torch.tensor(stream[:length], dtype=torch.long)
    return SynthCorpus(
        data=data,
        vocab_size=vocab_size,
        synonym_groups=synonym_groups,
        name="structured",
    )


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def ensure_tinyshakespeare(path: Optional[Path] = None) -> Path:
    """Download Karpathy tinyshakespeare if missing (~1MB)."""
    path = path or (_data_dir() / "tinyshakespeare.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        return path
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, path)
    return path


def ensure_wikitext2_train(path: Optional[Path] = None) -> Path:
    """Download WikiText-2 raw train if missing."""
    root = _data_dir()
    root.mkdir(parents=True, exist_ok=True)
    out = path or (root / "wiki.train.raw")
    if out.exists() and out.stat().st_size > 10_000:
        return out
    zip_path = root / "wikitext-2-raw-v1.zip"
    url = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # extract train
        name = "wikitext-2-raw/wiki.train.raw"
        with zf.open(name) as src, open(out, "wb") as dst:
            dst.write(src.read())
    return out


def build_char_corpus(
    source: str = "shakespeare",
    max_chars: int = 1_000_000,
    seed: int = 0,
    vocab_cap: Optional[int] = None,
) -> SynthCorpus:
    """Character-level LM corpus. ``source``: shakespeare | wikitext2.

    WikiText-2 raw has 1013 distinct characters, nearly all of them rare unicode.
    ``vocab_cap`` keeps the ``cap - 1`` most frequent chars and folds the tail into
    a single UNK id (``cap - 1``), so ``vocab_size`` is exactly ``cap`` regardless
    of which slice of text is loaded. That last property matters: the matched-budget
    planner probes a small slice to learn the vocab before building any model, and
    an uncapped vocab would disagree with the full run.
    """
    if source == "shakespeare":
        path = ensure_tinyshakespeare()
    elif source == "wikitext2":
        path = ensure_wikitext2_train()
    else:
        raise ValueError(f"unknown source: {source}")

    full = path.read_text(encoding="utf-8", errors="ignore")

    if vocab_cap is not None:
        if vocab_cap < 2:
            raise ValueError("vocab_cap must be >= 2 (one real char + UNK)")
        from collections import Counter

        # Rank over the *whole* file, never the slice: the budget planner probes a
        # 50k slice to learn vocab_size before any model exists, and the run then
        # loads the full text. Ranking on the slice would give those two the same
        # size but different char->id maps.
        ranked = sorted(Counter(full).items(), key=lambda kv: (-kv[1], kv[0]))
        keep = [ch for ch, _ in ranked[: vocab_cap - 1]]
        stoi = {ch: i for i, ch in enumerate(sorted(keep))}
        vocab_size = vocab_cap

    text = full
    if len(text) > max_chars:
        # deterministic slice
        g = torch.Generator().manual_seed(seed)
        start = int(torch.randint(0, len(text) - max_chars, (1,), generator=g).item())
        text = text[start : start + max_chars]

    if vocab_cap is None:
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        vocab_size = len(chars)
    unk = vocab_size - 1 if vocab_cap is not None else None
    itos = {i: ch for ch, i in stoi.items()}
    if unk is not None:
        itos[unk] = "�"
        data = torch.tensor([stoi.get(c, unk) for c in text], dtype=torch.long)
    else:
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return SynthCorpus(
        data=data,
        vocab_size=vocab_size,
        synonym_groups=[],
        name=f"char_{source}",
        stoi=stoi,
        itos=itos,
    )


def build_corpus(
    kind: str = "structured",
    vocab_size: int = 96,
    length: int = 500_000,
    seed: int = 0,
    max_chars: int = 1_000_000,
    vocab_cap: Optional[int] = None,
) -> SynthCorpus:
    """Factory: synonym | structured | shakespeare | wikitext2."""
    if kind == "synonym":
        return build_synonym_language(
            vocab_size=vocab_size, length=length, seed=seed
        )
    if kind == "structured":
        return build_structured_language(
            vocab_size=vocab_size, length=length, seed=seed
        )
    if kind in ("shakespeare", "wikitext2"):
        return build_char_corpus(
            source=kind, max_chars=max_chars, seed=seed, vocab_cap=vocab_cap
        )
    raise ValueError(f"unknown corpus kind: {kind}")


class TokenBatcher:
    def __init__(
        self,
        data: torch.Tensor,
        seq_len: int,
        batch_size: int,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.g = torch.Generator().manual_seed(seed)
        self.n = data.numel() - seq_len - 1
        if self.n <= 0:
            raise ValueError("corpus too short for seq_len")

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(0, self.n, (self.batch_size,), generator=self.g)
        x = torch.stack([self.data[i : i + self.seq_len] for i in ix.tolist()])
        y = torch.stack(
            [self.data[i + 1 : i + 1 + self.seq_len] for i in ix.tolist()]
        )
        return x.to(self.device), y.to(self.device)
