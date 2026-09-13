"""Delayed identity: show a class token, wait τ silent steps, emit the class."""

from __future__ import annotations

import torch
from torch import Tensor


def delayed_identity_batch(
    batch: int,
    n_classes: int,
    delay: int,
    seed: int,
    *,
    silence: int = 255,
    vocab_size: int = 256,
) -> tuple[Tensor, Tensor]:
    """Return tokens `[cue, silence × delay]` and the cue class ids.

    `delay=0` is immediate copy (I/O probe). Cue tokens are `0 .. n_classes-1`.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if delay < 0:
        raise ValueError("delay must be >= 0")
    if not 0 <= silence < vocab_size:
        raise ValueError("silence token out of vocab")
    if silence < n_classes:
        raise ValueError("silence token collides with cue classes")
    rng = torch.Generator().manual_seed(seed)
    y = torch.randint(0, n_classes, (batch,), generator=rng)
    t_total = delay + 1
    x = torch.full((batch, t_total), silence, dtype=torch.long)
    x[:, 0] = y
    return x, y


# Binding task seed and connectome seed must not share a Generator stream:
# both start with randperm(n_features)[:k], so seed=s for patterns and seed=s
# for unique claws plants Kenyon[i] == pattern[i].
BINDING_GRAPH_SEED_OFFSET = 10_007


def binding_graph_seed(task_seed: int) -> int:
    return int(task_seed) + BINDING_GRAPH_SEED_OFFSET


def make_sparse_patterns(
    n_classes: int,
    n_features: int,
    k: int,
    seed: int,
) -> Tensor:
    """Unique k-hot patterns. Overlap is required so class = conjunction."""
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if k < 2:
        raise ValueError("k must be >= 2 for conjunction")
    if n_features < k:
        raise ValueError("n_features must be >= k")
    rng = torch.Generator().manual_seed(seed)
    seen: set[tuple[int, ...]] = set()
    rows: list[tuple[int, ...]] = []
    attempts = 0
    while len(rows) < n_classes:
        attempts += 1
        if attempts > 20_000:
            raise ValueError("could not sample unique sparse patterns")
        pat = tuple(sorted(int(i) for i in torch.randperm(n_features, generator=rng)[:k]))
        if pat in seen:
            continue
        seen.add(pat)
        rows.append(pat)
    return torch.tensor(rows, dtype=torch.long)


def sparse_binding_batch(
    batch: int,
    patterns: Tensor,
    seed: int,
    *,
    n_features: int | None = None,
    n_distractors: int = 0,
) -> tuple[Tensor, Tensor]:
    """Present a class's features in random order; label is the class id.

    `n_distractors` extra feature tokens (not in the class set) are mixed in
    so a bag-of-tokens linear head cannot ignore conjunction.
    """
    if patterns.ndim != 2:
        raise ValueError("patterns must be [n_classes, k]")
    n_classes, k = int(patterns.shape[0]), int(patterns.shape[1])
    n_features = int(patterns.max().item()) + 1 if n_features is None else int(n_features)
    if n_distractors < 0:
        raise ValueError("n_distractors must be >= 0")
    if n_distractors > n_features - k:
        raise ValueError("n_distractors exceeds unused features")
    rng = torch.Generator().manual_seed(seed)
    y = torch.randint(0, n_classes, (batch,), generator=rng)
    feats = patterns[y]
    if n_distractors == 0:
        perm = torch.rand(batch, k, generator=rng).argsort(dim=1)
        return torch.gather(feats, 1, perm), y
    extras = torch.empty(batch, n_distractors, dtype=torch.long)
    all_ids = torch.arange(n_features)
    for i in range(batch):
        true_set = set(int(t) for t in feats[i].tolist())
        pool = torch.tensor([int(t) for t in all_ids.tolist() if int(t) not in true_set])
        pick = torch.randperm(pool.numel(), generator=rng)[:n_distractors]
        extras[i] = pool[pick]
    bag = torch.cat([feats, extras], dim=1)
    perm = torch.rand(batch, k + n_distractors, generator=rng).argsort(dim=1)
    return torch.gather(bag, 1, perm), y


def copy_batch(
    batch: int,
    seq_len: int = 64,
    span: int = 16,
    seed: int = 0,
    *,
    vocab_size: int = 256,
) -> tuple[Tensor, Tensor, Tensor]:
    """Uniform random bytes with one random `span` repeated later in the window.

    Returns (x, y, mask): y is x shifted by one; mask marks the positions whose
    target is byte 2..span of the second occurrence (the only predictable bytes).
    Unigram floor on masked positions is log2(vocab); a perfect copy scores 0.
    """
    if seq_len < 2 * span + 2:
        raise ValueError("seq_len too short for two spans")
    rng = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab_size, (batch, seq_len + 1), generator=rng)
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    for b in range(batch):
        p1 = int(torch.randint(0, seq_len // 2 - span + 1, (1,), generator=rng))
        p2 = int(torch.randint(p1 + span, seq_len - span + 1, (1,), generator=rng))
        x[b, p2 : p2 + span] = x[b, p1 : p1 + span]
        mask[b, p2 : p2 + span - 1] = True
    return x[:, :-1], x[:, 1:], mask
