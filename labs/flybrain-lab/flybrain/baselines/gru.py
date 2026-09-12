"""Unique-weight GRU baseline sized to the fly model's trainable-scalar budget."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _gru_numel(embed_dim: int, hidden: int, vocab_size: int) -> int:
    embed = vocab_size * embed_dim
    gru = 3 * (embed_dim * hidden + hidden * hidden + 2 * hidden)
    head = hidden * vocab_size + vocab_size
    return embed + gru + head


class ByteGRU(nn.Module):
    def __init__(self, embed_dim: int, hidden: int, vocab_size: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.encoder = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.decoder = nn.Linear(hidden, vocab_size)

    def persistent_state_scalars(self, batch: int = 1) -> int:
        return int(batch * self.hidden)

    def reset_state(self) -> None:
        return None

    def forward(
        self,
        tokens: Tensor,
        state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        x = self.encoder(tokens)
        out, hidden = self.gru(x, state)
        return self.decoder(out), hidden


def matched_gru(fly: nn.Module, vocab_size: int | None = None) -> ByteGRU:
    if vocab_size is None:
        vocab_size = int(getattr(fly, "n_out", None) or getattr(fly, "vocab_size", 256))
    n_fly = sum(p.numel() for p in fly.parameters() if p.requires_grad)
    best: tuple[int, int] | None = None
    best_err: int | None = None
    for embed_dim in range(4, 48):
        for hidden in range(4, 96):
            n = _gru_numel(embed_dim, hidden, vocab_size)
            err = abs(n - n_fly)
            if best_err is None or err < best_err:
                best_err = err
                best = (embed_dim, hidden)
    assert best is not None
    return ByteGRU(best[0], best[1], vocab_size=vocab_size)
