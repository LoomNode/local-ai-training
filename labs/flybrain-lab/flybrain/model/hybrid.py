"""GRU + fast-weight memory head. The GRU wins bytes; give it the exact-match memory."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flybrain.model.sparsela import Block, KeyCode

WHERE = ("pre", "post")


class GRUHybrid(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        hidden: int = 330,
        key: str = "sparse",
        where: str = "post",
        n_key: int = 1024,
        top_k: int = 4,
        claws: int = 4,
        theta: float = 1.0,
        learned_key: int = 64,
        normalize: bool = True,
        key_conv: int = 2,
        decay: float = 1.0,
        vocab_size: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if where not in WHERE:
            raise ValueError("where must be pre or post")
        self.d_model, self.hidden, self.where, self.key = int(d_model), int(hidden), where, key
        self.vocab_size = self.n_out = int(vocab_size)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, hidden, batch_first=True)
        self.decoder = nn.Linear(hidden, vocab_size)
        dim = d_model if where == "pre" else hidden
        nk = learned_key if key in ("learned", "softmax") else n_key
        self.mem = Block(dim, KeyCode(key, dim, nk, claws, top_k, theta, seed), decay, d_ff=1, key_conv=key_conv, normalize=normalize)

    def persistent_state_scalars(self, batch: int = 1) -> int:
        return int(batch * self.hidden)

    def flops_per_step(self, batch: int = 1) -> int:
        e, h, v = self.d_model, self.hidden, self.vocab_size
        base = 2 * 3 * (e * h + h * h) + 2 * h * v
        c = self.mem.code; dim = e if self.where == "pre" else h; k = c.active_units()
        key_cost = 2 * c.claws * c.n_key if c.kind == "sparse" else 2 * dim * c.n_key * (2 if c.kind in ("learned", "softmax") else 1)
        return int(batch * (base + key_cost + 2 * dim * dim * 2 + 4 * k * dim + 2 * dim * self.mem.key_conv))

    def forward(self, tokens: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        e = self.embed(tokens)
        if self.where == "pre":
            e = e + self.mem.memory(self.mem.ln1(e))
        out, hstate = self.gru(e, state)
        if self.where == "post":
            out = out + self.mem.memory(self.mem.ln1(out))
        return self.decoder(out), hstate
