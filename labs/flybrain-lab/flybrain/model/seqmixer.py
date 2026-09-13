"""Coincidence block as a recurrent byte LM. No connectome.

Per byte:
    u   = embed(x_t) + U h_{t-1}                      # width d
    z   = E u              (E frozen sparse ±1, `claws` per row, d → D)
    c   = ReLU(z − θ)      (coincidence: several claws must agree)
    a   = TopK(c, k)       (most expanded units stay off)
    h_t = tanh(u + W R a)  (thin read D → m → d, residual)
    logits = decoder(h_t)

The dense control keeps the same skeleton with a GELU FFN in place of the
sparse expand/AND/top-k block, matched on trainable scalars.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def signed_expand_matrix(n_expand: int, d_in: int, claws: int, seed: int) -> Tensor:
    """Frozen 0/±1 matrix: each expanded unit reads `claws` distinct input dims."""
    if claws < 1 or claws > d_in:
        raise ValueError("claws must be in 1..d_in")
    rng = torch.Generator().manual_seed(seed)
    e = torch.zeros(n_expand, d_in)
    for i in range(n_expand):
        idx = torch.randperm(d_in, generator=rng)[:claws]
        sign = torch.randint(0, 2, (claws,), generator=rng).float() * 2.0 - 1.0
        e[i, idx] = sign
    return e


class CoincidenceLM(nn.Module):
    def __init__(
        self,
        d_model: int = 24,
        n_expand: int = 256,
        claws: int = 4,
        top_k: int = 16,
        n_read: int = 24,
        theta_init: float = 1.0,
        vocab_size: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_expand = int(n_expand)
        self.claws = int(claws)
        self.top_k = int(top_k)
        self.n_read = int(n_read)
        self.vocab_size = int(vocab_size)
        self.n_out = int(vocab_size)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.recur = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("expand", signed_expand_matrix(n_expand, d_model, claws, seed))
        self.theta = nn.Parameter(torch.full((n_expand,), float(theta_init)))
        self.read = nn.Linear(n_expand, n_read)
        self.back = nn.Linear(n_read, d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def persistent_state_scalars(self, batch: int = 1) -> int:
        return int(batch * self.d_model)

    def flops_per_step(self, batch: int = 1) -> int:
        d, D, m, k = self.d_model, self.n_expand, self.n_read, self.top_k
        per = (
            2 * d * d  # recur
            + 2 * self.claws * D  # sparse expand (nnz)
            + 2 * D  # threshold
            + 2 * min(k, D) * m  # read only the live units
            + 2 * m * d  # back
            + 2 * d * self.vocab_size  # decoder
        )
        return int(batch * per)

    def block(self, u: Tensor) -> tuple[Tensor, Tensor]:
        z = u.matmul(self.expand.t())
        c = F.relu(z - self.theta)
        k = min(self.top_k, self.n_expand)
        if 0 < k < self.n_expand:
            idx = c.topk(k, dim=-1).indices
            keep = torch.zeros_like(c).scatter_(-1, idx, 1.0)
            c = c * keep
        h = torch.tanh(u + self.back(self.read(c)))
        return h, c

    def forward_with_activation(
        self, tokens: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, seq = tokens.shape
        h = state if state is not None else tokens.new_zeros(batch, self.d_model, dtype=torch.float)
        emb = self.embed(tokens)
        outs = []
        acts = []
        for t in range(seq):
            u = emb[:, t] + self.recur(h)
            h, c = self.block(u)
            outs.append(h)
            acts.append(c)
        hs = torch.stack(outs, dim=1)
        return self.decoder(hs), h, torch.stack(acts, dim=1)

    def forward(self, tokens: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        logits, h, _ = self.forward_with_activation(tokens, state)
        return logits, h


class DenseFFNLM(nn.Module):
    """Same recurrence, dense GELU FFN instead of sparse coincidence."""

    def __init__(self, d_model: int, d_ff: int, vocab_size: int = 256) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.vocab_size = int(vocab_size)
        self.n_out = int(vocab_size)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.recur = nn.Linear(d_model, d_model, bias=False)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def persistent_state_scalars(self, batch: int = 1) -> int:
        return int(batch * self.d_model)

    def flops_per_step(self, batch: int = 1) -> int:
        d, D = self.d_model, self.d_ff
        return int(batch * (2 * d * d + 2 * d * D + 2 * D * d + 2 * d * self.vocab_size))

    def forward(self, tokens: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch, seq = tokens.shape
        h = state if state is not None else tokens.new_zeros(batch, self.d_model, dtype=torch.float)
        emb = self.embed(tokens)
        outs = []
        for t in range(seq):
            u = emb[:, t] + self.recur(h)
            h = torch.tanh(u + self.down(F.gelu(self.up(u))))
            outs.append(h)
        return self.decoder(torch.stack(outs, dim=1)), h


def _dense_numel(d: int, d_ff: int, vocab: int) -> int:
    return vocab * d + d * d + d * d_ff + d_ff + d_ff * d + d + d * vocab + vocab


def matched_dense_ffn(model: CoincidenceLM) -> DenseFFNLM:
    target = sum(p.numel() for p in model.parameters() if p.requires_grad)
    d, vocab = model.d_model, model.vocab_size
    best, best_err = 1, None
    for d_ff in range(1, 4096):
        err = abs(_dense_numel(d, d_ff, vocab) - target)
        if best_err is None or err < best_err:
            best, best_err = d_ff, err
    return DenseFFNLM(d, best, vocab_size=vocab)
