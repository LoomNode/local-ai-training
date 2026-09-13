"""Coincidence block + KC→MBON-style fast weights.

Per byte (d = model width, D = expanded code, m = thin read):
    u   = embed(x_t) + U h_{t-1}
    a   = code(u)                 sparse: TopK(ReLU(E u − θ)); dense: ReLU(E_g u), all units
    key = a / ‖a‖
    r   = R a + F key             F: per-sequence fast weights [m, D], starts at 0
    h_t = tanh(u + W r)
    g   = σ(G h_t)                learned modulatory scalar (the "dopamine")
    F  ← λ F + g · (V u) ⊗ key_{t−1}   Hebbian: "key_{t−1} was followed by u_t"

Retrieval F key_t then returns what followed the last time this code appeared.
Gradients flow through the writes (fast-weight-programmer style, BPTT).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F_

from flybrain.model.seqmixer import signed_expand_matrix

CODE_KINDS = ("sparse", "dense")


class FastWeightLM(nn.Module):
    def __init__(
        self,
        d_model: int = 48,
        n_expand: int = 1024,
        claws: int = 4,
        top_k: int = 64,
        n_read: int = 48,
        theta_init: float = 1.0,
        vocab_size: int = 256,
        seed: int = 0,
        code: str = "sparse",
        fast: bool = True,
        decay: float = 0.95,
    ) -> None:
        super().__init__()
        if code not in CODE_KINDS:
            raise ValueError(f"code must be one of {CODE_KINDS}")
        self.d_model, self.n_expand, self.claws = int(d_model), int(n_expand), int(claws)
        self.top_k, self.n_read, self.vocab_size = int(top_k), int(n_read), int(vocab_size)
        self.n_out = int(vocab_size)
        self.code, self.fast, self.decay = code, bool(fast), float(decay)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.recur = nn.Linear(d_model, d_model, bias=False)
        if code == "sparse":
            self.register_buffer("expand", signed_expand_matrix(n_expand, d_model, claws, seed))
            self.theta = nn.Parameter(torch.full((n_expand,), float(theta_init)))
        else:
            g = torch.Generator().manual_seed(seed)
            self.register_buffer("expand", torch.randn(n_expand, d_model, generator=g) / d_model**0.5)
        self.read = nn.Linear(n_expand, n_read)
        self.back = nn.Linear(n_read, d_model)
        if fast:
            self.val = nn.Linear(d_model, n_read, bias=False)
            self.mod = nn.Linear(d_model, 1)
        self.decoder = nn.Linear(d_model, vocab_size)

    # ---- accounting -------------------------------------------------------
    def active_units(self) -> int:
        return min(self.top_k, self.n_expand) if self.code == "sparse" else self.n_expand

    def persistent_state_scalars(self, batch: int = 1) -> int:
        n = self.d_model + (self.n_read * self.n_expand if self.fast else 0)
        return int(batch * n)

    def flops_per_step(self, batch: int = 1) -> int:
        d, D, m, k = self.d_model, self.n_expand, self.n_read, self.active_units()
        per = 2 * d * d + 2 * D  # recur, threshold/relu
        per += 2 * self.claws * D if self.code == "sparse" else 2 * d * D
        per += 2 * k * m  # slow read over live units
        per += 2 * m * d + 2 * d * self.vocab_size
        if self.fast:
            per += 2 * k * m  # F key (read)
            per += 2 * k * m  # outer-product write over live units (decay is a scalar trick)
            per += 2 * d * m + 2 * d  # value, modulator
        return int(batch * per)

    # ---- compute ------------------------------------------------------------
    def encode(self, u: Tensor) -> Tensor:
        z = u.matmul(self.expand.t())
        if self.code == "dense":
            return F_.relu(z)
        c = F_.relu(z - self.theta)
        k = min(self.top_k, self.n_expand)
        if 0 < k < self.n_expand:
            idx = c.topk(k, dim=-1).indices
            c = c * torch.zeros_like(c).scatter_(-1, idx, 1.0)
        return c

    def forward(self, tokens: Tensor, state=None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, seq = tokens.shape
        if state is None:
            h = tokens.new_zeros(batch, self.d_model, dtype=torch.float)
            F = tokens.new_zeros(batch, self.n_read, self.n_expand, dtype=torch.float)
        else:
            h, F = state
        emb = self.embed(tokens)
        key_prev = None
        outs = []
        for t in range(seq):
            u = emb[:, t] + self.recur(h)
            a = self.encode(u)
            r = self.read(a)
            if self.fast:
                key = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
                r = r + torch.bmm(F, key.unsqueeze(-1)).squeeze(-1)
            h = torch.tanh(u + self.back(r))
            if self.fast:
                if key_prev is not None:
                    g = torch.sigmoid(self.mod(h))  # [B,1]
                    v = self.val(u)  # [B,m]
                    F = self.decay * F + (g * v).unsqueeze(-1) * key_prev.unsqueeze(1)
                key_prev = key
            outs.append(h)
        logits = self.decoder(torch.stack(outs, dim=1))
        return logits, (h, F)


def flop_matched_dense_fw(model: FastWeightLM) -> FastWeightLM:
    """Dense (all-active) code with the same fast weights, D chosen for FLOP parity."""
    target = model.flops_per_step(1)
    best, best_err = 1, None
    for D in range(1, model.n_expand + 1):
        cand = FastWeightLM(
            d_model=model.d_model, n_expand=D, n_read=model.n_read, vocab_size=model.vocab_size,
            seed=0, code="dense", fast=model.fast, decay=model.decay,
        )
        err = abs(cand.flops_per_step(1) - target)
        if best_err is None or err < best_err:
            best, best_err = D, err
        if cand.flops_per_step(1) > target * 1.5:
            break
    return FastWeightLM(
        d_model=model.d_model, n_expand=best, n_read=model.n_read, vocab_size=model.vocab_size,
        seed=0, code="dense", fast=model.fast, decay=model.decay,
    )
