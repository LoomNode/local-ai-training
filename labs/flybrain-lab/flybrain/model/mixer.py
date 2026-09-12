"""Coincidence mixer: expand → unique AND → top-k → thin read. No connectome."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from flybrain.data.task import binding_graph_seed


def sparse_expand_matrix(
    n_expand: int,
    n_features: int,
    claws: int,
    seed: int,
) -> Tensor:
    """Each of `n_expand` units has `claws` distinct feature 1s. Frozen 0/1."""
    if claws < 1 or claws > n_features:
        raise ValueError("claws must be in 1..n_features")
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(n_expand, n_features)
    for i in range(n_expand):
        idx = torch.randperm(n_features, generator=rng)[:claws]
        w[i, idx] = 1.0
    return w


class CoincidenceMixer(nn.Module):
    """Bag of feature ids → class logits.

    z_i = how many of unit i's claws are on (set, not bag-with-duplicates).
    h = TopK(ReLU(z − θ)). Thin Linear(D → m → n_classes).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_expand: int = 80,
        claws: int = 4,
        top_k: int = 8,
        n_read: int = 16,
        theta: float = 3.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if n_read < 1:
            raise ValueError("n_read must be >= 1")
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.n_expand = int(n_expand)
        self.claws = int(claws)
        self.top_k = int(top_k)
        self.n_read = int(n_read)
        self.theta = float(theta)
        self.n_out = int(n_classes)
        self.vocab_size = int(n_classes)
        w = sparse_expand_matrix(n_expand, n_features, claws, binding_graph_seed(seed))
        self.register_buffer("expand", w)
        self.read = nn.Linear(n_expand, n_read)
        self.decoder = nn.Linear(n_read, n_classes)

    def bag_to_set(self, tokens: Tensor) -> Tensor:
        batch, _t = tokens.shape
        x = torch.zeros(batch, self.n_features, device=tokens.device, dtype=self.expand.dtype)
        clipped = tokens.clamp(0, self.n_features - 1)
        ones = torch.ones(batch, clipped.size(1), device=tokens.device, dtype=x.dtype)
        x.scatter_add_(1, clipped, ones)
        return (x > 0).to(x.dtype)

    def encode(self, tokens: Tensor) -> Tensor:
        x = self.bag_to_set(tokens)
        z = x.matmul(self.expand.t())
        c = F.relu(z - self.theta)
        k = min(self.top_k, self.n_expand)
        if k <= 0:
            return c
        idx = c.topk(k, dim=1).indices
        keep = torch.zeros_like(c)
        keep.scatter_(1, idx, 1.0)
        return c * keep

    def forward(self, tokens: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        h = self.encode(tokens)
        logits = self.decoder(self.read(h)).unsqueeze(1)
        dummy = tokens.new_zeros(tokens.size(0), 1)
        return logits, dummy

    def persistent_state_scalars(self, batch: int = 1) -> int:
        return int(batch * self.n_read)
