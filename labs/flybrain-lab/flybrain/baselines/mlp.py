"""Dense bag MLP sized to another model's trainable-scalar budget."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flybrain.model.mixer import CoincidenceMixer


def _mlp_numel(n_in: int, hidden: int, n_out: int) -> int:
    return n_in * hidden + hidden + hidden * n_out + n_out


class BagMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.n_out = int(n_classes)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def bag_to_set(self, tokens: Tensor) -> Tensor:
        batch, _t = tokens.shape
        x = torch.zeros(batch, self.n_features, device=tokens.device, dtype=self.net[0].weight.dtype)
        clipped = tokens.clamp(0, self.n_features - 1)
        ones = torch.ones(batch, clipped.size(1), device=tokens.device, dtype=x.dtype)
        x.scatter_add_(1, clipped, ones)
        return (x > 0).to(x.dtype)

    def forward(self, tokens: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        logits = self.net(self.bag_to_set(tokens)).unsqueeze(1)
        dummy = tokens.new_zeros(tokens.size(0), 1)
        return logits, dummy

    def persistent_state_scalars(self, batch: int = 1) -> int:
        return 0


def matched_mlp(mixer: CoincidenceMixer) -> BagMLP:
    n_target = sum(p.numel() for p in mixer.parameters() if p.requires_grad)
    n_in, n_out = mixer.n_features, mixer.n_classes
    best_h, best_err = 4, None
    for hidden in range(4, 256):
        err = abs(_mlp_numel(n_in, hidden, n_out) - n_target)
        if best_err is None or err < best_err:
            best_err = err
            best_h = hidden
    return BagMLP(n_in, best_h, n_out)
