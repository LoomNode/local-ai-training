"""Delayed identity: remember a one-hot cue across a silent delay."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def delayed_identity_batch(
    batch: int,
    n_classes: int,
    delay: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return cue_batch("first", batch=batch, n_classes=n_classes, delay=delay, seed=seed)


def last_cue_batch(
    batch: int,
    n_classes: int,
    delay: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return cue_batch("last", batch=batch, n_classes=n_classes, delay=delay, seed=seed)


def cue_batch(
    cue: str,
    *,
    batch: int,
    n_classes: int,
    delay: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    y = torch.randint(0, n_classes, (batch,), generator=rng)
    t_total = delay + 1
    x = torch.zeros(batch, t_total, n_classes)
    if cue == "first":
        t = 0
    elif cue == "last":
        t = t_total - 1
    else:
        raise ValueError(f"unknown cue {cue}")
    x[torch.arange(batch), t, y] = 1.0
    return x, y


@torch.no_grad()
def evaluate_accuracy(
    net: nn.Module,
    n_classes: int = 4,
    delay: int = 5,
    batches: int = 8,
    batch_size: int = 32,
    seed: int = 1,
    cue: str = "first",
) -> float:
    net.eval()
    correct = 0
    total = 0
    for i in range(batches):
        x, y = cue_batch(cue, batch=batch_size, n_classes=n_classes, delay=delay, seed=seed + i)
        logits = net.forward_sequence(x)
        correct += int((logits.argmax(dim=-1) == y).sum())
        total += batch_size
    return correct / max(total, 1)


def train_task(
    net: nn.Module,
    cue: str = "first",
    n_classes: int = 4,
    delay: int = 4,
    steps: int = 80,
    batch_size: int = 32,
    lr: float = 0.05,
    seed: int = 0,
    perm: torch.Tensor | None = None,
) -> dict[str, Any]:
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    last_loss = 0.0
    last_acc = 0.0
    for step in range(steps):
        x, y = cue_batch(cue, batch=batch_size, n_classes=n_classes, delay=delay, seed=seed + step)
        if perm is not None:
            y = perm[y]
        logits = net.forward_sequence(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        last_acc = float((logits.argmax(dim=-1) == y).float().mean())
    return {"loss": last_loss, "train_acc": last_acc, "steps": steps}


def train_delayed_identity(
    net: nn.Module,
    n_classes: int = 4,
    delay: int = 4,
    steps: int = 80,
    batch_size: int = 32,
    lr: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    return train_task(
        net,
        cue="first",
        n_classes=n_classes,
        delay=delay,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )
