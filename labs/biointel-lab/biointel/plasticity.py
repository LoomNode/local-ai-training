"""Inner-loop adapters: update synapse/readout state while the shared program stays frozen."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from biointel.network import SharedRuleNet
from biointel.task import cue_batch, delayed_identity_batch, evaluate_accuracy

ADAPTER_NAMES = frozenset({"synapse_state", "readout_adapter"})


def derangement(n: int, seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    eye = torch.arange(n)
    spins = 0
    while bool((perm == eye).any()) and spins < 32:
        perm = torch.randperm(n, generator=rng)
        spins += 1
    if bool((perm == eye).any()):
        perm = torch.arange(n)
        perm = torch.roll(perm, 1)
    return perm


def program_tensors(net: nn.Module) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for name, param in net.named_parameters():
        if name in ADAPTER_NAMES:
            continue
        out.append(param.detach().clone())
    return out


def _set_adapter_training(net: nn.Module, *, adapters: bool) -> dict[str, bool]:
    previous: dict[str, bool] = {}
    for name, param in net.named_parameters():
        previous[name] = bool(param.requires_grad)
        if name in ADAPTER_NAMES:
            param.requires_grad = adapters
        else:
            param.requires_grad = not adapters
    return previous


def _restore_requires_grad(net: nn.Module, previous: dict[str, bool]) -> None:
    for name, param in net.named_parameters():
        param.requires_grad = previous[name]


@torch.no_grad()
def evaluate_accuracy_perm(
    net: nn.Module,
    perm: torch.Tensor,
    n_classes: int = 4,
    delay: int = 5,
    batches: int = 8,
    batch_size: int = 32,
    seed: int = 1,
) -> float:
    net.eval()
    correct = 0
    total = 0
    for i in range(batches):
        x, y = delayed_identity_batch(batch_size, n_classes, delay, seed=seed + i)
        y = perm[y]
        logits = net.forward_sequence(x)
        correct += int((logits.argmax(dim=-1) == y).sum())
        total += batch_size
    return correct / max(total, 1)


def inner_adapt(
    net: SharedRuleNet,
    *,
    n_classes: int,
    delay: int,
    steps: int,
    perm: torch.Tensor,
    lr: float = 0.05,
    batch_size: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    if net.synapse_state is None or net.readout_adapter is None:
        raise ValueError("inner_adapt requires synapse='stateful'")
    frozen = evaluate_accuracy_perm(
        net, perm, n_classes=n_classes, delay=delay, batches=6, batch_size=batch_size, seed=seed + 50
    )
    before = program_tensors(net)
    previous = _set_adapter_training(net, adapters=True)
    net.train()
    opt = torch.optim.Adam(
        [p for n, p in net.named_parameters() if n in ADAPTER_NAMES],
        lr=lr,
    )
    last_loss = 0.0
    for step in range(steps):
        x, y = delayed_identity_batch(batch_size, n_classes, delay, seed=seed + step)
        y = perm[y]
        logits = net.forward_sequence(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
    _restore_requires_grad(net, previous)
    for name, param in net.named_parameters():
        if name in ADAPTER_NAMES:
            param.requires_grad = False
        else:
            param.requires_grad = True
    after = program_tensors(net)
    program_changed = any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    adapt_acc = evaluate_accuracy_perm(
        net, perm, n_classes=n_classes, delay=delay, batches=6, batch_size=batch_size, seed=seed + 50
    )
    return {
        "shift_frozen_acc": frozen,
        "shift_adapt_acc": adapt_acc,
        "program_changed": program_changed,
        "loss": last_loss,
        "steps": steps,
    }


def reset_adapters(net: nn.Module) -> None:
    synapse = getattr(net, "synapse_state", None)
    adapter = getattr(net, "readout_adapter", None)
    if synapse is not None:
        synapse.fill_(1.0)
    if adapter is not None:
        adapter.zero_()


def _pooled_features(net: nn.Module) -> torch.Tensor:
    state = getattr(net, "state", None)
    if state is None:
        raise ValueError("net has no state after forward_sequence")
    if state.ndim == 3:
        return state.mean(dim=1)
    return state


def local_adapt(
    net: SharedRuleNet,
    *,
    cue: str,
    n_classes: int,
    delay: int,
    steps: int,
    lr: float = 0.05,
    batch_size: int = 32,
    seed: int = 0,
    perm: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Delta-rule adapters. No autograd through the recurrent program."""
    adapter = getattr(net, "readout_adapter", None)
    if adapter is None:
        raise ValueError("local_adapt requires a readout_adapter")
    if perm is None:
        frozen = evaluate_accuracy(
            net, n_classes=n_classes, delay=delay, batches=6, batch_size=batch_size, seed=seed + 50, cue=cue
        )
    else:
        frozen = evaluate_accuracy_perm(
            net, perm, n_classes=n_classes, delay=delay, batches=6, batch_size=batch_size, seed=seed + 50
        )
    before = program_tensors(net)
    last_loss = 0.0
    net.eval()
    for step in range(steps):
        x, y = cue_batch(cue, batch=batch_size, n_classes=n_classes, delay=delay, seed=seed + step)
        if perm is not None:
            y = perm[y]
        with torch.no_grad():
            logits = net.forward_sequence(x)
            hidden = _pooled_features(net)
            target = F.one_hot(y, n_classes).to(dtype=logits.dtype)
            err = torch.softmax(logits, dim=-1) - target
            grad_w = err.T @ hidden / max(x.size(0), 1)
            adapter.add_(-lr * grad_w)
            synapse = getattr(net, "synapse_state", None)
            state = getattr(net, "state", None)
            edge_index = getattr(net, "edge_index", None)
            if synapse is not None and state is not None and edge_index is not None and edge_index.shape[1] > 0:
                src = edge_index[0]
                weight = net.readout.weight + adapter
                delta_h = err @ weight
                pre = state[:, src, :]
                post_err = delta_h[:, None, :] / max(state.size(1), 1)
                ds = (pre * post_err).mean(dim=(0, 2))
                synapse.add_(-lr * ds)
            last_loss = float((-(target * torch.log_softmax(logits, dim=-1)).sum() / max(x.size(0), 1)))
    after = program_tensors(net)
    program_changed = any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    if perm is None:
        adapt_acc = evaluate_accuracy(
            net, n_classes=n_classes, delay=delay, batches=6, batch_size=batch_size, seed=seed + 50, cue=cue
        )
    else:
        adapt_acc = evaluate_accuracy_perm(
            net, perm, n_classes=n_classes, delay=delay, batches=6, batch_size=batch_size, seed=seed + 50
        )
    return {
        "task_frozen_acc": frozen,
        "task_adapt_acc": adapt_acc,
        "program_changed": program_changed,
        "used_backward": False,
        "loss": last_loss,
        "steps": steps,
    }
