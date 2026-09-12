"""Next-byte training loop shared by fly / shuffle / GRU arms."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from flybrain.data.task import delayed_identity_batch
from flybrain.data.text import batch_windows


def _forward_task(model: nn.Module, x: Tensor) -> tuple[Tensor, Tensor]:
    odor = getattr(model, "forward_odor", None)
    if callable(odor) and getattr(model, "inject", "embed") == "local":
        return odor(x)
    return model(x)


def bits_per_byte(cross_entropy_nats: Tensor | float) -> float:
    value = float(cross_entropy_nats)
    return value / math.log(2.0)


def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def eval_bpb(
    model: nn.Module,
    data: Tensor,
    seq_len: int,
    batch: int,
    n_batches: int,
    seed: int,
) -> float:
    model.eval()
    device = _device_of(model)
    losses: list[Tensor] = []
    for i in range(n_batches):
        x, y = batch_windows(data, seq_len=seq_len, batch=batch, seed=seed + i)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)))
    model.train()
    return bits_per_byte(torch.stack(losses).mean())


def train_steps(
    model: nn.Module,
    data: Tensor,
    steps: int,
    seq_len: int,
    batch: int,
    lr: float,
    seed: int,
    val_data: Tensor | None = None,
    log_every: int = 0,
    eval_batches: int = 4,
) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    device = _device_of(model)
    initial: float | None = None
    last = 0.0
    curve: list[dict[str, float | int]] = []
    log_every = log_every if log_every > 0 else steps

    def _record(step: int, train_bpb: float) -> None:
        row: dict[str, float | int] = {"step": step, "train_bpb": train_bpb}
        if val_data is not None:
            row["val_bpb"] = eval_bpb(
                model, val_data, seq_len=seq_len, batch=batch, n_batches=eval_batches, seed=10_000 + step
            )
        curve.append(row)

    for step in range(steps):
        x, y = batch_windows(data, seq_len=seq_len, batch=batch, seed=seed + step)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        train_bpb = bits_per_byte(loss.detach())
        if initial is None:
            initial = float(loss.detach())
            _record(0, train_bpb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if (step + 1) % log_every == 0 or step + 1 == steps:
            _record(step + 1, bits_per_byte(last))
    assert initial is not None
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out: dict = {
        "initial_loss": initial,
        "final_loss": last,
        "initial_bpb": bits_per_byte(initial),
        "final_bpb": bits_per_byte(last),
        "trainable_scalars": int(trainable),
        "steps": steps,
        "curve": curve,
    }
    if val_data is not None:
        out["val_bpb"] = curve[-1]["val_bpb"]
        out["initial_val_bpb"] = curve[0]["val_bpb"]
    return out


@torch.no_grad()
def eval_delay(
    model: nn.Module,
    n_classes: int,
    delay: int,
    batch: int,
    n_batches: int,
    seed: int,
    silence: int = 255,
) -> dict[str, float]:
    model.eval()
    device = _device_of(model)
    losses: list[Tensor] = []
    correct = 0
    total = 0
    for i in range(n_batches):
        x, y = delayed_identity_batch(
            batch, n_classes, delay, seed + i, silence=silence
        )
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        last = logits[:, -1]
        losses.append(F.cross_entropy(last, y))
        correct += int((last.argmax(dim=-1) == y).sum())
        total += int(y.numel())
    model.train()
    return {
        "val_acc": correct / max(total, 1),
        "val_bpb": bits_per_byte(torch.stack(losses).mean()),
    }


def train_delay(
    model: nn.Module,
    n_classes: int,
    delay: int,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    silence: int = 255,
    log_every: int = 0,
    eval_batches: int = 8,
) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    device = _device_of(model)
    initial: float | None = None
    last = 0.0
    last_acc = 0.0
    curve: list[dict[str, float | int]] = []
    log_every = log_every if log_every > 0 else steps

    def _record(step: int, train_bpb: float, train_acc: float) -> None:
        row: dict[str, float | int] = {
            "step": step,
            "train_bpb": train_bpb,
            "train_acc": train_acc,
        }
        held = eval_delay(
            model,
            n_classes=n_classes,
            delay=delay,
            batch=batch,
            n_batches=eval_batches,
            seed=10_000 + step,
            silence=silence,
        )
        row.update(held)
        curve.append(row)

    for step in range(steps):
        x, y = delayed_identity_batch(
            batch, n_classes, delay, seed + step, silence=silence
        )
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        last_logits = logits[:, -1]
        loss = F.cross_entropy(last_logits, y)
        train_bpb = bits_per_byte(loss.detach())
        train_acc = float((last_logits.argmax(dim=-1) == y).float().mean())
        if initial is None:
            initial = float(loss.detach())
            _record(0, train_bpb, train_acc)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())
        last_acc = train_acc
        if (step + 1) % log_every == 0 or step + 1 == steps:
            _record(step + 1, bits_per_byte(last), last_acc)
    assert initial is not None
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    held = curve[-1]
    return {
        "initial_loss": initial,
        "final_loss": last,
        "initial_bpb": bits_per_byte(initial),
        "final_bpb": bits_per_byte(last),
        "train_acc": last_acc,
        "val_acc": float(held["val_acc"]),
        "val_bpb": float(held["val_bpb"]),
        "initial_val_acc": float(curve[0]["val_acc"]),
        "initial_val_bpb": float(curve[0]["val_bpb"]),
        "trainable_scalars": int(trainable),
        "steps": steps,
        "curve": curve,
    }


@torch.no_grad()
def eval_last_step(
    model: nn.Module,
    batch_fn,
    batch: int,
    n_batches: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    device = _device_of(model)
    losses: list[Tensor] = []
    correct = 0
    total = 0
    for i in range(n_batches):
        x, y = batch_fn(batch, seed + i)
        x, y = x.to(device), y.to(device)
        logits, _ = _forward_task(model, x)
        last = logits[:, -1]
        losses.append(F.cross_entropy(last, y))
        correct += int((last.argmax(dim=-1) == y).sum())
        total += int(y.numel())
    model.train()
    return {
        "val_acc": correct / max(total, 1),
        "val_bpb": bits_per_byte(torch.stack(losses).mean()),
    }


def train_last_step(
    model: nn.Module,
    batch_fn,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    log_every: int = 0,
    eval_batches: int = 8,
) -> dict:
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr) if trainable else None
    device = _device_of(model)
    initial: float | None = None
    last = 0.0
    last_acc = 0.0
    curve: list[dict[str, float | int]] = []
    log_every = log_every if log_every > 0 else steps

    def _record(step: int, train_bpb: float, train_acc: float) -> None:
        row: dict[str, float | int] = {
            "step": step,
            "train_bpb": train_bpb,
            "train_acc": train_acc,
        }
        held = eval_last_step(
            model,
            batch_fn,
            batch=batch,
            n_batches=eval_batches,
            seed=10_000 + step,
        )
        row.update(held)
        curve.append(row)

    for step in range(steps):
        x, y = batch_fn(batch, seed + step)
        x, y = x.to(device), y.to(device)
        logits, _ = _forward_task(model, x)
        last_logits = logits[:, -1]
        loss = F.cross_entropy(last_logits, y)
        train_bpb = bits_per_byte(loss.detach())
        train_acc = float((last_logits.argmax(dim=-1) == y).float().mean())
        if initial is None:
            initial = float(loss.detach())
            _record(0, train_bpb, train_acc)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        last = float(loss.detach())
        last_acc = train_acc
        if (step + 1) % log_every == 0 or step + 1 == steps:
            _record(step + 1, bits_per_byte(last), last_acc)
    assert initial is not None
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    held = curve[-1]
    return {
        "initial_loss": initial,
        "final_loss": last,
        "initial_bpb": bits_per_byte(initial),
        "final_bpb": bits_per_byte(last),
        "train_acc": last_acc,
        "val_acc": float(held["val_acc"]),
        "val_bpb": float(held["val_bpb"]),
        "initial_val_acc": float(curve[0]["val_acc"]),
        "initial_val_bpb": float(curve[0]["val_bpb"]),
        "trainable_scalars": int(n_trainable),
        "steps": steps,
        "curve": curve,
    }


@torch.no_grad()
def generate(model: nn.Module, prompt: str, n_bytes: int) -> str:
    device = _device_of(model)
    tokens = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long, device=device).unsqueeze(0)
    logits, state = model(tokens)
    out = tokens[0].tolist()
    for _ in range(n_bytes):
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        nxt_t = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits, state = model(nxt_t, state=state)
    return bytes(out).decode("utf-8", errors="replace")
