"""Deterministic eager-PyTorch ratchet training loop."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .checkpoint import save_checkpoint
from .config import ExperimentConfig
from .data import CharCorpus, batch_from_starts, make_batch_schedule
from .metrics import collect_ratchet_metrics
from .model import RatchetGPT, build_seeded_model
from .ratchet import RatchetUpdateStats, audit_no_master_weights


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    metrics_csv: Path
    checkpoint: Path
    initial_validation_loss: float
    final_validation_loss: float
    total_code_moves: int


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


@torch.no_grad()
def evaluate(
    model: RatchetGPT,
    data: Tensor,
    schedule: Tensor,
    *,
    block_size: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for starts in schedule:
        inputs, targets = batch_from_starts(data, starts, block_size=block_size)
        _, loss = model(inputs.to(device), targets.to(device))
        assert loss is not None
        losses.append(float(loss.item()))
    model.train(was_training)
    return sum(losses) / len(losses)


def _metric_row(
    model: RatchetGPT,
    *,
    step: int,
    train_loss: float,
    validation_loss: float,
    tokens_per_second: float,
    update: RatchetUpdateStats,
) -> dict[str, object]:
    row: dict[str, object] = {
        "step": step,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "perplexity": math.exp(min(validation_loss, 20.0)),
        "tokens_per_second": tokens_per_second,
        "positive_moves": update.positive_moves,
        "negative_moves": update.negative_moves,
        "blocked_positive_moves": update.blocked_positive_moves,
        "blocked_negative_moves": update.blocked_negative_moves,
        "code_moves": update.code_moves,
        "move_percent": 100.0 * update.code_moves / max(update.total_weights, 1),
        "gradient_rms_mean": update.gradient_rms_mean,
        "cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(model.token_embedding.weight.device)
            if model.token_embedding.weight.is_cuda
            else 0
        ),
    }
    row.update(collect_ratchet_metrics(model))
    return row


def train_run(
    *,
    corpus: CharCorpus,
    config: ExperimentConfig,
    max_code: int,
    seed: int,
    run_dir: str | Path,
) -> TrainResult:
    if max_code not in (2, 3):
        raise ValueError("max_code must be 2 or 3")
    device = resolve_device(config.device)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    model = build_seeded_model(
        config.model_config(vocab_size=len(corpus.vocabulary)), max_code=max_code, seed=seed
    ).to(device)
    audit_no_master_weights(model, raise_on_violation=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.support_learning_rate, weight_decay=0.0
    )
    train_schedule = make_batch_schedule(
        data_length=corpus.train_ids.numel(),
        steps=config.steps,
        batch_size=config.batch_size,
        block_size=config.block_size,
        seed=seed + 10_000,
    )
    validation_schedule = make_batch_schedule(
        data_length=corpus.validation_ids.numel(),
        steps=config.eval_batches,
        batch_size=config.batch_size,
        block_size=config.block_size,
        seed=seed + 20_000,
    )
    initial_validation = evaluate(
        model,
        corpus.validation_ids,
        validation_schedule,
        block_size=config.block_size,
        device=device,
    )
    empty_update = RatchetUpdateStats(0, 0, 0, 0, 0, 0.0)
    rows = [
        _metric_row(
            model,
            step=0,
            train_loss=float("nan"),
            validation_loss=initial_validation,
            tokens_per_second=0.0,
            update=empty_update,
        )
    ]
    total_moves = 0
    last_loss = float("nan")
    interval_started = time.perf_counter()
    interval_tokens = 0
    model.train()
    for step_index, starts in enumerate(train_schedule, start=1):
        inputs, targets = batch_from_starts(
            corpus.train_ids, starts, block_size=config.block_size
        )
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs.to(device), targets.to(device))
        assert loss is not None
        if not torch.isfinite(loss):
            model.discard_pending_gradients()
            raise FloatingPointError(f"non-finite training loss at step {step_index}")
        loss.backward()
        update = model.ratchet_update()
        optimizer.step()
        last_loss = float(loss.item())
        total_moves += update.code_moves
        interval_tokens += config.batch_size * config.block_size

        if step_index % config.eval_interval == 0 or step_index == config.steps:
            elapsed = max(time.perf_counter() - interval_started, 1e-12)
            validation_loss = evaluate(
                model,
                corpus.validation_ids,
                validation_schedule,
                block_size=config.block_size,
                device=device,
            )
            rows.append(
                _metric_row(
                    model,
                    step=step_index,
                    train_loss=last_loss,
                    validation_loss=validation_loss,
                    tokens_per_second=interval_tokens / elapsed,
                    update=update,
                )
            )
            interval_started = time.perf_counter()
            interval_tokens = 0

    metrics_path = run_path / "metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint = save_checkpoint(
        run_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=config.steps,
        max_code=max_code,
        vocabulary=corpus.vocabulary,
        experiment_config=config.to_dict(),
    )
    return TrainResult(
        run_dir=run_path,
        metrics_csv=metrics_path,
        checkpoint=checkpoint,
        initial_validation_loss=initial_validation,
        final_validation_loss=float(rows[-1]["validation_loss"]),
        total_code_moves=total_moves,
    )

