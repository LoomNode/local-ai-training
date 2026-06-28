"""Deterministic eager-PyTorch ratchet training loop."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from .checkpoint import load_checkpoint, save_checkpoint
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
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(inputs.to(device), targets.to(device))
        assert loss is not None
        losses.append(float(loss.item()))
    model.train(was_training)
    return sum(losses) / len(losses)


def _cuda_peak(device: torch.device) -> int:
    """Cumulative CUDA allocator high-water mark since the last reset, or 0 on CPU."""
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0


def _reset_cuda_peak(device: torch.device) -> None:
    """Reset the CUDA allocator peak so the next region is measured in isolation."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _metric_row(
    model: RatchetGPT,
    *,
    step: int,
    train_loss: float,
    validation_loss: float,
    tokens_per_second: float,
    update: RatchetUpdateStats,
    cumulative_code_moves: int,
    tokens_seen: int,
    target_tokens: int,
    tokens_per_step: int,
    resolved_steps: int,
    cuda_train_peak_bytes: int = 0,
) -> dict[str, object]:
    # `cuda_train_peak_bytes` is the peak of the completed forward/backward/update
    # step, captured by the caller BEFORE eval and collect_ratchet_metrics run. The
    # observability peak is read here, after collect_ratchet_metrics, so the two are
    # reported as distinct columns and the ~6.9 GiB histogram spike can never be
    # conflated with the training-step peak (it was, via the old cumulative
    # `cuda_memory_bytes` column).
    row: dict[str, object] = {
        "step": step,
        "tokens_seen": tokens_seen,
        "target_tokens": target_tokens,
        "tokens_per_step": tokens_per_step,
        "resolved_steps": resolved_steps,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "perplexity": math.exp(min(validation_loss, 20.0)),
        "tokens_per_second": tokens_per_second,
        "positive_moves": update.positive_moves,
        "negative_moves": update.negative_moves,
        "blocked_positive_moves": update.blocked_positive_moves,
        "blocked_negative_moves": update.blocked_negative_moves,
        "code_moves": update.code_moves,
        "cumulative_code_moves": cumulative_code_moves,
        "move_percent": 100.0 * update.code_moves / max(update.total_weights, 1),
        "gradient_rms_mean": update.gradient_rms_mean,
        "cuda_train_peak_bytes": cuda_train_peak_bytes,
    }
    row.update(collect_ratchet_metrics(model))
    _emb = model.token_embedding
    _emb_device = _emb.weight.device if hasattr(_emb, "weight") else next(_emb.buffers()).device
    row["cuda_observability_peak_bytes"] = _cuda_peak(_emb_device)
    return row


def train_run(
    *,
    corpus: CharCorpus,
    config: ExperimentConfig,
    max_code: int | None,
    seed: int,
    run_dir: str | Path,
    resume_from: str | Path | None = None,
    weight_mode: str = "ratchet",
) -> TrainResult:
    if weight_mode not in {"ratchet", "frozen", "fp32", "qat"}:
        raise ValueError("weight_mode must be ratchet, frozen, fp32, or qat")
    if weight_mode == "fp32" and max_code is not None:
        raise ValueError("fp32 mode requires max_code=None")
    if weight_mode != "fp32" and max_code not in (1, 2, 3, 4, 5, 6, 7):
        raise ValueError("ratchet, frozen, and qat modes require max_code in 1..7")
    checkpoint_code = max_code or 0
    if config.target_tokens is not None:
        tokens_per_step = config.batch_size * config.block_size
        calculated_steps = max(1, config.target_tokens // tokens_per_step)
        config = replace(config, steps=calculated_steps)
    elif config.epochs is not None:
        tokens_per_epoch = corpus.train_ids.numel()
        tokens_per_step = config.batch_size * config.block_size
        calculated_steps = int(config.epochs * tokens_per_epoch / tokens_per_step)
        config = replace(config, steps=calculated_steps)
    tokens_per_step = config.batch_size * config.block_size
    target_tokens = config.target_tokens or (config.steps * tokens_per_step)
    
    if config.eval_tokens is not None:
        calculated_eval_interval = max(1, config.eval_tokens // tokens_per_step)
        config = replace(config, eval_interval=calculated_eval_interval)

    device = resolve_device(config.device)
    if config.matmul_mode == "int8" and device.type != "cuda":
        raise RuntimeError("int8_matmul requires CUDA; the Triton int8 path is GPU-only")
    if config.matmul_mode == "bf16" and device.type != "cuda":
        raise RuntimeError("bf16 matmul requires CUDA; the BF16 comparison path is GPU-only")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    vocab_size = getattr(corpus, "vocab_size", None) or len(corpus.vocabulary)
    model_config = config.model_config(vocab_size=vocab_size)
    if weight_mode == "qat":
        model_config = replace(model_config, qat=True)
    model = build_seeded_model(model_config, max_code=max_code, seed=seed).to(device)
    audit_no_master_weights(model, raise_on_violation=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.support_learning_rate, weight_decay=0.0
    )
    start_step = 0
    if resume_from is not None:
        metadata = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            expected_max_code=checkpoint_code,
            expected_vocabulary=getattr(corpus, "vocabulary", ()),
            expected_matmul_mode=config.matmul_mode,
        )
        start_step = int(metadata["step"])
        if start_step >= config.steps:
            raise ValueError("checkpoint step must be lower than configured training steps")
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
    _reset_cuda_peak(device)
    current_validation = evaluate(
        model,
        corpus.validation_ids,
        validation_schedule,
        block_size=config.block_size,
        device=device,
    )
    warmup_peak = _cuda_peak(device)
    empty_update = RatchetUpdateStats(0, 0, 0, 0, 0, 0.0)
    metrics_path = run_path / "metrics.csv"

    def append_metric_row(row: dict[str, object], *, write_header: bool) -> None:
        # Flush every eval row to disk immediately so progress is observable and a
        # crash mid-run keeps the metrics computed so far instead of losing everything.
        with metrics_path.open("w" if write_header else "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    if resume_from is not None and metrics_path.is_file():
        with metrics_path.open(newline="") as handle:
            rows: list[dict[str, object]] = list(csv.DictReader(handle))
        initial_validation = float(rows[0]["validation_loss"])
        total_moves = int(rows[-1]["cumulative_code_moves"])
    else:
        initial_validation = current_validation
        total_moves = 0
        seed_row = _metric_row(
            model,
            step=start_step,
            train_loss=float("nan"),
            validation_loss=current_validation,
            tokens_per_second=0.0,
            update=empty_update,
            cumulative_code_moves=total_moves,
            tokens_seen=start_step * tokens_per_step,
            target_tokens=target_tokens,
            tokens_per_step=tokens_per_step,
            resolved_steps=config.steps,
            cuda_train_peak_bytes=warmup_peak,
        )
        rows = [seed_row]
        append_metric_row(seed_row, write_header=True)
    last_loss = float("nan")
    # Cumulative move counter lives on-device so the hot path needn't sync it every step;
    # it is read to host only on the eval cadence (and once at the end).
    total_moves_t = torch.tensor(total_moves, device=device, dtype=torch.long)
    interval_started = time.perf_counter()
    interval_tokens = 0
    interval_train_peak = 0
    model.train()
    for step_index, starts in enumerate(
        train_schedule[start_step:], start=start_step + 1
    ):
        inputs, targets = batch_from_starts(
            corpus.train_ids, starts, block_size=config.block_size
        )
        # Reset the allocator peak so this step's forward/backward/update is measured
        # in isolation, uncontaminated by the prior step's eval/observability spike.
        _reset_cuda_peak(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, loss = model(inputs.to(device), targets.to(device))
        assert loss is not None
        if not torch.isfinite(loss):
            model.discard_pending_gradients()
            raise FloatingPointError(f"non-finite training loss at step {step_index}")
        loss.backward()
        # Validate + materialize the ratchet stats only on the eval cadence; the hot path
        # (validate=False) keeps the GPU launch-pipelined (no per-step .item() syncs).
        is_eval_step = step_index % config.eval_interval == 0 or step_index == config.steps
        if weight_mode == "ratchet":
            update = model.ratchet_update(validate=is_eval_step)
        else:
            model.discard_pending_gradients()
            update = RatchetUpdateStats(0, 0, 0, 0, 0, 0.0)
        optimizer.step()
        # Peak of the completed training step, captured before eval/metrics run.
        interval_train_peak = max(interval_train_peak, _cuda_peak(device))
        last_loss = float(loss.item())
        total_moves_t = total_moves_t + update.code_moves  # stays on device, no host sync
        interval_tokens += config.batch_size * config.block_size

        if is_eval_step:
            total_moves = int(total_moves_t.item())
            elapsed = max(time.perf_counter() - interval_started, 1e-12)
            validation_loss = evaluate(
                model,
                corpus.validation_ids,
                validation_schedule,
                block_size=config.block_size,
                device=device,
            )
            row = _metric_row(
                model,
                step=step_index,
                train_loss=last_loss,
                validation_loss=validation_loss,
                tokens_per_second=interval_tokens / elapsed,
                update=update,
                cumulative_code_moves=total_moves,
                tokens_seen=step_index * tokens_per_step,
                target_tokens=target_tokens,
                tokens_per_step=tokens_per_step,
                resolved_steps=config.steps,
                cuda_train_peak_bytes=interval_train_peak,
            )
            rows.append(row)
            append_metric_row(row, write_header=False)
            print(
                f"step {step_index}/{config.steps} "
                f"train={last_loss:.4f} val={validation_loss:.4f} "
                f"tok/s={interval_tokens / elapsed:.0f}",
                flush=True,
            )
            interval_started = time.perf_counter()
            interval_tokens = 0
            interval_train_peak = 0
            _tok = getattr(corpus, "tokenizer", None)
            save_checkpoint(
                run_path / "checkpoint",
                model=model,
                optimizer=optimizer,
                step=step_index,
                max_code=checkpoint_code,
                vocabulary=getattr(corpus, "vocabulary", ()),
                experiment_config={**config.to_dict(), "weight_mode": weight_mode},
                tokenizer_kind=("subword" if _tok is not None else "char"),
                tokenizer_json=(_tok.to_json() if _tok is not None else None),
            )

    _tok = getattr(corpus, "tokenizer", None)
    checkpoint = save_checkpoint(
        run_path / "checkpoint",
        model=model,
        optimizer=optimizer,
        step=config.steps,
        max_code=checkpoint_code,
        vocabulary=getattr(corpus, "vocabulary", ()),
        experiment_config={**config.to_dict(), "weight_mode": weight_mode},
        tokenizer_kind=("subword" if _tok is not None else "char"),
        tokenizer_json=(_tok.to_json() if _tok is not None else None),
    )
    return TrainResult(
        run_dir=run_path,
        metrics_csv=metrics_path,
        checkpoint=checkpoint,
        initial_validation_loss=initial_validation,
        final_validation_loss=float(rows[-1]["validation_loss"]),
        total_code_moves=total_moves,
    )
