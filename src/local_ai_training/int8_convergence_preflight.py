"""Hard-gate checks for the matched BF16 versus int8 convergence experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .config import ExperimentConfig
from .data import TEXT8_EXPECTED_CHARS, make_batch_schedule
from .model import RatchetGPT, build_seeded_model
from .ratchet import DiscreteRatchetLinear


class PreflightMismatch(RuntimeError):
    """A condition that would confound the matched-arm experiment."""


def assert_configs_matched(bf16: ExperimentConfig, int8: ExperimentConfig) -> str:
    bf16_values = bf16.to_dict()
    int8_values = int8.to_dict()
    differences = sorted(key for key in bf16_values if bf16_values[key] != int8_values[key])
    if differences != ["matmul_mode"]:
        raise PreflightMismatch(
            "configs must differ only at matmul_mode; differing fields: "
            f"{differences or 'none'}"
        )
    if (bf16.matmul_mode, int8.matmul_mode) != ("bf16", "int8"):
        raise PreflightMismatch("matmul_mode arms must be ordered bf16 then int8")
    if bf16.seeds != (1337,):
        raise PreflightMismatch(f"configs must contain only seed 1337, got {bf16.seeds}")
    return differences[0]


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def assert_model_states_matched(bf16: RatchetGPT, int8: RatchetGPT) -> dict[str, Any]:
    bf16_state = bf16.state_dict()
    int8_state = int8.state_dict()
    if bf16_state.keys() != int8_state.keys():
        missing = sorted(bf16_state.keys() ^ int8_state.keys())
        raise PreflightMismatch(f"model state keys differ: {missing}")

    dtype_bytes: Counter[str] = Counter()
    for name, bf16_tensor in bf16_state.items():
        int8_tensor = int8_state[name]
        if bf16_tensor.dtype != int8_tensor.dtype or bf16_tensor.shape != int8_tensor.shape:
            raise PreflightMismatch(f"model state metadata differs at {name}")
        if not torch.equal(bf16_tensor, int8_tensor):
            raise PreflightMismatch(f"model state tensor differs at {name}")
        dtype_bytes[str(bf16_tensor.dtype)] += _tensor_bytes(bf16_tensor)

    return {
        "tensor_count": len(bf16_state),
        "persistent_bytes": sum(dtype_bytes.values()),
        "persistent_bytes_by_dtype": dict(sorted(dtype_bytes.items())),
        "packed_codes": "identical",
        "row_scales": "identical",
        "support_parameters": "identical",
        "positional_state": "identical",
    }


def _schedule_summary(schedule: Tensor) -> dict[str, Any]:
    contiguous = schedule.detach().cpu().contiguous()
    digest = hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()
    return {"shape": list(contiguous.shape), "dtype": str(contiguous.dtype), "sha256": digest}


def assert_schedules_matched(
    train_bf16: Tensor,
    train_int8: Tensor,
    evaluation_bf16: Tensor,
    evaluation_int8: Tensor,
) -> dict[str, Any]:
    for name, left, right in (
        ("training", train_bf16, train_int8),
        ("evaluation", evaluation_bf16, evaluation_int8),
    ):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise PreflightMismatch(f"{name} schedules differ")
    return {
        "training": _schedule_summary(train_bf16),
        "evaluation": _schedule_summary(evaluation_bf16),
    }


def assert_rng_unchanged(name: str, operation: Callable[[], object]) -> dict[str, str]:
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    try:
        operation()
        cpu_after = torch.random.get_rng_state()
        cuda_after = torch.cuda.get_rng_state_all()
    finally:
        torch.random.set_rng_state(cpu_before)
        if cuda_before:
            torch.cuda.set_rng_state_all(cuda_before)
    if not torch.equal(cpu_before, cpu_after):
        raise PreflightMismatch(f"{name} changed CPU RNG state")
    if len(cuda_before) != len(cuda_after) or any(
        not torch.equal(before, after)
        for before, after in zip(cuda_before, cuda_after, strict=True)
    ):
        raise PreflightMismatch(f"{name} changed CUDA RNG state")
    return {"name": name, "cpu": "unchanged", "cuda": "unchanged"}


def _matmul_probe(mode: str, device: torch.device) -> Callable[[], object]:
    reference = torch.linspace(-0.5, 0.5, 32 * 24, dtype=torch.float32).reshape(24, 32)
    layer = DiscreteRatchetLinear(
        32, 24, max_code=4, matmul_mode=mode, initial_weight=reference
    ).to(device)
    layer.eval()
    inputs = torch.linspace(-1.0, 1.0, 7 * 32, device=device).reshape(7, 32)

    def run() -> object:
        with torch.no_grad():
            return layer(inputs)

    return run


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def cuda_uuid_text(value: object | None) -> str | None:
    """Normalize PyTorch's private CUDA UUID wrapper for JSON provenance."""
    return None if value is None else str(value)


def run_preflight(
    *, bf16_config_path: Path, int8_config_path: Path, dataset_path: Path
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise PreflightMismatch("CUDA is required before either experiment arm may run")
    if not dataset_path.is_file() or dataset_path.stat().st_size != TEXT8_EXPECTED_CHARS:
        raise PreflightMismatch(
            f"text8 must be the pinned {TEXT8_EXPECTED_CHARS}-byte corpus: {dataset_path}"
        )

    bf16 = ExperimentConfig.from_toml(bf16_config_path)
    int8 = ExperimentConfig.from_toml(int8_config_path)
    differing_field = assert_configs_matched(bf16, int8)

    # text8 has a fixed 90/10 split in build_char_corpus and a pinned 27-character vocabulary.
    train_length = TEXT8_EXPECTED_CHARS - TEXT8_EXPECTED_CHARS // 10
    validation_length = TEXT8_EXPECTED_CHARS // 10
    bf16_model = build_seeded_model(
        bf16.model_config(vocab_size=27), max_code=4, seed=1337
    )
    int8_model = build_seeded_model(
        int8.model_config(vocab_size=27), max_code=4, seed=1337
    )
    model_summary = assert_model_states_matched(bf16_model, int8_model)

    def schedules(config: ExperimentConfig) -> tuple[Tensor, Tensor]:
        return (
            make_batch_schedule(
                data_length=train_length,
                steps=config.steps,
                batch_size=config.batch_size,
                block_size=config.block_size,
                seed=1337 + 10_000,
            ),
            make_batch_schedule(
                data_length=validation_length,
                steps=config.eval_batches,
                batch_size=config.batch_size,
                block_size=config.block_size,
                seed=1337 + 20_000,
            ),
        )

    bf16_train, bf16_eval = schedules(bf16)
    int8_train, int8_eval = schedules(int8)
    schedule_summary = assert_schedules_matched(
        bf16_train, int8_train, bf16_eval, int8_eval
    )
    device = torch.device("cuda")
    rng_summary = [
        assert_rng_unchanged("bf16_matmul", _matmul_probe("bf16", device)),
        assert_rng_unchanged("int8_matmul", _matmul_probe("int8", device)),
    ]
    return {
        "status": "passed",
        "commit": _git_commit(),
        "seed": 1337,
        "max_code": 4,
        "codes": 9,
        "configs": {
            "bf16": str(bf16_config_path),
            "int8": str(int8_config_path),
            "only_difference": differing_field,
        },
        "dataset": {"path": str(dataset_path), "bytes": dataset_path.stat().st_size},
        "model_state": model_summary,
        "schedules": schedule_summary,
        "rng": rng_summary,
        "cuda": {
            "device_index": torch.cuda.current_device(),
            "name": torch.cuda.get_device_name(),
            "uuid": cuda_uuid_text(getattr(torch.cuda.get_device_properties(0), "uuid", None)),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bf16-config", type=Path, default=Path("configs/int8_convergence_25m_bf16.toml")
    )
    parser.add_argument(
        "--int8-config", type=Path, default=Path("configs/int8_convergence_25m_int8.toml")
    )
    parser.add_argument("--dataset-path", type=Path, default=Path("data/text8/text8"))
    args = parser.parse_args(argv)
    try:
        summary = run_preflight(
            bf16_config_path=args.bf16_config,
            int8_config_path=args.int8_config,
            dataset_path=args.dataset_path,
        )
    except (PreflightMismatch, ValueError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
