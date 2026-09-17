"""Restricted tensor-only checkpoints with validated JSON metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from .ratchet import DiscreteRatchetLinear

FORMAT_VERSION = 2
_SUPPORTED_FORMAT_VERSIONS = {1, FORMAT_VERSION}
_RESUME_CONFIG_DEFAULTS: dict[str, Any] = {
    "matmul_mode": "fp32",
    "dropout": 0.0,
    "int8_backward": False,
    "gradient_checkpointing": False,
    "deterministic_attention": False,
    "pressure_leak_period": 0,
    "rms_ema_beta": 0.0,
    "stochastic_bucket": False,
    "scale_multiplier": 1.0,
    "pressure_weight": 0.0,
    "weight_mode": "ratchet",
    "int8_lr": 0.1,
    "int8_lr_final": 0.0,
}


def _paths(base_path: str | Path) -> tuple[Path, Path, Path]:
    base = Path(base_path)
    if base.suffix in {".safetensors", ".json"}:
        base = base.with_suffix("")
    return base, base.with_suffix(".safetensors"), base.with_suffix(".json")


def _canonical_tokenizer_json(tokenizer_json: str) -> str:
    try:
        value = json.loads(tokenizer_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("tokenizer_json must be valid JSON") from error
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _model_device(model: nn.Module) -> torch.device:
    tensor = next(iter(model.parameters()), None)
    if tensor is None:
        tensor = next(iter(model.buffers()), None)
    return tensor.device if tensor is not None else torch.device("cpu")


def _ratchet_update_counts(model: nn.Module) -> dict[str, int]:
    return {
        name: int(module._update_count)
        for name, module in model.named_modules()
        if isinstance(module, DiscreteRatchetLinear)
    }


def restore_checkpoint_rng(
    base_path: str | Path, *, training_device: str | torch.device = "cpu"
) -> None:
    """Restore RNG streams from a checkpoint that has already passed validation."""
    _, tensor_path, _ = _paths(base_path)
    device = torch.device(training_device)
    with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
        if "rng::cpu" in keys:
            torch.set_rng_state(tensors.get_tensor("rng::cpu"))
        if device.type == "cuda" and "rng::cuda" in keys:
            torch.cuda.set_rng_state(tensors.get_tensor("rng::cuda"), device=device)


def save_checkpoint(
    base_path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    max_code: int,
    vocabulary: tuple[str, ...],
    experiment_config: dict[str, Any],
    tokenizer_kind: str = "char",
    tokenizer_json: str | None = None,
    run_seed: int | None = None,
) -> Path:
    base, tensor_path, metadata_path = _paths(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Tensor] = {
        f"model::{name}": tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    named_parameters = dict(model.named_parameters())
    for parameter_name, parameter in named_parameters.items():
        for state_name, state_value in optimizer.state.get(parameter, {}).items():
            if not isinstance(state_value, Tensor):
                state_value = torch.tensor(state_value)
            tensors[f"optimizer::{parameter_name}::{state_name}"] = (
                state_value.detach().cpu().contiguous()
            )
    tensors["rng::cpu"] = torch.get_rng_state()
    device = _model_device(model)
    if device.type == "cuda":
        tensors["rng::cuda"] = torch.cuda.get_rng_state(device)
    save_file(tensors, tensor_path)
    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "step": int(step),
        "max_code": int(max_code),
        "vocabulary": list(vocabulary),
        "experiment_config": experiment_config,
        "tokenizer_kind": tokenizer_kind,
        "ratchet_update_counts": _ratchet_update_counts(model),
    }
    if run_seed is not None:
        metadata["run_seed"] = int(run_seed)
    if tokenizer_json is not None:
        metadata["tokenizer_json"] = _canonical_tokenizer_json(tokenizer_json)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return base


def load_checkpoint(
    base_path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_max_code: int,
    expected_vocabulary: tuple[str, ...],
    expected_matmul_mode: str = "fp32",
    expected_tokenizer_kind: str = "char",
    expected_tokenizer_json: str | None = None,
    expected_experiment_config: dict[str, Any] | None = None,
    expected_run_seed: int | None = None,
    training_device: str | torch.device | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    _, tensor_path, metadata_path = _paths(base_path)
    metadata = json.loads(metadata_path.read_text())
    format_version = metadata.get("format_version")
    if format_version not in _SUPPORTED_FORMAT_VERSIONS:
        raise ValueError("unsupported checkpoint format version")
    if metadata.get("max_code") != expected_max_code:
        raise ValueError("checkpoint code range does not match requested model")

    saved_kind = metadata.get("tokenizer_kind", "char")
    if saved_kind != expected_tokenizer_kind:
        raise ValueError("checkpoint tokenizer kind does not match dataset")
    if saved_kind == "char":
        if tuple(metadata.get("vocabulary", ())) != expected_vocabulary:
            raise ValueError("checkpoint vocabulary does not match dataset")
    else:
        saved_tokenizer_json = metadata.get("tokenizer_json")
        if not isinstance(saved_tokenizer_json, str) or expected_tokenizer_json is None:
            raise ValueError("subword checkpoint requires tokenizer identity")
        if _canonical_tokenizer_json(saved_tokenizer_json) != _canonical_tokenizer_json(
            expected_tokenizer_json
        ):
            raise ValueError("checkpoint tokenizer does not match dataset")

    saved_config = metadata.get("experiment_config", {})
    saved_mode = saved_config.get("matmul_mode", "fp32")
    if saved_mode != expected_matmul_mode:
        raise ValueError("checkpoint matmul_mode does not match requested run")
    if expected_experiment_config is not None:
        for field, default in _RESUME_CONFIG_DEFAULTS.items():
            if field not in expected_experiment_config:
                continue
            if saved_config.get(field, default) != expected_experiment_config[field]:
                raise ValueError(f"checkpoint {field} does not match requested run")
    if expected_run_seed is not None:
        if "run_seed" not in metadata:
            if format_version == 1:
                raise ValueError("legacy checkpoint is missing run seed")
            raise ValueError("checkpoint is missing run seed")
        if metadata["run_seed"] != expected_run_seed:
            raise ValueError("checkpoint run seed does not match requested run")

    tensors = load_file(tensor_path)
    device = torch.device(training_device) if training_device is not None else _model_device(model)
    ratchet_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, DiscreteRatchetLinear)
    }
    ratchet_updates_enabled = saved_config.get("weight_mode", "ratchet") == "ratchet"
    leak_active = (
        bool(ratchet_modules)
        and ratchet_updates_enabled
        and int(saved_config.get("pressure_leak_period", 0)) > 0
    )
    counts = metadata.get("ratchet_update_counts")
    if leak_active and not isinstance(counts, dict):
        if format_version == 1:
            raise ValueError("legacy checkpoint is missing ratchet leak counters")
        raise ValueError("checkpoint is missing ratchet leak counters")
    cuda_rng_required = device.type == "cuda" and float(saved_config.get("dropout", 0.0)) > 0
    if cuda_rng_required and "rng::cuda" not in tensors:
        if format_version == 1:
            raise ValueError("legacy CUDA checkpoint is missing CUDA RNG state")
        raise ValueError("CUDA checkpoint is missing CUDA RNG state")
    cpu_rng_required = (
        (device.type == "cpu" and float(saved_config.get("dropout", 0.0)) > 0)
        or (
            bool(ratchet_modules)
            and saved_mode == "int8"
            and bool(saved_config.get("int8_backward", False))
        )
    )
    if cpu_rng_required and "rng::cpu" not in tensors:
        if format_version == 1:
            raise ValueError("legacy checkpoint is missing CPU RNG state")
        raise ValueError("checkpoint is missing CPU RNG state")

    if isinstance(counts, dict):
        valid_values = all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in counts.values()
        )
        if set(counts) != set(ratchet_modules) or not valid_values:
            raise ValueError("checkpoint ratchet leak counters do not match model")

    model_state = {
        key.removeprefix("model::"): value
        for key, value in tensors.items()
        if key.startswith("model::")
    }
    model.load_state_dict(model_state, strict=True)
    if isinstance(counts, dict):
        for name, module in ratchet_modules.items():
            module._update_count = int(counts[name])

    named_parameters = dict(model.named_parameters())
    optimizer.state.clear()
    for key, value in tensors.items():
        if not key.startswith("optimizer::"):
            continue
        _, parameter_name, state_name = key.split("::", 2)
        if parameter_name not in named_parameters:
            raise ValueError(f"checkpoint optimizer parameter is unknown: {parameter_name}")
        parameter = named_parameters[parameter_name]
        optimizer.state[parameter][state_name] = value.to(parameter.device)
    if restore_rng:
        if "rng::cpu" in tensors:
            torch.set_rng_state(tensors["rng::cpu"])
        if device.type == "cuda" and "rng::cuda" in tensors:
            torch.cuda.set_rng_state(tensors["rng::cuda"], device=device)
    return metadata
