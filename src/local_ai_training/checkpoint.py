"""Restricted tensor-only checkpoints with validated JSON metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

FORMAT_VERSION = 1


def _paths(base_path: str | Path) -> tuple[Path, Path, Path]:
    base = Path(base_path)
    if base.suffix in {".safetensors", ".json"}:
        base = base.with_suffix("")
    return base, base.with_suffix(".safetensors"), base.with_suffix(".json")


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
    save_file(tensors, tensor_path)
    metadata: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "step": int(step),
        "max_code": int(max_code),
        "vocabulary": list(vocabulary),
        "experiment_config": experiment_config,
        "tokenizer_kind": tokenizer_kind,
    }
    if tokenizer_json is not None:
        metadata["tokenizer_json"] = tokenizer_json
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
) -> dict[str, Any]:
    _, tensor_path, metadata_path = _paths(base_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    if metadata.get("max_code") != expected_max_code:
        raise ValueError("checkpoint code range does not match requested model")
    if metadata.get("tokenizer_kind", "char") == "char":
        if tuple(metadata.get("vocabulary", ())) != expected_vocabulary:
            raise ValueError("checkpoint vocabulary does not match dataset")
    saved_mode = metadata.get("experiment_config", {}).get("matmul_mode", "fp32")
    if saved_mode != expected_matmul_mode:
        raise ValueError("checkpoint matmul_mode does not match requested run")

    tensors = load_file(tensor_path)
    model_state = {
        key.removeprefix("model::"): value
        for key, value in tensors.items()
        if key.startswith("model::")
    }
    model.load_state_dict(model_state, strict=True)
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
    if "rng::cpu" in tensors:
        torch.set_rng_state(tensors["rng::cpu"])
    return metadata
