"""Observable ratchet-state metrics."""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import nn

from .ratchet import DiscreteRatchetLinear, audit_no_master_weights


def _histogram(values: torch.Tensor) -> str:
    # Vectorized count: torch.unique runs on-device in O(N) and the Python loop below
    # touches only the handful of distinct code/pressure values, not every element.
    # (A Counter over values.tolist() iterated all ~1.6B elements in pure Python at
    # width-4096 — minutes of GPU-idle stall per metric row that read as a hang.)
    unique, counts = torch.unique(values.detach().flatten(), sorted=True, return_counts=True)
    unique = unique.cpu().tolist()
    counts = counts.cpu().tolist()
    return json.dumps(
        {str(int(key)): int(count) for key, count in zip(unique, counts, strict=True)},
        separators=(",", ":"),
    )


def collect_ratchet_metrics(model: nn.Module) -> dict[str, Any]:
    layers = [module for module in model.modules() if isinstance(module, DiscreteRatchetLinear)]
    if not layers:
        support_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        )
        return {
            "ratchet_layers": 0,
            "ratchet_weights": 0,
            "ratchet_state_bytes": 0,
            "support_parameter_bytes": support_bytes,
            "zero_percent": 0.0,
            "saturated_percent": 0.0,
            "code_histogram": "{}",
            "pressure_histogram": "{}",
        }
    codes = torch.cat([layer.code.flatten() for layer in layers])
    pressure = torch.cat([layer.pressure.flatten() for layer in layers])
    saturated = torch.cat([(layer.code.abs() == layer.max_code).flatten() for layer in layers])
    audit = audit_no_master_weights(model, raise_on_violation=True)
    return {
        "ratchet_layers": audit.ratchet_layers,
        "ratchet_weights": audit.ratchet_weights,
        "ratchet_state_bytes": audit.ratchet_state_bytes,
        "support_parameter_bytes": audit.support_parameter_bytes,
        "zero_percent": 100.0 * float((codes == 0).float().mean().item()),
        "saturated_percent": 100.0 * float(saturated.float().mean().item()),
        "code_histogram": _histogram(codes),
        "pressure_histogram": _histogram(pressure),
    }
