"""Observable ratchet-state metrics."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import torch
from torch import nn

from .ratchet import DiscreteRatchetLinear, audit_no_master_weights


def _histogram(values: torch.Tensor) -> str:
    counts = Counter(int(value) for value in values.detach().cpu().flatten().tolist())
    return json.dumps({str(key): counts[key] for key in sorted(counts)}, separators=(",", ":"))


def collect_ratchet_metrics(model: nn.Module) -> dict[str, Any]:
    layers = [module for module in model.modules() if isinstance(module, DiscreteRatchetLinear)]
    if not layers:
        raise ValueError("model contains no ratchet layers")
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

