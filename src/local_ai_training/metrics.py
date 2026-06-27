"""Observable ratchet-state metrics."""

from __future__ import annotations

import json
from typing import Any

import torch
from torch import nn

from .ratchet import DiscreteRatchetLinear, audit_no_master_weights


def _accumulate_counts(values: torch.Tensor, counts: dict[int, int]) -> None:
    # Count via bounded boolean reductions over the (tiny) integer range, accumulating into
    # `counts`. Code/pressure are small nibble-packed ints, so min..max spans only a handful
    # of buckets and each (flat == v).sum() is a cheap on-device reduction with a single
    # transient mask the size of the (per-layer) input.
    flat = values.detach().flatten()
    if flat.numel() == 0:
        return
    lo = int(flat.min().item())
    hi = int(flat.max().item())
    for value in range(lo, hi + 1):
        count = int((flat == value).sum().item())
        if count:
            counts[value] = counts.get(value, 0) + count


def _format_histogram(counts: dict[int, int]) -> str:
    return json.dumps({str(key): counts[key] for key in sorted(counts)}, separators=(",", ":"))


def _histogram(values: torch.Tensor) -> str:
    counts: dict[int, int] = {}
    _accumulate_counts(values, counts)
    return _format_histogram(counts)


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
    # Accumulate per layer so we never materialize the whole state at once. At width-4096 the
    # full state is ~1.6B elements; a torch.cat of every code/pressure plus a float() cast for
    # the mean (float32 of 1.6B == ~6.4 GiB) added ~11 GiB of transients on top of the model
    # and OOM'd. Per-layer unpacking keeps the working set to one layer (tens of MiB), and the
    # percentages come from integer sums rather than a float-cast mean.
    code_counts: dict[int, int] = {}
    pressure_counts: dict[int, int] = {}
    zero = 0
    saturated = 0
    total = 0
    for layer in layers:
        code = layer.code
        _accumulate_counts(code, code_counts)
        _accumulate_counts(layer.pressure, pressure_counts)
        zero += int((code == 0).sum().item())
        saturated += int((code.abs() == layer.max_code).sum().item())
        total += code.numel()
    audit = audit_no_master_weights(model, raise_on_violation=True)
    return {
        "ratchet_layers": audit.ratchet_layers,
        "ratchet_weights": audit.ratchet_weights,
        "ratchet_state_bytes": audit.ratchet_state_bytes,
        "support_parameter_bytes": audit.support_parameter_bytes,
        "zero_percent": 100.0 * zero / total,
        "saturated_percent": 100.0 * saturated / total,
        "code_histogram": _format_histogram(code_counts),
        "pressure_histogram": _format_histogram(pressure_counts),
    }
