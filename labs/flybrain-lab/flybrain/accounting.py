"""Count trainable scalars, not marketing 'parameters'."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn


@dataclass(frozen=True)
class ResourceReport:
    trainable_scalars: int
    trainable_bits: int
    persistent_state_scalars: int
    neuron_count: int
    synapse_count: int
    flops_per_step: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def param_breakdown(module: nn.Module) -> dict[str, int]:
    """Split trainable scalars into encoder / decoder / synapse / program / other."""
    parts = {"encoder": 0, "decoder": 0, "synapse": 0, "program": 0, "other": 0}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        if name.startswith("encoder") or name.startswith("embed"):
            parts["encoder"] += n
        elif name.startswith("decoder"):
            parts["decoder"] += n
        elif "synapse" in name:
            parts["synapse"] += n
        elif (
            name.startswith("cells")
            or name.startswith("emit")
            or name.startswith("gru")
            or name.startswith("unique")
        ):
            parts["program"] += n
        else:
            parts["other"] += n
    parts["total"] = sum(v for k, v in parts.items() if k != "total")
    return parts


def count_resources(
    module: nn.Module,
    *,
    persistent_state_scalars: int,
    neuron_count: int,
    synapse_count: int,
    flops_per_step: int = 0,
    bits_per_scalar: int = 32,
) -> ResourceReport:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return ResourceReport(
        trainable_scalars=int(trainable),
        trainable_bits=int(trainable * bits_per_scalar),
        persistent_state_scalars=int(persistent_state_scalars),
        neuron_count=int(neuron_count),
        synapse_count=int(synapse_count),
        flops_per_step=int(flops_per_step),
    )
