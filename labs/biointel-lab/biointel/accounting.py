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
    activation_scalars: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def count_resources(
    module: nn.Module,
    *,
    persistent_state_scalars: int,
    neuron_count: int,
    synapse_count: int,
    flops_per_step: int = 0,
    activation_scalars: int | None = None,
    bits_per_scalar: int = 32,
) -> ResourceReport:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if activation_scalars is None:
        activation_scalars = persistent_state_scalars
    return ResourceReport(
        trainable_scalars=int(trainable),
        trainable_bits=int(trainable * bits_per_scalar),
        persistent_state_scalars=int(persistent_state_scalars),
        neuron_count=int(neuron_count),
        synapse_count=int(synapse_count),
        flops_per_step=int(flops_per_step),
        activation_scalars=int(activation_scalars),
    )
