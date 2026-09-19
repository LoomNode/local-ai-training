"""Honest resource accounting: a fat tensor is many scalars, not one 'complex parameter'."""

from __future__ import annotations

import torch
from torch import nn

from biointel.accounting import ResourceReport, count_resources


class FatScalar(nn.Module):
    """Looks like one learned object; stores 256 independent floats."""

    def __init__(self) -> None:
        super().__init__()
        self.blob = nn.Parameter(torch.zeros(8, 32))


def test_fat_parameter_counts_as_256_scalars_not_one():
    report = count_resources(FatScalar(), persistent_state_scalars=0, neuron_count=0, synapse_count=0)
    assert report.trainable_scalars == 256
    assert report.trainable_bits == 256 * 32


def test_frozen_tensor_is_not_trainable():
    m = FatScalar()
    m.blob.requires_grad_(False)
    report = count_resources(m, persistent_state_scalars=12, neuron_count=3, synapse_count=4)
    assert report.trainable_scalars == 0
    assert report.persistent_state_scalars == 12
    assert report.neuron_count == 3
    assert report.synapse_count == 4


def test_report_is_jsonable_with_required_keys():
    report = count_resources(FatScalar(), persistent_state_scalars=1, neuron_count=1, synapse_count=1)
    payload = report.as_dict()
    for key in (
        "trainable_scalars",
        "trainable_bits",
        "persistent_state_scalars",
        "neuron_count",
        "synapse_count",
        "flops_per_step",
        "activation_scalars",
    ):
        assert key in payload
        assert isinstance(payload[key], int)
