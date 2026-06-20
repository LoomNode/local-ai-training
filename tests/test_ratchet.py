import math

import pytest
import torch
from torch import nn

from local_ai_training.ratchet import (
    DiscreteRatchetLinear,
    audit_no_master_weights,
    bucket_pressure,
)


def test_bucket_pressure_uses_gradient_descent_direction() -> None:
    z = torch.tensor([-1.5, -1.49, -0.5, -0.49, 0.0, 0.49, 0.5, 1.49, 1.5])

    pressure = bucket_pressure(z, low=0.5, high=1.5)

    assert pressure.tolist() == [2, 1, 1, 0, 0, 0, -1, -1, -2]


@pytest.mark.parametrize("max_code", [2, 3])
def test_reference_initialization_has_valid_codes_and_row_scales(max_code: int) -> None:
    reference = torch.tensor([[0.0, 0.5, -1.0], [3.0, -2.0, 1.0]])

    layer = DiscreteRatchetLinear.from_reference(reference, max_code=max_code)

    assert layer.code.dtype == torch.int8
    assert layer.pressure.dtype == torch.int8
    assert layer.scale.dtype == torch.float32
    assert layer.code.min().item() >= -max_code
    assert layer.code.max().item() <= max_code
    assert layer.scale.shape == (2,)
    assert torch.all(layer.scale > 0)
    assert layer.effective_weight().shape == reference.shape


def test_positive_gradient_clicks_code_down_and_retains_residual_pressure() -> None:
    layer = DiscreteRatchetLinear.from_reference(
        torch.tensor([[1.0, 1.0]]), max_code=2, pressure_threshold=2
    )

    stats = layer.apply_normalized_gradient(torch.tensor([[1.5, 0.5]]))

    assert layer.code.tolist() == [[1, 2]]
    assert layer.pressure.tolist() == [[0, -1]]
    assert stats.negative_moves == 1
    assert stats.positive_moves == 0


def test_saturated_outward_click_is_consumed_and_recorded() -> None:
    layer = DiscreteRatchetLinear.from_reference(
        torch.tensor([[1.0]]), max_code=2, pressure_threshold=2
    )
    layer.code.fill_(2)

    stats = layer.apply_normalized_gradient(torch.tensor([[-2.0]]))

    assert layer.code.item() == 2
    assert layer.pressure.item() == 0
    assert stats.blocked_positive_moves == 1
    assert stats.positive_moves == 0


def test_training_forward_keeps_temporary_weight_until_update_then_releases_it() -> None:
    layer = DiscreteRatchetLinear.from_reference(torch.eye(2), max_code=2)
    layer.train()
    inputs = torch.ones(1, 2, requires_grad=True)

    layer(inputs).sum().backward()

    assert layer.has_pending_gradient
    stats = layer.ratchet_update()
    assert stats.total_weights == 4
    assert not layer.has_pending_gradient
    assert layer._effective_weight is None


def test_audit_rejects_float_matrix_parameter_inside_ratchet_module() -> None:
    good = nn.Sequential(DiscreteRatchetLinear(3, 2, max_code=2))
    report = audit_no_master_weights(good)
    assert report.ratchet_layers == 1
    assert report.violations == ()
    assert report.ratchet_state_bytes > 0

    bad_layer = good[0]
    bad_layer.register_parameter("master_weight", nn.Parameter(torch.zeros(2, 3)))

    with pytest.raises(RuntimeError, match="master_weight"):
        audit_no_master_weights(good, raise_on_violation=True)


def test_row_rms_normalization_is_finite_for_zero_gradient() -> None:
    layer = DiscreteRatchetLinear(4, 2, max_code=3)

    stats = layer.apply_weight_gradient(torch.zeros(2, 4))

    assert stats.code_moves == 0
    assert torch.count_nonzero(layer.pressure).item() == 0
    assert math.isfinite(stats.gradient_rms_mean)

