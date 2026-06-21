import math

import pytest
import torch
from torch import nn

from local_ai_training.ratchet import (
    DiscreteRatchetLinear,
    audit_no_master_weights,
    bucket_pressure,
    compare_persistent_footprint,
    pack_code_pressure,
    unpack_code_pressure,
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



def test_persistent_footprint_compares_ratchet_against_fp32_adam() -> None:
    model = nn.Sequential(
        DiscreteRatchetLinear(16, 8, max_code=2),
        DiscreteRatchetLinear(8, 4, max_code=3),
    )
    audit = audit_no_master_weights(model)
    footprint = compare_persistent_footprint(model)

    weights = audit.ratchet_weights
    # Ratchet keeps only its int8 code/pressure plus the per-row scale.
    assert footprint.ratchet_weights == weights
    assert footprint.ratchet_matrix_bytes == audit.ratchet_state_bytes
    # FP32 + AdamW must persist a 4-byte master plus two 4-byte moment buffers.
    assert footprint.fp32_master_bytes == weights * 4
    assert footprint.fp32_optimizer_bytes == weights * 8
    assert footprint.fp32_matrix_bytes == weights * 12
    # The whole point: storing the trainable matrices costs far less.
    assert footprint.reduction_ratio == footprint.fp32_matrix_bytes / footprint.ratchet_matrix_bytes
    assert footprint.reduction_ratio > 5


def test_trainable_scale_receives_gradient_and_stays_master_weight_free() -> None:
    torch.manual_seed(0)
    layer = DiscreteRatchetLinear(4, 3, max_code=2, trainable_scale=True)
    # Exactly one trainable scale per output row, not a per-weight matrix.
    params = dict(layer.named_parameters())
    assert set(params) == {"log_scale"}
    assert params["log_scale"].shape == (3,)
    # The audit still sees no floating matrix parameter.
    assert audit_no_master_weights(nn.Sequential(layer)).violations == ()

    layer.train()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.1)
    before = layer.scale.detach().clone()
    output = layer(torch.randn(5, 4))
    output.square().mean().backward()
    assert layer.log_scale.grad is not None
    # Codes still update from the captured effective-weight gradient.
    layer.ratchet_update()
    optimizer.step()
    assert not torch.equal(layer.scale.detach(), before)
    assert torch.all(layer.scale > 0)


def test_default_scale_is_a_fixed_buffer_not_a_parameter() -> None:
    layer = DiscreteRatchetLinear(4, 3, max_code=2)
    assert dict(layer.named_parameters()) == {}
    assert "_scale" in dict(layer.named_buffers())


def test_nonary_max_code_four_is_supported() -> None:
    torch.manual_seed(0)
    layer = DiscreteRatchetLinear(6, 4, max_code=4)
    assert int(layer.code.abs().max()) <= 4
    assert audit_no_master_weights(nn.Sequential(layer)).violations == ()

    layer.train()
    output = layer(torch.randn(3, 6))
    output.square().mean().backward()
    stats = layer.ratchet_update()
    assert stats.total_weights == 24
    assert int(layer.code.abs().max()) <= 4  # codes never escape the 9-state range


def test_nibble_pack_unpack_round_trip_is_lossless() -> None:
    for max_code in (2, 3, 4):
        codes = torch.arange(-max_code, max_code + 1, dtype=torch.int8)
        pressures = torch.arange(-7, 8, dtype=torch.int8)
        code_grid, pressure_grid = torch.meshgrid(codes, pressures, indexing="ij")
        packed = pack_code_pressure(code_grid, pressure_grid, max_code)
        assert packed.dtype == torch.uint8
        out_code, out_pressure = unpack_code_pressure(packed, max_code)
        assert torch.equal(out_code, code_grid)
        assert torch.equal(out_pressure, pressure_grid)
        assert out_code.dtype == torch.int8 and out_pressure.dtype == torch.int8


@pytest.mark.parametrize(
    "max_code, code_sum, pressure_sum, total_moves",
    [(2, 15, -32, 59), (3, 14, -32, 62), (4, 9, -32, 65)],
)
def test_update_matches_golden_reference(max_code, code_sum, pressure_sum, total_moves) -> None:
    torch.manual_seed(1234)
    layer = DiscreteRatchetLinear(8, 6, max_code=max_code, pressure_threshold=8)
    total = 0
    for _ in range(60):
        total += layer.apply_normalized_gradient(torch.randn(6, 8) * 2.0).code_moves
    assert int(layer.code.sum()) == code_sum
    assert int(layer.pressure.sum()) == pressure_sum
    assert total == total_moves
