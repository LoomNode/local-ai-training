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


@pytest.mark.parametrize("max_code", [1, 2, 3, 4, 5, 6, 7])
def test_ratchet_accepts_max_code_up_to_nibble_cap(max_code):
    # max_code=1 is ternary (3 states); 7 is the 4-bit nibble cap (15 states).
    layer = DiscreteRatchetLinear(8, 4, max_code=max_code)
    assert layer.max_code == max_code
    assert layer.code.abs().max().item() <= max_code


def test_ratchet_rejects_max_code_above_nibble_cap():
    with pytest.raises(ValueError):
        DiscreteRatchetLinear(8, 4, max_code=8)


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


def test_packed_footprint_is_one_byte_per_weight() -> None:
    model = nn.Sequential(
        DiscreteRatchetLinear(256, 128, max_code=2),
        DiscreteRatchetLinear(128, 64, max_code=3),
    )
    audit = audit_no_master_weights(model)
    weights = audit.ratchet_weights
    scale_bytes = (128 + 64) * 4  # per-row fp32 scale
    assert audit.ratchet_state_bytes == weights * 1 + scale_bytes
    fp = compare_persistent_footprint(model)
    assert fp.reduction_ratio > 10  # ~12x now (was ~6x at int8)


def test_stored_pressure_stays_within_nibble_under_adversarial_updates() -> None:
    torch.manual_seed(0)
    for max_code in (2, 3, 4):
        for threshold in (1, 2, 4, 8):
            layer = DiscreteRatchetLinear(8, 6, max_code=max_code, pressure_threshold=threshold)
            for step in range(200):
                sign = 1.0 if step % 7 < 5 else -1.0  # sustained push -> saturate codes
                layer.apply_normalized_gradient(sign * (3.0 + torch.rand(6, 8)))
                assert int(layer.pressure.abs().max()) <= 7
                assert int(layer.code.abs().max()) <= max_code


def test_compiled_update_matches_eager() -> None:
    # Seed before each construction so both layers start from identical random codes.
    torch.manual_seed(1234)
    eager = DiscreteRatchetLinear(8, 6, max_code=4, pressure_threshold=8)
    torch.manual_seed(1234)
    comp = DiscreteRatchetLinear(8, 6, max_code=4, pressure_threshold=8, compile_update=True)
    assert torch.equal(eager.packed, comp.packed)  # identical starting state
    for _ in range(20):
        g = torch.randn(6, 8) * 2.0
        eager.apply_normalized_gradient(g.clone())
        comp.apply_normalized_gradient(g.clone())
    assert torch.equal(eager.packed, comp.packed)


@pytest.mark.parametrize("mode", ["bf16", "int8"])
def test_opt_in_matmul_modes_add_no_persistent_state(mode: str) -> None:
    baseline = DiscreteRatchetLinear(7, 5, max_code=2)
    layer = DiscreteRatchetLinear(7, 5, max_code=2, matmul_mode=mode)
    assert dict(layer.named_parameters()) == {}
    assert set(dict(layer.named_buffers())) == {"packed", "_scale"}
    assert layer.persistent_state_bytes == baseline.persistent_state_bytes
    assert audit_no_master_weights(nn.Sequential(layer)).violations == ()


@pytest.mark.parametrize("mode", ["bf16", "int8"])
def test_opt_in_matmul_modes_capture_effective_gradient_and_clear(mode: str) -> None:
    if mode == "int8" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(2)
    device = "cuda" if mode == "int8" else "cpu"
    layer = DiscreteRatchetLinear(35, 53, max_code=3, matmul_mode=mode).to(device).train()
    inputs = torch.randn(2, 3, 35, device=device, requires_grad=True)
    output = layer(inputs)
    assert output.shape == (2, 3, 53)

    loss = output.sum()
    loss.backward()

    assert layer.has_pending_gradient
    assert layer._pending_weight_gradient is not None
    assert layer._pending_weight_gradient.shape == (53, 35)
    assert torch.isfinite(layer._pending_weight_gradient).all()

    # Second backward should fail because ratchet_update was not called
    output2 = layer(inputs)
    with pytest.raises(RuntimeError, match="missing ratchet_update"):
        output2.sum().backward()

    layer.ratchet_update()
    output = layer(inputs)
    output.sum().backward()
    layer.discard_pending_gradient()
    assert not layer.has_pending_gradient


def test_bf16_mode_matches_eager_bf16_forward_and_effective_gradient() -> None:
    torch.manual_seed(4)
    layer = DiscreteRatchetLinear(17, 11, max_code=2, matmul_mode="bf16").train()
    inputs = torch.randn(7, 17, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    reference_weight = layer.effective_weight().to(torch.bfloat16).detach().requires_grad_(True)
    expected = torch.nn.functional.linear(reference_inputs.to(torch.bfloat16), reference_weight)
    expected.float().square().mean().backward()

    actual = layer(inputs)
    actual.float().square().mean().backward()

    assert torch.equal(actual, expected.to(actual.dtype))
    assert torch.equal(inputs.grad, reference_inputs.grad)
    assert torch.equal(layer._pending_weight_gradient, reference_weight.grad.float())


def test_trainable_scale_gradient_matches_eager_bf16_autograd() -> None:
    torch.manual_seed(5)
    layer = DiscreteRatchetLinear(
        13, 9, max_code=4, matmul_mode="bf16", trainable_scale=True
    ).train()
    inputs = torch.randn(6, 13)
    code = layer.code.float()
    reference_log_scale = layer.log_scale.detach().clone().requires_grad_(True)
    reference_weight = (code * reference_log_scale.exp()[:, None]).to(torch.bfloat16)
    reference = torch.nn.functional.linear(inputs.to(torch.bfloat16), reference_weight)
    reference.float().square().mean().backward()

    layer(inputs).float().square().mean().backward()

    assert torch.allclose(layer.log_scale.grad, reference_log_scale.grad, atol=2e-4, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_mode_forward_matches_integer_reference() -> None:
    from local_ai_training.int8_matmul import quantize_rows

    torch.manual_seed(6)
    layer = DiscreteRatchetLinear(35, 53, max_code=4, matmul_mode="int8").cuda().eval()
    inputs = torch.randn(2, 3, 35, device="cuda")
    quantized, input_scale = quantize_rows(inputs.flatten(0, -2))
    expected = (
        (quantized.float() @ layer.code.t().float()) * input_scale[:, None] * layer.scale[None, :]
    ).to(torch.bfloat16)

    actual = layer(inputs)

    assert torch.equal(actual, expected.reshape(2, 3, 53).float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_int8_effective_weight_gradient_stays_close_to_eager_autograd() -> None:
    torch.manual_seed(7)
    layer = DiscreteRatchetLinear(35, 53, max_code=3, matmul_mode="int8").cuda().train()
    inputs = torch.randn(67, 35, device="cuda", requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    reference_weight = layer.effective_weight().detach().requires_grad_(True)
    reference = torch.nn.functional.linear(reference_inputs, reference_weight)
    reference.float().square().mean().backward()

    layer(inputs).float().square().mean().backward()

    relative_error = (
        layer._pending_weight_gradient - reference_weight.grad
    ).norm() / reference_weight.grad.norm()
    assert relative_error < 0.03

@pytest.mark.parametrize("max_code", [2, 3, 4])
@pytest.mark.parametrize("trainable_scale", [False, True])
@pytest.mark.parametrize("matmul_mode", ["fp32", "bf16", "int8"])
def test_fused_backward_equivalence(max_code: int, trainable_scale: bool, matmul_mode: str) -> None:
    if matmul_mode in ("bf16", "int8") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda" if matmul_mode in ("bf16", "int8") else "cpu"
    
    torch.manual_seed(42)
    inputs = torch.randn(5, 7, 16, device=device, requires_grad=True)
    
    weight = torch.randn(32, 16, device=device)
    
    eager = DiscreteRatchetLinear(
        16, 32, max_code=max_code, trainable_scale=trainable_scale, 
        matmul_mode=matmul_mode, fuse_backward_update=False, initial_weight=weight
    ).to(device)
    
    fused = DiscreteRatchetLinear(
        16, 32, max_code=max_code, trainable_scale=trainable_scale, 
        matmul_mode=matmul_mode, fuse_backward_update=True, tile_size=11, initial_weight=weight
    ).to(device)
    
    eager.train()
    fused.train()
    
    inputs_eager = inputs.detach().clone().requires_grad_()
    inputs_fused = inputs.detach().clone().requires_grad_()
    
    out_eager = eager(inputs_eager)
    out_fused = fused(inputs_fused)
    
    grad_out = torch.randn_like(out_eager)
    out_eager.backward(grad_out)
    out_fused.backward(grad_out)
    
    stats_eager = eager.ratchet_update()
    stats_fused = fused.ratchet_update()
    
    assert stats_fused.total_weights == stats_eager.total_weights
    assert stats_fused.positive_moves == stats_eager.positive_moves
    assert stats_fused.negative_moves == stats_eager.negative_moves
    assert stats_fused.blocked_positive_moves == stats_eager.blocked_positive_moves
    assert stats_fused.blocked_negative_moves == stats_eager.blocked_negative_moves
    
    assert torch.isclose(
        torch.tensor(stats_fused.gradient_rms_mean),
        torch.tensor(stats_eager.gradient_rms_mean),
        rtol=1e-4,
        atol=1e-4,
    )
    
    assert torch.equal(fused.packed, eager.packed)
    
    if trainable_scale:
        assert torch.allclose(fused.log_scale.grad, eager.log_scale.grad, rtol=1e-4, atol=1e-4)


def test_rms_ema_beta_zero_matches_instantaneous_normalization():
    torch.manual_seed(0)
    ref = torch.randn(6, 8)
    base = DiscreteRatchetLinear(8, 6, max_code=2, initial_weight=ref.clone())
    ema = DiscreteRatchetLinear(8, 6, max_code=2, rms_ema_beta=0.0, initial_weight=ref.clone())
    grad = torch.randn(6, 8)
    n_base = base._normalize(grad, 0, 6)
    n_ema = ema._normalize(grad, 0, 6)
    assert torch.equal(n_base, n_ema)  # beta=0 is bit-identical to the current rule


def test_rms_ema_first_step_matches_instantaneous_then_smooths():
    torch.manual_seed(0)
    layer = DiscreteRatchetLinear(8, 6, max_code=2, rms_ema_beta=0.9)
    g1 = torch.randn(6, 8)
    # first step: EMA seeds from this step's mean-square, so identical to instantaneous
    rms1 = g1.float().square().mean(dim=1, keepdim=True).sqrt()
    assert torch.allclose(layer._normalize(g1, 0, 6), g1.float() / (rms1 + layer.eps))
    # second step: denominator is the EMA, NOT this step's rms
    g2 = torch.randn(6, 8) * 5.0
    ms1 = g1.float().square().mean(dim=1)
    ms2 = g2.float().square().mean(dim=1)
    expected_ema = 0.9 * ms1 + 0.1 * ms2
    expected = g2.float() / (expected_ema.unsqueeze(1).sqrt() + layer.eps)
    assert torch.allclose(layer._normalize(g2, 0, 6), expected)


def test_rms_ema_buffer_is_per_row_and_audit_clean():
    layer = DiscreteRatchetLinear(8, 6, max_code=2, rms_ema_beta=0.9)
    assert layer.rms_ema.shape == (6,)  # one scalar per output row
    assert layer.rms_ema.ndim == 1
    assert audit_no_master_weights(nn.Sequential(layer)).violations == ()


def _force_pressure(layer, value):
    # set every weight's pressure to `value`, codes unchanged, via the packing helpers
    code, _ = unpack_code_pressure(layer.packed, layer.max_code)
    pressure = torch.full_like(code, value, dtype=torch.int8)
    layer.packed.copy_(pack_code_pressure(code, pressure, layer.max_code))


def test_pressure_leak_period_zero_never_leaks():
    layer = DiscreteRatchetLinear(8, 4, max_code=2, pressure_leak_period=0)
    _force_pressure(layer, 5)
    for _ in range(10):
        layer._maybe_leak_pressure()
    _, pressure = unpack_code_pressure(layer.packed, layer.max_code)
    assert int(pressure.min()) == 5 and int(pressure.max()) == 5  # untouched


def test_pressure_leak_fires_every_k_and_moves_toward_zero():
    layer = DiscreteRatchetLinear(8, 4, max_code=2, pressure_leak_period=3)
    _force_pressure(layer, 5)
    for _ in range(3):  # fires on the 3rd call (count 1,2,3 -> leak at 3)
        layer._maybe_leak_pressure()
    _, pressure = unpack_code_pressure(layer.packed, layer.max_code)
    assert int(pressure.max()) == 4  # one unit toward zero, exactly once


def test_pressure_leak_moves_negative_toward_zero_and_never_enlarges():
    layer = DiscreteRatchetLinear(8, 4, max_code=2, pressure_leak_period=1)
    _force_pressure(layer, -2)
    layer._maybe_leak_pressure()
    _, pressure = unpack_code_pressure(layer.packed, layer.max_code)
    assert int(pressure.min()) == -1  # toward zero, |pressure| shrank
    layer._validate_state()  # still within the nibble range
