"""Tests for sparse-gradient-safe per-row RMS-EMA normalization in DiscreteRatchetLinear.

The bug: when a gradient is sparse (many all-zero rows, e.g. embedding rows not in a batch),
_normalize with rms_ema_beta > 0 decays those rows' EMA toward zero because ms==0 triggers the
exponential decay formula. On the next firing step, the now-tiny EMA inflates the normalized
gradient and can cause runaway updates.

The fix: the EMA update must be masked to only touch rows where ms > 0 (fired rows).
Dense behavior (all rows fire) must be bit-exact with the old formula.
"""
import torch

from local_ai_training.ratchet import DiscreteRatchetLinear


def _layer(*, rms_ema_beta: float = 0.9) -> DiscreteRatchetLinear:
    torch.manual_seed(42)
    return DiscreteRatchetLinear(
        in_features=16,
        out_features=32,
        max_code=7,
        rms_ema_beta=rms_ema_beta,
    )


# ---------------------------------------------------------------------------
# Test 1: unfired rows preserve their EMA across a step
# ---------------------------------------------------------------------------
def test_sparse_ema_unfired_rows_preserved() -> None:
    """Rows with zero gradient (not in batch) must not have their EMA updated."""
    layer = _layer()

    # Manually set some non-zero EMA values to simulate prior history
    torch.manual_seed(0)
    layer.rms_ema.copy_(torch.rand(32) + 0.1)  # all > 0
    ema_before = layer.rms_ema.clone()

    # Build a sparse gradient: only rows 0-3 fire, rows 4-31 are zero
    grad = torch.zeros(32, 16)
    grad[:4] = torch.randn(4, 16)

    layer.apply_weight_gradient(grad, validate=False)

    # Rows 0-3 should have updated EMA (ms > 0)
    assert not torch.allclose(layer.rms_ema[:4], ema_before[:4]), (
        "Fired rows should have updated EMA"
    )
    # Rows 4-31 must be exactly unchanged — no decay
    assert torch.equal(layer.rms_ema[4:], ema_before[4:]), (
        "Unfired rows must NOT have their EMA decayed; got decay: "
        f"max delta = {(layer.rms_ema[4:] - ema_before[4:]).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 2: dense gradient (all rows fire) is mathematically identical to old formula
# ---------------------------------------------------------------------------
def test_dense_ema_matches_reference_formula() -> None:
    """When all rows fire, masked update == original update (no regression)."""
    layer = _layer()

    torch.manual_seed(1)
    layer.rms_ema.copy_(torch.rand(32) * 0.5 + 0.1)
    ema_before = layer.rms_ema.clone()

    # Dense gradient: every row has non-zero signal
    torch.manual_seed(2)
    grad = torch.randn(32, 16)

    # Compute reference EMA update (original formula)
    grad_f = grad.float()
    ms = grad_f.square().mean(dim=1)  # [32]
    beta = 0.9
    ema_ref = torch.where(
        ema_before == 0,
        ms,
        beta * ema_before + (1.0 - beta) * ms,
    )

    layer.apply_weight_gradient(grad, validate=False)

    assert torch.allclose(layer.rms_ema, ema_ref, atol=1e-6), (
        f"Dense step EMA mismatch; max delta = {(layer.rms_ema - ema_ref).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 3: repeated sparse steps don't accumulate EMA decay on unfired rows
# ---------------------------------------------------------------------------
def test_sparse_ema_no_accumulation_over_multiple_steps() -> None:
    """EMA of unfired rows must stay stable across many steps, not slowly decay."""
    layer = _layer()

    torch.manual_seed(3)
    layer.rms_ema.copy_(torch.ones(32) * 0.5)

    # Only row 0 fires for 50 steps
    grad = torch.zeros(32, 16)
    torch.manual_seed(4)
    for _ in range(50):
        grad[0] = torch.randn(16)
        layer.apply_weight_gradient(grad.clone(), validate=False)

    # Rows 1-31 should still be 0.5 (no decay at all)
    assert torch.allclose(layer.rms_ema[1:], torch.full((31,), 0.5), atol=1e-6), (
        f"Unfired rows decayed after 50 sparse steps; "
        f"min={layer.rms_ema[1:].min().item():.6f}, "
        f"max={layer.rms_ema[1:].max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 4: zero-initialized EMA correctly initializes on first firing
# ---------------------------------------------------------------------------
def test_sparse_ema_zero_init_initializes_on_first_fire() -> None:
    """A row starting at EMA=0 (uninitialized) should initialize from ms on first fire."""
    layer = _layer()
    assert (layer.rms_ema == 0).all(), "EMA buffer should start zeroed"

    torch.manual_seed(5)
    grad = torch.zeros(32, 16)
    grad[7] = torch.randn(16)

    layer.apply_weight_gradient(grad, validate=False)

    ms7 = grad[7].float().square().mean().item()
    assert abs(layer.rms_ema[7].item() - ms7) < 1e-6, (
        "First fire for uninitialized row should set EMA = ms (not blend with 0)"
    )
    # Other rows stay at 0 (not decayed from 0)
    mask = torch.ones(32, dtype=torch.bool)
    mask[7] = False
    assert (layer.rms_ema[mask] == 0).all(), "Unfired uninitialized rows must stay at 0"
