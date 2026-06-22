import pytest
import torch

from local_ai_training.int8_matmul import quantize_columns, quantize_rows, scaled_int8_mm


def test_row_and_column_quantization_keep_zero_inputs_finite() -> None:
    values = torch.zeros(3, 5, dtype=torch.bfloat16)
    rows, row_scale = quantize_rows(values)
    columns, column_scale = quantize_columns(values)
    assert rows.dtype == columns.dtype == torch.int8
    assert torch.count_nonzero(rows) == torch.count_nonzero(columns) == 0
    assert torch.isfinite(row_scale).all() and torch.all(row_scale > 0)
    assert torch.isfinite(column_scale).all() and torch.all(column_scale > 0)
    assert row_scale.shape == (3,)
    assert column_scale.shape == (5,)


def test_scaled_int8_mm_rejects_cpu_operands() -> None:
    lhs = torch.ones(2, 3, dtype=torch.int8)
    rhs = torch.ones(3, 4, dtype=torch.int8)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        scaled_int8_mm(lhs, rhs, torch.ones(2), torch.ones(4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scaled_int8_mm_matches_scaled_reference_for_non_tile_shape() -> None:
    torch.manual_seed(3)
    lhs = torch.randint(-20, 21, (67, 35), device="cuda", dtype=torch.int8)
    rhs = torch.randint(-8, 9, (35, 53), device="cuda", dtype=torch.int8)
    lhs_scale = torch.rand(67, device="cuda", dtype=torch.float32) + 0.01
    rhs_scale = torch.rand(53, device="cuda", dtype=torch.float32) + 0.01

    actual = scaled_int8_mm(lhs, rhs, lhs_scale, rhs_scale)
    expected = ((lhs.float() @ rhs.float()) * lhs_scale[:, None] * rhs_scale[None, :]).to(
        torch.bfloat16
    )

    assert actual.dtype == torch.bfloat16
    assert actual.shape == (67, 53)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scaled_int8_mm_consumes_noncontiguous_operands_bit_exactly() -> None:
    # The Triton kernel takes full operand strides, so a transposed (non-contiguous)
    # operand must give bit-identical results without first being copied to a
    # contiguous buffer. This guards the memory optimization that drops the forced
    # .contiguous() materialization of transposed int8 weights in the int8 path.
    torch.manual_seed(5)
    lhs = torch.randint(-20, 21, (48, 40), device="cuda", dtype=torch.int8)
    weight = torch.randint(-8, 9, (53, 40), device="cuda", dtype=torch.int8)
    rhs_view = weight.t()  # (40, 53), non-contiguous
    assert not rhs_view.is_contiguous()
    lhs_scale = torch.rand(48, device="cuda", dtype=torch.float32) + 0.01
    rhs_scale = torch.rand(53, device="cuda", dtype=torch.float32) + 0.01

    strided = scaled_int8_mm(lhs, rhs_view, lhs_scale, rhs_scale)
    contiguous = scaled_int8_mm(lhs, rhs_view.contiguous(), lhs_scale, rhs_scale)

    assert torch.equal(strided, contiguous)
