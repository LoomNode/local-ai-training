import pytest
import torch

from local_ai_training.int8_matmul import (
    _quantize_columns_reference,
    _quantize_rows_reference,
    quantize_columns,
    quantize_rows,
    scaled_int8_mm,
)


def _bit_exact(fused, reference) -> None:
    fused_q, fused_scale = fused
    ref_q, ref_scale = reference
    assert torch.equal(fused_q, ref_q)
    assert torch.equal(fused_scale, ref_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(67, 35), (128, 256), (1, 513), (255, 1)])
def test_quantize_rows_fused_matches_reference_bit_exact(shape, dtype) -> None:
    from local_ai_training.int8_matmul import _quantize_rows_fused

    torch.manual_seed(7)
    values = torch.randn(shape, device="cuda", dtype=dtype) * 3.0
    _bit_exact(_quantize_rows_fused(values), _quantize_rows_reference(values))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quantize_rows_fused_handles_noncontiguous_input() -> None:
    from local_ai_training.int8_matmul import _quantize_rows_fused

    torch.manual_seed(11)
    base = torch.randn(70, 48, device="cuda", dtype=torch.bfloat16) * 2.0
    view = base.t()  # (48, 70), non-contiguous
    assert not view.is_contiguous()
    _bit_exact(_quantize_rows_fused(view), _quantize_rows_reference(view))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(67, 35), (128, 256), (513, 1), (1, 255)])
def test_quantize_columns_fused_matches_reference_bit_exact(shape, dtype) -> None:
    from local_ai_training.int8_matmul import _quantize_columns_fused

    torch.manual_seed(9)
    values = torch.randn(shape, device="cuda", dtype=dtype) * 3.0
    _bit_exact(_quantize_columns_fused(values), _quantize_columns_reference(values))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_quantizers_keep_zero_inputs_finite() -> None:
    from local_ai_training.int8_matmul import _quantize_columns_fused, _quantize_rows_fused

    zeros = torch.zeros(3, 5, device="cuda", dtype=torch.bfloat16)
    for fused in (_quantize_rows_fused(zeros), _quantize_columns_fused(zeros)):
        q, scale = fused
        assert torch.count_nonzero(q) == 0
        assert torch.isfinite(scale).all() and torch.all(scale > 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quantize_columns_slice_equals_quantize_rows_transpose(dtype) -> None:
    # The int8 backward restructure (Step 2A) replaces per-tile
    # quantize_rows(grad.t()[a:b]) with a single quantize_columns(grad) sliced and
    # transposed. This is the exact identity that makes that bit-exact.
    torch.manual_seed(13)
    grad = torch.randn(384, 200, device="cuda", dtype=dtype) * 2.0
    cols_q, cols_scale = quantize_columns(grad)
    a, b = 64, 192  # a tile of output features
    rows_q, rows_scale = quantize_rows(grad.t()[a:b])
    assert torch.equal(cols_q[:, a:b].t(), rows_q)
    assert torch.equal(cols_scale[a:b], rows_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(67, 35), (128, 256), (1, 255), (300, 1)])
def test_quantize_rows_colscaled_matches_reference_bit_exact(shape, dtype) -> None:
    from local_ai_training.int8_matmul import quantize_rows_colscaled

    torch.manual_seed(21)
    g = torch.randn(shape, device="cuda", dtype=dtype) * 3.0
    col_scale = torch.rand(shape[1], device="cuda", dtype=torch.float32) + 0.01
    # Equivalent to pre-scaling each column then row-quantizing (the grad_input path).
    expected = quantize_rows(g.float() * col_scale[None, :])
    _bit_exact(quantize_rows_colscaled(g, col_scale), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quantize_rows_colscaled_handles_noncontiguous_input() -> None:
    from local_ai_training.int8_matmul import quantize_rows_colscaled

    torch.manual_seed(23)
    base = torch.randn(50, 96, device="cuda", dtype=torch.bfloat16) * 2.0
    g = base.t()  # (96, 50) non-contiguous
    assert not g.is_contiguous()
    col_scale = torch.rand(50, device="cuda", dtype=torch.float32) + 0.01
    expected = quantize_rows(g.float() * col_scale[None, :])
    _bit_exact(quantize_rows_colscaled(g, col_scale), expected)


def test_quantize_rows_rejects_cpu_operands() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        quantize_rows(torch.ones(2, 3))


def test_quantize_columns_rejects_cpu_operands() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        quantize_columns(torch.ones(2, 3))


def test_row_and_column_quantization_keep_zero_inputs_finite() -> None:
    values = torch.zeros(3, 5, dtype=torch.bfloat16)
    rows, row_scale = _quantize_rows_reference(values)
    columns, column_scale = _quantize_columns_reference(values)
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
