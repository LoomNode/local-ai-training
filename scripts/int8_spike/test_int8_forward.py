# scripts/int8_spike/test_int8_forward.py
import pytest
import torch

from local_ai_training.ratchet import pack_code_pressure
from scripts.int8_spike.int8_forward import int8_ratchet_forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_int8_pipeline_matches_float_reference_within_quant_tolerance():
    # A per-token-quantized pipeline cannot be bit-exact (the max/127 scale does not put
    # integers on exact grid points, and the output is cast to bf16). So this verifies the
    # *dequant math* -- both scales applied, correct matmul orientation -- which a real bug
    # (wrong transpose, dropped scale) would blow far past the ~1% quantization noise.
    torch.manual_seed(0)
    dev = "cuda"
    N, K, T, max_code = 256, 256, 128, 4
    code = torch.randint(-max_code, max_code + 1, (N, K), dtype=torch.int8, device=dev)
    packed = pack_code_pressure(code, torch.zeros_like(code), max_code).to(torch.uint8)
    code_scale = (torch.rand(N, device=dev) + 0.1).to(torch.float32)
    x = torch.randn(T, K, device=dev, dtype=torch.bfloat16)

    effective = code.to(torch.float32) * code_scale[:, None]
    ref = x.to(torch.float32) @ effective.t()
    out = int8_ratchet_forward(packed, code_scale, x, max_code).to(torch.float32)
    rel = (out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)
    assert rel < 3e-2, f"rel err {rel:.4f} -- dequant math likely wrong"
