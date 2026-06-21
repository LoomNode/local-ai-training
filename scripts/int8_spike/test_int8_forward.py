# scripts/int8_spike/test_int8_forward.py
import pytest
import torch

from local_ai_training.ratchet import pack_code_pressure
from scripts.int8_spike.int8_forward import int8_ratchet_forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_int8_pipeline_is_exact_when_inputs_quantize_exactly():
    # Activations that are already small integers, unit code scale -> int8 path must equal the
    # float reference bit-for-bit, proving the dequant math (not quantization error) is correct.
    torch.manual_seed(0)
    dev = "cuda"
    N, K, T, max_code = 64, 128, 32, 4
    code = torch.randint(-max_code, max_code + 1, (N, K), dtype=torch.int8, device=dev)
    packed = pack_code_pressure(code, torch.zeros_like(code), max_code).to(torch.uint8)
    code_scale = torch.ones(N, device=dev)
    # integer-valued activations in [-127, 127] -> per-token round-trip is lossless
    x = torch.randint(-5, 6, (T, K), device=dev).to(torch.bfloat16)

    out = int8_ratchet_forward(packed, code_scale, x, max_code).to(torch.float32)
    ref = (x.to(torch.float32) @ code.to(torch.float32).t())
    assert torch.equal(out, ref.to(torch.bfloat16).to(torch.float32))
