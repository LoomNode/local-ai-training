# scripts/kernel_prototype/test_ratchet_forward.py
import pytest
import torch

from local_ai_training.ratchet import pack_code_pressure, unpack_code_pressure
from scripts.kernel_prototype.ratchet_forward import ratchet_forward

SHAPES = [(768, 768, 256), (2304, 768, 256), (3072, 768, 256)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel requires CUDA")
@pytest.mark.parametrize("max_code", [2, 4])
@pytest.mark.parametrize("n,k,t", SHAPES)
def test_kernel_matches_bf16_eager(max_code, n, k, t):
    torch.manual_seed(0)
    dev = "cuda"
    code = torch.randint(-max_code, max_code + 1, (n, k), dtype=torch.int8, device=dev)
    pressure = torch.zeros_like(code)
    packed = pack_code_pressure(code, pressure, max_code).to(torch.uint8)
    scale = (torch.rand(n, device=dev) + 0.1).to(torch.float32)
    x = torch.randn(t, k, device=dev, dtype=torch.bfloat16)

    # bf16-eager reference: materialize effective weight, matmul, same precision
    decoded, _ = unpack_code_pressure(packed, max_code)
    effective = (decoded.to(torch.float32) * scale[:, None]).to(torch.bfloat16)
    ref = (x @ effective.t()).to(torch.float32)

    out = ratchet_forward(packed, scale, x, max_code).to(torch.float32)
    rel = (out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)
    assert rel < 2e-2, f"rel err {rel:.4f} too high for max_code={max_code} shape={(n, k, t)}"
