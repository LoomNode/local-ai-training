# scripts/int8_spike/int8_forward.py
import torch


def int8_ratchet_forward(packed, code_scale, x, max_code, per_token=True):
    """int8(x) @ int8(code) via torch._int_mm, dequant by per-token x scale and per-row code scale."""
    assert packed.dtype == torch.uint8 and packed.is_cuda
    code_int8 = ((packed & 0x0F).to(torch.int16) - max_code).to(torch.int8)   # [N, K]
    x = x.to(torch.float32)
    if per_token:
        x_scale = x.abs().amax(dim=1, keepdim=True) / 127.0                   # [T, 1]
    else:
        x_scale = x.abs().amax().reshape(1, 1) / 127.0                        # [1, 1]
    x_scale = x_scale.clamp_min(1e-12)
    x_int8 = torch.clamp(torch.round(x / x_scale), -127, 127).to(torch.int8)  # [T, K]
    acc = torch._int_mm(x_int8, code_int8.t().contiguous())                   # [T, N] int32
    out = acc.to(torch.float32) * x_scale * code_scale.to(torch.float32)[None, :]
    return out.to(torch.bfloat16)
