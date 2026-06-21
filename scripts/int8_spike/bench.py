# scripts/int8_spike/bench.py
import time

import torch

from local_ai_training.ratchet import pack_code_pressure
from scripts.int8_spike.int8_forward import int8_ratchet_forward

SHAPES = [(768, 768, 16384), (2304, 768, 16384), (3072, 768, 16384)]
MAX_CODE = 4


def _time(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters * 1000  # ms/call


def _rel_err(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def main():
    dev = "cuda"
    for n, k, t in SHAPES:
        code = torch.randint(-MAX_CODE, MAX_CODE + 1, (n, k), dtype=torch.int8, device=dev)
        packed = pack_code_pressure(code, torch.zeros_like(code), MAX_CODE).to(torch.uint8)
        code_scale = (torch.rand(n, device=dev) + 0.1).to(torch.float32)
        x = torch.randn(t, k, device=dev, dtype=torch.bfloat16)
        eff_bf16 = (code.to(torch.float32) * code_scale[:, None]).to(torch.bfloat16)
        code_int8 = code  # already int8
        cb = code_int8.t().contiguous()

        ref = (x @ eff_bf16.t()).to(torch.float32)
        out_pt = int8_ratchet_forward(packed, code_scale, x, MAX_CODE, per_token=True).to(
            torch.float32
        )
        out_pte = int8_ratchet_forward(packed, code_scale, x, MAX_CODE, per_token=False).to(
            torch.float32
        )
        err_pt = _rel_err(out_pt, ref)
        err_pte = _rel_err(out_pte, ref)

        ms_int8 = _time(
            lambda packed=packed, code_scale=code_scale, x=x: int8_ratchet_forward(
                packed, code_scale, x, MAX_CODE
            )
        )
        x_i8 = torch.clamp(
            torch.round(x.float() / (x.float().abs().amax(1, keepdim=True) / 127).clamp_min(1e-12)),
            -127,
            127,
        ).to(torch.int8)
        ms_mm = _time(lambda x_i8=x_i8, cb=cb: torch._int_mm(x_i8, cb))
        ms_bf16 = _time(lambda x=x, eff_bf16=eff_bf16: x @ eff_bf16.t())

        print(f"shape N={n} K={k} T={t}")
        print(f"  int8 pipeline : {ms_int8:.3f} ms   ({ms_bf16 / ms_int8:.2f}x vs bf16-eager)")
        print(f"  bare _int_mm  : {ms_mm:.3f} ms   ({ms_bf16 / ms_mm:.2f}x vs bf16-eager)")
        print(f"  bf16-eager    : {ms_bf16:.3f} ms")
        print(f"  rel err  per-token={err_pt:.4f}  per-tensor={err_pte:.4f}")


if __name__ == "__main__":
    main()
