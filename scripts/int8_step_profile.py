"""Profile where time goes in an int8 ratchet training step: quantization vs the GEMMs.

If the int8 path is "just unoptimized", the memory-bound quantization/elementwise kernels (per-token
activation quant in forward, gradient/input quant in both backward GEMMs, the full-FP32 scaled_gradient
materialization) should dominate over the int8 GEMM itself — meaning a fused int8 linear has real
headroom. If the int8 GEMM dominates, there is little left to win at this shape.

    CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
      uv run python scripts/int8_step_profile.py --mode int8 --embd 512
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from local_ai_training.model import ModelConfig, build_seeded_model

VOCAB = 65


def classify(name: str) -> str:
    n = name.lower()
    if "_scaled_int8_kernel" in n or "int8" in n:
        return "int8_gemm (tuned)"
    if any(k in n for k in ("gemm", "cutlass", "ampere", "cublas", "sgemm", "wgrad", "dgrad")):
        return "fp/bf16 gemm (cublas)"
    if any(k in n for k in ("elementwise", "vectorized_elementwise", "copy", "cast", "convert")):
        return "elementwise/cast"
    if any(k in n for k in ("reduce", "amax", "norm", "sum")):
        return "reduce (quant scale)"
    if any(k in n for k in ("round", "clamp", "mul", "div", "add")):
        return "elementwise arith (quant)"
    return name[:40]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="int8")
    parser.add_argument("--embd", type=int, default=512)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda")
    max_code = None if args.mode == "fp32" else 2
    matmul_mode = "fp32" if args.mode == "fp32" else args.mode
    config = ModelConfig(
        vocab_size=VOCAB, block_size=args.block, n_layer=args.layer, n_head=args.head,
        n_embd=args.embd, dropout=0.0, matmul_mode=matmul_mode, gradient_checkpointing=False,
    )
    model = build_seeded_model(config, max_code=max_code, seed=1337).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    gen = torch.Generator(device="cpu").manual_seed(0)
    tokens = torch.randint(0, VOCAB, (args.batch, args.block + 1), generator=gen).to(device)
    inputs, targets = tokens[:, :args.block], tokens[:, 1:]

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        loss.backward()
        if max_code is not None:
            model.ratchet_update()
        optimizer.step()

    for _ in range(8):
        step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(10):
            step()
        torch.cuda.synchronize()

    buckets: dict[str, float] = defaultdict(float)
    total = 0.0
    for evt in prof.key_averages():
        cuda_us = getattr(evt, "self_device_time_total", 0) or getattr(evt, "self_cuda_time_total", 0)
        if cuda_us <= 0:
            continue
        buckets[classify(evt.key)] += cuda_us
        total += cuda_us

    print(f"\nmode={args.mode} embd={args.embd}  total GPU us/10 steps = {total:,.0f}\n")
    for name, us in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {us/total*100:5.1f}%  {us/10:9.1f} us/step   {name}")


if __name__ == "__main__":
    main()
