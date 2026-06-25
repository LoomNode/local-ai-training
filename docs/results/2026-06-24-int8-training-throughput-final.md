# End-to-End int8 Training Pipeline (Final Throughput)

> **SUPERSEDED (2026-06-24)** by `2026-06-24-int8-per-token-speed.md`. The "int8 beats bf16 at
> width 4096 (1.08×)" claim below compares int8 against the **bf16 *ratchet*** (`matmul_mode="bf16"`,
> which pays the ratchet's effective-weight materialization overhead) — not **dense bf16** (plain
> `nn.Linear` under bf16 autocast = real mixed-precision training, the actual bf16-then-PTQ cost).
> Against the correct dense baseline, int8 is **~0.66× and never wins per token** at fittable width;
> the width-4096 advantage is **memory only** (dense OOMs there). The crossover and "int8 strictly
> faster at frontier" framing here is an artifact of the weak baseline. Numbers retained below as a
> record. The `code.t().contiguous()` forward fix and the ~19B-param memory ceiling remain valid.

**Date:** 2026-06-24

## The Core Finding
We re-architected the `int8` ACT pipeline to heavily rely on native PyTorch memory management and Flash Attention via `torch.autocast(dtype=torch.bfloat16)`. The backward pass now strictly avoids python-side tiling and utilizes PyTorch's native `.to(torch.bfloat16)` `cuBLAS` path.

## Benchmark Results
At lower widths, the memory allocation overhead of expanding `int8` to `bfloat16` for intermediate operations swamps the tensor core speedups. But at frontier scales (Width 4096), the raw mathematical speed of Tensor Cores completely dominates the memory expansion overhead, making `int8` strictly faster and substantially smaller.

**Width 512 (Batch 64):**
- **fp32:** 236,750 tok/s (1.8 GB VRAM)
- **bf16:** 111,613 tok/s (1.6 GB VRAM)
- **int8:** 42,674 tok/s (1.2 GB VRAM)
*Result:* At small widths, memory operations dominate. `bf16` is over 2x faster than `int8`.

**Width 2048 (Batch 16):**
- **fp32:** 17,903 tok/s (7.8 GB VRAM)
- **bf16:** 7,347 tok/s (4.7 GB VRAM)
- **int8:** 8,536 tok/s (3.7 GB VRAM)
*Result:* Crossover point. `int8` ACT overtakes `bf16` (~16% faster) and uses 1 GB less memory.

**Width 4096 (Batch 8):**
- **fp32:** CRASH (Out of Memory - exceeded 24GB VRAM)
- **bf16:** 1,432 tok/s (7.2 GB VRAM)
- **int8:** 1,551 tok/s (6.3 GB VRAM)
*Result:* Frontier scale. Standard 32-bit math becomes physically impossible on an RTX 3090. `int8` ACT is **1.08x faster** than `bf16` and saves roughly a gigabyte of VRAM.

## Architectural Notes
- The theoretical maximum parameter limit for this architecture on an RTX 3090 (24GB) is now roughly **~19 Billion Parameters**.
- A massive forward-pass bug was fixed where `code.t()` was passed without `.contiguous()`, forcing PyTorch to clone a 16MB tensor uncoalesced every step. This restored the tensor-core advantage at Width 4096.
