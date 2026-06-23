# Ratchet Forward Kernel Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a custom Triton dequantize-and-matmul kernel that reads the ratchet's packed 4-bit codes directly beats the eager "materialize FP weight + cuBLAS" forward path -- a go/no-go feasibility spike.

**Architecture:** One standalone Triton kernel computing `out = scale * (x @ code.T)` with the weight read as packed uint8 and unpacked to bf16 in-register (fp32 accumulate), plus a benchmark/correctness harness comparing it to bf16-eager and fp32-eager. Nothing is wired into the package.

**Tech Stack:** Python 3.10+, PyTorch >=2.5 (Triton bundled), CUDA GPU required. Run with `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run ...`.

## Global Constraints

- Weight-only quantization: 4-bit weights, **bf16 activations, fp32 accumulation**. Activations are never quantized.
- Reuses the repo's existing packed format: `pack_code_pressure` / `unpack_code_pressure` from `local_ai_training.ratchet` (code in the low nibble, `code = (packed & 0x0F) - max_code`).
- Supported nibble ranges: `max_code in (2, 4)` for the prototype.
- Measurement only. Out of scope: backward, autograd, integration into `DiscreteRatchetLinear.forward`.
- Success bar: kernel >= 1.2x faster than bf16-eager at the representative shapes, output within ~2e-2 relative error (bf16 tolerance).
- Lives entirely under `scripts/kernel_prototype/`. Does not touch the existing test suite.
- Commit messages end with the repo's Co-Authored-By / Claude-Session trailers.

---

### Task 1: Triton forward kernel + correctness test

**Files:**
- Create: `scripts/kernel_prototype/__init__.py` (empty)
- Create: `scripts/kernel_prototype/ratchet_forward.py`
- Create: `scripts/kernel_prototype/test_ratchet_forward.py`

**Interfaces:**
- Produces: `ratchet_forward(packed: Tensor, scale: Tensor, x: Tensor, max_code: int) -> Tensor`
  -- `packed` uint8 `[N, K]`, `scale` fp32 `[N]`, `x` bf16 `[T, K]`; returns bf16 `[T, N]`.

- [ ] **Step 1: Write the kernel and wrapper**

```python
# scripts/kernel_prototype/ratchet_forward.py
import torch
import triton
import triton.language as tl


@triton.jit
def _ratchet_forward_kernel(
    x_ptr, packed_ptr, scale_ptr, out_ptr,
    T, N, K, max_code,
    stride_xt, stride_xk,
    stride_pn, stride_pk,
    stride_ot, stride_on,
    BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        x_tile = tl.load(
            x_ptr + offs_t[:, None] * stride_xt + k[None, :] * stride_xk,
            mask=(offs_t[:, None] < T) & (k[None, :] < K), other=0.0,
        ).to(tl.bfloat16)
        p_tile = tl.load(
            packed_ptr + offs_n[:, None] * stride_pn + k[None, :] * stride_pk,
            mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0,
        )
        code = (p_tile & 0x0F).to(tl.int32) - max_code          # [BLOCK_N, BLOCK_K]
        code_bf = code.to(tl.bfloat16)
        acc += tl.dot(x_tile, tl.trans(code_bf))                # [BLOCK_T, BLOCK_N], fp32 accumulate
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc * scale[None, :]
    tl.store(
        out_ptr + offs_t[:, None] * stride_ot + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=(offs_t[:, None] < T) & (offs_n[None, :] < N),
    )


def ratchet_forward(packed: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, max_code: int) -> torch.Tensor:
    assert packed.dtype == torch.uint8 and packed.is_cuda
    T, K = x.shape
    N, Kp = packed.shape
    assert Kp == K
    x = x.to(torch.bfloat16)
    scale = scale.to(torch.float32)
    out = torch.empty((T, N), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_T"]), triton.cdiv(N, meta["BLOCK_N"]))
    _ratchet_forward_kernel[grid](
        x, packed, scale, out, T, N, K, max_code,
        x.stride(0), x.stride(1), packed.stride(0), packed.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_T=64, BLOCK_N=64, BLOCK_K=32,
    )
    return out
```

Also create an empty `scripts/kernel_prototype/__init__.py`.

- [ ] **Step 2: Write the correctness test**

```python
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
```

- [ ] **Step 3: Run the test**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache CUDA_VISIBLE_DEVICES=1 uv run pytest scripts/kernel_prototype/test_ratchet_forward.py -v`
Expected: PASS (6 cases). If it fails on relative error, the kernel math is wrong (check the nibble unpack and the `tl.trans`); if it errors in Triton compilation, reduce `BLOCK_K` to 16. Do not loosen the 2e-2 tolerance to force a pass -- a failing correctness test is a real signal.

- [ ] **Step 4: Commit**

```bash
git add scripts/kernel_prototype/
git commit -m "feat: triton ratchet forward dequant-matmul kernel + correctness test"
```

---

### Task 2: Benchmark harness

**Files:**
- Create: `scripts/kernel_prototype/bench.py`

**Interfaces:**
- Consumes: `ratchet_forward` (Task 1).

- [ ] **Step 1: Write the benchmark**

```python
# scripts/kernel_prototype/bench.py
import time

import torch

from local_ai_training.ratchet import pack_code_pressure
from scripts.kernel_prototype.ratchet_forward import ratchet_forward

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


def main():
    dev = "cuda"
    for n, k, t in SHAPES:
        code = torch.randint(-MAX_CODE, MAX_CODE + 1, (n, k), dtype=torch.int8, device=dev)
        packed = pack_code_pressure(code, torch.zeros_like(code), MAX_CODE).to(torch.uint8)
        scale = (torch.rand(n, device=dev) + 0.1).to(torch.float32)
        x_bf16 = torch.randn(t, k, device=dev, dtype=torch.bfloat16)
        x_fp32 = x_bf16.to(torch.float32)
        eff_bf16 = (code.to(torch.float32) * scale[:, None]).to(torch.bfloat16)
        eff_fp32 = code.to(torch.float32) * scale[:, None]

        ms_kernel = _time(lambda: ratchet_forward(packed, scale, x_bf16, MAX_CODE))
        ms_bf16 = _time(lambda: x_bf16 @ eff_bf16.t())
        ms_fp32 = _time(lambda: x_fp32 @ eff_fp32.t())

        # peak memory of the eager path (materializes the effective weight) vs kernel
        torch.cuda.reset_peak_memory_stats()
        _ = (code.to(torch.float32) * scale[:, None]).to(torch.bfloat16)
        _ = x_bf16 @ eff_bf16.t()
        mem_eager = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.reset_peak_memory_stats()
        _ = ratchet_forward(packed, scale, x_bf16, MAX_CODE)
        mem_kernel = torch.cuda.max_memory_allocated() / 1e6

        print(f"shape N={n} K={k} T={t}")
        print(f"  kernel     : {ms_kernel:.3f} ms")
        print(f"  bf16-eager : {ms_bf16:.3f} ms   (kernel {ms_bf16 / ms_kernel:.2f}x vs bf16-eager)")
        print(f"  fp32-eager : {ms_fp32:.3f} ms   (kernel {ms_fp32 / ms_kernel:.2f}x vs fp32-eager)")
        print(f"  peak mem   : kernel {mem_kernel:.0f}MB vs eager {mem_eager:.0f}MB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.kernel_prototype.bench`
Expected: prints per-shape timings and the `kernel Xx vs bf16-eager` ratios. Record the numbers.

- [ ] **Step 3: Commit**

```bash
git add scripts/kernel_prototype/bench.py
git commit -m "feat: benchmark harness for ratchet forward kernel"
```

---

### Task 3: Record results and verdict

**Files:**
- Create: `docs/results/2026-06-21-forward-kernel-prototype.md`

- [ ] **Step 1: Write the results note**

Capture, from the actual `bench.py` output: per-shape kernel ms, bf16-eager ms, the
`kernel vs bf16-eager` ratio, the `vs fp32-eager` ratio, and the peak-memory comparison.
Then state the verdict explicitly:

- **GO** if kernel >= 1.2x faster than bf16-eager across the shapes with the correctness
  test passing -> the full forward+backward training kernel is worth building.
- **NO-GO / needs tuning** if marginal or slower -> note whether it is a tuning issue
  (fixed block sizes, no autotune) or a structural ceiling, and stop.

Use this skeleton, filling in the real numbers:

```markdown
# Ratchet Forward Kernel Prototype: Results

## Setup
Triton dequant-matmul (4-bit weights, bf16 activations, fp32 accumulate) vs eager
materialize+matmul, single RTX 3090, nonary-100M layer shapes at a 16384-token batch.

## Numbers
| shape (N x K, T) | kernel ms | bf16-eager ms | kernel vs bf16 | vs fp32 | peak mem k/e |
| --- | ---: | ---: | ---: | ---: | ---: |
| ... | ... | ... | ...x | ...x | ... |

Correctness: kernel matches bf16-eager within 2e-2 (test passes).

## Verdict
GO / NO-GO -- <one paragraph: is it >=1.2x faster, is the win bandwidth or tensor-core,
is any shortfall tuning vs structural, recommendation on building the full kernel>.
```

- [ ] **Step 2: Commit**

```bash
git add docs/results/2026-06-21-forward-kernel-prototype.md
git commit -m "docs: record forward kernel prototype results and verdict"
```

---

## Self-Review

**Spec coverage:** kernel (spec "Component: the kernel") -> Task 1; benchmark + correctness
(spec "Component: the benchmark + correctness harness") -> Tasks 1 (correctness) + 2
(benchmark); success criteria/decision (spec) -> Task 3; testing (spec) -> Task 1 test +
Task 2 script. All covered. Out-of-scope items (backward, integration) correctly absent.

**Placeholder scan:** none -- kernel, wrapper, test, and benchmark are complete code; the
results note has a fill-in-the-numbers skeleton (expected, since numbers come from running it).

**Type consistency:** `ratchet_forward(packed, scale, x, max_code)` signature identical in
Task 1 definition, Task 1 test, and Task 2 benchmark; `pack_code_pressure`/`unpack_code_pressure`
imported from `local_ai_training.ratchet` matching the repo. uint8 packed / fp32 scale / bf16
x / bf16 out consistent throughout.

## Handoff note

Self-contained for an external executor (e.g. Gemini): all code is inline, commands are
exact, and the go/no-go bar is explicit. Needs a free CUDA GPU (GPU 1 in this environment).
Branch: `feat/packed-matmul-kernel` (worktree `/tmp/lat-kernel`).
