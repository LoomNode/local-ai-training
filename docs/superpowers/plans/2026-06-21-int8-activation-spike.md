# Int8 Activation Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether int8 tensor-core math (codes already int8; activations quantized per-token via `torch._int_mm`) is faster than bf16 at training shapes *and* how much int8-activation quantization perturbs a layer's output -- a cheap go/no-go screen gating an expensive int8-activation training run.

**Architecture:** A standalone int8 pipeline (per-token activation quant -> `torch._int_mm` -> dequant by both scales) plus a benchmark/accuracy harness comparing it to bf16-eager. Nothing wired into the package.

**Tech Stack:** Python 3.10+, PyTorch >=2.5 (`torch._int_mm` in-tree, no custom kernel), CUDA GPU. Run with `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run ...`.

## Global Constraints

- Codes are already int8 (`code = (packed & 0x0F) - max_code`); only **activations** are quantized, to int8, **per-token** (per-tensor measured only for contrast). Activations are never quantized below int8 in this spike.
- Reuses the repo's packed format via `pack_code_pressure` from `local_ai_training.ratchet`.
- Shapes: weight `[N, 768]` for N in {768, 2304, 3072}, activations `[16384, 768]`, max_code 4.
- Two gates: speed (int8 pipeline >= 1.2x faster than bf16-eager) AND accuracy (per-token output rel error small, low single-digit %). Both must pass to recommend the training run.
- Measurement only. Out of scope: training run, model integration, int4 activations, backward, SmoothQuant outlier handling.
- Lives under `scripts/int8_spike/`. Does not touch the existing test suite.
- `torch._int_mm` needs int8 contiguous inputs and tile-aligned dims; the chosen shapes satisfy this.
- Commit messages end with the repo's Co-Authored-By / Claude-Session trailers.

---

### Task 1: int8 pipeline + correctness test

**Files:**
- Create: `scripts/int8_spike/__init__.py` (empty)
- Create: `scripts/int8_spike/int8_forward.py`
- Create: `scripts/int8_spike/test_int8_forward.py`

**Interfaces:**
- Produces: `int8_ratchet_forward(packed: Tensor, code_scale: Tensor, x: Tensor, max_code: int, per_token: bool = True) -> Tensor`
  -- `packed` uint8 `[N,K]`, `code_scale` fp32 `[N]`, `x` bf16 `[T,K]`; returns bf16 `[T,N]`.

- [ ] **Step 1: Write the pipeline**

```python
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
```

Also create an empty `scripts/int8_spike/__init__.py`.

- [ ] **Step 2: Write the correctness test (isolates pipeline math from quantization loss)**

```python
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
```

- [ ] **Step 3: Run the test**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache CUDA_VISIBLE_DEVICES=1 uv run pytest scripts/int8_spike/test_int8_forward.py -v`
Expected: PASS. If `torch._int_mm` rejects the shapes, pad K/N to a multiple of 16 and note it. Do not change the int8 rounding to force the test -- a failure means the dequant math is wrong.

- [ ] **Step 4: Commit**

```bash
git add scripts/int8_spike/
git commit -m "feat: int8 per-token activation ratchet forward pipeline + correctness test"
```

---

### Task 2: Speed + accuracy harness

**Files:**
- Create: `scripts/int8_spike/bench.py`

**Interfaces:**
- Consumes: `int8_ratchet_forward` (Task 1).

- [ ] **Step 1: Write the harness**

```python
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
        out_pt = int8_ratchet_forward(packed, code_scale, x, MAX_CODE, per_token=True).to(torch.float32)
        out_pte = int8_ratchet_forward(packed, code_scale, x, MAX_CODE, per_token=False).to(torch.float32)
        err_pt = _rel_err(out_pt, ref)
        err_pte = _rel_err(out_pte, ref)

        ms_int8 = _time(lambda: int8_ratchet_forward(packed, code_scale, x, MAX_CODE))
        x_i8 = torch.clamp(torch.round(x.float() / (x.float().abs().amax(1, keepdim=True) / 127).clamp_min(1e-12)), -127, 127).to(torch.int8)
        ms_mm = _time(lambda: torch._int_mm(x_i8, cb))
        ms_bf16 = _time(lambda: x @ eff_bf16.t())

        print(f"shape N={n} K={k} T={t}")
        print(f"  int8 pipeline : {ms_int8:.3f} ms   ({ms_bf16 / ms_int8:.2f}x vs bf16-eager)")
        print(f"  bare _int_mm  : {ms_mm:.3f} ms   ({ms_bf16 / ms_mm:.2f}x vs bf16-eager)")
        print(f"  bf16-eager    : {ms_bf16:.3f} ms")
        print(f"  rel err  per-token={err_pt:.4f}  per-tensor={err_pte:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the harness**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.int8_spike.bench`
Expected: per-shape timings (int8 pipeline, bare `_int_mm`, bf16-eager) and per-token/per-tensor relative errors. Record the numbers.

- [ ] **Step 3: Commit**

```bash
git add scripts/int8_spike/bench.py
git commit -m "feat: int8 spike speed + accuracy harness"
```

---

### Task 3: Record results and verdict

**Files:**
- Create: `docs/results/2026-06-21-int8-activation-spike.md`

- [ ] **Step 1: Write the results note**

Capture, from the actual `bench.py` output: per-shape int8-pipeline ms, bare `_int_mm` ms,
bf16-eager ms, the `int8 vs bf16` ratios, and per-token / per-tensor relative errors. Then
state the verdict, judged on BOTH gates:

- **GO** (justify the training run) if the int8 pipeline is >= 1.2x faster than bf16-eager
  AND per-token rel error is small (low single-digit %).
- **NO-GO** if too slow (note whether the bottleneck is the matmul or the quantization
  overhead, from the bare-`_int_mm` number) or if per-token error is large (note the
  per-tensor contrast -- if per-token is much better, outliers are the issue and would need
  SmoothQuant-style handling).

Skeleton, fill in real numbers:

```markdown
# Int8 Activation Spike: Results

## Setup
int8 per-token activation quant + int8 codes via torch._int_mm vs bf16-eager, single RTX 3090,
nonary-100M layer shapes at a 16384-token batch, max_code 4.

## Numbers
| shape (N x K, T) | int8 pipeline ms | bare _int_mm ms | bf16-eager ms | int8 vs bf16 | rel err (per-token / per-tensor) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ... | ... | ... | ... | ...x | ... / ... |

Correctness test (exact-quantization pipeline check): passes.

## Verdict
GO / NO-GO -- <one paragraph: speed (and whether quant overhead is the bottleneck, from
bare _int_mm), accuracy (per-token error magnitude, per-tensor contrast / outlier signal),
recommendation on the int8-activation training run>.
```

- [ ] **Step 2: Commit**

```bash
git add docs/results/2026-06-21-int8-activation-spike.md
git commit -m "docs: record int8 activation spike results and verdict"
```

---

## Self-Review

**Spec coverage:** the int8 pipeline (spec "Component: the int8 pipeline") -> Task 1; benchmark
+ accuracy screen (spec "Component: benchmark + accuracy screen") -> Task 2; success
criteria/decision (spec) -> Task 3; testing (spec) -> Task 1 test. Both gates (speed, accuracy)
are measured in Task 2 and judged in Task 3. Out-of-scope items correctly absent.

**Placeholder scan:** none -- pipeline, test, and harness are complete code; the results note is
a fill-in-the-numbers skeleton (numbers come from running it).

**Type consistency:** `int8_ratchet_forward(packed, code_scale, x, max_code, per_token=True)`
signature identical in Task 1 definition, Task 1 test, and Task 2 harness. `pack_code_pressure`
imported from `local_ai_training.ratchet` matching the repo. uint8 packed / fp32 code_scale /
bf16 x / bf16 out consistent throughout.

## Handoff note

Self-contained for an external executor (e.g. Gemini): all formulas/code inline, commands
exact, both go/no-go gates explicit. Needs a free CUDA GPU (GPU 1). Branch:
`feat/packed-matmul-kernel` (worktree `/tmp/lat-kernel`).
