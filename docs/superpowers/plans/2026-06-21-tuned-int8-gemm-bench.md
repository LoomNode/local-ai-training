# Tuned int8 GEMM Re-Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether a *tuned* int8 GEMM (torchao) beats bf16 cuBLAS at our training shapes -- the decisive test the `torch._int_mm` spike couldn't make, reopening or definitively closing the training-speed question.

**Architecture:** A standalone layer-level benchmark: bf16 `nn.Linear` vs the same linear quantized with torchao `int8_dynamic_activation_int8_weight`, at the nonary-100M layer shapes. Measurement only.

**Tech Stack:** Python 3.10+, PyTorch >=2.5, **torchao** (new dep), CUDA GPU. Run with `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run ...`.

## Global Constraints

- Shapes: weight `[N, 768]` for N in {768, 2304, 3072}, input `[16384, 768]`.
- Comparison is bf16 vs torchao-int8 forward, same shapes, warmup + `cuda.synchronize()` + many iters.
- Decision bar: torchao int8 >= 1.2x faster than bf16 = GO (speed reopens); <= bf16 = NO-GO (closed for real). K=768 is small, so ~1.3-1.6x is a plausible GO, not 2x.
- Measurement only. Out of scope: model integration, training, custom kernels, fusion.
- New dependency: torchao. Lives under `scripts/int8_spike/` (extends the existing spike). Does not touch the existing test suite.
- Commit messages end with the repo's Co-Authored-By / Claude-Session trailers.

---

### Task 1: Add torchao and a tuned-int8 vs bf16 benchmark

**Files:**
- Modify: `pyproject.toml` (add torchao dependency)
- Create: `scripts/int8_spike/tuned_bench.py`

**Interfaces:**
- Produces: a runnable `python -m scripts.int8_spike.tuned_bench` printing per-shape `torchao-int8 vs bf16` ratios.

- [ ] **Step 1: Add the dependency**

```bash
MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv add torchao
```
If `uv add` cannot resolve torchao against the pinned torch, instead run
`MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv pip install torchao` and note in the
results which install path was used. Verify: `uv run python -c "import torchao; print(torchao.__version__)"`.

- [ ] **Step 2: Write the benchmark**

```python
# scripts/int8_spike/tuned_bench.py
import time

import torch
from torch import nn

from torchao.quantization import int8_dynamic_activation_int8_weight, quantize_

SHAPES = [(768, 768, 16384), (2304, 768, 16384), (3072, 768, 16384)]  # (N, K, T)


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
        x = torch.randn(t, k, device=dev, dtype=torch.bfloat16)

        bf16 = nn.Linear(k, n, bias=False).to(dev).to(torch.bfloat16).eval()
        ms_bf16 = _time(lambda: bf16(x))

        # torchao int8: per-token int8 activations + int8 weight, tuned int8 matmul
        ao = nn.Linear(k, n, bias=False).to(dev).to(torch.bfloat16).eval()
        quantize_(ao, int8_dynamic_activation_int8_weight())
        with torch.no_grad():
            ms_ao = _time(lambda: ao(x))

        print(f"shape N={n} K={k} T={t}")
        print(f"  bf16        : {ms_bf16:.3f} ms")
        print(f"  torchao int8: {ms_ao:.3f} ms   ({ms_bf16 / ms_ao:.2f}x vs bf16)")


if __name__ == "__main__":
    main()
```

If the import path `from torchao.quantization import ...` differs in the installed torchao
version, adapt it (e.g. `torchao.quantization.quant_api`) and note what was used. The stable
concept is `quantize_(linear, int8_dynamic_activation_int8_weight())`.

- [ ] **Step 3: Run the benchmark**

Run: `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache CUDA_VISIBLE_DEVICES=1 uv run python -m scripts.int8_spike.tuned_bench`
Expected: per-shape `bf16` ms, `torchao int8` ms, and the ratio. Record the numbers. If torchao
errors on the quantize call, report the error verbatim and try the documented alternative API
for that version before giving up.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock scripts/int8_spike/tuned_bench.py
git commit -m "feat: tuned int8 (torchao) vs bf16 GEMM benchmark"
```

---

### Task 2: Record results and verdict; correct the roadmap

**Files:**
- Create: `docs/results/2026-06-21-tuned-int8-gemm-bench.md`
- Modify: `docs/ROADMAP.md` (the "Closed" section's int8 bullet)

- [ ] **Step 1: Write the results note**

Capture from the actual run: per-shape bf16 ms, torchao-int8 ms, the ratio, and the torchao
install path/version. State the verdict:

- **GO** if torchao int8 >= 1.2x faster than bf16 at the shapes -> speed reopens; next step is a
  custom fused Triton/CUTLASS kernel (dequant + ratchet update fused into the GEMM).
- **NO-GO** if still <= bf16 even tuned -> int8 genuinely does not beat bf16 on this hardware at
  these shapes; the speed door is closed for real (not an unoptimized-kernel artifact).

Skeleton, fill in real numbers:

```markdown
# Tuned int8 GEMM Re-Benchmark: Results

## Setup
torchao int8_dynamic_activation_int8_weight vs bf16 nn.Linear, single RTX 3090, nonary-100M
layer shapes at a 16384-token batch. torchao install: <uv add | uv pip>, version <x>.

## Numbers
| shape (N x K, T) | bf16 ms | torchao int8 ms | torchao vs bf16 |
| --- | ---: | ---: | ---: |
| ... | ... | ... | ...x |

## Verdict
GO / NO-GO -- <one paragraph: did a tuned int8 kernel beat bf16; if NO-GO this closes the
speed question definitively (not a torch._int_mm artifact); if GO, recommend the fused custom
kernel as the next step>.
```

- [ ] **Step 2: Correct the roadmap's int8 bullet**

In `docs/ROADMAP.md`, under "Closed: training-speed investigation", update the int8 bullet so it
no longer rests only on `torch._int_mm`. Replace the int8 bullet's conclusion with the
tuned-kernel result: either "tuned int8 (torchao) also fails to beat bf16 -> closed for real"
(NO-GO) or "tuned int8 (torchao) beats bf16 by Nx -> speed reopened; custom fused kernel is the
next step" (GO). Keep the int4 bullet unchanged.

- [ ] **Step 3: Commit**

```bash
git add docs/results/2026-06-21-tuned-int8-gemm-bench.md docs/ROADMAP.md
git commit -m "docs: record tuned int8 GEMM result and update roadmap verdict"
```

---

## Self-Review

**Spec coverage:** dependency (spec "Dependency") -> Task 1 Step 1; benchmark (spec "Component:
the benchmark") -> Task 1; success criteria/decision (spec) -> Task 2; roadmap correction
(spec "Output") -> Task 2 Step 2. Optional bare-`int_mm` comparison is spec-optional and omitted
to keep the decisive high-level test clean. All required items covered.

**Placeholder scan:** none -- the benchmark is complete code; the results note is a
fill-in-the-numbers skeleton (numbers come from running it); the roadmap edit is specified as a
conditional with both branches.

**Type consistency:** shapes `(N, K, T)` consistent; `quantize_(linear, int8_dynamic_activation_int8_weight())`
used consistently; bf16 input throughout.

## Handoff note

Self-contained for an external executor (e.g. Gemini): commands exact, decision bar explicit,
torchao API adaptation flagged. New dep torchao; needs a free CUDA GPU (GPU 1). Branch:
`feat/int8-tuned-bench`.
