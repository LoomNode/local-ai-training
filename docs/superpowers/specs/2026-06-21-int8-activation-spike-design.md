# Int8 Activation Speed/Accuracy Spike

## Goal

The forward-kernel prototype showed weight-only 4-bit gives no training speedup (compute-bound
regime; cuBLAS near peak -- see `docs/results/2026-06-21-forward-kernel-prototype.md`). The
only remaining lever for a real *training* speedup is cheaper math: int8 tensor-core ops,
which on the 3090 run ~2x bf16. That requires quantizing **activations** to int8 -- and the
open question is whether activation quantization wrecks accuracy (the outlier problem).

This is a cheap screen: for one ratchet layer at training shapes, measure (A) whether the
int8 pipeline is actually faster than bf16 net of quantization overhead, and (B) how much
int8-activation quantization perturbs the output. It gates the expensive next step (a full
ratchet training run with int8 activations) -- if the screen fails, we stop having spent
hours; if it passes, a training run is justified.

The ratchet is well-suited: the weights (codes, +-max_code) are already int8, so only the
activations need quantizing.

## Scope

In scope: a standalone forward-only harness (speed benchmark + per-layer accuracy screen +
a correctness test), one GPU, producing a results note and a verdict.

Out of scope (gated on a GO): the full training run, model integration, int4 activations,
the backward pass, and SmoothQuant-style outlier handling.

## Component: the int8 pipeline

`int8_ratchet_forward(packed, code_scale, x, max_code, per_token=True) -> Tensor`

- Inputs: `packed` uint8 `[N, K]`, `code_scale` fp32 `[N]` (the ratchet's per-row scale),
  `x` bf16 `[T, K]`.
- Steps:
  1. Unpack codes: `code_int8 = (packed & 0x0F).to(int8) - max_code`  (shape `[N, K]`).
  2. Quantize activations to int8:
     - per-token: `x_scale = x.abs().amax(dim=1, keepdim=True) / 127` (shape `[T, 1]`)
     - per-tensor: `x_scale = x.abs().amax() / 127` (scalar)
     - `x_int8 = torch.clamp(torch.round(x / x_scale), -127, 127).to(torch.int8)`
  3. int8 matmul: `acc = torch._int_mm(x_int8, code_int8.t().contiguous())`  -> `[T, N]` int32.
  4. Dequant (both scales factor out of the K-sum):
     `out = acc.to(torch.float32) * x_scale * code_scale[None, :]`  -> cast to bf16.
- Returns bf16 `[T, N]`.

Both scales factoring out is the reason per-token (per-T) activation scale and per-row (per-N)
weight scale are matmul-compatible: neither lives on the reduced K dimension.

Lives under `scripts/int8_spike/`. Not wired into the package.

## Component: benchmark + accuracy screen

Shapes (nonary-100M layers at a full batch): weight `[N, 768]` for N in {768, 2304, 3072},
activations `[16384, 768]`, max_code 4.

Baseline: bf16-eager (`effective = (code.float()*scale).bfloat16(); x @ effective.t()`).

**Gate A -- speed:** time the full int8 pipeline (quantize + `_int_mm` + dequant) and the bare
`_int_mm` alone, vs bf16-eager. Warmup, `cuda.synchronize()`, many iters, ms/call. Pass if the
full int8 pipeline is >= 1.2x faster than bf16-eager. The bare-`_int_mm` number shows whether
quantization overhead is the bottleneck.

**Gate B -- accuracy screen:** relative error of int8-path output vs the bf16 reference, for
both per-token and per-tensor scaling. Read: per-token error in low single-digit % is
encouraging; tens of % means outliers dominate. The per-tensor number shows how much per-token
rescues. This is a screen, not proof -- compounding over depth/training is what the gated
training run tests.

## Success criteria / decision

GO to the expensive training-run accuracy test only if BOTH gates pass: int8 pipeline >= 1.2x
faster than bf16-eager AND per-token output error is small (low single-digit %). Either fails
-> NO-GO, recorded with the reason (too slow / quant overhead, or outliers too lossy). Output:
a results note in `docs/results/` with the timings, errors (per-token and per-tensor), and the
verdict.

## Testing

A correctness test isolating "is the dequant math right" from "how lossy is quantization":
construct activations and codes that quantize *exactly* (e.g. integer-valued activations within
int8 range, unit `code_scale`), and assert the int8-path output equals the float reference
bit-for-bit. This proves the pipeline math is correct independent of quantization error. Plus
the benchmark/accuracy script.

## Notes for execution

Self-contained for handoff (e.g. Gemini): all formulas specified, `torch._int_mm` is in-tree
(no custom kernel), success bar explicit. Requires a free CUDA GPU (GPU 1 here). `torch._int_mm`
needs int8 contiguous inputs and dimensions that are multiples of the tensor-core tile (the
chosen shapes satisfy this); if a shape is rejected, pad K/N up to a multiple of 16 and note it.
