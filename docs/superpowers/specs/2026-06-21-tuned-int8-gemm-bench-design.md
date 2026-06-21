# Tuned int8 GEMM Re-Benchmark

## Goal

The int8 activation spike concluded NO-GO using `torch._int_mm` -- but that is an unoptimized
kernel, so the conclusion may be wrong. This re-benchmark is the decisive test: does a
*properly tuned* int8 GEMM (torchao's int8 path) beat bf16 cuBLAS at our training shapes on the
3090? It reopens (or definitively closes) the training-speed question for ~an hour of work.

The ratchet maps cleanly onto weight-only-int8-already + per-token int8 activations, which is
exactly `int8_dynamic_activation_int8_weight`: the codes are the int8 weight; activations are
dynamically quantized per-token to int8; the matmul runs on int8 tensor cores.

## Scope

In scope: a standalone layer-level benchmark comparing a bf16 linear vs a torchao-int8 linear
(and, if cleanly accessible, torchao's bare autotuned `int_mm` vs `torch._int_mm`), at the
training shapes, producing a results note and a verdict.

Out of scope: model integration, training runs, a custom CUTLASS/Triton kernel (gated on a GO),
fusing the ratchet update into the GEMM.

## Dependency

Add **torchao** (`uv add torchao`). It is PyTorch's quantization library with tuned int8
kernels. If `uv add` fails to resolve, fall back to `uv pip install torchao` and note it.

## Component: the benchmark

Shapes: weight `[N, 768]` for N in {768, 2304, 3072}, input `[16384, 768]` (the nonary-100M
layers at a full batch -- same as prior spikes).

For each shape, build an `nn.Linear(768, N, bias=False)` on CUDA and time the forward of:
1. **bf16** -- `linear.to(torch.bfloat16)`, bf16 input.
2. **torchao int8** -- the same linear quantized with
   `torchao.quantization.quantize_(linear, int8_dynamic_activation_int8_weight())`, bf16 input.
   (Exact import path may vary by torchao version; use the high-level `quantize_` +
   `int8_dynamic_activation_int8_weight` API and adapt the import if needed.)

Timing: warmup, `torch.cuda.synchronize()`, many iterations, ms/call. Report the
`torchao-int8 vs bf16` ratio per shape.

Optionally, if torchao exposes an autotuned int8 matmul (e.g. `torchao.kernel.intmm`), also time
it vs `torch._int_mm` on the raw int8 operands, to separate "tuned kernel" from "quantize
overhead". Skip if the API is unclear -- the high-level comparison is the decisive one.

## Success criteria / decision

- **GO (speed reopens):** torchao int8 >= 1.2x faster than bf16 at the representative shapes ->
  a custom fused Triton/CUTLASS kernel (dequant + ratchet update fused into the GEMM) becomes
  worth building.
- **NO-GO (closed):** torchao int8 still <= bf16 even tuned -> int8 genuinely does not help on
  this hardware at these shapes, and no custom kernel will (they are at the ceiling). The
  speed door is then closed for real, not on an unoptimized-kernel technicality.

Expectation calibration: K=768 is smallish, so even a perfect int8 kernel may land ~1.3-1.6x,
not 2x -- still a GO, just calibrated.

Output: a results note in `docs/results/` with the per-shape ratios and the verdict, and an
update to the roadmap's "Closed" section (correcting the earlier `torch._int_mm`-based
conclusion to reflect the tuned-kernel result).

## Notes for execution

Self-contained for handoff (e.g. Gemini): the comparison and bar are explicit. New dep
torchao; needs a free CUDA GPU (GPU 1). torchao's exact API may shift by version -- the
high-level `quantize_(linear, int8_dynamic_activation_int8_weight())` is the stable entry point;
adapt the import if the path differs and note what was used.
