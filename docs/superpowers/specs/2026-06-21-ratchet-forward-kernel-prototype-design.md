# Ratchet Forward Kernel Prototype (Feasibility Spike)

## Goal

A measurement-only prototype to answer one question: **can a custom Triton dequantize-and-
matmul kernel that reads the ratchet's packed 4-bit codes directly beat the current eager
"materialize FP weight + cuBLAS" forward path?** It is a go/no-go gate -- if the kernel is
faster (with correct output), the full forward+backward training kernel is worth building; if
not, we stop having spent a day instead of weeks.

This is the cheapest way to de-risk the one genuine unknown ("can a custom matmul beat
cuBLAS?") before committing to the tedious-but-known work (backward, autograd, integration).

## Background

The eager ratchet forward materializes a full FP weight (`code * scale`) and runs the same
dense matmul FP32 training uses -- so low-bit codes save *storage*, not *compute*, and the
ratchet trains at ~0.91x FP32 throughput (see `docs/results/2026-06-21-eager-throughput.md`).
Genuine speedup requires the matmul itself to read packed codes and never materialize the FP
weight. The win is *weight-only* quantization: 4-bit weights, bf16 activations, fp32
accumulation -- the standard GPTQ/AWQ regime. (Activations stay bf16 because their outlier
distribution does not survive 4-bit; the persistent-memory prize is the weights.)

## Scope

In scope: a standalone Triton forward kernel + a benchmark/correctness harness, run on one
GPU, producing a results note and a verdict.

Out of scope (gated on a GO): the backward pass, autograd `Function` wiring, integration into
`DiscreteRatchetLinear.forward`, and the ratchet-update interplay.

## Component: the kernel

`ratchet_forward(packed, scale, x, max_code) -> Tensor`

- Inputs: `packed` uint8 `[out_features, in_features]` (the existing packed format -- code in
  the low nibble), `scale` fp32 `[out_features]`, `x` bf16 `[tokens, in_features]`.
- Output: bf16 `[tokens, out_features]`.
- Computes `out[t,o] = scale[o] * sum_i x[t,i] * code(packed[o,i])`, where
  `code = (packed & 0x0F) - max_code`.
- Implementation: a standard tiled Triton matmul with two twists. (1) The weight operand is
  loaded as packed uint8 and unpacked to bf16 in-register (`(packed & 0xF) - max_code`),
  never materializing a full weight matrix -- this is the bandwidth win (0.5 bytes/weight
  read vs 2-4). (2) The per-row `scale` factors out of the sum, so it multiplies the fp32
  accumulator before the bf16 store. Multiply via `tl.dot` in bf16, accumulate in fp32.

Lives under `scripts/kernel_prototype/` -- deliberately not wired into the package, since this
is a spike.

## Component: the benchmark + correctness harness

Baselines per shape:
- **bf16-eager** (apples-to-apples): `effective = (code.to(torch.float32) * scale[:, None]).bfloat16(); x @ effective.t()`.
- **fp32-eager** (current path, for context): same with fp32.

Shapes (nonary-100M layers at a full batch): weight `[out, 768]` for out in {768, 2304,
3072}; activation `[16384, 768]` (64 x 256 tokens). Multiple shapes so the result is not a
single-point fluke.

Timing: warmup, `torch.cuda.synchronize()`, many iterations, report ms/call. Also report peak
memory (the kernel never allocates the `[out,in]` weight, so it should save transient memory).

Correctness gate: kernel output vs the **bf16-eager** output, max relative error within bf16
tolerance (~1e-2). Same-precision comparison, so agreement should be tight. Tested across the
shapes and both nibble ranges (max_code 4 and 2). A fast-but-wrong kernel fails.

## Success criteria / decision

GO if the kernel beats bf16-eager by >= 1.2x at the representative shapes with output within
tolerance. Marginal or slower -> NO-GO (or "needs tuning"). Output: a results note in
`docs/results/` with the timings, memory, correctness, and the verdict.

## Testing

A correctness test (kernel matches bf16-eager within tolerance, across shapes and max_code
in {2, 4}) plus the benchmark script. Standalone -- does not touch the existing test suite.

## Notes for execution

Self-contained and suitable for handoff to an external executor (e.g. Gemini): the kernel,
the harness, and the success bar are fully specified here. Requires Triton (bundled with the
torch in this environment) and a free CUDA GPU.
