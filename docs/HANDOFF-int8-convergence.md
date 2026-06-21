# Handoff: int8 training path — convergence test

**For Codex (cold start).** Read `CLAUDE.md` and `AGENTS.md` first for invariants. This handoff is
self-contained. Git identity is already `LoomNode` — never expose any real name/email. Run cmds with
`MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run ...`. 2x RTX 3090; use
`CUDA_VISIBLE_DEVICES` to pick an idle one (`nvidia-smi`).

## Context (what just happened)

The "int8 gives no training speedup" NO-GO was **reversed** this session (PR #3, branch
`feat/int8-kernel-reversal`, may still be open — check `gh pr view 3`). A hand-written autotuned
Triton int8 GEMM hits ~100% of int8 peak: **2.10x bf16** bare, **1.6-1.7x** for a full training step
(fwd + both backward GEMMs) at width >=8192, gradients ~1.4% accurate. Crossover ~K=4096; toy widths
(our 25M, K~512-2048) get NO speedup but still let us test accuracy. int4 activations are dead (~62%
err); int8 is the activation precision. Full writeup: `docs/results/2026-06-21-int8-tuned-kernel-reversal.md`.

Reference kernels/benches (standalone, NOT yet wired into training):
`scripts/int8_spike/triton_int8_gemm.py` (bare GEMM), `fused_int8_linear.py` (fwd, per-token quant +
fused dequant epilogue), `int8_backward_bench.py` (full step + grad accuracy), `activation_precision.py`.

## The ONE open question

Per-step int8 gradients are ~1.4% noisy. **Does that compound over thousands of steps into worse final
loss, or wash out like benign noise?** Single-step error does NOT answer this — must actually train.

## Task (do in this order)

1. **Wire an int8 matmul path into the linear layer**, behind a flag, forward AND backward. The model's
   linears live in `src/local_ai_training/model.py` (`_linear`, `RatchetLinear`) and `ratchet.py`.
   Reuse the int8 mm helper from `scripts/int8_spike/int8_backward_bench.py` (`int8_mm`: quantize both
   operands per-row/per-col, Triton int8 GEMM, fused dequant -> bf16). Key alignment: the ratchet's
   stored `code` (int8-representable, +-7) IS the int8 weight operand; the per-row scale IS the weight
   scale — so no extra weight quant is needed. Use a custom autograd.Function so backward also uses int8.
   USE TDD (project rule). Do NOT add any FP32/BF16 Parameter mirroring a code matrix (`lat audit` must
   stay clean — invariant).

2. **Small convergence test** (cheap, minutes): same config twice — bf16 matmuls vs int8 matmuls in the
   loop — on `configs/scaleup_text8_25m.toml` or a smaller smoke config, same seed. Compare validation
   loss curves. This answers convergence, NOT speed (toy width is below the crossover — expected).
   Write results to `docs/results/`.

3. **Only if int8 converges ~= bf16:** a wide run (n_embd >=4096, K>=4096) to show speed + accuracy
   together. This is the expensive one; don't do it until step 2 passes.

## Guardrails

- Preserve the existing 6 ratchet runs in `runs/tiny-shakespeare` — do not overwrite (write elsewhere).
- All numbers so far are 3090-specific.
- If int8 diverges in step 2: that's a real finding — record it, don't silently tune. The fix space is
  per-token vs per-tensor scales, keeping some ops in bf16, or stochastic rounding.
- Brainstorm -> spec -> plan before the model.py change (it's real feature work on the training loop).
