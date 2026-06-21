# Roadmap

Tracks the big-picture trajectory of the master-weight-free ratchet training research.
Detailed results live in `docs/results/`; designs in `docs/superpowers/specs/`.

## Done

- **Trainability** — discrete ratchet matrices learn with no FP master weights; frozen/FP32
  controls confirm the learning comes from code moves, not the FP support params.
- **States-vs-quality curve** (text8, 25M) — 5/7/9-state codes give a clean monotonic dial;
  more states -> lower loss + less saturation. Gains taper; states alone won't reach FP32.
- **Iso-memory (MB-for-MB)** — the right axis for "params per GB". Ratchet wins at small
  budgets (nonary-25M beats FP32-2M at ~22MB); FP32 wins at ~89MB. Caveat: big eager ratchets
  were undertrained at fixed step budgets; convergence runs stopped early.
- **4-bit packed storage** — code+pressure in one uint8 (lossless); ~6x -> ~12x params/GB.
- **torch.compile-fused update** — 3.17x on the isolated update; opt-in `compile_update`.
- **Eager throughput finding** — the ratchet is already ~0.91x FP32 throughput (the "2.3x
  slower" was GPU contention); update fusion is only ~4% end-to-end. Real speedup needs the
  matmul to exploit the low bits, not the update.

## REOPENED: training-speed investigation (the earlier NO-GO was wrong)

**Correction (2026-06-21).** The prior NO-GO was an artifact of trusting *vendor* int8 kernels
(`torch._int_mm` / cuBLASLt IMMA, torchao autotuned Triton) — all of which stall at ~33-42% of
int8 peak on the 3090 at *every* width. A **hand-written autotuned Triton int8 GEMM** reaches
~100% of int8 peak and **delivers the full 2x the silicon spec promises**. The bare-GEMM NO-GO
is dead; the int8 tensor-core speedup is real and reachable — it was a kernel-quality problem,
not a hardware ceiling and not a model-size problem. See
`docs/results/2026-06-21-int8-tuned-kernel-reversal.md`.

- **Bare int8 GEMM** (`scripts/int8_spike/triton_int8_gemm.py`, correctness err=0): **2.10x bf16**
  at width 8192-12288, **1.88x even at K=768**. Vendor kernels (~35% of peak) were the problem.
- **End-to-end fused int8 linear, forward** (`scripts/int8_spike/fused_int8_linear.py`): per-token
  activation quant + int8 GEMM + dequant-fused epilogue. **1.8-1.95x at width 8192-12288**
  (~1.4% rel err), break-even ~2048, but **0.60x at K=768** — the fixed quant overhead only
  amortizes at large width. So the *end-to-end* win is width-gated, switching on exactly where
  the ratchet's memory win lives (frontier-scale models). Bare GEMM wins at all widths.
- **Activation precision** (`scripts/int8_spike/activation_precision.py`): int8 activations
  ~1-6% rel err (per-token); **int4 activations ~62% — dead.** int8 is the activation precision.
- **Ratchet alignment:** the stored `code` (int8-representable, +-7) IS the int8 weight operand;
  the per-row scale IS the weight scale. The ratchet maps onto int8xint8 with no extra weight quant.
- **Full training step (fwd + both backward GEMMs)** (`scripts/int8_spike/int8_backward_bench.py`):
  **1.6-1.7x at width 8192-12288**, crossover ~K=4096, **<1x at toy width** (the x6 quant passes
  swamp small matmuls). Gradients ~1.4% accurate vs bf16. So the per-step speedup holds through
  backward, gated on frontier width.
- **STILL UNPROVEN:** end-to-end *convergence* under int8 activations — per-step grads are ~1.4%
  noisy; does that compound over thousands of steps into worse final loss, or wash out? Needs an
  actual int8-in-the-loop training run vs bf16 (at width >=4096 to also show the speedup). This is
  the last gate before "the ratchet trains faster at scale" is a standing claim.
- **Caveat:** all 3090-specific; A100/H100/FP8 differ.

## Upcoming

### 1. BitNet b1.58 evaluation review
Review/validate the parallel BitNet evaluation harness (`src/local_ai_training/bitnet_*.py`)
as a low-bit reference point. Goal: a clean apples-to-apples baseline (pretrained BitNet
b1.58, ternary 1.58-bit) to compare the ratchet against.

### 2. Fairer iso-memory re-run
Re-run MB-for-MB with `compile_update` and enough steps that the big ratchet models actually
converge, on dedicated GPUs (avoid the contention/undertraining confound). Optionally pair
with algorithmic update-rule improvements (the per-bit gains taper; the update rule, not the
state count, is the likely ceiling).

### 3. (Open question) algorithmic update-rule improvements
The remaining lever for closing the quality gap to FP32 is the pressure/bucket update rule,
not bits or speed. Lower priority unless the memory thesis warrants pushing accuracy.
