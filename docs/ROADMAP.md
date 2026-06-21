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

## Closed: training-speed investigation (NO-GO)

Two cheap forward spikes settled this for ~a day of work. **Conclusion: there is no cheap
training speedup; the ratchet is a memory technique, not a speed one, for training.** Do not
re-attempt on a false premise.

- **Weight-only int4 dequant-matmul kernel** (custom Triton, `scripts/kernel_prototype/`,
  `docs/results/2026-06-21-forward-kernel-prototype.md`): correct but ~0.6x bf16 cuBLAS.
  Structural, not a tuning gap: at training batch sizes the matmul is **compute-bound**
  (cuBLAS at ~87% of peak) and the weight is reused over thousands of tokens, so reading
  4-bit weights saves negligible bandwidth. Weight-only int4 helps *bandwidth-bound small-batch
  inference*, not compute-bound training.
- **int8 activations + int8 math** (`scripts/int8_spike/`,
  `docs/results/2026-06-21-tuned-int8-gemm-bench.md`): accuracy was fine (~1% error, outliers
  not even a problem). While earlier tests using `torch._int_mm` were slow, a re-benchmark with a tuned int8 kernel (torchao) also fails to beat bf16 (0.24x-0.41x vs bf16) -> closed for real. The int8 tensor-core speedup does not materialize at these shapes on this hardware.

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
