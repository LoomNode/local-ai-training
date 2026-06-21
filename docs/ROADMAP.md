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

## Upcoming

### 1. Packed dequant-matmul kernel (the real speed/transient-memory win)
The ratchet cannot beat FP32 in eager mode (it materializes an FP32 weight and runs the same
cuBLAS matmul). A fused dequantize-and-matmul kernel that reads packed int4 codes directly --
never materializing the FP weight -- is what makes it faster AND realizes the memory win at
scale. Existing int4 GEMMs assume frozen inference weights + heavy prepack; ratchet codes
change every step, so a purpose-built Triton kernel (signed int4 + per-row scale, no prepack,
training forward+backward) is the path.
- **First: forward-pass feasibility prototype** -- measure whether a custom int4 dequant-matmul
  (including any per-step repack) actually beats the eager FP32 path before building the rest.
- Then: forward+backward training integration, bit-exact vs the eager reference.

### 2. BitNet b1.58 evaluation review
Review/validate the parallel BitNet evaluation harness (`src/local_ai_training/bitnet_*.py`)
as a low-bit reference point. Goal: a clean apples-to-apples baseline (pretrained BitNet
b1.58, ternary 1.58-bit) to compare the ratchet against.

### 3. Fairer iso-memory re-run (lower priority)
Re-run MB-for-MB with `compile_update` and enough steps that the big ratchet models actually
converge, on dedicated GPUs (avoid the contention/undertraining confound). Optionally pair
with algorithmic update-rule improvements (the per-bit gains taper; the update rule, not the
state count, is the likely ceiling).

### 4. (Gated, large) full packed training loop at scale
Only after the kernel proves out: forward+backward+update all on packed integer state, to push
toward the 100B "most parameters per GB" thesis.
