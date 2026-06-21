# Eager Training Throughput: the Ratchet Is Already Near FP32 (and the Speed Win Lives in the Matmul)

## Measurement

25M-param model, text8, batch 64 x block 256, single uncontended RTX 3090, real vectorized
batching. Tokens/second over 40 timed training steps after warmup:

| Arm | tok/s | vs FP32 |
| --- | ---: | ---: |
| FP32 | 90,500 | 1.00x |
| Ratchet (eager) | 82,774 | 0.91x |
| Ratchet (eager + compile_update) | 85,952 | 0.95x |

## Findings

1. **The ratchet is already ~0.91x FP32 throughput, not 2.3x slower.** The earlier "2.3x"
   figure was GPU contention (multiple runs sharing two GPUs in the Gemini experiments), not
   an intrinsic cost of the update.

2. **The `torch.compile` update fusion gives only ~4% end-to-end** (0.91x -> 0.95x), despite
   being 3.17x faster on the isolated update kernel. The update is only ~6% of a training
   step; the two matmuls dominate. Amdahl caps the whole-step gain.

3. **The ratchet cannot beat FP32 in the eager path, by construction.** The forward
   materializes a full FP32 effective weight (`code * scale`) and runs the same FP32 cuBLAS
   matmul as FP32 training, then does extra work (materialization + update). Low-bit codes
   save *storage*, not *compute*, in eager mode. Best case it approaches FP32 (~0.95x), never
   exceeds it.

## Implication

Genuine speedup (beating FP32) requires the matmul itself to exploit the low bits: a fused
dequantize-and-matmul kernel that reads packed int4 codes directly and never materializes the
FP weight. Existing int4 GEMMs (tinygemm/Marlin/GPTQ) assume *frozen* inference weights and a
heavy prepack; ratchet codes change every step, so a purpose-built Triton kernel (signed int4
+ per-row scale, no prepack, training forward+backward) is the path. That is the next, larger
effort -- gated on a forward-pass feasibility prototype that measures whether a custom int4
dequant-matmul actually beats the eager FP32 path.

`compile_update` is kept as an opt-in flag (off by default): correct, ~4% when enabled, and
harmless when not.
