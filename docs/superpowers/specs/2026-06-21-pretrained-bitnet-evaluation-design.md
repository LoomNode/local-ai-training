# Pretrained BitNet Evaluation Design

## Purpose

Evaluate Microsoft's official `BitNet-b1.58-2B-4T` checkpoint through its optimized,
packed CPU runtime and provide an interactive local chat. This is an inference experiment,
not another ratchet training arm.

## Reproducibility Boundary

Pin the Microsoft runtime commit, Hugging Face model revision, GGUF filename, byte count,
SHA-256, and local toolchain bootstrap. Keep the checkout, build, model, setup manifest,
raw generations, and benchmark evidence under ignored `data/` and `runs/` directories.
Never execute model-supplied Python or use a floating-point conversion of the checkpoint.

## Evaluation

Use the x86 `I2_S` CPU kernel. Preserve deterministic qualitative generations separately
from performance results. Benchmark 128- and 512-token prompt processing plus 128-token
generation at 1, 2, 4, 8, and 16 threads with five repetitions. Refuse CPU-intensive runs
while `lat train` is active unless the operator explicitly allows contention.

Report the packed matrix encoding, total GGUF artifact bytes, peak resident memory, CPU and
toolchain details, and raw prompt/generation throughput. The GGUF size includes metadata and
non-matrix tensors; runtime allocations and KV cache are reported separately. Do not compare
the pretrained model's token metrics directly with character-level text8 validation loss.
