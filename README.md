# Local AI Training

Research code for testing training methods that avoid persistent full-precision
master copies of low-state weight matrices.

The first experiment compares quinary (`{-2, -1, 0, 1, 2}`) and septenary
(`{-3, ..., 3}`) ratchet weights on character-level Tiny Shakespeare.

This repository initially tests **trainability**, not speed. The eager PyTorch
implementation materializes temporary floating-point effective weights and gradients.
Codes and pressure use `int8`, not packed 2.32/2.81-bit storage. Optimized packed
Triton/CUDA kernels are intentionally out of scope until the update rule learns.

Detailed setup and experiment commands are added with the runnable implementation.
See [the design](docs/superpowers/specs/2026-06-20-ratchet-training-design.md)
for the precision boundary and scientific constraints.

