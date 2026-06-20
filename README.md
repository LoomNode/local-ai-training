# Local AI Training

Research code for testing training methods that avoid persistent full-precision
master copies of low-state weight matrices.

The first experiment compares quinary (`{-2, -1, 0, 1, 2}`) and septenary
(`{-3, ..., 3}`) ratchet weights on character-level Tiny Shakespeare.

This repository initially tests **trainability**, not speed. The eager PyTorch
implementation materializes temporary floating-point effective weights and gradients.
Codes and pressure use `int8`, not packed 2.32/2.81-bit storage. Optimized packed
Triton/CUDA kernels are intentionally out of scope until the update rule learns.

See [the design](docs/superpowers/specs/2026-06-20-ratchet-training-design.md)
for the precision boundary and scientific constraints.

## Install

Install [uv](https://docs.astral.sh/uv/), then create the locked environment:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv sync --extra dev
```

PyTorch chooses CUDA automatically when `device = "auto"`. The smoke configuration is
explicitly CPU-only. Override the cache path if this checkout is moved.

## Dataset

Download the script-free Hugging Face corpus once:

```bash
uv run lat dataset
```

The command pins `SamPIngram/tinyshakespeare` to commit
`6d8bc3fdfca13bf8a128bb0e0914cead1e2d208c` and downloads only `input.txt`.
No Hub Python code is executed. Later runs reuse the Hugging Face cache. A local text file
can be supplied with `--dataset-path`.

## Run

Quick CPU checks:

```bash
uv run lat train --config configs/smoke.toml --codes 5 --output runs/smoke/quinary
uv run lat train --config configs/smoke.toml --codes 7 --output runs/smoke/septenary
uv run lat plot runs/smoke
```

Matched three-seed research comparison:

```bash
uv run lat compare --config configs/ratchet_tiny.toml --output runs/tiny-shakespeare
```

Resume a run when the new configuration has a larger `steps` value:

```bash
uv run lat train --config configs/ratchet_tiny.toml --codes 5 \
  --output runs/quinary-seed-1337 --resume runs/quinary-seed-1337/checkpoint
```

Runs write `metrics.csv`, `checkpoint.safetensors`, `checkpoint.json`, and comparison PNGs.
Checkpoints contain model tensors, AdamW tensor state for the small FP support parameters,
and RNG state. Metadata and vocabulary are validated before loading.

## Audit

Inspect the persistent state boundary without training:

```bash
uv run lat audit --model configs/ratchet_tiny.toml --codes 5
uv run lat audit --model configs/ratchet_tiny.toml --codes 7
```

Ratchet matrices have no trainable PyTorch `Parameter`. They persist `int8` code and
pressure matrices plus one FP32 scale per output row. Token embeddings and RMSNorm weights
are normal floating-point support parameters and are reported separately.

## Metrics And Interpretation

CSV logs include training/validation loss, perplexity, tokens/second, code and pressure
histograms, zero/saturation percentages, positive/negative/blocked moves, state bytes, and
CUDA peak memory when applicable.

Early evidence requires validation loss below its initial/random-character level, code
moves across multiple seeds, and no immediate near-total saturation. A positive result only
justifies investigating packed Triton/CUDA kernels. This eager implementation is expected to
be slower than ordinary BF16/FP32 training.

## Development

```bash
uv run pytest
uv run ruff check .
git diff --check
```

Agent-specific invariants and handoff rules are in [AGENTS.md](AGENTS.md).
