# Agent Guide

## Purpose

This repository tests whether low-state Transformer matrices can learn without
persistent floating-point master weights. Favor correctness, observability, and
reproducibility over performance.

## Non-Negotiable Invariants

- Ratchet matrices persist only integer codes, integer pressure, and explicit row scales.
- Never add an FP32/BF16 `Parameter` that mirrors a ratchet code matrix.
- Temporary floating-point effective weights and gradients are allowed during eager
  forward/backward, but must be released after each ratchet update.
- Quinary and septenary comparisons must share logical initialization, batch schedules,
  evaluation batches, token budgets, and seeds.
- Do not claim this eager implementation accelerates training or stores packed 2.32/2.81
  bits per weight.
- Hugging Face dataset loading must pin a revision and disable remote code.

## Workflow

Use test-driven development for behavior changes. Run before committing:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv run pytest
UV_CACHE_DIR=/games/ailab/.uv-cache uv run ruff check .
UV_CACHE_DIR=/games/ailab/.uv-cache uv run lat audit --model configs/ratchet_tiny.toml
git diff --check
```

Generated datasets, checkpoints, plots, and run logs belong under ignored `data/` and
`runs/` directories. Keep architecture decisions in `docs/` and update the README when
the command surface changes.

The completed ratchet runs are under `runs/tiny-shakespeare`. Run controls separately with
`uv run lat controls --config configs/ratchet_tiny.toml --output runs/controls`; do not
repeat or overwrite the existing six ratchet arms.

## Scientific Reporting

Report all persistent tensor dtypes and byte counts. Separate ratchet state from floating-
point embeddings and normalization. Preserve failed runs and their configurations when
they contain useful evidence; do not silently tune or discard seeds.
