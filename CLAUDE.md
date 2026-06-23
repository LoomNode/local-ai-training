# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A research experiment testing whether tiny Transformer matrices can learn with **no persistent floating-point master weights** — only low-state integer codes. It tests *trainability*, not throughput. The eager PyTorch implementation materializes temporary FP effective weights/gradients each step; packed low-bit kernels are intentionally out of scope until the update rule is shown to learn. Read `docs/superpowers/specs/2026-06-20-ratchet-training-design.md` for the precision boundary and update rule.

## Commands

The environment uses `uv` with a shared cache. Prefix commands as needed:

```bash
UV_CACHE_DIR=/games/ailab/.uv-cache uv sync --extra dev   # one-time setup
uv run pytest                          # full test suite
uv run pytest tests/test_ratchet.py    # single file
uv run pytest tests/test_ratchet.py::test_name   # single test
uv run ruff check .                     # lint (line-length 100, rules E/F/I/UP/B)
git diff --check                        # whitespace check (part of pre-commit gate)
uv run lat audit --model configs/ratchet_tiny.toml   # assert no master weights exist
```

The `lat` CLI (entry point `local_ai_training.cli:main`) exposes: `dataset` (download pinned corpus), `train` (one arm), `compare` (matched quinary+septenary arms), `controls` (FP32 + frozen arms), `plot`, `audit`. See README for full invocations. Use `uv run lat dataset` once before training.

**Step budget.** Default to **5k steps for screening/iteration** (`configs/scaleup_text8_25m_5k.toml`) — it clears the early transient window (effects can show the *wrong sign* before ~step 3000) at ~6× the speed of a full run. Reserve **30k** (`configs/scaleup_text8_25m_30k.toml`) only for final converged magnitudes and iso-comparison against the stored 30k baselines in `runs/text8-25m-qat/`. Pure *throughput* work needs neither — use the `scripts/int8_training_throughput.py`-style microbench (~30 timed steps). Rationale: `docs/results/2026-06-23-adaptive-scale-ratchet.md`.

## Non-Negotiable Invariants

These are scientific constraints, not style preferences. Violating them invalidates results:

- Ratchet matrices persist **only** `int8` code, `int8` pressure, and one FP32 scale per output row. Never add an FP32/BF16 `Parameter` that mirrors a code matrix. `audit_no_master_weights` (run via `lat audit`) enforces this at runtime — keep it passing.
- Temporary FP effective weights/gradients during eager forward/backward are fine, but must be released after each ratchet update.
- Quinary (codes `[-2,2]`) and septenary (`[-3,3]`) arms must share logical initialization, batch schedule, eval batches, token budget, and seeds. A seed yields one logical FP init per matrix; each arm quantizes it with `row_max_abs / max_code`.
- Never claim this implementation accelerates training or stores packed 2.32/2.81-bit weights.
- Hugging Face dataset loading must pin a revision and disable remote code.
- Preserve failed runs and their configs when they hold evidence; do not silently tune or discard seeds.

## Architecture

Single package `src/local_ai_training/`. Data flow: `config` → `model` (built per seed) → `train` → `metrics`/`checkpoint`/`plotting`.

- **`ratchet.py`** — the core. `RatchetLinear` holds the integer state; `bucket_pressure` converts RMS-normalized gradients into integer pressure (descent direction); the post-backward update accumulates pressure and, at `pressure_threshold`, moves a code one step keeping residual pressure. Boundary outward-moves are recorded as **blocked moves** so pressure can't wind up. `audit_no_master_weights` walks a model and reports violations + state bytes.
- **`model.py`** — `RatchetGPT`: bias-free linears, fixed sinusoidal positions, RMSNorm. `_linear(max_code=None)` produces a plain `nn.Linear` (used by the FP32 control); a non-None `max_code` produces a `RatchetLinear`. `ratchet_update()` applies the update across all ratchet layers and returns `RatchetUpdateStats`; `discard_pending_gradients()` is how frozen controls drop ratchet grads while still training embeddings/RMSNorm.
- **`train.py`** — `train_run` is the loop; AdamW only ever sees the small FP support tensors (embeddings, norms). Writes `metrics.csv` per step, fails on non-finite loss/scale.
- **`config.py`** — `ExperimentConfig.from_toml` parses configs in `configs/` (`smoke.toml` is CPU-only; `ratchet_tiny.toml` is the seeds 1337/1338/1339 research config). `.model_config(vocab_size=...)` bridges to `ModelConfig`.
- **`checkpoint.py`** — safetensors tensors + validated JSON metadata + RNG state; vocab and metadata validated before load. `data.py` builds the char corpus with a deterministic final-10% validation split.

The three control arms (`lat controls`): **FP32** swaps every ratchet matrix for a trained `nn.Linear`; **frozen-quinary/septenary** train embeddings + RMSNorm but discard ratchet gradients (codes never move).

## Artifacts & Workflow

- Generated datasets, checkpoints, plots, and run logs live under git-ignored `data/` and `runs/`. Completed ratchet runs are in `runs/tiny-shakespeare`; **do not overwrite** the six existing ratchet arms — run controls into a separate `--output`.
- Use TDD for behavior changes. Architecture decisions go in `docs/`; update the README when the command surface changes.
- `AGENTS.md` holds the authoritative handoff rules and mirrors the invariants above.
