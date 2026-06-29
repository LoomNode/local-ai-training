# Local AI Training

Research code for testing a pressure-ratchet approach to training low-state
weight matrices without persistent full-precision master copies.

The ratchet representation persists integer codes, integer pressure, and explicit
row scales, with audit-visible byte counts and no hidden floating-point matrix
parameters. Nearby work includes QAT/STE, BitNet-style low-bit models, ECO-style
master-weight-free quantized optimization, and memory-efficient optimizer methods;
this repository focuses specifically on the pressure/code-ratchet update mechanism.

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

Stream a pinned FineWeb-Edu `sample-10BT` shard into local `uint16` token IDs:

```bash
uv run lat shard fineweb-edu \
  --target-tokens 1000000000 \
  --output data/fineweb_edu_sample10bt_1b
```

The shard command uses Hugging Face streaming with remote code disabled, trains an 8K byte-BPE
tokenizer from the first streamed rows, writes `tokens.uint16` plus `metadata.json`, and records
the dataset revision, tokenizer JSON/hash, row counts, and actual token count. Train from the local
shard by passing the metadata path:

```bash
uv run lat train --config configs/rtx3090_optimized_25m.toml \
  --dataset-path data/fineweb_edu_sample10bt_1b/metadata.json
```

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

Run only the nine control runs without repeating completed ratchet arms:

```bash
uv run lat controls --config configs/ratchet_tiny.toml --output runs/controls
```

This produces FP32, frozen-quinary, and frozen-septenary arms for seeds 1337, 1338, and
1339. Frozen arms still train embeddings and RMSNorm parameters, but discard ratchet
gradients without changing codes. FP32 replaces every ratchet matrix with a bias-free
`nn.Linear` and trains all weights with AdamW.

Resume a run when the new configuration has a larger `steps` value:

```bash
uv run lat train --config configs/ratchet_tiny.toml --codes 5 \
  --output runs/quinary-seed-1337 --resume runs/quinary-seed-1337/checkpoint
```

Runs write `metrics.csv`, `checkpoint.safetensors`, `checkpoint.json`, and comparison PNGs.
Checkpoints contain model tensors, AdamW tensor state for the small FP support parameters,
and RNG state. Metadata and vocabulary are validated before loading.

Configs may specify either `steps` or `target_tokens` under `[training]`. `target_tokens`
is resolved at run setup with `ceil(target_tokens / (batch_size * block_size))`, so
large-batch experiments can preserve the same sampled-token budget without manual step
math. Logs still use `step` as the checkpoint/schedule unit.

### Matmul Precision

Ratchet configs may opt into a linear-matmul backend under `[training]`:

```toml
matmul_mode = "fp32"     # "fp32", "bf16", or "int8"
int8_backward = false    # int8 grad_input (with stochastic rounding); requires matmul_mode="int8"
```

`fp32` is the default and preserves the existing CPU-capable eager path. `bf16` and `int8`
are CUDA-only experimental paths and fail at setup when CUDA is unavailable; neither silently
falls back to FP32. The int8 path uses integer ratchet codes directly for the linear GEMMs,
int32 accumulation, and BF16 dequantized outputs in forward and backward without adding a
floating-point master matrix.

`int8_backward` additionally runs the input-gradient GEMM in int8 (folding the per-row weight
scale into the gradient, then a stochastic-rounded int8 quant against the persistent codes); the
weight-gradient GEMM stays BF16. It converges on par with the BF16 backward (within seed noise)
but is **slower per token than plain int8** — a preserved negative result, not a recommended mode.
For throughput, prefer `compile_update = true`; int8's real advantage is memory, letting models
train at widths where dense BF16 OOMs, not per-token speed. See
`docs/results/2026-06-24-int8-per-token-speed.md`.

Matched convergence experiments must compare `bf16` against `int8` with identical seeds,
initialization, batch/evaluation schedules, and token budgets. This isolates int8 quantization
from the separate FP32-to-BF16 precision change. A checkpoint can only resume with the same
`matmul_mode`; changing it would combine experimental conditions in one logical run and is
rejected.

## Audit

Inspect the persistent state boundary without training:

```bash
uv run lat audit --model configs/ratchet_tiny.toml --codes 5
uv run lat audit --model configs/ratchet_tiny.toml --codes 7
```

Ratchet matrices have no trainable PyTorch `Parameter`. They persist `int8` code and
pressure matrices plus one FP32 scale per output row. Token embeddings and RMSNorm weights
are normal floating-point support parameters and are reported separately.

## Pretrained BitNet Inference

The repository also provides a separate evaluation harness for Microsoft's official
`BitNet-b1.58-2B-4T` checkpoint. This is packed ternary CPU inference through the external
`bitnet.cpp` runtime; it is not a ratchet training arm and its language-model metrics are
not directly comparable to this repository's character-level validation losses.

Provision the pinned runtime, model, and project-local build tools under ignored `data/`:

```bash
uv run python scripts/bitnet_eval.py setup
uv run python scripts/bitnet_eval.py doctor
```

After other training processes finish, run the deterministic qualitative prompts and the
full CPU benchmark. Both commands refuse to contend with an active `lat train` process by
default and write timestamped evidence under ignored `runs/bitnet/`:

```bash
uv run python scripts/bitnet_eval.py smoke
uv run python scripts/bitnet_eval.py benchmark
```

Start an interactive 4096-token conversation with eight CPU threads:

```bash
uv run python scripts/bitnet_eval.py chat
```

Use `--system-prompt` to replace the default assistant instruction. `--allow-contention`
exists for deliberate overrides, but results collected while training is active should not
be used as clean performance evidence.

## Metrics And Interpretation

CSV logs include training/validation loss, perplexity, tokens/second, token-budget progress,
code and pressure histograms, zero/saturation percentages, positive/negative/blocked moves,
state bytes, and CUDA peak memory when applicable.

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
