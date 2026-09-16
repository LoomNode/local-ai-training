# Local AI Training

Research code for testing a pressure-ratchet approach to training low-state
weight matrices without persistent full-precision master copies.

The ratchet representation persists codes and pressure packed together in one
`uint8` per weight, plus explicit FP32 row scales and an optional FP32 row RMS EMA,
with audit-visible byte counts and no hidden floating-point matrix
parameters. Nearby work includes QAT/STE, BitNet-style low-bit models, ECO-style
master-weight-free quantized optimization, and memory-efficient optimizer methods;
this repository focuses specifically on the pressure/code-ratchet update mechanism.

This repository initially tested **trainability**, not speed. The current implementation
has a fused tiled backward/update backend and experimental BF16/int8 CUDA matmul paths.
The persistent code and pressure nibbles occupy one `uint8` byte per weight; this is
lossless state packing, not entropy-packed 2.32/2.81-bit storage. Floating-point support
state and temporary compute storage remain separate from that one-byte matrix state.

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
1339. Frozen arms train ordinary floating-point token embeddings (when configured) and RMSNorm
parameters while freezing packed code/pressure, row scale, optional row RMS EMA, leak counters,
ratchet statistics, and ratcheted embeddings. FP32 replaces every ratchet matrix with a bias-free
`nn.Linear` and trains all weights with AdamW.

Two more control/competitor arms share the `--weight-mode` flag. `--weight-mode qat` (`QATLinear`)
is a straight-through-estimator control: it keeps a full-precision master weight trained by AdamW
and quantizes it every forward pass with the ratchet's exact per-row quantizer, so it isolates the
cost of dropping master weights; it deliberately does have a master weight and sits outside
`audit_no_master_weights`' scope. `--weight-mode int8master` (`Int8MasterLinear`) is the iso-state
competitor: an int8 buffer plus one frozen FP32 per-row scale (the same one-byte-per-weight-plus-
row-scale budget the ratchet spends on its packed code/pressure nibble), updated by a stateless,
stochastically-rounded sign step (`--int8-lr`, grid units per step, default 0.1; also `[int8master]
lr` in TOML) instead of the ratchet's pressure accumulator. This arm genuinely *is* an 8-bit master
weight -- unlike the ratchet's nibble-packed code, `audit_no_master_weights` counts it (never as a
violation, since the master is an int8 buffer, not a floating `Parameter`) rather than skipping it,
so its state bytes and layer counts show up in the same audit and `metrics.csv` columns as the
ratchet. See
`docs/superpowers/specs/2026-09-15-int8-master-iso-state-design.md`.

Resume a run when the new configuration has a larger `steps` value:

```bash
uv run lat train --config configs/ratchet_tiny.toml --codes 5 \
  --output runs/quinary-seed-1337 --resume runs/quinary-seed-1337/checkpoint
```

Runs write `metrics.csv`, `checkpoint.safetensors`, `checkpoint.json`, and comparison PNGs.
New version-2 training checkpoints contain model tensors, AdamW tensor state for the small FP
support parameters, CPU RNG, active CUDA RNG, the selected seed, and per-module pressure-leak
counters. Resume validates the tokenizer kind and canonical serialized subword tokenizer before
mutating training state, then restores RNG immediately before continuation. Legacy checkpoints
remain usable for generation. Training resume requires an explicit matching selected `run_seed` in
both format versions; original version-1 checkpoints did not record it and therefore cannot resume,
even for deterministic models, because the seed selects the batch schedule. There is no unsafe
override.

Bit-exact resume additionally needs a repeatable backward pass. The default flash attention
backend accumulates the query gradient with atomic adds, which is bit-different on every pass
once the head size reaches 128 (the 1B recipe: `n_embd = 3072`, `n_head = 24`); head size 64
happens to be repeatable on RTX 3090. Set `deterministic_attention = true` under `[training]` or
pass `--deterministic-attention` to pin the efficient/math backends (about 1.7x slower attention,
a small share of a step). The setting is recorded in the checkpoint and a resume must match it.
See `docs/results/2026-09-13-provenance-reruns.md`.

Two opt-in, zero-new-persistent-state update-rule levers (both default off; see
`docs/superpowers/specs/2026-09-14-update-rule-levers-design.md`): `--stochastic-bucket` replaces
the deterministic round-to-nearest pressure bucket with a stochastic round, so sub-bucket gradient
magnitude accumulates pressure in expectation instead of being discarded. `--pressure-weight FLOAT`
(λ, default 0) blends accumulated pressure into the forward pass's effective weight
(`code + λ × pressure / pressure_threshold`); it requires `matmul_mode` `fp32` or `bf16`.
`--scale-multiplier FLOAT` (default 1.0; `[ratchet] scale_multiplier` in TOML) multiplies each
row's init scale (`row_max × scale_multiplier / max_code`) before quantizing that row's codes
against it, trading code-magnitude headroom for coarser resolution near zero; see
`docs/superpowers/specs/2026-09-15-init-scale-screen-design.md`.

Sample from a checkpoint once it exists:

```bash
uv run lat generate --checkpoint runs/fineweb_25m_subword_1b/checkpoint \
  --prompt "The history of" --max-new-tokens 200
```

For manual conversation testing, use the same checkpoint loader in an interactive loop:

```bash
uv run lat chat --checkpoint runs/fineweb_25m_subword_1b/checkpoint
```

`lat chat` keeps a simple `System`/`User`/`Assistant` transcript and supports `/reset`,
`/quit`, and `/exit`. FineWeb-trained checkpoints are base language models, so this command is
for qualitative probing, not a substitute for instruction tuning. Generation defaults to
`--inference-matmul-mode fp32`; use `checkpoint`, `bf16`, or `int8` to probe native ratchet
inference modes. The int8 generation path uses fixed-shape context windows to avoid Triton
recompiling on every generated token, and `lat chat`/`lat generate` run one startup warmup forward
when int8 inference is active.

Configs may specify either `steps` or `target_tokens` under `[training]`. `target_tokens`
is resolved at run setup with `ceil(target_tokens / (batch_size * block_size))`, so
large-batch experiments can preserve the same sampled-token budget without manual step
math. `lat train --target-tokens N` overrides the config for one-off extensions or
resumes without editing the TOML. Logs still use `step` as the checkpoint/schedule unit.

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

Ratchet matrices have no floating-point master `Parameter`. Each matrix persists one packed
`uint8` code/pressure tensor plus one FP32 scale per output row and, when enabled, one FP32 RMS
EMA value per row. A trainable scale is floating-point support state, not a code-matrix mirror.
Token embeddings and RMSNorm weights are normal floating-point support parameters and are
reported separately.

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
