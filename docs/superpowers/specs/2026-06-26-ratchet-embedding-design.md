# Ratchet the token embedding — design (2026-06-26)

**Goal.** Close the last FP master-weight carve-out: make the input `token_embedding`
master-weight-free like the rest of the model, and run a controlled A/B to measure whether
forcing the embedding to low-state integer codes costs quality. Keep the current **byte-level
vocab (~205)**; this validates the mechanism at small vocab and is *not* an attempt to crack
large-vocab sparse updates (deferred — see "Out of scope").

## Context

The output `lm_head` is already a `DiscreteRatchetLinear` (master-free). The **only** remaining FP
`nn.Parameter` among the big tables is the input `token_embedding` (`nn.Embedding`, AdamW-trained).
At byte-level the "sparse gradient" obstacle that normally makes embedding-ratcheting hard mostly
evaporates: with vocab 205 and batch 64×256 = 16,384 tokens, essentially every row appears every
step (~80 occurrences each), so the embedding's weight gradient is effectively **dense** and the
existing dense per-row update applies directly. This is the right stepping stone before large vocab.

Value: thesis completeness ("every learnable matrix in the model is master-weight-free") and a
validated per-row-scale + dense-update mechanism to build on when vocab scales. It is *not* a memory
win — at byte-level the embedding is ~0.4% of the model (205×512 ≈ 0.1M params).

ROADMAP item #5 (`docs/ROADMAP.md`).

## Architecture

### `RatchetEmbedding` (in `src/local_ai_training/ratchet.py`)

A sibling of `DiscreteRatchetLinear` that **shares all update internals** — `pack_code_pressure` /
`unpack_code_pressure`, `bucket_pressure`, `_ratchet_update_core`, and the per-row RMS normalization.
It differs only in (a) its forward is an embedding lookup, and (b) how it captures the weight
gradient.

**Persistent state (audit-clean):**
- `packed`: `uint8` buffer, shape `(num_embeddings, embedding_dim)` = `(vocab, n_embd)`. Code in the
  low nibble, pressure in the high nibble (identical layout to the linears).
- `scale`: FP32 buffer, shape `(num_embeddings,)` — one scale per token-row.
- Initialization: quantize the seeded logical embedding init (the same FP init the FP32 control
  would use) by `row_max_abs / max_code` → code; pressure = 0; `scale = row_max_abs / max_code`.
  Identical recipe to `DiscreteRatchetLinear`, so a seed yields one logical init quantized
  consistently across arms.
- **No `nn.Parameter`.** `audit_no_master_weights` must report the embedding as ratchet state bytes,
  not a violation.

**Effective weight (property):** `code.to(scale.dtype) * scale[:, None]` → `(vocab, n_embd)` FP,
exactly as `DiscreteRatchetLinear.effective_weight`.

**Forward:**
```python
eff = (self.code.float() * self.scale[:, None]).detach().requires_grad_(True)  # transient leaf
out = F.embedding(token_ids, eff)
self._effective = eff   # stash to read .grad after backward
return out
```
code/scale are buffers (no grad), so the effective weight cannot accrue a gradient on its own.
Making `eff` a transient **leaf** with `requires_grad_(True)` lets autograd populate `eff.grad` =
the full `(vocab, n_embd)` weight gradient — exactly the signal the ratchet consumes. `eff` is
released (`self._effective = None`) after each update so no persistent FP weight survives the step.

**Update** (folded into the existing `RatchetGPT.ratchet_update()` sweep): per-row-normalize
`eff.grad` (dim over `n_embd`), run `_ratchet_update_core`, move codes one step where pressure
crosses `pressure_threshold`, keep residual pressure, record boundary blocked moves. A row whose
gradient was zero (token absent this step) simply does not move. Returns move-count stats consistent
with `RatchetUpdateStats`.

### Wiring (`src/local_ai_training/model.py`)

Mirror the existing `_linear(max_code)` pattern: `token_embedding = nn.Embedding(...)` when
`max_code is None` (FP32 control, untouched), else `RatchetEmbedding(num_embeddings=vocab_size,
embedding_dim=n_embd, max_code=max_code, ...)`. `RatchetGPT.ratchet_update()` and
`discard_pending_gradients()` must include the embedding alongside the linears.

**Invariant impact:** AdamW no longer sees the embedding (it is no longer a Parameter); it is trained
purely by code moves. The `frozen` controls discard its pending gradient (codes never move) while
still training the remaining FP support (RMSNorm).

## Data flow

forward (lookup on transient effective weight) → loss → `loss.backward()` (fills `eff.grad`) →
`model.ratchet_update()` sweep (normalizes per row, moves codes, releases `eff`) → AdamW steps only
the remaining FP support tensors (RMSNorm). One logical init per seed; both arms quantize it the
same way.

## The A/B experiment

| Arm | Embedding | Everything else |
|---|---|---|
| Control | FP32 `nn.Embedding` (AdamW-trained) | ratchet, codes 15, enwik8 25M |
| Treatment | `RatchetEmbedding`, codes 15 | identical |

- Matched seed 1337, data schedule, steps, eval batches; the only variable is embedding precision.
- Corpus/scale: enwik8 25M codes-15 (continuity with the existing run; a 205-row embedding is a more
  meaningful quantization test than text8's 27 rows).
- Budget: **5k screen** both arms first (~15 min each); promote to **30k** only if the gap is
  interesting (per the project screening-budget rule).
- Both arms run fresh (the control is a clean FP-embedding run, not the earlier resumed one).
- Metric: val-loss gap. Secondary observables: the embedding's own code histogram + saturation %, so
  a quality gap can be attributed (is the embedding rail-pinning like coarse linears did?).
- **Preserve the result either way.** If the treatment lands within seed noise of the control, the
  carve-out closes essentially free. If there is a gap, that quantifies the embedding's quantization
  cost — a real result; do not tune it away.

## Error handling / edge cases

- The update sweep skips a `RatchetEmbedding` with no pending gradient (e.g. an eval-only forward).
- Non-finite-gradient handling mirrors the linears but is **gated** (no per-step `.all()` host sync —
  the width-4096 "hang" lesson: per-step host syncs over large tensors stall the GPU).
- `eff` is cleared every step to honor "release temporary FP weights after each update."
- Effective-weight dtype follows `scale.dtype` (FP32), matching the linears.

## Testing (TDD)

**Unit (`tests/test_ratchet.py` / a new `tests/test_ratchet_embedding.py`):**
- Effective weight equals `code × scale` (per-row).
- `forward(token_ids)` equals `F.embedding(token_ids, effective_weight)`.
- `audit_no_master_weights` reports no violation and counts the embedding's bytes as ratchet state.
- Init recipe matches the linears' (`row_max_abs / max_code`); a fixed seed gives a deterministic
  packed buffer + scale.
- Codes move after accumulated pressure crosses `pressure_threshold`; a zero-gradient row does not
  move.
- Gradient is captured into `_effective.grad`, consumed by the update, and `eff` is released.

**Integration (`tests/test_experiment.py`):**
- One end-to-end train step with `RatchetEmbedding`: audit clean, finite loss, codes move.
- `frozen` control: `discard_pending_gradients()` leaves embedding codes frozen.

**Regression:** the full suite stays green — the FP32 control path (`max_code is None`) is untouched.

## Out of scope (YAGNI / future)

- **Large-vocab sparse updates.** At larger vocab the gradient becomes genuinely sparse (rows update
  rarely; per-row RMS over few samples; pressure/scale staleness). That needs a sparse-aware variant
  (gather only in-batch rows, update only those) — deferred until vocab actually scales.
- **Fused-inline backward** (a custom autograd Function à la `_RatchetMatmul`) — a speed lever, premature.
- **Trainable embedding scale** — fixed per-row scale this round, matching the default config.
- **Weight tying** input↔output — confounds the A/B and changes the baseline; a separate experiment.
