# Ratcheting the token embedding — A/B (2026-06-26)

**TL;DR.** Making the input `token_embedding` master-weight-free (integer codes + per-row scale,
no FP `Parameter`) costs **~0.018 nats at 5k** (val 1.1828 → 1.2008) on enwik8 25M codes-15. The
embedding tolerates ~15 states cleanly — its saturation matches the rest of the model — and the
ratcheted-embedding model still generates legible text. This **closes the last FP master-weight
carve-out at byte-level** and validates the per-row-scale + dense-update mechanism. The cost is
small but real, so the flag stays **opt-in** (`ratchet_embedding`, default False) pending a 30k
confirmation. Large-vocab sparse updates remain future work.

## Setup

Matched arms, the only variable is the input embedding's precision: seed 1337, enwik8 (byte-level,
vocab 205), 25M model (n_layer 8, n_head 8, n_embd 512), codes 15, 5k steps, identical data
schedule and eval batches. Control = FP32 `nn.Embedding` (AdamW-trained); treatment =
`RatchetEmbedding` (`--ratchet-embedding`). Both ratchet every other matrix (including the output
`lm_head`, which was already master-free).

## Result

| Arm | Embedding | val loss (5k) |
|-----|-----------|---------------|
| Control | FP32 `nn.Embedding` | 1.1828 |
| Treatment | `RatchetEmbedding` (codes 15) | 1.2008 |

**Optimization/representation cost of ratcheting the embedding: +0.0179 nats** (~1.5% relative).

### Secondary observables

- **The embedding is now ratchet state, not FP support.** `ratchet_weights` rises by 104,960
  (= 205 × 512, the embedding) and `support_parameter_bytes` drops 454,656 → 34,816 — the embedding
  left the AdamW-trained FP set. The treatment checkpoint has **no** `optimizer::token_embedding`
  tensor and only `model::token_embedding.packed` (uint8 code+pressure) + `model::token_embedding._scale`
  (FP32 per-row). No `.weight` `Parameter` anywhere → master-free, confirmed.
- **The embedding tolerates ~15 states.** Model-wide saturation is 19.22% (control) vs 19.57%
  (treatment) — nearly identical. The embedding rows do not rail-pin more than the transformer
  matrices, so the small loss gap is not a saturation failure; codes 15 has enough resolution for
  the input representation.
- **It still writes.** Treatment sample (`[[History of `, temp 0.8): *"[[History of Austin American
  Representatives]] *[http://www.clemening.com/ Clemening] Delivery Language Company ..."* — wiki
  links, URLs, and structure, indistinguishable in kind from the FP-embedding model.

## Why this works at byte-level (and what's deferred)

At vocab 205 with batch 64×256 = 16,384 tokens, essentially every embedding row appears every step
(~80 occurrences each), so the embedding's weight gradient is effectively **dense** and the existing
per-row dense ratchet update applies directly — `RatchetEmbedding` is just a `DiscreteRatchetLinear`
(rows = tokens) with an `F.embedding` forward. This validates the mechanism and closes the carve-out
*at small vocab*. It does **not** solve large-vocab sparse updates: at a subword/word vocab the
gradient becomes genuinely sparse (rows update rarely; per-row RMS over few samples; pressure/scale
staleness), which needs a sparse-aware variant. That is the remaining work before large vocab.

## Decision

- **Keep `ratchet_embedding` opt-in (default False).** The ~0.018-nat cost is small but real at 5k;
  do not flip the default until a 30k run shows whether it shrinks (the ratchet converges slower than
  AdamW, so the screening gap is an upper bound) or holds. Preserve the result either way — do not
  tune it away.
- **Thesis status:** every learnable matrix in the model can now train master-weight-free, including
  the input embedding, at a quantified small cost. ROADMAP #5 mechanism: **done at byte-level**;
  large-vocab sparse updates: open.

CAVEAT: single seed, 5k screen. A 30k converged re-run and a second seed would firm the magnitude and
whether the gap is within seed noise.
