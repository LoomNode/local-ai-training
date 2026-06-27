# Subword + sparse-update ratchet embedding → polished PoC — Design (2026-06-26)

## Goal

Train a properly-converged, interactible, **fully master-weight-free** 25M model on enwik8 using
an **8K BPE subword vocabulary** with a **ratcheted (master-free) input embedding** driven by a
**sparse-aware per-row update rule**. Ship it with a working `generate` path. Alongside, run a
matched A/B that quantifies the sparse update's cost vs an FP embedding at subword sparsity.

This closes the project's thesis at a useful scale: every learnable matrix — including the input
embedding — trains with no persistent floating-point master weights, while the model reads/writes
subword tokens like a standard small LM.

## Background / motivation

- The input `token_embedding` is the last FP master-weight carve-out; `lm_head` and every transformer
  matrix are already ratcheted.
- At byte-level (~205 vocab) the ratchet embedding works with the **dense** per-row update and costs
  ~0.017 nats (measured, holds to 30k — see `docs/results/2026-06-26-ratchet-embedding-ab.md`). But
  byte-level is wasteful (one char/token) and is not where the ratchet embedding's memory win matters.
- Subword vocab is the standard route to a coherent small LM (≈4 chars/token → ~4× more text per
  context window, ~4× fewer transformer steps per unit of corpus). It is also exactly what makes the
  embedding gradient **sparse**: at 8K vocab with batch 64×256 = 16,384 tokens/step, a row averages
  ~2 hits/step (rare tokens: zero for many steps), so the dense-update assumption breaks. **Raising
  the vocab is what forces the sparse-update work** — this project does that work.
- The ratchet embedding's memory advantage (~12× vs an FP embedding incl. AdamW moments) is modest at
  8K (~4 MB vs ~48 MB) but is the reason the mechanism matters at large vocab (50K+: ~25 MB vs
  ~300 MB). 8K is the tractable first sparsity regime to validate the sparse update.

## Non-negotiable invariants (carry from CLAUDE.md / AGENTS.md)

- Ratchet matrices persist ONLY int8 code, int8 pressure (nibble-packed), and one FP32 scale/row.
  No FP/BF16 `Parameter` mirroring a code matrix. `lat audit` must stay violation-free.
- The subword ratchet-embedding model must have **zero** FP master weights: no
  `optimizer::token_embedding`, only `model::token_embedding.packed` + `model::token_embedding._scale`.
- Quinary/septenary matching rules, pinned/reproducible corpus discipline, no remote code.
- Never claim acceleration or packed sub-byte storage. Preserve failed runs/configs and the measured
  A/B magnitudes — do not tune them away.
- The existing **char path stays untouched and fully backward-compatible**; byte-level baselines and
  tests remain valid.

## Section 1 — Core research: sparse-aware per-row update

### The bug

`RatchetLinear._normalize` (ratchet.py ~540-548) updates the per-row RMS EMA for **every row in
range**, including rows with zero gradient this step:

```python
ema = where(ema == 0, ms, beta*ema + (1-beta)*ms)   # ms == 0 for non-firing rows
```

For a rarely-firing subword row this decays the EMA toward zero on every step it does **not** appear;
when it finally fires, `gradient / sqrt(ema)` explodes into a spurious maxed-out pressure step. This
is the sparse degradation. (With `rms_ema_beta == 0`, the failure is different but also fatal: a row
firing once normalizes by its single sample → normalized gradient is always ±1 → every appearance
forces a unit code step regardless of magnitude.)

### The fix

The per-row RMS EMA must update **only on rows that fired this step** (`ms > 0`). Non-firing rows:

1. retain their EMA untouched (stable normalization constant across sparse appearances),
2. contribute exactly zero pressure (zero gradient → zero bucketed pressure → no move),
3. never move a code spuriously.

A rare row then accumulates pressure across the steps it fires (pressure is already persistent
integer state) and moves a code only at `pressure_threshold` — the dense behavior stretched over
time. `pressure_leak_period` remains available to bleed stale pressure on sign-flipping rows.

This is a surgical change to `_normalize` (mask the EMA update to firing rows) plus turning
`rms_ema_beta > 0` on for the embedding by default.

### Acceptance (TDD)

- A row that fires every N steps reaches the **same code** as a row firing every step, given equal
  total accumulated gradient (within one code step).
- A non-firing row never moves and its EMA is unchanged across non-firing steps.
- A firing row's EMA updates only from its own samples (masked update verified).

## Section 2 — Pure-Python BPE tokenizer (zero new dependencies)

New module `src/local_ai_training/tokenizer.py`, hand-rolled byte-level BPE (~150 lines), no new
dependency:

- `train_bpe(text, vocab_size=8000) -> BpeTokenizer` — repeatedly merge the most frequent adjacent
  pair until `vocab_size` is reached. Trained on a **~10 MB slice of the train split only** (never the
  validation tail). Vocab quality at 8K saturates well before 100 MB; merges generalize to the full
  corpus at encode time. Training is a cached one-time artifact.
- `encode(text) -> list[int]` — apply learned merges in rank order.
- `decode(ids) -> str` — concatenate token byte-strings, latin-1/utf-8 safe.
- `to_json()` / `from_json()` — serialize merges + vocab. The trained JSON is the **pinned artifact**
  (same discipline as the SHA-pinned corpus); it is the source of truth and is not retrained per run.
  Saved under `data/` next to the corpus.

New `SubwordCorpus` (parallel to `CharCorpus`, in `data.py`), leaving `CharCorpus` untouched:

- Same deterministic final-10% train/val split.
- `train_ids` / `validation_ids` produced by the tokenizer.
- `vocab_size` = the tokenizer's actual vocabulary size.

### Acceptance (TDD)

- `decode(encode(text)) == text` (round-trip) on sample text incl. markup/capitals/digits.
- Training is deterministic given (corpus slice, vocab_size).
- `from_json(to_json(t))` is identical (artifact stability).
- `SubwordCorpus` split is deterministic and leakage-free (tokenizer trained only on train slice).

## Section 3 — Checkpoint + generation for subword

- Checkpoint metadata gains `tokenizer_kind: "char" | "subword"`. For subword, the trained
  **tokenizer JSON is embedded directly in the metadata** (few hundred KB → self-contained, portable
  checkpoint). Char checkpoints keep storing `vocabulary` exactly as today (**backward compatible**).
- `load_for_generation` branches on `tokenizer_kind`: char → existing char-vocab path; subword →
  rebuild `BpeTokenizer.from_json(...)`.
- `generate()` swaps encode-in / decode-out only: subword encodes the prompt and detokenizes produced
  ids via the tokenizer. Autoregressive loop, temperature, top-k, seed logic unchanged.

### Acceptance (TDD)

- Subword generate round-trips from a saved checkpoint (prompt encodes, output decodes to text).
- A char checkpoint still loads and generates unchanged (no regression).

## Section 4 — Model, config, training, and the A/B

- **Config** (`config.py` + `cli.py`): add `tokenizer = "char" | "subword"` (default `"char"`) and
  surface `rms_ema_beta` / `pressure_leak_period` as config fields wired through to `RatchetEmbedding`
  (they exist in `ratchet.py` but are not currently config-threaded). `ratchet_embedding` already
  exists. `--tokenizer` CLI flag; `_corpus` builds `SubwordCorpus` when subword, training the
  tokenizer once if the artifact is absent and loading it otherwise.
- **Tuning step** (5k screens per the screening budget): a small sweep over `rms_ema_beta` and
  `pressure_leak_period` to settle values before the full run. Preserve the screen results.
- **Polished run:** 25M arch (n_layer 8, n_head 8, n_embd 512, block_size 256), 8K subword,
  `ratchet_embedding=True`, settled `rms_ema_beta>0`, 30k steps, enwik8, seed 1337. Fully master-free.
- **A/B (the unmeasured number):** matched arms at 8K subword — control = FP `nn.Embedding`,
  treatment = ratchet embedding + sparse update. Same tokenizer, seed, schedule, eval batches.
  Measures the sparse update's cost at subword sparsity (distinct from the byte-level dense 0.017).
  A new `docs/results/` note records it; preserve the magnitude.

## Section 5 — Testing summary

TDD throughout. Suites: BPE round-trip + determinism + artifact stability; the sparse-EMA fix
(rare-row convergence, non-firing invariance, masked EMA); `SubwordCorpus` determinism/leakage;
subword generate round-trip + char no-regression; `lat audit` clean on the subword ratchet-embedding
model (zero FP master weights). Full `uv run pytest` green; `ruff` clean on changed files;
`git diff --check`.

## Out of scope (deferred)

- Vocab beyond 8K (50K+ is where the ratchet embedding's memory win is large; revisit after 8K works).
- Any claim of training acceleration or packed sub-byte storage.
- Replacing or deprecating the char path.

## Decisions (resolved during brainstorming)

- Subword + ratchet embedding **now** (pull the sparse-update research into this PoC).
- Sparse normalization via **per-row RMS EMA** (reuse existing buffer; fix the non-firing update bug).
- **Pure-Python BPE**, zero new dependencies (repo's auditable/self-contained ethos; `tokenizers`
  would be a genuinely new top-level dep).
- **8K** vocab; **25M @ 30k steps**; enwik8; seed 1337.
