# Subword + sparse-update ratchet embedding — polished PoC + A/B (2026-06-26)

**TL;DR.** A 25M model trains **fully master-weight-free** (zero persistent FP/BF16 `Parameter`
anywhere, including the input embedding) on enwik8 with an **8K byte-level BPE** vocab, and produces
**legible Wikipedia-style text**. At subword scale the master-free (ratcheted) input embedding costs
**nothing** — with matched settings it is **0.035 nats better** than an FP `nn.Embedding`, the
opposite of the small +0.017-nat *cost* seen at byte-level. The enabler is a **sparse-aware per-row
RMS-EMA update** (update the EMA only for rows that fired this step). A side finding: turning that
per-row RMS-EMA on globally (`rms_ema_beta=0.9`) is a **large win for the whole ratchet model**
(val 2.92 → 2.41 at 30k), independent of the embedding.

## Setup

- Model: 25M (n_layer 8, n_head 8, n_embd 512, block_size 256), codes 15 (max_code 7), seed 1337.
- Data: enwik8 (byte-level latin-1 source), **8K pure-Python BPE** vocab. The BPE is trained on a
  2 MB slice of the **train split only** (frequency-driven merges saturate well before the full
  corpus; pure-Python full-corpus training is impractical — see "Tokenizer cost"). Artifact:
  `data/enwik8/enwik8.bpe8000.json`, built+cached once.
- Steps: 30000. Sparse knobs settled by 5k screens: `rms_ema_beta=0.9`, `pressure_leak_period=0`.
- Configs: `configs/enwik8_subword_25m_5k.toml` (screening), `configs/enwik8_subword_25m_30k.toml`
  (polished). Runs are reported as the mean of the last 4 evals to beat ~0.003 per-eval noise.

## Tuning (5k screens, last-4-eval mean val, lower is better)

| rms_ema_beta | pressure_leak_period | val (5k) |
|---|---|---|
| **0.9** | **0** | **2.5603** |
| 0.9 | 500 | 2.5618 |
| 0.9 | 200 | 2.5631 |
| 0.99 | 0 | 2.5642 |

`rms_ema_beta=0.9`, no pressure leak. Differences are small (~0.004), but 0.9/leak-0 is consistently
best; pressure leak did not help the embedding.

## Result 1 — Embedding A/B (the question this PoC set out to answer)

Matched arms, identical except the input embedding's precision: both `rms_ema_beta=0.9` on every
ratchet matrix, same seed/schedule/tokenizer, 30k steps.

| Arm | Embedding | val (30k) | bits/char |
|---|---|---|---|
| Control | FP32 `nn.Embedding` (AdamW) | 2.4485 | 1.290 |
| Treatment | `RatchetEmbedding` (codes 15, sparse update) | **2.4139** | **1.272** |

**Ratcheting the embedding costs nothing at 8K subword — it is 0.035 nats *better* than FP.** This
is the reverse of the byte-level finding (+0.017-nat cost, see
`docs/results/2026-06-26-ratchet-embedding-ab.md`). Interpretation: at subword vocab the FP embedding
(AdamW) appears to slightly over-fit, while the integer-code + per-row-scale embedding with
sparse-aware normalization acts as a mild regularizer. The honest, conservative claim is that the
**master-free embedding is free at subword scale** (≥ as good as FP); we do not over-claim a
consistent advantage from one seed.

The treatment is **fully master-weight-free**: its checkpoint has `model::token_embedding.packed`
(uint8 code+pressure) + `model::token_embedding._scale` (FP32 per-row) + a non-Parameter
`rms_ema` normalization buffer, and **no** `optimizer::token_embedding`. The control, by contrast,
carries `model::token_embedding.weight` plus AdamW `exp_avg`/`exp_avg_sq` — the last FP master block.

### Sample (treatment, prompt `[[History of `, temp 0.8, seed 1)

> [[History of Europe#Limited#Economic Monetary and Work for Social Grants|GEDGS]] (since 1957)
> *Cyprus - GDP (current in the new-market economy and enjoying continuous economic development
> during the [[World War I]] and the euro)  ==Births==  *[[1448]] - [[Nicole B. De Mon]], French
> writer (d. [[19...

Wiki links, section headers, date entries, parenthetical clauses — structurally indistinguishable
from real enwik8, from a model with zero floating-point master weights.

## Result 2 — Global RMS-EMA is a large win (side finding)

`rms_ema_beta` is applied to **all** ratchet matrices, not just the embedding. An accidental first
control run with `rms_ema_beta=0` (default) plateaued at val ~2.92 from step ~14k, while the matched
`rms_ema_beta=0.9` arms reached ~2.41 — a **~0.5-nat** improvement across the whole 30k run.

| rms_ema_beta (all ratchet matrices) | val (30k) |
|---|---|
| 0.0 (default) | 2.9186 |
| 0.9 | 2.4139 (ratchet emb) / 2.4485 (FP emb) |

Per-row EMA normalization of the bucketed gradient is therefore a broadly beneficial change to the
ratchet update, not merely a sparse-embedding fix. The β=0 run is preserved at
`runs/subword_ab_control_beta0` as evidence. (This was not used in prior byte-level baselines, which
defaulted to β=0 — worth revisiting there.)

## Subword vs byte-level

The subword model is a materially better language model: **1.272 bits/char** (treatment) vs the
byte-level model's ~1.56 bits/char, at ~2.74 chars/token — fewer transformer steps per character and
better compression, as expected from subword tokenization.

## Tokenizer cost (honest note)

The pure-Python BPE now updates pair counts incrementally instead of doing a full pair-recount after
every merge. Training still uses a 2 MB train-split slice (default `train_chars=2_000_000`) because
encoding the full corpus is pure-Python and single-threaded, and because the frequency-driven merges
saturate well before the full 100 MB corpus. The optimization is semantics-preserving: tests compare
the incremental trainer against the old naive reference on merge JSON and encoded samples, so it does
not change the reported model result.

## Decisions

- The master-free subword model is the PoC deliverable: `runs/subword_ab_ratchet`, reproducible via
  `configs/enwik8_subword_25m_30k.toml` (+ `--ratchet-embedding --rms-ema-beta 0.9`).
- Keep `ratchet_embedding` opt-in by default, but at subword scale there is **no measured reason not
  to use it** — it matches/beats FP and removes the last master-weight block.
- `rms_ema_beta=0.9` is recommended as a global ratchet default pending a byte-level re-check.
- Large-vocab (50K+) sparse updates — where the ratchet embedding's *memory* win is large — remain
  future work.

CAVEAT: single seed at 30k per arm. A second seed would firm the embedding A/B sign/magnitude and the
global rms_ema_beta gain. Both are well outside per-eval noise (~0.003), and the rms_ema_beta effect
(~0.5 nats) is far outside any plausible seed spread.
