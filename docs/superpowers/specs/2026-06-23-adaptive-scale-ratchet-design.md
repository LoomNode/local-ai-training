# Adaptive-per-row-scale ratchet: attributing the master-weight-free gap

**Date:** 2026-06-23
**Status:** Approved design, pending implementation plan
**Predecessor:** `docs/results/2026-06-22-qat-deconfounding.md`

## Background

The STE-QAT de-confounding experiment (`docs/results/2026-06-22-qat-deconfounding.md`)
established that **master-weight-free training, not few states, owns 76–82% of the
ratchet's FP32 quality gap**. Mining the final ratchet checkpoints traced that penalty to
a concrete mechanism: **frozen-scale saturation**. The ratchet freezes each output row's
FP32 scale in a buffer at init; QAT recomputes scale live from the moving master every
forward. As a result a large fraction of ratchet codes pile up against the `±max_code`
rail and cannot express the magnitude the gradient asks for:

| arm | mean saturation (codes at rail) | master-free gap (nats) |
| ---: | ---: | ---: |
| ratchet5 | 45% | +0.174 |
| ratchet7 | 35% | +0.147 |
| ratchet9 | 29% | +0.109 |

Saturation tracks the gap one-to-one, is systemic across every matrix (worst in `lm_head`
and feed-forward), and is a stable equilibrium (≈unchanged from step 1k to 30k), which is
why more steps do not close the gap.

## Hypothesis

If the per-row scale is allowed to adapt during training, saturated rows can grow their
scale to fit the weights the loss wants, relieving saturation and closing part of the
master-weight-free gap. The effect should be **largest where saturation is worst (5
states) and taper toward higher state counts**.

## Mechanism: reuse the existing `trainable_scale`

No new adaptation mechanism is designed. `DiscreteRatchetLinear` already supports
`trainable_scale=True`, which stores the per-row scale as an AdamW-trained `log_scale`
`nn.Parameter` (log space keeps it strictly positive). It is tested
(`tests/test_ratchet.py`) and **audit-clean**: `audit_no_master_weights` flags only
floating *matrix* parameters (`ndim >= 2`); a 1-D per-row scale with optimizer state is
permitted persistent state, not a master-weight mirror. The QAT-experiment ratchet arms
ran with this flag **off** (frozen scale). This experiment turns it on.

This is the YAGNI choice: test the diagnosis with the tested mechanism that already
exists. If gradient-trained scale only partially closes the gap, the *residual* defines
what a richer mechanism (e.g. saturation-triggered re-quantization) would need to do —
that is a separate, later spec.

## Scope decisions

- **Ratchet-only comparison.** This is a "where does the loss come from" attribution, not a
  QAT study. At each state count we compare **frozen-scale vs trainable-scale ratchet**;
  the existing QAT{5,7,9} arms serve as the FP-master ceiling at those counts. QAT is not
  extended to the new state counts.
- **State sweep to the 4-bit nibble cap.** States `{5,7,9,11,13,15}` → `max_code`
  `{2,3,4,5,6,7}`. 15 states (`max_code=7`) is the largest the current packing supports:
  codes are stored as `code + max_code` in the low nibble, so `code + max_code ≤ 14 < 16`.
  `max_code=8` (17 states) overflows the nibble and must be rejected.
- **30000 steps, iso-everything.** Same config as the existing arms
  (`scaleup_text8_25m_30k.toml`: n_embd 512 / 8L / 8H / block 256 / batch 64, lr 3e-4,
  pressure_threshold 8), data, eval schedule (every 200 steps, 40 batches), token budget,
  seed 1337. A different budget would confound "adaptive scale helped" with "trained less".
  Note the budget is set by iso-comparison, **not** by the saturation-equilibrium finding
  above: that finding is a property of the *frozen* arm (its saturation is stuck by ~1k, so
  more steps cannot help it) and does not license a shorter run here. The baselines being
  compared against were still descending in val loss at 30k, and the trainable-scale
  dynamic (`log_scale` under AdamW) is new — there is no evidence it equilibrates early, so
  it gets the full budget.

## Code changes (minimal, TDD)

All behavior changes are test-first. No change to the update rule, packing layout, or
audit logic.

1. **Relax `max_code` validation** from `(2, 3, 4)` to the inclusive range `2..7` in both
   `DiscreteRatchetLinear.__init__` and `QATLinear.__init__`. Add a test asserting
   `max_code=7` is accepted and `max_code=8` raises (nibble overflow).
2. **Expand `--codes` choices** to `(5, 7, 9, 11, 13, 15)` in the `train` and `audit`
   subcommands. The existing `max_code = (codes - 1) // 2` mapping already yields
   `{2,3,4,5,6,7}` correctly; no other CLI logic changes.
3. **Add `--trainable-scale`** (`store_true`) to the `train` subcommand, threaded into the
   model config (the `ExperimentConfig.trainable_scale` field and `model.py` plumbing
   already exist; only the CLI surface is missing). Default off preserves current behavior.

## Experiment matrix

9 runs, seed 1337, into the **new** output dir `runs/text8-25m-adascale/` (existing
`runs/text8-25m-qat/` and `runs/text8-25m/` untouched — preserve-runs invariant):

| arm | states | max_code | trainable_scale | status |
| --- | ---: | ---: | --- | --- |
| frozen11 | 11 | 5 | off | new |
| frozen13 | 13 | 6 | off | new |
| frozen15 | 15 | 7 | off | new |
| ada5 | 5 | 2 | on | new |
| ada7 | 7 | 3 | on | new |
| ada9 | 9 | 4 | on | new |
| ada11 | 11 | 5 | on | new |
| ada13 | 13 | 6 | on | new |
| ada15 | 15 | 7 | on | new |

Baselines reused (not re-run): frozen{5,7,9} and QAT{5,7,9} from `runs/text8-25m-qat/`.

All arms share one logical FP init per seed, so frozen, trainable, and QAT start from the
identical quantized init at each state count (iso-init).

**Execution:** two idle 3090s, serial per-GPU driver modeled on
`runs/text8-25m-qat/run_all7.sh`, with `CUDA_VISIBLE_DEVICES`, `MPLCONFIGDIR=/tmp/mpl`,
`UV_CACHE_DIR=/games/ailab/.uv-cache`. ~110 min/run, ~8h wall across two GPUs. `runs/`
artifacts are not committed.

## Measurement & deliverable

For each arm record best validation loss + the full curve, and **per-row saturation from
the final checkpoint** (unpack int8 codes, fraction at `±max_code`; the script from the
QAT analysis). The results note (`docs/results/`) reports:

- **Frozen → trainable improvement** at each state count, with the matching saturation
  drop, validating (or not) that the gap closes in proportion to relieved saturation.
- The **state-count gradient**: hypothesis predicts the largest gain at 5 states, tapering
  toward 15.
- **Residual gap** of the best trainable-scale arm vs the QAT ceiling at 5/7/9 — how much
  of the master-weight-free penalty the frozen scale explained, and how much remains.

## Invariants & guardrails

- `lat audit` stays violation-free for every ratchet arm (1-D trainable scale is permitted;
  no FP/bf16 matrix parameter mirrors a code matrix).
- Existing runs preserved: new output dir only.
- HF dataset revision stays pinned (inherited from the existing config).
- Single seed (1337), 25M/30k: a trainability/attribution test, not a converged-scale
  claim. Carry the caveat into the results note.

## Limitations

- One mechanism (gradient-trained scale), one seed. If it under-closes the gap, a
  saturation-triggered re-quantization mechanism is a possible follow-up — out of scope
  here.
- Trainable scale adds per-row AdamW moment state (two FP32 per output row per matrix):
  negligible vs a master weight, but worth noting it is not literally zero extra state.
