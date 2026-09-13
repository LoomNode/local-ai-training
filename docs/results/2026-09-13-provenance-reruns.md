# Provenance reruns on the corrected backend: text8 30k, subword seed 1338, historical 1B — 2026-09-13

> **Commit hash note (2026-09-13):** the corrected-source commit cited below as `5567c51`
> (full `5567c51d4c7e370183016532b878c1952f0d320e`) was rewritten to `1c6f3b6` before the push,
> changing only the author identity; the tree is identical (`79a2c497…`). Run manifests under
> `runs/` keep the original hash. Likewise the follow-up commit `a33b508` became `948ff62`
> (tree `dd204d2c…`).

## Outcome

All seven precautionary reruns from the [rerun tracker](2026-09-12-rerun-tracker.md) completed on
corrected source `5567c51` with fresh version-2 checkpoints and no resume, retry, or tuning.
The historical conclusions hold:

- **text8 states curve:** FP32 still wins at iso-parameters, and quality still improves
  monotonically from 5 to 7 to 9 states. Quinary landed 0.016 nats worse than the June best;
  septenary 0.008 worse; nonary 0.005 better at the matched 29,400-step budget.
- **Subword sparse-embedding A/B, seed 1338:** the ratcheted input embedding still beats the
  FP32 control, by 0.034 nats on the last-four-evaluation mean (historical: about 0.031).
- **Historical-config 1B screen:** the 5,000-step run fit on one RTX 3090, descended cleanly, and
  passed the artifact audit, both generation modes, and the exact split/resume gate. The gate
  exposed that the default flash attention backward is not repeatable at head size 128; a new
  opt-in `deterministic_attention` setting fixes it and the gate passed with it on.

These runs close the launch/resume provenance gaps the audit found. They were never evidence of
numerical corruption, and this note does not claim the historical numbers were wrong.

## Why these three, and only these

The 2026-09-12 integrity review found two defect paths: frozen controls that still updated
ratchet state (fused-update regression `d11fe888`, fixed in `5567c51`), and training resume
that skipped tokenizer-identity, seed, CUDA RNG, and pressure-leak-phase checks. The tracker
classified all 26 historical result reports against those paths. Three comparisons lacked
launch or resume provenance:

| Historical report | Gap |
| --- | --- |
| [text8 states curve](2026-06-20-text8-states-curve.md) | FP32/quinary/septenary explicitly resumed from 12k checkpoints whose provenance is unavailable |
| [subword sparse embedding A/B](2026-06-26-subword-sparse-embedding-ab.md) | seed-1338 arms added at `19b745e` with no launch history |
| ROADMAP 1B feasibility screen | no launch/resume record; the named config had since changed to batch 224 / LR 0.005 |

Every other report was judged not exposed by its documented protocol. See the tracker inventory.

## Protocol and provenance

- Source: `5567c51d4c7e370183016532b878c1952f0d320e`; working tree edits limited to docs and one
  test. All 21 recorded Python source hashes matched at the final audit.
- Python 3.12.13, PyTorch 2.12.1+cu130, CUDA 13.0, two RTX 3090s with the user's power caps
  (GPU 0 at 300 W, GPU 1 at 250 W) unchanged.
- **text8:** `runs/integrity-text8-30k-2026-09-12/`, `configs/scaleup_text8_25m_30k.toml`
  (SHA256 `a8cf8f1a…`), seed 1337, four arms fresh 0→30,000 steps, 491,520,000 tokens per arm,
  151 evaluation rows each. Dataset SHA256 `6e890197…`.
- **Subword seed 1338:** `runs/integrity-followups-2026-09-12/`, `subword-30k.toml`: 30k steps,
  batch 64, context 256, LR 0.0003, codes 15, RMS EMA 0.9, no pressure leak, pinned enwik8,
  one shared rebuilt 8,000-token / 7,744-merge BPE artifact. Only the treatment ratchets the
  input embedding. The original June BPE artifact is unavailable, so this is an internally
  matched A/B, not an exact historical-tokenizer replication.
- **1B:** same directory, `historical-1b.toml` recovered from `528946d`: batch 96,
  support LR 0.00075, 5,000 steps, 122,880,000 tokens, codes 15, RMS EMA 0.9, ratchet
  embedding, int8 forward, compiled updates, gradient checkpointing, seed 1337.
- Execution: text8 serial on GPU 1; after the user authorized a dual-GPU trial, the subword and
  1B queue ran on GPU 0 concurrently. Each job ran under `thermal_guard.py` (admission needs
  three samples with free VRAM, core temperature below 80 °C, and the expected power cap;
  85 °C stops only the managed process). Temperatures held at 77–79 °C on GPU 0 and 75 °C on
  GPU 1; no thermal stop, OOM, or GPU fault occurred during the trial.
- One failed attempt is preserved: the first text8 quinary launch hit CUDA OOM while an
  unrelated process held 20 GiB on GPU 1. It was relaunched unchanged as `quinary-retry1`.
- Artifact audit (`runs/integrity-final-checks-2026-09-13/audit.md`): every checkpoint SHA256
  matches its manifest, all floating payloads are finite, every packed nibble is valid, and no
  ratchet matrix has an FP/BF16 master or optimizer state.

## text8 30k results

Best validation loss, lower is better. The historical nonary job died at step 29,400, so its
matched-budget comparison uses the new best at or before that step.

| Arm | Historical best | New best | Δ | New final (30k) |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 0.9726 | 0.972721 | +0.0001 | 0.972908 |
| Quinary (5) | 1.2096 | 1.225254 | +0.0157 | 1.236806 |
| Septenary (7) | 1.1434 | 1.151153 | +0.0078 | 1.156952 |
| Nonary (9) | 1.1065 at 29,400 | 1.101799 at ≤29,400 (1.101661 at 30k) | −0.0047 | 1.104519 |

Ordering is unchanged: FP32 < nonary < septenary < quinary. The gap to FP32 is 0.129 (nonary),
0.178 (septenary), and 0.253 (quinary), versus 0.134 / 0.171 / 0.237 historically. Single seed,
no confidence interval; the sub-0.02 deltas are within what a backend change (the current loop
uses BF16 autocast; June did not) and a different resume history would produce, and are not
read as an effect.

## Subword seed-1338 A/B results

Last-four-evaluation mean validation loss.

| Arm | Historical | New | Δ |
| --- | ---: | ---: | ---: |
| Control, FP32 `nn.Embedding` | 2.4505 | 2.452436 | +0.0019 |
| Treatment, `RatchetEmbedding` | 2.4191 | 2.418420 | −0.0007 |
| Control − treatment | ≈0.031 | 0.034017 | |

The treatment checkpoint has no FP token-embedding weight or optimizer state (34,816 bytes of
support parameters versus 16,418,816 for the control, whose FP32 embedding is intentional).

## Historical-config 1B screen

| Metric | Historical | New |
| --- | ---: | ---: |
| Final validation, step 5,000 | 2.3648 (statistic unstated) | 2.368821 |
| Last-four-evaluation mean | | 2.375096 |
| Ratchet weights | | 955,121,664 in 34 layers |
| Support parameter bytes | | 208,896 |

The historical note does not say whether 2.3648 is a final or smoothed value; against either
reading the new run is within 0.011 nats. Feasibility on one 3090 is confirmed again.

**Generation gate:** from the saved checkpoint, both the default FP32 path and the native int8
path produced finite logits (256 and 257 positions checked), reproduced the identical sample
under the same seed, and left every persistent tensor unchanged. Both modes produced the same
Wikipedia-style link text. Details: `generation-results.json`.

**Exact split/resume gate:** PASS. With `deterministic_attention` on every branch, the uninterrupted 5000→5004 continuation and the 5000→5002→5004 split match bitwise: 120 model tensors (packed codes, row scales, RMS-EMA buffers, support parameters), 51 optimizer tensors, and both RNG states, plus identical checkpoint metadata and per-step metrics. The original step-5000 checkpoint is unchanged (hash verified). Artifacts: `continuous-det/`, `split-a-det/`, `split-b-det/`, `resume-status.json`; the failed default-backend attempts are preserved as `resume-status.gpu0-flash-nondeterministic.json` and the `continuous/`, `continuous-gpu0/`, `split-a/`, `split-b/` directories.

The first gate attempts failed, and the failure was a real finding about the backend rather
than about resume. Protocol: continue the saved step-5000 checkpoint to 5004 in one process, and
separately 5000→5002 then 5002→5004 through a checkpoint, then require bitwise-equal tensors,
metadata, and metrics. With the default configuration, 119 tensors differed. But the two
branches that ran the identical first two steps from the same checkpoint on the same GPU
already disagreed at step 5002 (train loss 1.98954 versus 1.98947, cumulative code moves
18,647,211 versus 18,650,706), and a third copy of those steps on the other GPU disagreed with
both. The 1B training step itself was not repeatable across processes, so no resume comparison
could be exact.

Root cause, isolated one variable at a time (`determinism-probe/` in the final-checks directory,
two fresh processes per variant, checkpoints compared bitwise):

| Variant | Head size | Tokens | Attention backend | Matmul | Result |
| --- | ---: | ---: | --- | --- | --- |
| 1B recipe, tiny width | 64 | 256 | flash (default) | int8 | identical |
| same, head size 128 | 128 | 256 | flash | int8 | differ |
| same, no compiled update | 128 | 256 | flash | int8 | differ |
| same, no gradient checkpointing | 128 | 256 | flash | int8 | differ |
| same, bf16 matmul | 128 | 256 | flash | bf16 | differ |
| head size 64, 1,024 tokens | 64 | 1024 | flash | int8 | differ |
| head size 128, deterministic attention | 128 | 256 | efficient | int8 | identical |

A direct test of `scaled_dot_product_attention` at the 1B shape (batch 96, 24 heads, 256
tokens, head size 128) reproduced the query gradient on 0 of 40 backward passes under the
flash backend and on 40 of 40 under the efficient backend; the math backend was also exact.
The flash backward accumulates the query gradient across key blocks with atomic adds, so the
summation order varies per launch. At head size 64 and 256 tokens the kernel happens to use two
key blocks, and a two-term float sum is order-independent, which is why every 25M study and the
suite's exact-resume tests are repeatable while the 1B recipe (head size 128) and longer
contexts are not. The int8 GEMM was ruled out by inspection and by the bf16 variant: it
accumulates in int32, so Triton's autotune choice cannot change its result.

Fix: an opt-in `deterministic_attention` setting (TOML `[training]`, or
`--deterministic-attention`) pins the efficient/math backends inside the attention module, so
gradient-checkpoint recomputation re-enters it. Default off, because the efficient backward is
about 1.7x slower than flash at the 1B attention shape (24.9 versus 14.3 ms per forward+backward
for one layer's attention; a small share of a step). The setting is stored in checkpoint
metadata and a resume must match it. The gate was rerun with the flag on every branch, resuming
from a copy of the step-5000 checkpoint whose metadata requests the flag (tensors byte-identical
to the original, hash recorded; the original is untouched).

Consequence for the studies above: their numbers stand, because none of them claims bitwise
reproducibility and none resumed. Anyone rerunning the 1B recipe, or any context longer than
256 tokens, who needs an exact resume must set the flag; the same is true for exact
seed-replication claims at those shapes.

## Reporting defect found and fixed

The runtime `persistent_state_bytes` / `ratchet_state_bytes` helper omitted the optional
per-row RMS-EMA buffer, understating logged ratchet state by 4 bytes per output row whenever
`rms_ema_beta > 0` (179,456 bytes for the subword control, 948,736 for the 1B run). The EMA is a
one-dimensional support buffer, not a master weight, so the audit result stands; only the
memory figure in `metrics.csv` and the audit report was low. Fixed in this commit with a
regression test (`test_persistent_bytes_include_optional_row_ema`). Figures in the tables
above come from direct checkpoint accounting and are unaffected. The metrics CSVs of the seven
runs retain the old undercounted column; they were not rewritten.

## What this does and does not establish

- Establishes: the corrected backend reproduces the three historical findings with complete
  provenance, and the checkpoint/resume stack is exact at 1B scale.
- Does not establish: a throughput claim, an FP32-arithmetic control (autocast is on), any
  significance across seeds, or exact numerical reproduction of the June procedure.
