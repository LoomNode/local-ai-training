# Gap-versus-scale ladder: does the master-weight-free residual shrink with model size?

**Date:** 2026-09-14
**Status:** Approved design (user: "line up the next tests"), implementation in `scripts/gap_ladder.py`
**Predecessor:** `docs/results/2026-09-14-momentum-confirmation.md` (25M, 15 codes, 30k, three seeds:
plain ratchet 0.077 above QAT, QAT 0.010 above FP32; momentum recovers 6%)

## Question

At 15 codes and a fixed 491,520,000-token budget, how does the plain ratchet's converged gap to
matched QAT change with model size? Three rungs: 7.4M, 25.2M (already run), and 99.1M ratchet
weights. If the gap shrinks with size the one-byte-per-weight trade improves at scale; if it
holds or grows, the update rule is the bottleneck at every size.

## Design

- Rungs: `configs/ladder_text8_7m_30k.toml` (6 layers, 8 heads, width 320) and
  `configs/ladder_text8_99m_30k.toml` (14 layers, 12 heads, width 768). Both are the 25M 30k
  recipe with only the model geometry changed: block 256, batch 64, 30,000 steps, support LR
  0.0003, pressure threshold 8, buckets 0.5/1.5, text8. The 25M rung is not rerun; its plain,
  QAT, and FP32 arms come from `runs/momentum-30k-2026-09-13/results.json` (`per_seed`).
- Arms per rung and seed: plain ratchet (`--weight-mode ratchet --codes 15`, both momentum
  knobs off), QAT (`--weight-mode qat --codes 15`), FP32 (`--weight-mode fp32`). Momentum is
  not run; the confirmation retired it as a lever. Seeds 1337, 1338, 1339. 18 new runs.
- Matched arms share seed, logical initialization, batch schedule, evaluation batches, and
  token budget. Fresh runs, no resume, no retries with changed settings; failed runs preserved.
  `lat audit` zero violations on each rung config at 15 codes.
- Execution: two per-card queues under `scripts/thermal_guard.py`, round-robin over items
  ordered 99M rung first, then 7M, arm-major (plain, QAT, FP32), seed-minor, so the long
  99M ratchet runs start first. The 99M rung is undertrained at 5 tokens per weight; that is
  the iso-token design and is reported as a limitation, not corrected.
- Metrics per run: best validation loss, its step, final, last-four-evaluation mean, plus the
  final-row saturation, cumulative code moves, ratchet state bytes, support bytes.
- Comparisons per rung, per seed and three-seed mean with sample SD: plain minus QAT (the
  master-weight-free penalty), QAT minus FP32 (the state-count cost), plain minus FP32.
- Verdict, on the three-seed mean of plain minus QAT across the three rungs: **shrinking** if
  it decreases monotonically from 7M to 99M and the 99M value is at least 0.02 nats below the
  7M value; **growing** if the mirror holds; **flat** otherwise. Sub-0.02 differences across
  rungs are within what three seeds resolve and are reported as flat.

## Outputs

`runs/gap-ladder-2026-09-14/` with manifest (source commit, both config SHA256s, dataset
SHA256, seeds, arms, queue assignment), per-run logs, metrics, checkpoints, guard status,
`results.json`, `validation.png`; then `docs/results/2026-09-15-gap-ladder.md` with the tables
above, per-rung wall-clock, sharing conditions, and explicit limits (one corpus, iso-token
budget, three seeds, BF16 autocast loop, no throughput claim).

## Non-goals

Other corpora, state counts, the momentum knobs, throughput, or any rung above 99M.
