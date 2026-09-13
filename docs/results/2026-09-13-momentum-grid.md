# Momentum knob grid at the 5k screen, 15 codes: the EMA denominator recovers 35% of the QAT gap; every pressure leak is worse than none — 2026-09-13

**Date:** 2026-09-13
**Spec:** `docs/superpowers/specs/2026-09-13-momentum-confirmation-design.md` (Phase 1)
**Plan:** `docs/superpowers/plans/2026-09-13-momentum-confirmation.md` (Task 4)
**Status:** Screening result, one seed, 5,000 steps. Selects the arm for the 30k three-seed
confirmation (Phase 2, `docs/results/2026-09-14-momentum-confirmation.md`).

## Outcome

Winner: **`rms_ema_beta = 0.99` with the pressure leak off** (`pressure_leak_period = 0`),
best validation loss 1.1523 at step 5,000.

| Reference | Best@5k | Winner minus reference |
| --- | ---: | ---: |
| Plain ratchet (both knobs off) | 1.1671 | −0.0148 |
| QAT ceiling (FP32 master + Adam, 15 codes) | 1.1250 | +0.0273 |

The plain-minus-QAT gap at this budget is 0.0421 nats; the winner closes 35% of it. That is far
short of the 84% the June study reported at 5 codes, and the reason is visible in the grid: at
15 codes every pressure leak costs code moves and quality, so the whole recoverable effect here
comes from the second-moment EMA on the per-row RMS denominator, which June measured alone at
about 75% of a larger gap.

Tie set under the spec's 0.005-nat rule: `leak0-beta0.9` (1.1559) and `leak32-beta0.99`
(1.1543) tie with the winner. The rule prefers weaker forgetting, and no leak is the weakest, so
the leak-0 row wins the tie and its best cell is beta 0.99.

## Grid: best validation loss at step 5,000 (lower is better)

| Leak period \ beta | 0.9 | 0.99 | 0.999 |
| --- | ---: | ---: | ---: |
| 0 (no leak) | 1.1559 | **1.1523** | 1.1865 |
| 8 | 1.2476 | 1.2177 | 1.3205 |
| 12 | 1.1935 | 1.1918 | 1.2480 |
| 16 | 1.1719 | 1.1713 | 1.2276 |
| 24 | 1.1650 | 1.1637 | 1.2041 |
| 32 | 1.1591 | 1.1543 | 1.1960 |

Plain ratchet 1.1671; QAT 1.1250. Three regularities, each holding without exception:

- **Shorter leak periods are monotonically worse.** Leak 8 loses 0.05 to 0.15 nats to plain;
  the row improves with every step toward 32, and only the leak-24 and leak-32 cells at beta
  0.9 and 0.99 beat plain, by 0.002 to 0.013 nats. The leak-0 row beats every leaked row at
  beta 0.9 and 0.99, so a leak recovers part of the gap only when it is long enough to matter
  little, and never as much as no leak. The June leak-16 optimum at 5 codes does not
  carry to 15 codes, where a 15-state pressure accumulator needs more consecutive same-sign
  gradient than a 5-state one and any bleed starves it.
- **beta 0.999 is the worst column at every leak period,** including leak 0. A denominator that
  averages over a thousand updates lags the gradient scale during the fast early descent and
  under-buckets pressure for most of the 5k budget.
- **Quality tracks code-move count.** Cumulative code moves fall from 1.09 billion (plain) and
  1.00 billion (winner) to 127 million (leak 8, beta 0.999), and the ordering of the table is
  nearly the ordering of move counts. The ratchet at 15 codes is move-starved, not
  move-saturated; the leak removes moves it needs.

## Per-arm statistics at step 5,000

| Arm | Best@5k | Saturated % | Cumulative code moves | Ratchet state bytes | First eval below plain | Below plain from |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qat | 1.1250 | 0.0 | 0 | 0 | 200 | 200 |
| leak0-beta0.99 | 1.1523 | 18.5 | 1,000,945,937 | 25,474,776 | 600 | 600 |
| leak32-beta0.99 | 1.1543 | 17.9 | 762,926,528 | 25,474,776 | 800 | 1800 |
| leak0-beta0.9 | 1.1559 | 18.5 | 972,682,140 | 25,474,776 | 200 | 200 |
| leak32-beta0.9 | 1.1591 | 17.5 | 704,513,401 | 25,474,776 | 200 | 5000 |
| leak24-beta0.99 | 1.1637 | 17.2 | 650,969,225 | 25,474,776 | 800 | 5000 |
| leak24-beta0.9 | 1.1650 | 17.2 | 642,001,908 | 25,474,776 | 200 | 5000 |
| plain | 1.1671 | 19.9 | 1,091,815,262 | 25,327,212 | — | — |
| leak16-beta0.99 | 1.1713 | 16.3 | 516,451,672 | 25,474,776 | never | never |
| leak16-beta0.9 | 1.1719 | 16.2 | 506,480,321 | 25,474,776 | 200 | never |
| leak0-beta0.999 | 1.1865 | 15.7 | 438,068,888 | 25,474,776 | never | never |
| leak12-beta0.99 | 1.1918 | 15.2 | 372,918,889 | 25,474,776 | never | never |
| leak12-beta0.9 | 1.1935 | 14.8 | 352,752,144 | 25,474,776 | 200 | never |
| leak32-beta0.999 | 1.1960 | 14.4 | 310,436,525 | 25,474,776 | never | never |
| leak24-beta0.999 | 1.2041 | 13.9 | 273,732,552 | 25,474,776 | never | never |
| leak8-beta0.99 | 1.2177 | 14.1 | 247,989,032 | 25,474,776 | never | never |
| leak16-beta0.999 | 1.2276 | 13.1 | 218,316,273 | 25,474,776 | never | never |
| leak8-beta0.9 | 1.2476 | 12.9 | 187,333,392 | 25,474,776 | 400 | never |
| leak12-beta0.999 | 1.2480 | 12.5 | 178,026,036 | 25,474,776 | never | never |
| leak8-beta0.999 | 1.3205 | 11.9 | 127,393,479 | 25,474,776 | never | never |

Every momentum arm carries 147,564 more persistent bytes than plain: the per-row FP32 RMS-EMA
buffer, counted by the corrected `persistent_state_bytes` (one FP32 per output row across all
ratchet matrices). Best and final coincide for every arm: each arm's minimum is its last evaluation. Twelve of
the twenty arms do show transient rises between earlier evaluations (for example
`leak12-beta0.99` rose from 1.3616 at step 1,800 to 1.4750 at step 2,000), so the traces are
not monotone, only their endpoints are their minima. Support-parameter bytes are 90,112 for every ratchet arm and 100,808,704 for QAT,
whose FP32 master is the intended reference. The QAT arm reports zero ratchet state because it
keeps no packed tensor.

## Trace check: is the advantage early and never reversed?

June's finding at 5 codes was "positive from step 200, no transient trap". At 15 codes that holds
for beta 0.9 and not for beta 0.99:

| Step | 200 | 400 | 600 | 800 | 1000 | 2000 | 3000 | 4000 | 5000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qat | 2.2422 | 1.8332 | 1.5937 | 1.4738 | 1.3813 | 1.2374 | 1.1817 | 1.1518 | 1.1250 |
| leak0-beta0.99 (winner) | 2.3836 | 2.1251 | 1.7741 | 1.5313 | 1.4282 | 1.2503 | 1.2005 | 1.1700 | 1.1523 |
| leak0-beta0.9 | 2.2346 | 1.7668 | 1.5522 | 1.4093 | 1.3494 | 1.2388 | 1.1999 | 1.1735 | 1.1559 |
| leak32-beta0.99 | 2.3737 | 2.1475 | 1.8257 | 1.5826 | 1.4553 | 1.2610 | 1.2083 | 1.1808 | 1.1543 |
| plain | 2.3151 | 2.0330 | 1.7992 | 1.6273 | 1.5234 | 1.2826 | 1.2189 | 1.1867 | 1.1671 |

- `leak0-beta0.9` is below plain at every evaluation from step 200 and is the best ratchet arm
  through step 2,000, ahead of QAT itself at steps 400 through 1,000.
- `leak0-beta0.99` starts 0.07 to 0.09 nats *behind* plain at steps 200 and 400 (a
  hundred-update EMA warm-up during the fastest part of the descent), crosses below plain at
  step 600, and stays below thereafter. It overtakes beta 0.9 only at step 4,000 and wins at
  5,000 by 0.0036 nats, within the tie tolerance.
- Every leaked cell that ever beats plain does so late; `leak32-beta0.99` is not consistently
  below plain until step 1,800.

Consequence for Phase 2: the winner's margin over the beta-0.9 cell is a late-developing
0.0036 nats from one seed, and the two arms could reorder at 30k. Phase 2 confirms the winner
as selected by the rule; if the 30k verdict is "partial", the beta-0.9 cell is the first
alternative to test, not a new mechanism.

## Protocol and provenance

- Config `configs/scaleup_text8_25m_5k.toml` (SHA256 `18f53b47…`), text8 (SHA256
  `6e890197…`), seed 1337, 15 codes, batch schedule, evaluation batches, and token budget shared
  by all 20 arms; only `--weight-mode`, `--pressure-leak-period`, and `--rms-ema-beta` differ.
  Reference arms: `plain` (ratchet, both knobs off) and `qat` (`--weight-mode qat`).
- Driver `scripts/momentum_grid.py`. Manifest `runs/momentum-grid-2026-09-13/manifest.json`
  records source commit `592129d` for the first pass and the two `--continue` passes; the
  training package under `src/` is unchanged across every commit of this branch (the branch
  touches drivers, tests, and docs only), so all 20 arms ran the same training code as the
  provenance reruns at `948ff62`.
- `lat audit --model configs/scaleup_text8_25m_5k.toml --codes 15`: zero violations.
- Execution, at the user's direction to share both cards ("share the gpus"), superseding the
  spec's one-card serial plan. The first three arms (`plain`, `qat`, `leak8-beta0.9`) ran
  serially on GPU 1 (`study.log`). The driver was then stopped at that arm boundary (a
  `leak8-beta0.99` child that had just been spawned was stopped before writing any metrics and
  the arm was rerun from scratch) and relaunched with `--continue --gpus 0,1`, which skipped
  the three complete arms and round-robined the remaining fourteen across two per-card queues
  (`queue-gpu0.log`, `queue-gpu1.log`). A second `--continue` pass added the three leak-0
  cells after the first pass exited.
  - GPU 0 (300 W cap): leak8-beta0.99, leak12-beta0.9, leak12-beta0.999, leak16-beta0.99,
    leak24-beta0.9, leak24-beta0.999, leak32-beta0.99, leak0-beta0.9, leak0-beta0.999.
  - GPU 1 (250 W cap): plain, qat, leak8-beta0.9, leak8-beta0.999, leak12-beta0.99,
    leak16-beta0.9, leak16-beta0.999, leak24-beta0.99, leak32-beta0.9, leak32-beta0.999,
    leak0-beta0.99.
- Every arm ran under `scripts/thermal_guard.py`. Two 80 °C temperature warnings on GPU 0 at
  18:47 UTC; no managed stop, no OOM, no non-zero exit. The 25M model at head size 64 is
  bit-repeatable with the default attention backend, so which card ran an arm and what else
  the card was serving affect wall-clock only.
- The leak-0 row is an amendment to the spec's 5-by-3 grid, ruled mid-study when every leaked
  cell through leak 24 had lost to plain: the spec's grid never isolated the EMA denominator,
  which is the knob June credited with most of the effect. Recorded in the SDD ledger; the
  three extra cells cost about an hour of one card.
- Wall-clock: about 17 minutes per arm on a dedicated card; 28 to 38 minutes while sharing
  with the user's llama-server and other jobs. `leak0-beta0.99` waited about five minutes
  for admission (queue START 20:41:50 UTC, first metrics row 20:46:56 UTC) while the user's
  llama-server held 20.7 GB on GPU 1; the guard's live status showed `waiting_for_capacity`
  at 20:44 UTC. The guard keeps only its current status file, so that observation is recorded
  here rather than in an artifact. The arm then ran at reduced throughput alongside the
  user's jobs. Total grid time about 5.5 hours across both cards.

## Limitations

- One seed, 5,000 steps: a screening result. The 0.0036-nat margin inside the tie set is not
  distinguishable from seed noise; only the leak-period ordering and the beta-0.999 column are
  large enough to survive a seed change with any confidence.
- 15 codes only, 25M text8 only, BF16 autocast loop. No comparison to the June 5-code numbers
  is exact (different state count, different loop precision).
- No throughput claim. Per-arm tokens-per-second in `metrics.csv` reflect card sharing, not
  the update rule.
- "Gap closed" is measured against the QAT arm at 5k, where QAT is itself still descending
  fast; the converged fraction is Phase 2's job.

## Phase 2 launch

Launched 2026-09-13 22:02 UTC with the winner's knobs, both cards:

```bash
LAT_GUARD_DIR=runs/momentum-30k-2026-09-13/guard nohup .venv/bin/python -m scripts.momentum_30k \
  --leak 0 --beta 0.99 --gpus 0,1 > runs/momentum-30k-2026-09-13.launch.log 2>&1 &
```

Twelve runs (momentum, plain, QAT, FP32; seeds 1337, 1338, 1339) at
`configs/scaleup_text8_25m_30k.toml`, 15 codes, per-card queues under the thermal guard.
