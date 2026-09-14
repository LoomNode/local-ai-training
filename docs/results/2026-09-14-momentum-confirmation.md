# Momentum update rule at 30k, three seeds, 15 codes: partial — the EMA denominator recovers 6% of the master-weight-free gap; the converged residual to QAT is 0.073 nats — 2026-09-14

**Date:** 2026-09-13 to 2026-09-14
**Spec:** `docs/superpowers/specs/2026-09-13-momentum-confirmation-design.md` (Phase 2)
**Plan:** `docs/superpowers/plans/2026-09-13-momentum-confirmation.md` (Task 5)
**Phase 1:** `docs/results/2026-09-13-momentum-grid.md` (selected `rms_ema_beta = 0.99`, leak off)
**Status:** Converged result, three seeds, 30,000 steps, 15 codes, matched arms. Verdict **partial**.

## Outcome

The momentum ratchet (`rms_ema_beta = 0.99`, no pressure leak) finishes 0.0732 ± 0.0055 nats
above its matched QAT arm (three-seed mean ± sample SD, best validation loss), against the
0.03 threshold the spec set for "flipped". It beats the plain ratchet on every seed, by
0.0046 ± 0.0041 nats, which is 6% ± 5% of the plain-minus-QAT gap. The master-weight-free
penalty at this scale is **not** recovered by temporal EMA of the moments; the converged
residual that the next update-rule idea has to beat is about 0.073 nats at 15 codes.

The 5k screen's advantage over plain (0.008 to 0.014 nats at step 5,000) shrank on every seed
by step 30,000 (to 0.001 to 0.010) without reversing. The spec's "not confirmed" clause
("the 5k advantage shrinks or reverses by 30k on any seed") therefore also reads literally
true; the driver's verdict, and the one reported here, is "partial" because momentum still
finishes below plain on all three seeds. Either label carries the same conclusion: the EMA
denominator speeds the early descent and barely moves the asymptote.

Two reference numbers fall out of the matched design:

- **The state count is not the problem.** QAT at 15 codes sits only 0.0103 ± 0.0007 nats
  above FP32. Of the roughly 0.085 nats between the plain ratchet and FP32, about 0.010 is
  the cost of 15 states with an FP32 master, and about 0.077 is the cost of training without
  one.
- **The 15-code plain ratchet at 30k is 1.0608 ± 0.0011** (best, three seeds), the first
  converged multi-seed baseline at the repository's default state count. The September
  nonary rerun (one seed) was 1.1018; 15 codes buy about 0.04 nats over 9.

## Gaps, per seed and three-seed mean (best validation loss)

| Seed | Momentum − QAT | Momentum − plain | QAT − FP32 | Fraction of plain−QAT gap recovered |
| --- | ---: | ---: | ---: | ---: |
| 1337 | 0.0744 | −0.0024 | 0.0105 | 3.1% |
| 1338 | 0.0672 | −0.0094 | 0.0108 | 12.3% |
| 1339 | 0.0780 | −0.0020 | 0.0095 | 2.5% |
| Mean ± SD | 0.0732 ± 0.0055 | −0.0046 ± 0.0041 | 0.0103 ± 0.0007 | 5.9% ± 5.5% |

All three seeds contribute to every statistic (no zero-denominator exclusions). Seed 1338 is
the outlier in momentum's favour; on the other two seeds the momentum arm is within 0.0025
of plain, which is the size of the seed-to-seed spread of plain itself.

## Per-seed results

| Seed | Arm | Best | Best step | Final (30k) | Last-four mean |
| --- | --- | ---: | ---: | ---: | ---: |
| 1337 | momentum | 1.0571 | 29,800 | 1.0584 | 1.0584 |
| 1337 | plain | 1.0595 | 29,800 | 1.0614 | 1.0609 |
| 1337 | QAT | 0.9827 | 30,000 | 0.9827 | 0.9828 |
| 1337 | FP32 | 0.9722 | 30,000 | 0.9722 | 0.9726 |
| 1338 | momentum | 1.0521 | 30,000 | 1.0521 | 1.0530 |
| 1338 | plain | 1.0615 | 29,200 | 1.0618 | 1.0620 |
| 1338 | QAT | 0.9849 | 29,800 | 0.9861 | 0.9858 |
| 1338 | FP32 | 0.9741 | 29,600 | 0.9745 | 0.9747 |
| 1339 | momentum | 1.0595 | 29,200 | 1.0601 | 1.0604 |
| 1339 | plain | 1.0615 | 30,000 | 1.0615 | 1.0632 |
| 1339 | QAT | 0.9815 | 30,000 | 0.9815 | 0.9829 |
| 1339 | FP32 | 0.9720 | 29,000 | 0.9730 | 0.9732 |

Ordering is FP32 < QAT < momentum < plain on every seed and on every statistic (best, final,
last-four mean).

## Trace and tail

Validation loss at fixed steps:

| Seed | Arm | 5k | 10k | 15k | 20k | 25k | 30k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1337 | momentum | 1.1542 | 1.1090 | 1.0872 | 1.0719 | 1.0645 | 1.0584 |
| 1337 | plain | 1.1625 | 1.1158 | 1.0917 | 1.0758 | 1.0664 | 1.0614 |
| 1337 | QAT | 1.1244 | 1.0584 | 1.0260 | 1.0048 | 0.9913 | 0.9827 |
| 1337 | FP32 | 1.1169 | 1.0491 | 1.0149 | 0.9941 | 0.9792 | 0.9722 |
| 1338 | momentum | 1.1471 | 1.1019 | 1.0793 | 1.0676 | 1.0590 | 1.0521 |
| 1338 | plain | 1.1610 | 1.1133 | 1.0914 | 1.0785 | 1.0691 | 1.0618 |
| 1338 | QAT | 1.1264 | 1.0602 | 1.0287 | 1.0084 | 0.9934 | 0.9861 |
| 1338 | FP32 | 1.1192 | 1.0504 | 1.0162 | 0.9961 | 0.9827 | 0.9745 |
| 1339 | momentum | 1.1574 | 1.1071 | 1.0907 | 1.0731 | 1.0639 | 1.0601 |
| 1339 | plain | 1.1662 | 1.1160 | 1.0938 | 1.0762 | 1.0654 | 1.0615 |
| 1339 | QAT | 1.1269 | 1.0620 | 1.0275 | 1.0046 | 0.9912 | 0.9815 |
| 1339 | FP32 | 1.1193 | 1.0505 | 1.0156 | 0.9939 | 0.9803 | 0.9730 |

- **Momentum minus plain shrinks with training on every seed:** seed 1337 from −0.0082 at 5k
  to −0.0031 at 30k; seed 1338 from −0.0139 to −0.0097; seed 1339 from −0.0088 to −0.0014.
  Momentum is never above plain at any evaluation after step 5,000 on seeds 1337 and 1338;
  on seed 1339 the two arms are equal to four decimals at step 25,600 and momentum is below
  at every other evaluation. Before step 600 momentum is 0.08 to 0.13 nats *behind* plain
  on every seed (the beta-0.99 EMA warm-up seen at the screen).
- **Tail slope over the last 5,000 steps (validation at 25k → 30k):** momentum −0.0061,
  −0.0069, −0.0038; plain −0.0050, −0.0072, −0.0039; QAT −0.0085, −0.0073, −0.0097; FP32
  −0.0070, −0.0082, −0.0073. Every arm is still descending at 30k, and the FP-master arms
  descend faster than either ratchet arm, so the momentum-minus-QAT gap is still widening
  at the end of the budget, as the de-confounding study found for the plain ratchet. The
  ratchet tails are the flatter ones, which is the "flattening tail" the spec's partial
  criterion names.

## Saturation, code moves, and persistent state at step 30,000

| Arm | Saturated % | Zero % | Cumulative code moves | Ratchet state bytes | Support parameter bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| momentum, 3 seeds | 19.2–19.3 | 4.1 | 7.89–8.05 billion | 25,474,776 | 90,112 |
| plain, 3 seeds | 19.8–19.9 | 3.9–4.0 | 8.08–8.15 billion | 25,327,212 | 90,112 |
| QAT, 3 seeds | 0 | 0 | 0 | 0 | 100,808,704 |
| FP32, 3 seeds | 0 | 0 | 0 | 0 | 100,808,704 |

The momentum arm's extra 147,564 persistent bytes are the per-row FP32 RMS-EMA buffer,
counted by the corrected `persistent_state_bytes`. `lat audit --model
configs/scaleup_text8_25m_30k.toml --codes 15` reports zero violations. QAT and FP32 keep a
100.8 MB FP32 master by design; they are the reference, not the method. The momentum arm
makes 1 to 3% fewer code moves than plain and saturates slightly less.

## Protocol and provenance

- Config `configs/scaleup_text8_25m_30k.toml` (SHA256 `a8cf8f1a…`), text8 (SHA256
  `6e890197…`), 30,000 steps, 491,520,000 tokens per arm, 151 evaluation rows per arm, 15
  codes, seeds 1337, 1338, 1339. Per seed the four arms share logical initialization, batch
  schedule, evaluation batches, and token budget; only `--weight-mode` and `--rms-ema-beta`
  differ. Momentum: `--weight-mode ratchet --rms-ema-beta 0.99`; plain: `--weight-mode
  ratchet`; QAT: `--weight-mode qat --codes 15`; FP32: `--weight-mode fp32`.
- Driver `scripts/momentum_30k.py --leak 0 --beta 0.99 --gpus 0,1`, launched 2026-09-13
  22:02 UTC. Manifest `runs/momentum-30k-2026-09-13/manifest.json` records source commit
  `c6c9171`; the training package under `src/` is unchanged from `948ff62` across every
  commit of this branch (drivers, tests, and docs only). Fresh runs, no resume, no retries,
  no failures; all twelve `EXIT` codes are 0. `results.json` holds every number above;
  `validation.png` plots all twelve runs.
- Two per-card queues, round-robin in arm-major order, each under `scripts/thermal_guard.py`:
  - GPU 0 (300 W cap): momentum 1337, momentum 1339, plain 1338, QAT 1337, QAT 1339,
    FP32 1338.
  - GPU 1 (250 W cap): momentum 1338, plain 1337, plain 1339, QAT 1338, FP32 1337,
    FP32 1339.
- Guard events: 161 temperature warnings at 80 °C, all on GPU 0, during momentum seed 1337
  (118, 22:33 to 23:33 UTC) and momentum seed 1339 (43, 23:46 to 01:29 UTC); no managed stop
  (the stop threshold is 85 °C), no OOM. The guard held FP32 seed 1337 for about eight
  minutes at 07:35 UTC while the user's llama-server held about 21 GB on GPU 1, then
  admitted it when the server unloaded.
- Sharing: the user's flybrain-lab scale run occupied both cards from about 01:20 to 06:00
  UTC, cutting the ratchet arms' throughput from about 85k to 27k tokens per second; the
  QAT and FP32 arms, which have no ratchet update, ran at about 200k tokens per second on a
  free card. The 25M model at head size 64 is bit-repeatable with the default attention
  backend, so sharing changed wall-clock only. Total study time about 11 hours across both
  cards.

## Limitations

- One model size (25M), one corpus (text8, character level), one state count (15), one
  momentum setting (the Phase 1 winner; the tied beta-0.9 cell was not run at 30k).
- The training loop uses BF16 autocast; there is no FP32-arithmetic control, and no exact
  comparison to the June numbers (5 codes, no autocast).
- Three seeds give a sample SD, not a confidence interval; the momentum-minus-plain effect
  (0.0046 ± 0.0041) is one SD from zero.
- No throughput or speed claim. Tokens-per-second differences between arms reflect the
  ratchet update's eager cost and card sharing, not a property of the method.

## What this changes

- The repository's headline gap is now measured with momentum on, at convergence, across
  seeds, at the default state count: **0.073 ± 0.006 nats to QAT at 15 codes**, of which the
  momentum rule recovers 6%. The June screening estimate of 84% recovered (5 codes, 5k, one
  seed) was a transient-speed effect and should not be cited as a gap closure.
- The pressure leak is retired as a lever at 15 codes (Phase 1: every leak period is worse
  than none). `rms_ema_beta = 0.99` is a small, real, early-training gain and a safe default
  candidate, not a gap closer.
- The next update-rule levers in the roadmap backlog (adaptive pressure threshold, rail
  wind-up) inherit this 0.073 residual as the number to beat, and this study's twelve runs
  as their matched baselines. Since QAT at 15 codes is within 0.010 of FP32, the master-
  weight-free training dynamics are the whole remaining problem.
