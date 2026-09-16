# Where the master-weight-free gap lands: copyable versus non-copyable bytes — 2026-09-14

## Outcome

The ratchet's 0.078-nat deficit to its matched QAT arm is spread across every match-length bin at
0.05 to 0.09 nats, roughly in proportion to byte mass. Only 3% of it sits on bytes that could be
copied from an 8-gram match earlier in the window. It is a general deficit of the dense weights, not
lost in-context recall. A retrieval device bolted onto the model (the flybrain-lab sparse-key memory
head) would therefore absorb at most the copyable share of the gap, not the gap.

## Question

The flybrain-lab session suggested this split as the cheapest test of a merge idea: put the ratchet
under a recurrent backbone with a sparse-key fast-weight head, on the hypothesis that the head supplies
exact recall the backbone otherwise has to encode in precise dense weights. If the ratchet's loss were
concentrated on copyable bytes, the head would hide it. This note measures where the loss lands.

## Method

- Checkpoints: the twelve final 30k checkpoints of the momentum confirmation
  (`runs/momentum-30k-2026-09-13/`): plain ratchet, momentum ratchet, QAT (15 states, FP master),
  FP32; seeds 1337, 1338, 1339; 8 layers, 8 heads, width 512, 256-character windows.
- Data: the text8 validation split (final 10%, SHA256 `6e890197…`), scored in non-overlapping
  256-character windows, 39,062 windows, every target byte once. BF16 autocast as in training
  evaluation; per-byte NLL in nats from FP32 logits.
- Bins: for the target at position i+1, m[i] is the length of the longest suffix of the window up to
  i that also occurs earlier in the same window (its successor is therefore known). Bins 0, 1, 2–3,
  4–7, 8+. `copyable8` is m ≥ 8. Bins depend only on the text, so they are identical for every arm.
- Sanity: each checkpoint's overall NLL matches its recorded final validation loss (40 random
  batches) within 0.002.
- Driver: `scripts/copyable_mask_eval.py`; output `runs/copyable-mask-2026-09-14/results.json`.

## Results (three-seed means; seed sd of every gap ≤ 0.002)

| bin | byte fraction | plain | momentum | QAT | FP32 | plain − QAT | QAT − FP32 | share of plain − QAT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.092 | 1.060 | 1.057 | 0.994 | 0.984 | +0.066 | +0.010 | 7.7% |
| 1 | 0.382 | 1.001 | 0.995 | 0.910 | 0.898 | +0.091 | +0.012 | 44.3% |
| 2–3 | 0.370 | 1.196 | 1.191 | 1.115 | 1.104 | +0.082 | +0.011 | 38.5% |
| 4–7 | 0.108 | 0.982 | 0.979 | 0.935 | 0.929 | +0.046 | +0.006 | 6.4% |
| 8+ | 0.048 | 0.697 | 0.688 | 0.646 | 0.644 | +0.051 | +0.003 | 3.1% |
| overall | 1.000 | 1.062 | 1.057 | 0.983 | 0.973 | +0.078 | +0.010 | 100% |

Split at 8: copyable bytes are 4.8% of the validation set with a plain − QAT gap of 0.051
(per seed 0.049 / 0.052 / 0.051); the other 95.2% carry a gap of 0.080.

## Reading

- The state count costs nothing on retrieval: QAT is within 0.003 of FP32 on copyable bytes.
- The master-weight-free rule loses 0.05 even there, so it is worse at copying too, but no worse
  than on any other bin. Momentum sits 0.005 below plain in every bin; it does not change the shape.
- The overall plain − QAT figure here (0.078, final checkpoints, whole split) is consistent with the
  momentum note's best-so-far comparison (momentum − QAT 0.073).

## Limits

256-character windows, text8 prose, 27-symbol vocabulary. Copyable mass is small at this window;
the 2048-token code setting where the memory head earns its gain is not measured. The scan says
only that if the per-bin gap stays flat there, a retrieval head removes under half of it.
