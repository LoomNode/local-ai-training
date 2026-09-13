# Tiny Shakespeare rerun on the corrected backend — 2026-09-12

> **Commit hash note (2026-09-13):** the corrected-source commit cited below as `5567c51`
> (full `5567c51d4c7e370183016532b878c1952f0d320e`) was rewritten to `1c6f3b6` before the push,
> changing only the author identity; the tree is identical (`79a2c497…`). Run manifests under
> `runs/` keep the original hash. Likewise the follow-up commit `a33b508` became `948ff62`
> (tree `dd204d2c…`).

## Outcome

All 15 fresh runs completed: five arms, seeds 1337/1338/1339, 2,000 steps each.
The trainable ratchets substantially outperform their frozen counterparts on every seed.
All six frozen checkpoints retain bitwise-identical packed code/pressure and row scales;
all six ordinary FP32 support parameter tensors changed in each frozen run.

This supports trainability on this small benchmark. It does not establish general superiority,
a throughput advantage, or exact numerical reproduction of the original backend.

## Protocol and provenance

- Source: `5567c51d4c7e370183016532b878c1952f0d320e`, clean checkout at launch.
- Unmodified `configs/ratchet_tiny.toml`: 2 layers, width 128, 4 heads, context 128,
  batch 32, 2,000 steps, evaluation every 100 steps over 20 fixed batches, dropout 0.
- 8,192,000 sampled training tokens per run; 122,880,000 across all 15 runs.
- Pinned `SamPIngram/tinyshakespeare` revision `6d8bc3fdfca13bf8a128bb0e0914cead1e2d208c`;
  character tokenizer, last 10% validation split, matched initialization and schedules within seeds.
- Dataset SHA256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- Python 3.12.13, PyTorch 2.12.1+cu130, CUDA 13.0, RTX 3090 selected with
  `CUDA_VISIBLE_DEVICES=1`; OMP/MKL threads 2. Other GPU workloads were left running.
- No resume, dropped seeds, hyperparameter tuning, or failed run retries.
- Fresh artifacts: `runs/integrity-replication-2026-09-12/` (ignored), including
  `replicate.py`, `summarize.py`, config copy, manifest, per-run logs/metrics/checkpoints,
  `results.json`, `verified-results.json`, and `validation.png`. Original artifacts are untouched.

The current loop uses BF16 autocast, including the arm labeled FP32 (which describes its
parameter storage, not a guarantee of FP32 arithmetic). Ratchet and dense forward matmuls
are autocast-eligible; their backward implementations are not an iso-arithmetic control.
Trainable/frozen pairs share the same
ratchet arithmetic. The June loop did not use autocast, so this is a fresh current-backend
study rather than an exact replication of the historical numerical procedure. The FP32 arm
provides context; this small study with an untuned FP32 learning rate does not establish a
general accuracy advantage.

## Final validation loss

Lower is better. SD is the sample standard deviation of three seeds, not a confidence interval.

| Arm | 1337 | 1338 | 1339 | Mean | SD | Historical mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| quinary | 1.834103 | 1.820766 | 1.831085 | 1.828651 | 0.006994 | 1.826600 |
| frozen-quinary | 3.229235 | 3.273246 | 3.255845 | 3.252775 | 0.022165 | 3.252900 |
| septenary | 1.805253 | 1.792676 | 1.790435 | 1.796121 | 0.007987 | 1.794100 |
| frozen-septenary | 3.259924 | 3.284633 | 3.240787 | 3.261781 | 0.021982 | 3.261800 |
| fp32 | 1.909063 | 1.917739 | 1.963351 | 1.930051 | 0.029163 | 1.929900 |

The paired mean frozen-minus-trainable gap is 1.424124 nats for quinary and 1.465660
for septenary. Septenary has lower loss than quinary on all three seeds. These are descriptive
results for one small corpus/configuration; no significance test or historical pass tolerance
was specified. The FP32 learning rate was not independently tuned.

The historical Shakespeare result notes were committed at `840bdfc` and `fd3900a` on June 20,
before the fused-update regression in `d11fe88` on June 21. Their presence in that earlier
history means the later regression is not grounds for declaring those original recorded results
invalid. The new frozen results differ from the rounded historical per-seed values by less than
0.0002 nats, providing descriptive agreement across backends.

## Integrity and persistent state

Verification inspected all 15 checkpoint steps, selected seeds, and token budgets, plus all
21 metric rows per run. Each trainable/frozen pair has identical initial validation loss.
For each of the six frozen runs, all 18 ratchet state tensors (9 packed buffers and 9 row scales)
match seeded initialization exactly, all saved update counters are zero, and all six FP32
support tensors changed. Trainable runs have changed packed state and positive cumulative moves.

Checkpoint tensor storage by arm type is below. These are persistent tensor payloads, excluding
file headers, JSON metadata, activations, temporary workspaces, and allocator overhead.

| Arm type | Tensor group | Dtype | Bytes |
| --- | --- | --- | ---: |
| quinary | ratchet | torch.float32 | 9,476 |
| quinary | ratchet | torch.uint8 | 401,536 |
| quinary | support_or_fp32_model | torch.float32 | 101,376 |
| quinary | optimizer | torch.float32 | 71,704 |
| quinary | rng | torch.uint8 | 5,072 |
| frozen-quinary | ratchet | torch.float32 | 9,476 |
| frozen-quinary | ratchet | torch.uint8 | 401,536 |
| frozen-quinary | support_or_fp32_model | torch.float32 | 101,376 |
| frozen-quinary | optimizer | torch.float32 | 71,704 |
| frozen-quinary | rng | torch.uint8 | 5,072 |
| fp32 | support_or_fp32_model | torch.float32 | 1,707,520 |
| fp32 | optimizer | torch.float32 | 3,284,028 |
| fp32 | rng | torch.uint8 | 5,072 |

Septenary has the same storage layout and counts as quinary. Ratchet matrix state is 401,536
uint8 elements plus 9,476 bytes of FP32 row scales, totaling 411,012 bytes; ordinary support
parameters add 35,840 FP32 bytes. The model also persists a fixed FP32 positional encoding
buffer of 65,536 bytes, included in the model group above. Optimizer and RNG payloads are
listed separately above.
No throughput claim is made from this shared-GPU run or the logged interval timing.

See [integrity correction](2026-09-12-experiment-integrity.md) for the separate 303-test GPU
verification, including stochastic resume and optional feature coverage. This experiment uses
the default configuration and does not rerun the larger text8, QAT, or memory/speed studies.
