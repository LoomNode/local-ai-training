# Corrected Memory Scaling Sweep: observability was capping model size, not training

**Date:** 2026-06-22
**GPU:** NVIDIA GeForce RTX 3090, 24,576 MiB (`CUDA_VISIBLE_DEVICES=1`)
**Supersedes the memory framing in:** `docs/results/2026-06-21-packed-memory-scaling.md` and the
Direction 2 premise in `docs/HANDOFF-direction2-activation-memory.md`.

## Summary

The historical scaling sweep recorded a single `cuda_memory_bytes` column read from the never-reset
cumulative `torch.cuda.max_memory_allocated`, while `collect_ratchet_metrics` ran on every metrics
row. From step 1 on, every recorded "peak" carried the prior step's ~6.9 GiB observability spike, and
the sweep took the max across rows. **The curve measured `collect_ratchet_metrics`, not training.**

After separating the per-step training peak (`cuda_train_peak_bytes`, allocator reset each step) from
the cumulative observability peak (`cuda_observability_peak_bytes`), two load-bearing Direction 2
claims collapse:

1. **"int8 OOMs at 3072-width before bf16."** False. The int8@3072 OOM was the observability spike
   (~23 GiB) hitting the 24 GiB ceiling. With `collect_ratchet_metrics` neutralised, int8@3072 trains
   in **3,048 MiB** — 57 MiB above bf16, nowhere near OOM.
2. **"The 12× storage win is stranded behind ~6 GB of unchanged activations."** False. That ~6 GB was
   observability. At the real training frontier the persistent 1-byte weight state — not activations —
   is what binds.

The 12× *is* realized in trainable model size: fp32 OOMs at 3072 while the ratchet trains to
**6144-width (bf16)** / **5120-width (int8)** on the same 24 GiB card.

## Method

`scripts/memory_sweep.py` (corrected) records both peaks per run; for sizes that the observability
spike pushed into OOM (>=3072), `scripts/int8_3072_observability_probe.py` re-runs the **real
`train_run` path** in a fresh process with `collect_ratchet_metrics` monkeypatched to a no-op, so the
only thing that can OOM is the training step. Config matches the sweep: batch 2, context 32,
checkpointing on, quinary codes (`max_code=2`), seed 1337, 3 steps, layers/heads scaled with width
(512→8/8 … 6144→48/48). This is the weights-dominate regime the sweep targets by design.

Commands:

```bash
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
  uv run python scripts/memory_sweep.py
CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache \
  uv run python scripts/int8_3072_observability_probe.py --modes bf16 int8 --embd 3072
```

Raw JSON under ignored `runs/sweep/` and `runs/int8-3072-probe/`.

## Training peak vs observability peak

Training peaks: <=2048 from the corrected sweep (completed runs); >=3072 from the
observability-excluded probe. Observability peaks from the sweep (cumulative `max_memory_allocated`
including the `collect_ratchet_metrics` spike).

| width | fp32 train | bf16 train | int8 train | int8−bf16 | bf16 **obs** peak | ratchet state (1B) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512  | 508.5  | 68.3   | 292.0   | +223.7 | 234.1  | 24.2 |
| 1024 | 2,899.4 | 226.8 | 433.1   | +206.3 | 1,316.8 | 144.5 |
| 2048 | 15,382.9 | 954.7 | 1,112.4 | +157.7 | 6,933.8 | 769.3 |
| 3072 | **OOM** | 2,991.1 | 3,047.8 | +56.7 | 23,378.2 | 2,594.7 |
| 4096 | — | 6,792.6 | 6,836.7 | +44.1 | (OOM in sweep) | 6,148.8 |
| 5120 | — | 13,079.6 | 13,151.4 | +71.7 | — | 12,007.3 |
| 6144 | — | 22,163.1 | **OOM** | — | — | 20,746.5 |

All MiB. The bf16 obs peak at 2048 (6,933.8) reproduces the historical ~6.9 GiB almost exactly,
confirming that number was `collect_ratchet_metrics`. fp32 has no ratchet histogram, so its train and
obs peaks coincide — the contamination mechanism showing itself.

## Trainable frontier on one 24 GiB 3090

| mode | largest width that completes a step | binding constraint |
| --- | --- | --- |
| fp32 (nn.Linear + AdamW) | **2048** (OOM at 3072) | fp32 matrix + master + Adam state |
| bf16 ratchet | **6144** (OOM beyond) | persistent 1-byte weight state (20.7 GiB of 22.2) |
| int8 ratchet | **5120** (Triton OOM at 6144) | weight state + int8 backward working buffers |

The ratchet trains ~3 width-doublings past fp32's frontier — the 12× persistent-storage win converts
directly into trainable model size once observability is excluded from the peak.

## What actually binds (the Phase C gate)

At the frontier the training peak is dominated by the ratchet's own persistent state, not activations:

- 6144 bf16: train peak 22,163 MiB vs persistent state 20,746 MiB → activations + working set ≈
  **1.4 GiB (~6%)**.
- 5120 bf16: 13,080 vs 12,007 → ≈ 1.1 GiB (~8%).

Generic activation reduction (Direction 2 Phase C — int8 activation storage at the checkpoint
boundary) can recover at most this ~1 GiB slice, which is far less than the ~2× state growth between
adjacent widths. **It cannot push the frontier by a size, so the Phase C gate does not open in this
regime.** Engineering it here would repeat the Phase B mistake: optimizing a column the measurement
shows is not binding. (Caveat: this is the deliberately weights-dominate regime — batch 2, context
32. Under a realistic large-batch/long-context regime activations would dominate and Phase C would
matter; but that changes what the sweep is comparing.)

## The remaining real, narrow target (Phase B)

int8's training peak exceeds bf16 by a small, shrinking margin (+224 MiB at 512 down to +44 MiB at
4096), but at the very frontier it costs **one model size**: int8 OOMs at 6144 (Triton workspace)
while bf16 fits at 22.2 GiB. So the int8-specific backward working set is a genuine but narrow
liability — Phase B (reduce/stream it) would buy back that one size, not the multiples Direction 2
imagined. This is now a scoped optimization decision, not a blocker.

## Limitations

- `memory_allocated` is live PyTorch allocation, not reserved cache or driver memory.
- 3090-specific; deterministic here, not a cross-driver claim.
- Probe excludes observability by no-op'ing `collect_ratchet_metrics`; it does not change training
  math or persistent state (`lat audit` remains clean).
- The 6144 int8 failure is a Triton kernel-launch OOM, not a clean allocator OOM; the exact int8
  workspace footprint at that size is not separately attributed here.
