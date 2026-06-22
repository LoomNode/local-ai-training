# Packed Training Memory: Scaling Sweep (the real evidence)

**Date:** 2026-06-21
**TL;DR:** The ratchet's persistent footprint is a clean, constant **12× smaller** than fp32+Adam at
every model size — but that win is **stranded behind activation/working memory** at training time.
Measured training-peak reduction over fp32+Adam is only **~2.2×**, and the largest trainable model on
one 24 GB 3090 goes from 2048-width (fp32) to **3072-width (bf16-ratchet)** — about 1.5× the params.
**int8 gives no training-memory benefit — it is marginally *worse* than bf16** (OOMs one size
earlier); int8's value is speed at frontier width, not memory.

This supersedes the "int8 strictly optimal" claim, which came from an interrupted step-0 run that
recorded an invalid (artificially low) peak.

## Method

`scripts/memory_sweep.py`: sweep width/depth with batch=2, block=32 (so **weights dominate**),
3 training steps, activation checkpointing on. Record peak `max_memory_allocated` **only from runs
that reached the final step** (completion guard — the earlier int8@2048 "win" was a step-0 row
recorded before its backward). Persistent footprint from `lat audit` (`ratchet_state_bytes` vs
`fp32_matrix_bytes` = fp32 master + AdamW state). Three modes: `fp32` (nn.Linear + AdamW),
`bf16`-ratchet, `int8`-ratchet. One idle RTX 3090, sequential.

Note both ratchet modes are 1-byte persistent; `bf16` vs `int8` isolates the *matmul materialization*,
not storage.

## Results (peak CUDA MB; OOM = did not fit in 24 GB)

| n_embd | fp32 | bf16 | int8 | persistent redux |
| ---: | ---: | ---: | ---: | ---: |
| 512  | 509   | 234   | 292   | 11.9× |
| 1024 | 2899  | 1317  | 1307  | 12.0× |
| 2048 | 15383 | 6934  | 6937  | 12.0× |
| 3072 | OOM   | 23378 | OOM   | 12.0× |
| 4096 | —     | OOM   | —     | 12.0× |

Largest model that completes a training step on one 24 GB 3090:
- **fp32 + Adam:** 2048-width
- **int8-ratchet:** 2048-width (no gain over fp32)
- **bf16-ratchet:** 3072-width (~1.5× the params)

## Why 12× storage becomes only ~2× training

Training peak splits into two columns; the ratchet only attacks one of them. At 2048-width:

| | weight + optimizer | everything else (activations, grads, working buffers) | total |
| --- | ---: | ---: | ---: |
| fp32    | 9217 MB | ~6166 MB | 15383 MB |
| ratchet |  769 MB | ~6166 MB |  6935 MB |

The weight column drops 12× (1 byte vs 4+8). The "everything else" column is **identical** — activation
memory depends on batch×seq×width×layers, not on how weights are stored. Once weights are tiny, the
un-optimized activation term dominates, capping the total reduction. Even weights→0 would cap at
`15383/6166 ≈ 2.5×`. **Reducing "everything else" is the lever that releases the stranded 12×** — and
because the ratchet's weight floor is the only one small enough to expose, it benefits most (the ratio
climbs toward 12× as activations shrink).

## Why int8 is *worse* than bf16 at the margin

At 2048 they tie (~6935 MB), but at 3072 bf16 fits (23.4 GB) while **int8 OOMs**. The int8 backward
materializes transient transposed quantized operands; even tiled, that working set makes int8's peak
marginally higher than bf16's at scale. So int8 buys no training memory and is a slight liability —
consistent with int8's real role being **throughput at frontier width**, not footprint.

## Honest framing

- **Storage: 12×** (constant, audit-confirmed) — but this is ≈int8 density and *worse* than a 4-bit
  GGUF; it ties tools that already exist, and it does not translate to training capacity.
- **Training capacity: ~1.5× bigger model** than fp32+Adam (via bf16-ratchet), ~2.2× lower peak at
  fixed size. The cap is activations, a generic cost the ratchet's weight-packing can't reach.
- **int8: zero memory benefit, marginally negative.** Speed lever only.
- The genuinely distinctive property remains *capability* (full-parameter training with no master
  weights), not the byte count — gated on the unproven convergence + BitNet-quality comparisons.

## Next

The leverage is **reducing "everything else."** First step: decompose the ~6 GB into generic
activations vs ratchet/int8-specific working buffers (the latter is what makes int8 OOM before bf16 —
tractable and pure upside). Then activation quantization / selective recompute / offload. Pushing
ratchet density below 1 byte (4-bit at rest) is low value — it optimizes the already-small column and
only reaches storage parity with existing 4-bit.
