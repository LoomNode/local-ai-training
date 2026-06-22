# Handoff: Direction 2 — release the stranded 12× by cutting "everything else"

**For Codex.** Read `CLAUDE.md`, `AGENTS.md`, `docs/README.md`, and the evidence note
`docs/results/2026-06-21-packed-memory-scaling.md` first. Git identity is `LoomNode` — never expose
real identity. Run with `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run ...`.
Idle 3090 via `nvidia-smi` + `CUDA_VISIBLE_DEVICES`. Brainstorm -> spec -> plan, TDD, stop after each
phase for review. Keep `lat audit` clean (no FP/bf16 Parameter mirroring a code matrix). Coordinate
branches with any in-flight convergence work; base off `feat/packed-training-memory`.

## Why

The scaling sweep proved the ratchet's persistent state is a constant **12× smaller** than fp32+Adam,
but at training time that only yields **~2.2×** lower peak, because peak = `weights(tiny) +
"everything else"(~6 GB, unchanged)`. The 12× is *stranded behind the "everything else" column*
(activations + gradients + working buffers). Reducing that column is the only lever that converts the
storage win into training capacity, and the ratchet benefits most (its weight floor is the only one
small enough to expose — the peak ratio climbs toward 12× as that column shrinks).

A sharp, concrete sub-problem first: **int8 currently OOMs one model-size *before* bf16** (3072: bf16
fits at 23.4 GB, int8 OOMs) despite identical 1-byte storage. That extra memory is int8-specific
backward working buffers (transposed quantized operands) — pure overhead, tractable, and fixing it is
guaranteed upside (today int8 is a memory liability with only a speed upside).

## Phase A — decompose the ~6 GB (MEASURE first, no optimization)

Instrument a single training step (reuse the probe approach from
`docs/results/2026-06-21-packed-training-memory-probe.md`) to attribute peak memory into:
persistent state; saved activations (per block); gradients; and **ratchet/int8-specific backward
working buffers** (the `quantize_rows`/`quantize_columns` transposed copies in the fused backward).
Use `torch.cuda.memory_stats` / targeted `reset_peak_memory_stats` around regions, or
`torch.cuda.memory._record_memory_history` snapshots. Deliverable: a table attributing the ~6 GB at,
say, 2048-width, for bf16 and int8 modes. This decides where the leverage is. Write to `docs/results/`.

## Phase B — kill the int8-specific overhead (make int8 <= bf16)

Target the int8 backward working set identified in Phase A: avoid full transposed materialization,
reuse/stream buffers, smaller tiles, or compute the transposed product without a separate transpose.
**Success: int8 training peak <= bf16 at every size**, so int8 stops being a memory liability (it
should at least match bf16, ideally beat it since it avoids the bf16 effective-weight materialization).
Must stay bit-exact vs the current fused update — extend the existing equivalence tests, don't weaken
them.

## Phase C — generic activation reduction (push toward the 12× ceiling)

Beyond the checkpointing already in place: **activation quantization** is the natural fit — store
block activations in int8 between forward and backward (the ratchet already has int8 quant machinery
in `int8_matmul.py`), dequantizing on the recompute/backward. Optionally selective recompute or CPU
offload. Generic (helps all modes), but the ratchet's tiny weight floor means its *ratio* over
fp32 improves most. Watch accuracy: activation-storage quant adds noise — keep an equivalence/quality
check and preserve any divergence rather than tuning it away.

## Success metric

Re-run `scripts/memory_sweep.py` after each phase. Show: (1) int8 peak <= bf16 at all sizes
(Phase B); (2) the training-peak reduction over fp32 climbing **above ~2.2×**, and the largest
trainable model on one 24 GB 3090 growing past 3072-width (Phase C). Report the new curve in the
scaling results note.

## Guardrails

- `lat audit` violation-free; no persisted FP code-mirror.
- Equivalence tests before perf; preserve divergence, don't silently tune.
- Preserve `runs/tiny-shakespeare`; write elsewhere. Don't commit `runs/` artifacts.
- 3090-specific numbers; note it.
- Honest framing: activation reduction is generic, so the result is "ratchet + activation reduction
  turns 1-byte weights into real training capacity," not "the ratchet is uniquely efficient."
