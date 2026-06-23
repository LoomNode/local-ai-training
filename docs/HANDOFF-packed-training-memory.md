# Handoff: realize the 1-byte training PEAK (packed kernels)

**For Codex.** Read `CLAUDE.md`, `AGENTS.md`, `docs/README.md` first. Git is `LoomNode` — never
expose real identity. Run with `MPLCONFIGDIR=/tmp/mpl UV_CACHE_DIR=/games/ailab/.uv-cache uv run ...`.
2x RTX 3090; pick an idle one via `nvidia-smi` + `CUDA_VISIBLE_DEVICES`. Use brainstorm -> spec ->
plan, TDD, stop after each phase for review. Keep `lat audit` clean (no FP/bf16 Parameter mirroring
a code matrix — the core invariant).

## Why

Ratchet persistent state is ~1 byte/param. But the **training PEAK** memory is currently ~FP-sized,
because each eager step materializes a full FP **effective weight** (`code * scale`) and a full FP
**gradient w.r.t. that effective weight**, then converts the grad to pressure and frees them. So the
1-byte win is realized for storage/inference but NOT for training peak — which is what actually limits
how big a model you can train. Goal: make training peak approach ~1-2 bytes/param + activations, so a
genuinely larger model trains on the same VRAM.

## What's already done (build on it)

`matmul_mode="int8"` (committed this session, `src/local_ai_training/int8_matmul.py`,
`ratchet.py`) already consumes the integer `code` directly as the int8 matmul operand in the FORWARD —
it does NOT rebuild a full FP effective weight there. So part of the forward win may already exist.
The remaining FP-sized transients are: (a) the BACKWARD effective-weight gradient `grad_W_eff`
(shape `[out, in]`, full FP), computed then fed to `bucket_pressure`; (b) stored activations for
backprop. Do not re-solve the forward.

## Phase 0 — MEASURE first (cheap, decisive, do before any kernel)

Add a peak-memory probe: train a few steps at a fixed config under each `matmul_mode`
(`fp32`/`bf16`/`int8`) and record `torch.cuda.max_memory_allocated()`, broken down (weights,
transient effective-weight, grad_W_eff, activations, optimizer-on-support). Deliverable: a table
showing how far int8 already lowers training peak vs fp32/bf16, and which transient now dominates.
This SCOPES the kernel work. Write to `docs/results/`. Likely finding: `grad_W_eff` + activations
dominate; if so Phase 1 targets the backward.

## Phase 1 — fuse the backward (the main kernel work)

Today: backward forms the full FP `grad_W_eff = grad_out^T @ x` ([out,in]), then `apply_weight_gradient`
-> `bucket_pressure` -> integer pressure/code update. The full FP `grad_W_eff` is the transient to kill.
Build a fused path that produces the integer pressure update **without ever holding a full
`[out,in]` FP gradient** — e.g. tile/stream over output rows (the per-row scale and pressure are
already per-row), computing each row-block's gradient, its bucketed pressure, and the code move, then
discarding it. Peak backward memory becomes O(tile), not O(full matrix). MUST stay bit-identical (or
within a declared tolerance) to the current `bucket_pressure` update — add an equivalence test vs the
existing eager update BEFORE optimizing. Do not change the update RULE, only how it's computed.

## Phase 2 — activations

Apply activation checkpointing to the transformer blocks so the activation term stops dominating peak.
Standard PyTorch `torch.utils.checkpoint`; measure the peak reduction. Orthogonal to Phase 1.

## Success metric

`max_memory_allocated` per param during training drops from ~FP-sized toward ~1-2 bytes/param +
bounded activations, demonstrated by training a model at a size that OOMs in fp32/bf16 but fits with
the packed path. Report the largest model that trains on one 3090 (and on 2) under each mode.

## Guardrails

- `lat audit` stays violation-free; no persisted FP code-mirror.
- Preserve `runs/tiny-shakespeare`; write elsewhere.
- Equivalence test before perf work; preserve any divergence, don't tune it away.
- 3090-specific numbers; note it.
- This is a separate concern from the int8 convergence gate (task 2) — coordinate branches.
