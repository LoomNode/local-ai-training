# Docs Index — read this first

Entry point for understanding the ratchet research with no prior context. Start here, then
follow the pointers. `CLAUDE.md` and `AGENTS.md` hold the non-negotiable invariants; this file
holds the *concepts and the story*.

## The one-paragraph thesis

Can a Transformer learn with **no persistent floating-point master weights** — only low-state
integer codes? Standard training stores, per weight, an FP32 master (4 bytes) + optimizer state
(~8 bytes for Adam). The ratchet stores instead: an `int8` **code**, an `int8` **pressure**
accumulator (packed together into *one* `uint8` = 1 byte), and one FP32 **scale per output row**.
The effective weight is `code * row_scale`. There are no master weights and no per-weight
optimizer state. This is a **memory** technique: ~12x more parameters per GB of persistent state.

## Core concepts

- **code / pressure / scale.** A ratchet matrix persists only: `int8` code in
  `[-max_code, max_code]`, `int8` pressure, one FP32 scale per row. `max_code` sets the state
  count: 1=ternary (3 states), 2=quinary (5), 3=septenary (7), … 7=fifteen-state.
- **The update rule.** Eager forward/backward materialize a *temporary* FP effective weight and
  its gradient. `bucket_pressure` turns the RMS-normalized gradient into integer pressure (a
  descent direction); pressure accumulates, and at a threshold a code moves one step, keeping the
  residual. Boundary outward-moves are recorded as **blocked moves** so pressure can't wind up.
  The FP temporaries are released each step. See
  `superpowers/specs/2026-06-20-ratchet-training-design.md`.
- **Packed storage.** code (low nibble) + pressure (high nibble) in one `uint8`, lossless for
  threshold ≤ 8 and code ≤ ±7. All 3–15 state counts cost the same 1 byte/param.
- **Two comparison axes.** *Iso-parameters* (same param count): FP32 wins — quantization costs
  quality. *Iso-memory* (same MB): the ratchet packs ~12x more params, so it wins at budgets
  where FP32 is too small to be competent. The memory thesis lives on the iso-memory axis.
- **The matmul modes** (added 2026-06): `fp32` (default, eager, CPU-ok), `bf16`, and `int8`
  (Triton int8 GEMM in the linear layers). `bf16` vs `int8` is the controlled comparison that
  isolates quantization noise; FP32 is a separate absolute-quality reference.

## Status at a glance (see ROADMAP.md for detail)

- **Trainability: proven.** Codes learn without master weights (frozen controls confirm it).
- **States→quality: a clean monotonic dial**; gains taper, states alone won't reach FP32.
- **Memory win: real**, ~12x params/GB, lossless packing.
- **Training speed: int8 GEMM delivers ~2x** with a hand-tuned Triton kernel (the earlier
  "no speedup" NO-GO was a vendor-kernel artifact and was **overturned**). End-to-end speedup is
  width-gated (crossover ~K=4096), so it switches on at frontier scale — where the memory win
  also lives.
- **Open:** does int8 in the loop *converge* to the same loss? (the 1.4%-per-step gradient noise
  question). Plus algorithmic update-rule improvements as the lever toward FP32 quality.

## Reading order for results (`results/`)

1. `2026-06-20-smoke.md` — first end-to-end sanity check.
2. `2026-06-20-controls.md` — frozen/FP32 controls: learning comes from code moves.
3. `2026-06-20-scaleup-25m.md` + `2026-06-20-text8-states-curve.md` — states→quality + the
   iso-memory reframing (the key conceptual pivot).
4. `2026-06-20-trainable-scale.md` — trainable per-row scale: NULL result (keep it off).
5. `2026-06-21-eager-throughput.md` — the ratchet is already ~0.91x FP32; the speed lever is the
   matmul, not the update.
6. `2026-06-21-int8-tuned-kernel-reversal.md` — **the current speed story**: tuned Triton int8
   delivers 2x; forward/backward/activation-precision tables. Supersedes the three NO-GO notes
   below.

> **Superseded notes** (kept for the record, conclusions overturned — each carries a banner):
> `2026-06-21-forward-kernel-prototype.md`, `2026-06-21-int8-activation-spike.md`,
> `2026-06-21-tuned-int8-gemm-bench.md`. Their *measurements* stand; their NO-GO *conclusions*
> were overturned by the reversal note. Lesson: a vendor kernel underperforming is not evidence
> the hardware can't.

## Designs, plans, handoffs

- Specs: `superpowers/specs/` (ratchet design, packed storage, int8 training path, BitNet eval).
- Plans: `superpowers/plans/`.
- In-flight: `HANDOFF-int8-convergence.md` + `superpowers/specs/2026-06-21-int8-training-path-design.md`
  — the int8 forward+backward path (task 1 done) and the matched convergence experiment (task 2).
