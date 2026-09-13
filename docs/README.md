# Docs Index — read this first

Entry point for understanding the ratchet research with no prior context. Start here, then
follow the pointers. `CLAUDE.md` and `AGENTS.md` hold the non-negotiable invariants; this file
holds the *concepts and the story*.

## The one-paragraph thesis

Can a Transformer learn with **no persistent floating-point master weights** — only low-state
integer codes? Standard training stores, per weight, an FP32 master (4 bytes) + optimizer state
(~8 bytes for Adam). The ratchet stores instead: an `int8` **code**, an `int8` **pressure**
accumulator (packed together into *one* `uint8` = 1 byte), and one FP32 **scale per output row**.
The optional row RMS EMA is FP32 support state. The effective weight is `code * row_scale`.
There are no master weights and no per-weight optimizer state. Comparing ratchet matrix state with
an FP32 matrix plus Adam state gives roughly 12x more matrix parameters per persistent-state GB.
That ratio excludes row scales, other support parameters, activations, and transient workspaces; it
is not a whole-model training-memory guarantee.

## Core concepts

- **code / pressure / scale.** A ratchet matrix persists one `uint8` containing code and pressure
  nibbles per weight, one FP32 scale per row, and optional FP32 row RMS EMA. `max_code` sets the state
  count: 1=ternary (3 states), 2=quinary (5), 3=septenary (7), … 7=fifteen-state.
- **The update rule.** The current fused backend tiles the effective-weight gradient and applies
  integer state updates during backward; the FP32 mode may materialize a temporary effective weight.
  `bucket_pressure` turns the RMS-normalized gradient into integer pressure (a
  descent direction); pressure accumulates, and at a threshold a code moves one step, keeping the
  residual. Boundary outward-moves are recorded as **blocked moves** so pressure can't wind up.
  The FP temporaries are released each step. See
  `superpowers/specs/2026-06-20-ratchet-training-design.md`.
- **Packed storage.** code (low nibble) + pressure (high nibble) in one `uint8`, lossless for
  threshold ≤ 8 and code ≤ ±7. All 3–15 state counts cost the same 1 byte/param.
- **Two comparison axes.** *Iso-parameters* (same param count): FP32 wins — quantization costs
  quality. *Iso-matrix-state* compares the packed matrix with FP32 plus Adam at roughly 12x before
  row scales, support state, activations, and transients. Whole-model peak-memory measurements are
  reported separately.
- **The matmul modes** (added 2026-06): `fp32` (default, eager, CPU-ok), `bf16`, and `int8`
  (Triton int8 GEMM in the linear layers). `bf16` vs `int8` is the controlled comparison that
  isolates quantization noise; FP32 is a separate absolute-quality reference.

## Status at a glance (see ROADMAP.md for detail)

- **Trainability:** the corrected backend completed a fresh 15-run Shakespeare study. Trainable
  quinary/septenary mean validation losses were 1.829/1.796 versus frozen 3.253/3.262, with
  bitwise frozen-state checks. See `results/2026-09-12-shakespeare-rerun.md` for protocol and limits.
- **States→quality: a clean monotonic dial**; gains taper, states alone won't reach FP32.
- **Persistent matrix-state reduction:** roughly 12x versus FP32 matrix plus Adam before scales,
  support state, activations, and transients. Corrected whole-model memory measurements are in
  `results/2026-06-22-corrected-memory-sweep.md`.
- **Training speed:** a tuned Triton int8 GEMM reached about 2x in the bare-kernel benchmark. Later
  integrated tests found int8 was not faster than dense BF16 at widths that both fit; its observed
  advantage at width 4096 was capacity because BF16 OOMed. See
  `results/2026-06-24-int8-per-token-speed.md`.
- **Integrity correction verified:** frozen-control invariants, tokenizer-safe resume, and
  version-2 reproducible checkpoints passed the full CPU and RTX 3090 suites on 2026-09-12.

## Reading order for results (`results/`)

[Integrity audit and rerun tracker](results/2026-09-12-rerun-tracker.md).

Fresh verification: [2026-09-12 Shakespeare rerun](results/2026-09-12-shakespeare-rerun.md) and
[2026-09-13 provenance reruns](results/2026-09-13-provenance-reruns.md) (text8 30k, subword seed 1338, 1B).

Momentum update rule at 15 codes: [2026-09-13 momentum knob grid](results/2026-09-13-momentum-grid.md)
(5k screen: EMA denominator recovers 35% of the QAT gap, the pressure leak recovers nothing;
30k three-seed confirmation running).

1. `2026-06-20-smoke.md` — first end-to-end sanity check.
2. `2026-06-20-controls.md` — historical frozen/FP32 controls; apply the source-provenance caveat
   documented in `2026-09-12-experiment-integrity.md`.
3. `2026-06-20-scaleup-25m.md` + `2026-06-20-text8-states-curve.md` — states→quality + the
   iso-memory reframing (the key conceptual pivot).
4. `2026-06-20-trainable-scale.md` — trainable per-row scale: NULL result (keep it off).
5. `2026-06-21-eager-throughput.md` — the ratchet is already ~0.91x FP32; the speed lever is the
   matmul, not the update.
6. `2026-06-21-int8-tuned-kernel-reversal.md` — bare-kernel and microbenchmark reversal evidence.
7. `2026-06-22-corrected-memory-sweep.md` — corrected whole-model memory measurements.
8. `2026-06-24-int8-per-token-speed.md` — current integrated convergence, speed, and capacity
   interpretation. It supersedes broader speed conclusions drawn from the microbenchmark.

> **Superseded notes** (kept for the record, conclusions overturned — each carries a banner):
> `2026-06-21-forward-kernel-prototype.md`, `2026-06-21-int8-activation-spike.md`,
> `2026-06-21-tuned-int8-gemm-bench.md`. Their *measurements* stand; their NO-GO *conclusions*
> were overturned by the reversal note. Lesson: a vendor kernel underperforming is not evidence
> the hardware can't.

## Designs, plans, handoffs

- Specs: `superpowers/specs/` (ratchet design, packed storage, int8 training path, BitNet eval).
- Plans: `superpowers/plans/`.
- Current status: `ROADMAP.md` and `results/2026-09-12-experiment-integrity.md`.
- Historical handoffs are retained for provenance. `HANDOFF-int8-convergence.md` and
  `HANDOFF-packed-training-memory.md` describe work that has since landed and are marked obsolete.
