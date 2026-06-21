# 4-bit Packed Ratchet Storage and Update Fusion

## Goal

Two changes to `DiscreteRatchetLinear`, shipped together:

1. **4-bit packed storage** — halve the persistent footprint (2 bytes/weight -> 1
   byte/weight), doubling parameters-per-GB (~6x -> ~12x vs FP32+AdamW).
2. **Update fusion** — collapse the multi-pass elementwise ratchet update into a
   `torch.compile`-fused region so eager training is fast enough to converge the large
   (~100M) models that the iso-memory experiments need. The current eager update is ~2.3x
   slower per step than FP32, which prevents big models from getting enough optimizer steps.

Neither change requires a hand-written Triton/CUDA kernel. The update is pure elementwise
work plus a per-row reduction, which `torch.compile` fuses automatically. Triton is a
fallback only if profiling proves the compiler insufficient — expected: not needed.

Both changes are **behavior-preserving**: identical code moves, identical results. The only
externally visible change is the on-disk checkpoint format.

## Section 1: Packed Storage

Replace the two `int8` buffers `code` and `pressure` with a single `uint8` buffer `packed`,
one byte per weight:

- low nibble  = `code + max_code`      (range 0..2*max_code; <=8 for nonary)
- high nibble = `pressure + 7`         (range 0..14)

Lossless for every supported tier: code is in `[-max_code, max_code]` with `max_code <= 4`
(<=9 states), and stored pressure is provably in `[-7, 7]` for `pressure_threshold <= 8`
(any value reaching +-8 triggers a code move and has the threshold subtracted off the same
step, so the persisted residual never exceeds +-7). 15 pressure values and 9 code values
each fit a signed nibble.

API:
- `code` and `pressure` become read-only **properties** that unpack from `packed` (as
  `int8` tensors, matching today's dtype and sign).
- A private `_pack(code, pressure)` writes both nibbles back into `packed` after an update.
- `persistent_state_bytes` = `packed.numel() * 1 + scale.numel() * 4`.

The runtime audit (`audit_no_master_weights`) and `compare_persistent_footprint` are updated
to count the single packed buffer (1 byte/weight) and to validate `packed` is `uint8` with
the expected shape. The master-weight invariant is unaffected (no FP matrix parameter added).

## Section 2: Update Fusion

`apply_normalized_gradient` (and the RMS step in `apply_weight_gradient`) keep the same math
but are restructured into a single function compiled with `torch.compile`:

- One fused region: RMS-normalize gradient (per output row) -> `bucket_pressure` ->
  accumulate pressure -> threshold check -> move codes (respecting `+-max_code` bounds and
  recording blocked moves) -> clamp -> re-pack nibbles into `packed`.
- Today these are ~6-8 separate tensor ops each allocating a full-matrix temporary; fused
  they become ~1-2 kernels that read/write `packed` once.
- Returns the same `RatchetUpdateStats` (positive/negative/blocked moves, gradient RMS mean).
- Compilation is lazy/opt-in and falls back to eager: an eager code path is retained as the
  correctness reference and for debugging.

**Profile gate:** before finalizing, profile the current update to confirm it is the ~2.3x
bottleneck (vs FP effective-weight materialization). Pull whatever dominates into the
compiled region. The compiled update is the deliverable regardless.

## Section 3: Testing and Correctness

Correctness is the project's core value; behavior must be provably unchanged.

- **Golden-reference equality:** the packed + compiled update produces bit-identical `code`,
  `pressure`, and `RatchetUpdateStats` as the pre-change eager/int8 update, across random
  gradients and all tiers (`max_code` in {2,3,4}). Reference computed from a retained eager
  implementation.
- **Pack/unpack round-trip:** for every valid (code, pressure) pair in range,
  `unpack(pack(code, pressure))` is the identity.
- **Footprint:** audit and `compare_persistent_footprint` assert `packed` is `uint8`,
  1 byte/weight, and the halved persistent total.
- **Regression:** all existing tests (forward, resume, controls, audit invariant — currently
  36) pass unchanged.
- **Invariant:** `audit_no_master_weights` still reports zero violations.

## Section 4: Sequencing and Checkpoint Handling

- Implemented on a feature branch, test-first, as two staged commits (storage, then fusion)
  so regressions stay bisectable even though the changes ship together.
- **Checkpoint format breaks.** No migration shim. The in-flight 60k convergence runs finish
  and are analyzed on the old code (they hold their code in memory, unaffected). Pre-existing
  `runs/` checkpoints will not resume into the new format; they are terminal experiments.
  New runs use the packed format from the start.
- Land on the branch, fast-forward to `main`, push to the public repo.

## Out of Scope

- Packed-matmul GEMM (reading packed codes directly in the matmul). That is the separate,
  months-long "transient-memory + speed at scale" effort, gated on the science. This spec
  keeps the FP effective-weight materialization for the matmul unchanged.
- Sub-byte entropy-optimal packing (2.32/2.81/3.17 bits). The byte-aligned 4-bit nibble
  format is deliberately chosen for simplicity and kernel-ecosystem compatibility.
