# De-confounding the ratchet: an STE-QAT control arm

**Date:** 2026-06-22
**Status:** approved design, ready for spec review
**Scale:** 25M on text8 (the existing `text8-25m` config), idle 3090s.

## Why

The ratchet bundles two innovations that have only ever been measured *jointly* (vs FP32):
**(1) few weight states** (quinary 5 / septenary 7 / nonary 9 levels) and **(2) master-weight-free
training** (no FP latent weight, no optimizer state; codes move by the pressure/bucket rule). Every
quality number is FP32-vs-ratchet, which flips both switches at once, so the ~0.14–0.24 nat gap can't
be attributed: is it the cost of few states, or of dropping master weights?

The missing arm is **STE-QAT**: a control that quantizes to the *same* few states but *keeps* an FP
master weight + Adam, updated through a straight-through estimator. That fills the 2×2:

| | master weights | states | source |
| --- | --- | --- | --- |
| FP32 | yes | continuous | reuse `runs/text8-25m/fp32` |
| **QAT(s)** | **yes** | **few (s)** | **new — this work** |
| ratchet(s) | no | few (s) | reuse `runs/text8-25m/{quinary,septenary,nonary}` |

- **FP32 → QAT(s)** isolates the cost of *few states alone*.
- **QAT(s) → ratchet(s)** isolates the cost of *master-weight-free training alone*.

This tells us which lever owns the gap: if QAT≈FP32 but ratchet trails QAT, the update rule is the
problem (work the pressure/bucket rule); if QAT already trails FP32, few states is the inherent cost
and master-weight-free is nearly free on top.

## The QAT quantizer — identical to the ratchet's

`QATLinear` (new, in `model.py`) holds a normal fp32 `weight` Parameter (the master). Forward applies
the **same** quantizer the ratchet uses, per output row:

```
scale = (weight.detach().abs().amax(dim=1, keepdim=True) / max_code).clamp_min(tiny)   # detached
code  = clamp(round(weight / scale), -max_code, +max_code)
effective = weight + (code * scale - weight).detach()   # STE: d effective / d weight = 1
output = input @ effective.t()
```

The STE form makes the forward numerically equal to `code * scale` (bit-identical to a ratchet whose
code equals `round(weight/scale)`), while the gradient flows straight to the master, which Adam
updates. Scale is detached (standard absmax QAT / BitNet convention). **Gradient handling is pure
straight-through: the `round` passes gradient as identity and there is no gradient clamp on saturated
entries** (matching BitNet b1.58; saturated weights still receive gradient, so they can move back into
range). Because the quantizer and the per-row `row_max_abs/max_code` scale match the ratchet exactly,
**QAT→ratchet differs only in the update mechanism** — the clean isolation. At step 0, with shared
init, QAT's code equals the ratchet's initial code.

## Wiring — a new `weight_mode="qat"`

- `build_seeded_model` / `_linear` gain a mode so a non-None `max_code` produces `QATLinear` under qat
  and `RatchetLinear` under ratchet; `max_code=None` stays `nn.Linear` (fp32). Threaded through
  `RatchetGPT` → blocks → `_linear` (the existing `max_code` plumbing path).
- `train_run` accepts `weight_mode="qat"`: trained like the fp32 control (AdamW over all params
  including masters; no `ratchet_update`, no `discard_pending_gradients`).
- **Invariant boundary:** QAT legitimately *has* master weights — it is a control, not a ratchet.
  `audit_no_master_weights` governs only the ratchet arms; it is expected to report masters for a QAT
  model. No ratchet invariant is weakened.
- Shared init holds: same seed + config → one logical FP init; fp32 keeps it continuous, qat keeps it
  as master, ratchet quantizes it to codes (existing invariant in CLAUDE.md).

## Protocol — reuse + reproduce-check, run only the 3 new arms

The four FP32+ratchet arms already exist at the exact target config (`runs/text8-25m/*/seed-1337`,
30000 steps, seed 1337, n_embd 512 / 8L / 8H / block 256 / batch 64, lr 3e-4, pressure_threshold 8).
No QAT arm exists.

1. **Reproduce-check (1 short run):** re-run ratchet-quinary (`max_code=2`, `matmul_mode="fp32"`, no
   checkpointing) for ~300–500 steps at seed 1337 under current code; compare the val-loss trajectory
   to the stored `runs/text8-25m/quinary/seed-1337/metrics.csv` first rows. The eager fp32+ratchet
   path is deterministic, so it should match closely. **If it matches → reuse the 4 existing arms. If
   it has drifted → re-run all 7 fresh** (the session's bf16/int8/checkpointing changes shouldn't
   touch the eager path, but this verifies it rather than assuming).
2. **Run QAT{5,7,9}** (`max_code` 2/3/4, `weight_mode="qat"`) at the identical config/seed, writing to
   a **separate** output (`runs/text8-25m-qat/`) so existing runs are never overwritten.
3. **Analyze:** per state count, tabulate FP32 / QAT(s) / ratchet(s) best val loss and the two
   differences (states cost, master-weight-free cost). Write to `docs/results/`.

## Tests (TDD)

- `QATLinear` forward equals the ratchet effective-weight forward on a shared-seed init weight
  (identical quantizer → identical output).
- STE gradient reaches the master: after `backward`, `QATLinear.weight.grad` is non-None and nonzero,
  including for saturated entries (`|weight/scale| > max_code`) — confirms pure straight-through.
- A qat arm's loss decreases on a tiny repetitive corpus over a few steps (mirrors the existing
  short-corpus trainability test).
- `weight_mode="qat"` is accepted; `audit_no_master_weights` reports masters on a qat model (correct)
  and stays clean on ratchet models.

## Verification

- Tests + full suite green; ratchet-arm `lat audit` still clean.
- Reproduce-check passes (or triggers a documented full re-run).
- Results note with the 2×2 decomposition at 5/7/9, plus the headline attribution sentence.

## Risks

- **Codebase drift** since the existing runs — handled by the reproduce-check gate.
- **A weak QAT baseline would unfairly flatter the ratchet.** Mitigated by matching the ratchet's
  quantizer exactly and the FP32 control anchoring both; QAT uses the same lr/schedule as the fp32
  control. Preserve any QAT divergence rather than tuning it to a target.
- 25M / 30k-step / single-seed (1337) — a trainability+attribution test, not a converged-scale claim;
  note it. A multi-seed repeat is a follow-up if the gap is close.
