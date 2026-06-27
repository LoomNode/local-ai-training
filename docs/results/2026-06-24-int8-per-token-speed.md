# int8 Per-Token Speed: dense baseline, compile_update, and the int8-backward NO-GO

**Date:** 2026-06-24
**Hardware:** 2× RTX 3090 (24 GB). Throughput on GPU 1; convergence arms on both.
**Code:** branch `feat/int8-per-token-speed`. Throughput via `scripts/int8_training_throughput.py`
(sustained tok/s, one sync per timed block). Convergence via `lat train`, text8, 512-width 25M
config, 5k steps.

## Question

Now that the int8 ratchet *learns* (momentum ~84%), can it be **faster per token than the
bf16-then-PTQ workflow** — i.e. is master-weight-free int8 a time win given you quantize anyway?
The honest baseline for that question is **dense bf16**: a plain `nn.Linear` under bf16 autocast
(fp32 master + fp32 Adam + bf16 compute) — standard mixed-precision training, the real cost of the
"train in bf16, PTQ afterward" alternative.

> **Methodology correction.** The earlier throughput note
> (`2026-06-24-int8-training-throughput-final.md`, now superseded) compared int8 against the **bf16
> *ratchet*** (`matmul_mode="bf16"`), which carries the ratchet's effective-weight materialization
> overhead. int8 "beating" that baseline was beating a crippled opponent. Measured against **dense
> bf16** — the workflow int8 must actually beat — int8 does **not** win per token at any fittable
> width.

## Frontier throughput sweep (compile_update ON)

Sustained tok/s, integrated 25M model, batch shrinks with width to stay on-GPU. Speedup is vs
dense_bf16 at each width.

| width | dense_bf16 | int8 | int8_bwd (hybrid) |
|------:|-----------:|-----:|------------------:|
| 512   | 1.000 (236k t/s) | 0.547 | 0.235 |
| 1024  | 1.000 (71k)      | 0.617 | 0.327 |
| 2048  | 1.000 (18k)      | 0.658 | 0.455 |
| 3072  | 1.000 (7.8k)     | 0.662 | 0.522 |
| 4096  | **OOM**          | 2.7k t/s | 2.3k t/s |

**Reads:**

1. **Plain int8 never beats dense per token** at fittable width — it plateaus at ~0.66×. Bare-GEMM
   and isolated-backward microbenches (1.6–2.1× bf16) overstated the model-level win: the
   integrated model is dominated by bf16 attention, embeddings, and RMSNorm that the int8 GEMM
   does not touch. The per-token-vs-dense thesis is **lost in eager** on a 3090.
2. **The real win at 4096 is memory, not speed.** Dense bf16 OOMs at width 4096; the int8 ratchet
   trains there. The headline is "1-byte weights let you train where dense cannot run at all,"
   not "faster per token."
3. **`compile_update` is the banked speed win** — it collapses the ratchet update overhead
   (−31% @2048 / −47% @4096 vs the eager update). All numbers above already include it. It is the
   trusted, no-caveat lever (no `suppress_errors`, no accuracy change).

## The int8-backward (hybrid) experiment

`int8_backward=True` runs **grad_input in int8** (fold the per-out weight scale into grad, int8 GEMM
against the persistent `code` — no bf16 effective weight materialized) while **grad_weight stays
bf16** (the expensive, lower-payoff facet). Isolated-GEMM spikes suggested this hybrid crosses bf16
around width 2048. Integrated, it does **not**: int8_bwd is **slower than plain int8 at every
width** (0.455 vs 0.658 @2048). The grad quantization passes — and, with stochastic rounding, the
`tl.rand` pass over the full `[M, out]` gradient each backward — cost more than the bf16 grad_input
GEMM they replace.

### Convergence: stochastic rounding restores parity

Deterministic int8 grad_input failed convergence parity (round-to-nearest bias compounds over 5k
steps). **Stochastic rounding** (unbiased, `floor(y+u)`) fixes it. Seed-robustness A/B, 512-width,
5k steps, text8, `off` = int8 fwd + bf16 backward, `onSR` = int8 fwd + int8 grad_input with SR:

| seed | off (bf16 bwd) | onSR (int8 grad_input + SR) | gap |
|-----:|---------------:|----------------------------:|----:|
| 1337 | 1.3124 | 1.3325 | +0.020 |
| 1338 | 1.3750 | 1.3084 | −0.067 |
| 1339 | 1.3319 | 1.3964 | +0.065 |
| **mean** | **1.340** | **1.346** | **+0.006** |

The mean gap (+0.006 nats) is negligible and swamped by per-arm seed variance (±0.03–0.07); the
gap flips sign across seeds. **int8 grad_input + SR converges on par with bf16 backward.** The
single-seed "+0.020 penalty" first observed at seed 1337 was one draw of a noisy distribution.

### Verdict: int8_backward is a documented NO-GO

It converges fine but is a **per-token throughput regression** against plain int8 — and plain int8
already loses to dense. Converging on par does not rescue being slower. The `int8_backward` flag and
its SR kernel path are kept (default off) as a **preserved negative result**, not a recommended
mode. The deterministic-vs-SR throughput disentangle is moot: deterministic fails convergence, SR
fails speed, so neither config wins both axes.

## What to actually use

- **`compile_update=True`** for any int8 ratchet run — real, trusted −31–47% on the update.
- **int8 for memory, not per-token speed** — train at widths where dense bf16 OOMs (≥4096 on a
  3090); that is the genuine master-weight-free advantage, alongside skipping a separate PTQ step.
- **Do not** enable `int8_backward` for throughput.

## Thermals (overnight, 2×3090 sustained)

No throttle events. Peak GPU 78 °C, peak CPU (Tctl) 88.9 °C — both well under throttle (GPU ~83 °C,
CPU ~95 °C).
