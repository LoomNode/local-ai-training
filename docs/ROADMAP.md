# Roadmap

Tracks the big-picture trajectory of the master-weight-free ratchet training research.
Detailed results live in `docs/results/`; designs in `docs/superpowers/specs/`.

## Done

- **Trainability** — discrete ratchet matrices learn with no FP master weights; frozen/FP32
  controls confirm the learning comes from code moves, not the FP support params.
- **States-vs-quality curve** (text8, 25M) — 5/7/9-state codes give a clean monotonic dial;
  more states -> lower loss + less saturation. Gains taper; states alone won't reach FP32.
  State range now spans ternary..15 (`max_code` 1..7, the 4-bit nibble cap).
- **QAT de-confounding (2026-06-22)** — an STE-QAT control (keeps an FP32 master + Adam, quantizes
  to the *same* few states) splits the FP32->ratchet gap into its two confounded halves. Few-states
  cost is cheap (<=0.055 nats); **master-weight-free training owns 76-82% of the gap**, rising with
  state count. So the update rule — not the state count — is the ceiling. The master-free penalty is
  a quality *floor*, not a waitable slowdown: the ratchet asymptotes above matched QAT and the gap is
  stable-to-widening at 30k (more steps don't close it). `docs/results/2026-06-22-qat-deconfounding.md`.
- **Adaptive per-row scale (screening, 2026-06-23)** — trainable per-row scale (AdamW `log_scale`,
  audit-clean 1-D state) closes ~26% of the master-free gap at **5 states** (worst saturation, 46%),
  null at 7/9 — a clean gradient tracking saturation. Mechanism: scale *rescales* saturated codes but
  cannot *un-saturate* them (a rail-pinned code stays pinned), so it only helps the coarsest grid.
  Independently confirms the lever is code resolution / the update rule, not the scale.
  `docs/results/2026-06-23-adaptive-scale-ratchet.md`.
- **Iso-memory (MB-for-MB)** — the right axis for "params per GB". Ratchet wins at small
  budgets (nonary-25M beats FP32-2M at ~22MB); FP32 wins at ~89MB. Caveat: big eager ratchets
  were undertrained at fixed step budgets; convergence runs stopped early.
- **4-bit packed storage** — code+pressure in one uint8 (lossless); ~6x -> ~12x params/GB.
- **torch.compile-fused update** — 3.17x on the isolated update; opt-in `compile_update`.
- **Eager throughput finding** — the ratchet is already ~0.91x FP32 throughput (the "2.3x
  slower" was GPU contention); update fusion is only ~4% end-to-end. Real speedup needs the
  matmul to exploit the low bits, not the update.
- **int8 per-token speed: memory win, not a speed win (CORRECTED 2026-06-24)** — Measured against
  the right baseline — **dense bf16** (`nn.Linear` under bf16 autocast = standard mixed-precision
  training, the real bf16-then-PTQ cost), not the bf16 *ratchet* — the integrated int8 model is
  **~0.66× dense and never beats it** per token at fittable width (512→3072). Bare-GEMM microbenches
  overstated it; the model is dominated by bf16 attention/embeddings/norms the int8 GEMM doesn't
  touch. The genuine advantage at width 4096 is **memory**: dense OOMs there, the int8 ratchet
  trains. `compile_update` is the real banked speed lever (−31–47% on the update). An **int8
  backward** (`int8_backward`, grad_input in int8 + stochastic rounding) converges on par with bf16
  (mean +0.006 nats over seeds 1337/1338/1339, within seed noise) but is a **per-token regression vs
  plain int8** — kept default-off as a preserved negative result. Supersedes the earlier "int8 beats
  bf16 at 4096" claim, which compared against a crippled bf16-ratchet baseline. See
  `docs/results/2026-06-24-int8-per-token-speed.md`.

## REOPENED: training-speed investigation (the earlier NO-GO was wrong)

**Correction (2026-06-21).** The prior NO-GO was an artifact of trusting *vendor* int8 kernels
(`torch._int_mm` / cuBLASLt IMMA, torchao autotuned Triton) — all of which stall at ~33-42% of
int8 peak on the 3090 at *every* width. A **hand-written autotuned Triton int8 GEMM** reaches
~100% of int8 peak and **delivers the full 2x the silicon spec promises**. The bare-GEMM NO-GO
is dead; the int8 tensor-core speedup is real and reachable — it was a kernel-quality problem,
not a hardware ceiling and not a model-size problem. See
`docs/results/2026-06-21-int8-tuned-kernel-reversal.md`.

- **Bare int8 GEMM** (`scripts/int8_spike/triton_int8_gemm.py`, correctness err=0): **2.10x bf16**
  at width 8192-12288, **1.88x even at K=768**. Vendor kernels (~35% of peak) were the problem.
- **End-to-end fused int8 linear, forward** (`scripts/int8_spike/fused_int8_linear.py`): per-token
  activation quant + int8 GEMM + dequant-fused epilogue. **1.8-1.95x at width 8192-12288**
  (~1.4% rel err), break-even ~2048, but **0.60x at K=768** — the fixed quant overhead only
  amortizes at large width. So the *end-to-end* win is width-gated, switching on exactly where
  the ratchet's memory win lives (frontier-scale models). Bare GEMM wins at all widths.
- **Activation precision** (`scripts/int8_spike/activation_precision.py`): int8 activations
  ~1-6% rel err (per-token); **int4 activations ~62% — dead.** int8 is the activation precision.
- **Ratchet alignment:** the stored `code` (int8-representable, +-7) IS the int8 weight operand;
  the per-row scale IS the weight scale. The ratchet maps onto int8xint8 with no extra weight quant.
- **Full training step (fwd + both backward GEMMs)** (`scripts/int8_spike/int8_backward_bench.py`):
  **1.6-1.7x at width 8192-12288**, crossover ~K=4096, **<1x at toy width** (the x6 quant passes
  swamp small matmuls). Gradients ~1.4% accurate vs bf16. So the per-step speedup holds through
  backward, gated on frontier width.
- **STILL UNPROVEN:** end-to-end *convergence* under int8 activations — per-step grads are ~1.4%
  noisy; does that compound over thousands of steps into worse final loss, or wash out? Needs an
  actual int8-in-the-loop training run vs bf16 (at width >=4096 to also show the speedup). This is
  the last gate before "the ratchet trains faster at scale" is a standing claim.
- **Caveat:** all 3090-specific; A100/H100/FP8 differ.

## Upcoming

### 0. int8 training path + convergence (ACTIVE)
The int8 forward+backward matmul path is wired into ratchet linears behind `matmul_mode`
(`fp32` default / `bf16` / `int8`) — **task 1 done** (audit clean, matched-init tests pass).
Spec: `docs/superpowers/specs/2026-06-21-int8-training-path-design.md`. **Task 2 (pending):** a
matched `bf16` vs `int8` convergence run — does the ~1.4%/step gradient noise compound or wash
out? Run at width >=4096 to also show the speedup. Its own brainstorm->spec->plan cycle. This is
the gate that turns "1.6-1.7x per step" into "trains faster at scale."

### 1. BitNet b1.58 evaluation review
Review/validate the parallel BitNet evaluation harness (`src/local_ai_training/bitnet_*.py`)
as a low-bit reference point. Goal: a clean apples-to-apples baseline (pretrained BitNet
b1.58, ternary 1.58-bit) to compare the ratchet against.

### 2. Fairer iso-memory re-run
Re-run MB-for-MB with `compile_update` and enough steps that the big ratchet models actually
converge, on dedicated GPUs (avoid the contention/undertraining confound). Optionally pair
with algorithmic update-rule improvements (the per-bit gains taper; the update rule, not the
state count, is the likely ceiling).

### 3. Algorithmic update-rule improvements (PRIMARY quality direction)
**Two independent results now converge here.** QAT de-confounding showed master-weight-free training
owns 76-82% of the gap; the adaptive-scale screen showed scale adaptation only rescues the most
saturated regime (5 states) and cannot touch 7/9. Neither states, scale, nor bits is the ceiling —
the **pressure/bucket update rule** is. This is the lever for closing the quality gap to FP32.

Concrete first hypothesis: QAT keeps Adam's first/second *moments*; the ratchet's pressure
accumulator has none — it is a plain integer integrator. A **momentum/variance analogue in the
pressure accumulation** (the master-weight-free counterpart of Adam state, still 1-byte-ish per
weight) is the obvious first thing to test. Other angles: adaptive `pressure_threshold`, smarter
bucket boundaries, and reducing blocked-move pressure wind-up at the rail. Needs its own
brainstorm->spec->plan; screen at 5k steps (effects flip sign before ~step 3000) before any 30k
confirmation.

### 5. Ratchet the token embedding (close the last FP master carve-out → unlock large vocab)
The output `lm_head` is **already** a ratchet matrix (master-free), so the loss-bearing
`vocab × n_embd` table scales master-free today. The **only** remaining FP master weight among the
big tables is the input `token_embedding` (`nn.Embedding`, AdamW-trained). It's a defensible,
standard carve-out (most quantization work keeps embeddings higher precision) and is a small
fraction at real model size — but it is what *forces* small/byte-level vocab: a subword vocab
(30k–50k) would make this one FP table dominate the parameter count and dilute the thesis. So
large-vocab usable models are gated on ratcheting the embedding.

The real obstacle is **sparse gradients**: each step only the rows for in-batch tokens get a
gradient, whereas `bucket_pressure` assumes a dense, RMS-normalized gradient over the whole matrix.
So this needs a **per-row / sparse-aware update variant** (and a per-row scale, like the rest of the
ratchet). Cheaper interim levers that shrink but do not *remove* the carve-out: BF16 embedding
storage (~2× weights, but AdamW moments stay FP32 and it's still a master weight), or a **factorized
embedding** (`vocab × r` then `r × n_embd`, the second factor itself ratchetable). Own
brainstorm→spec→plan. Until then, **byte-level corpora (vocab ~200) keep every door open** — they
make the embedding cost trivial without touching this question. NOT a blocker for current research
or demos; only for the large-vocab/usable-model direction.

### 4. The honest next frontier for int8-class speed: int4 (accuracy experiment)
With the easy bit-exact int8-specific levers exhausted, the next throughput/memory frontier is
int4 — but it is fundamentally an **accuracy experiment**, not a free perf change, so it gets its
own brainstorm->spec->plan and an explicit quality A/B (it is NOT bit-exact). Two distinct angles,
because naive int4 is known-bad: `scripts/int8_spike/activation_precision.py` already showed int4
*activations* are ~62% rel err — dead as-is. So this means either (a) int4 *weights* with int8
activations (the ratchet code is +-7, already 4-bit-representable, so the weight side is free; the
question is a custom int4-weight GEMM and whether it actually beats int8 tensor cores), or (b)
revisiting int4 activations with a better scheme (per-group scales, outlier handling) to get the
error into a trainable range. Gate: does it improve throughput/memory *without* breaking
convergence vs the now-fast int8 baseline. Until that A/B is run, int8 is the floor.
