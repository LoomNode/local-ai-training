# scrt-lab — Self-Consolidating Recurrent Transformer (MVP)

Research lab for the **hard parameter budget + shared core + residual slots** idea:

> Store the common part once; spend capacity only on residuals; reorganize during training without ever growing physical parameter count.

Origin thread: ChatGPT share *Model Layer Inefficiency Discussion* (local-AI framing).

## Stack (MVP)

| Piece | What it is |
|--------|------------|
| **baseline** | Stacked transformer blocks (independent layers) |
| **looped** | Prelude + **shared core × N loops** + coda (depth without param depth) |
| **scrt** | Looped + **fixed residual slot bank** + periodic **consolidate** |
| **consolidate** | Functional signatures → merge similar slots → free weak → regrow on high pressure |
| **data** | Synthetic synonym language (multiple surface forms ↔ shared concept) |
| **train** | Tiny AdamW LM loop; writes JSON under `results/` |

Hard rule: consolidation **never** changes `numel()` of the model — only occupancy, assignment, and values inside a fixed slot bank.

## Quick start

```bash
cd /games/ailab/scrt-lab
source .venv/bin/activate   # torch 2.13 + cu130 already installed via uv

# unit + smoke tests (CPU)
pytest -q

# full three-arm MVP on research GPU (~1–2 min at default 200 steps)
python scripts/run_mvp.py --steps 200 --run-name mvp

# matched-budget stepped run (v1, easy synonym data)
python scripts/run_stepped.py --steps 3000

# v2: harder structured data + residual merge + throttle (defaults cuda:1)
python scripts/run_v2.py --steps 8000 --corpus structured
```

Results land in `results/{run}_{arm}.json` and `results/{run}_summary.json`.

## Architecture sketch

```
tokens → embed
      → prelude block
      → for loop_i in 1..N:
            shared_core(x)
            x += residual_bank(x, loop_i)   # fixed n_slots LoRA adapters
      → coda block
      → lm_head  (tied to embed)
```

Every `consolidate_every` steps (SCRT only):

1. Probe batch → **functional** signature per slot (mean residual delta — not weight cosine)
2. Merge pairs with cosine sim ≥ threshold → average into survivor, free loser
3. Free near-zero residual slots (skipped while bank still zero-init)
4. Regrow free slots onto loops with highest gradient pressure  
5. Assert param count unchanged

## What success looks like

**MVP (done):** plumbing works — train 3 arms, consolidate, hard param budget.

**v2 (next):** fair scientific signal

| Fix | Why |
|-----|-----|
| Harder data (`structured` / char) | v1 synonym bigrams let all arms tie at the same floor |
| Residual merge (shared + residual) | v1 average-and-drop + full regrow thrashed slots (214 merges/3k steps) |
| Throttle (`max_merges=2`, keep free slots) | Stop constant re-init of residuals |
| `late_improvement` metric | Did SCRT keep improving in the second half after baseline plateau? |

```bash
python scripts/run_v2.py --steps 8000 --corpus structured
# optional real text:
python scripts/run_v2.py --steps 8000 --corpus shakespeare
```

## capacity (2026-08-07) — the synthetic corpora were the bottleneck

`build_structured_language` has an **exact entropy rate of 2.1405 nats/token**
(count the nats the generator injects, divide by tokens emitted). Every born2 arm
finished at 2.1827–2.1860 — inside 0.05 nats of the floor, with the arms separated
by 0.0033. That is the same size as the seed-to-seed noise, so every synthetic run
in `results/` measured the dataset, not the architecture.

`capacity_config` fixes it: full char-level WikiText-2 (10.9M chars, vocab capped at
256), 4 seeds, param-matched at 1,452,672.

```bash
python scripts/run_capacity.py --steps 20000 --seeds 0 1 --device cuda:0 &
python scripts/run_capacity.py --steps 20000 --seeds 2 3 --device cuda:1 &
wait
python scripts/run_capacity.py --aggregate-only --seeds 0 1 2 3
```

| arm | best eval (mean ± std, n=4) | vs previous arm | wall_s |
|-----|------|------|--------|
| baseline | 1.3863 ± 0.0045 | — | 237 |
| looped | 1.3475 ± 0.0063 | **−0.0388 (7.1σ)** | 753 |
| scrt_static | 1.3418 ± 0.0101 | −0.0058 (0.7σ) | 803 |
| scrt | 1.3396 ± 0.0059 | −0.0022 (0.3σ) | 809 |

**Reusing a shared core is the whole effect.** The residual bank and the
consolidation loop are both inside the noise.

Two caveats on that headline, in order of importance:

1. **Not compute-matched.** Looped arms burn 3.2× the wall clock. Baseline was still
   improving at step 20000 (`last_quarter_improvement` +0.0118), so a baseline given
   the same FLOPs might close the 0.039 gap. Until that control runs, the claim is
   "better per parameter", not "better per unit compute" — and for the local-AI
   argument the second is the one that matters.
2. **Consolidation barely ran.** Across 4 seeds and 800 cycles: **50 merges, 1351
   frees, 1335 regrows.** The merge operation — the actual novel idea — fired on 6%
   of cycles; the rest was free/regrow churn. Frees outpace regrows, each regrow
   re-inits a slot near zero so it falls back under `residual_free_threshold=0.12`
   and is freed again. Active slots collapse 14 → 4 of 16 by step 8000 in every seed.
   The loop is not consolidating, it is reinitialising.

### thrash fix (2026-08-07) — fixed, and it did not matter

`min_slot_age_cycles=5` (a regrown slot cannot be freed until it has had time to
bootstrap `up` away from zero) plus `max_frees_per_cycle=2` (frees were the only
unthrottled op). Re-ran the `scrt` arm at full length on seeds 0 and 2
(`results_fixed/`):

| | old | fixed |
|---|---|---|
| active slots (mean) | 6.9 / 6.4 | **14.0 / 14.0** |
| active slots (min) | 4 / 4 | **10 / 10** |
| merges | 16 / 11 | 12 / 13 |
| best eval | 1.3341 / 1.3404 | 1.3361 / 1.3369 |

The bank stays fully employed now. **Merge count and loss are unchanged** (−0.0007
nats against a 0.0059 seed std).

Why: it was never a starvation problem. With a healthy bank, **192 of 200 cycles
contain no pair anywhere above sim 0.90** — max pair similarity falls from 0.985 at
step 100 to ~0.55 by mid-training and stays there. The residual slots learn distinct
functions and never re-converge, so there is no redundancy for the merge operation
to find. The early merges happen while slots are still near their shared
initialisation, i.e. they merge noise, not learned redundancy.

That is a negative result about the premise, not a tuning failure. "Slots will
redundantly learn the same thing, so detect and merge" does not hold in this setup.
Before spending more on the closed loop, the thing to establish is whether
*functionally redundant structure exists at all* at this scale — measure the
similarity distribution first, and only build consolidation if the mass is there.

### slot sweep (2026-08-08) — capacity does not bind, and the scope was 3.4%

`scripts/run_slotsweep.py` varies how many bank slots start occupied at **identical
parameter count** (occupancy is a buffer, not a parameter), using `scrt_static` so
occupancy is frozen. 3 levels x 3 seeds, 20k steps:

| occupied | best eval mean | std |
|---|---|---|
| 4 | 1.3419 | 0.0094 |
| 8 | 1.3420 | 0.0089 |
| 16 | 1.3424 | 0.0075 |

Spread 0.0005 nats over a 4x range of slots vs 0.0094 seed noise. Slot capacity is
not binding. Seed choice matters far more than slot count (seed 3 is worst at every
level, seed 2 best).

**Why none of this could have worked — parameter breakdown:**

| group | params | share |
|---|---|---|
| prelude | 443,136 | 30.5% |
| shared core | 443,136 | 30.5% |
| coda | 443,136 | 30.5% |
| embed/head | 74,112 | 5.1% |
| **bank** | **49,152** | **3.4%** |

Consolidation only ever touches the bank. A perfect merge halving the bank frees
1.7% of the model — below the seed noise by construction. **The experiment was
structurally incapable of a positive result**, independent of how well merge/free/
regrow work. This is a scope problem, not a scale problem: the model is not too
small (it was handed 4x the slots and did nothing with them), the mechanism is
aimed at the wrong 3%.

Two further observations from the same table:

- The looped/shared arms reuse only the **30.5%** that is the shared core. Prelude
  and coda are 61% of parameters and run exactly once each. The -0.039 nat win comes
  from reusing less than a third of the model; nobody has tried reusing the rest.
- When this model has spare capacity, unused slots go **quiet** (low residual
  energy), which is why `free` fired 1300+ times and `merge` 12. They do not
  duplicate each other. The premise "two structures redundantly learn the same
  thing, fold them together" does not reproduce here — decay-to-zero does. Pruning
  has work to do; merging does not.

### compute-matched baseline (2026-08-08) — the shared-core win does not survive

Looped costs 3.2x baseline wall clock (10 block-applications/token vs 3). Gave
baseline 67,000 steps (>= both the 3.18x wall and 3.33x FLOPs ratios), 3 seeds:

| arm | compute | best eval | std | n |
|---|---|---|---|---|
| baseline @20k | 1x | 1.3863 | 0.0045 | 4 |
| looped @20k | 3.2x | 1.3475 | 0.0063 | 4 |
| baseline @67k (as run) | 3.4x | **1.3362** | 0.0064 | 3 |
| baseline @ strict wall parity | 3.2x | **1.3419** | 0.0035 | 3 |

At strict parity: looped - baseline = +0.0056 nats, 1.1 sigma — **inside the noise,
nominally baseline**. The -0.039 nat "shared core wins" result was a compute
advantage read off the parameter axis, not parameter efficiency.

What survives is the exchange rate, which is the useful number: **the shared core
delivers equal quality at fewer parameters for ~3.2x the FLOPs — break-even.** That
is a memory-for-compute trade, not a free lunch. It is worth taking only when VRAM
binds and compute is cheap (the local-inference case the origin thread argued for),
and it is worth nothing when compute binds.

Corollary for the whole lab: **every comparison here must report loss-vs-compute,
not just loss-vs-params.** Param-matching alone made a 3.2x compute gap invisible.

### does the redundancy exist at all? (2026-08-08) — no, at any scale tested

`scripts/measure_redundancy.py`, `rank_truncate.py`, `measure_gguf_redundancy.py`,
`measure_expert_redundancy.py`. Structural redundancy — two structures computing the
same function, the thing merge-and-free requires — was looked for in four places:

| where | scale | pairwise twins | basis compressibility |
|---|---|---|---|
| lab FFN | 1.45M | 0 pairs >=0.8 | rank 128/192 costs +0.084 nats |
| lab FFN, 0.5-11M chars | 22x params/char range | 0 pairs >=0.8 | — |
| GLM-5.2 dense FFN | 226M/block | 0 pairs >=0.7 of 75M pairs | rank@90% = 0.69-0.70 vs 0.66 random |
| GLM-5.2 MoE experts | 120 expert pairs | 0 pairs >=0.5 | subspace overlap 1.03x chance |

Nothing, anywhere. Notably:

- **Over-parameterising does not create it.** Shrinking the corpus 22x (0.13 -> 2.91
  params/char) *lowered* measured redundancy. Over-parameterised models on small data
  memorise, and memorisation is individuated — it consumes slack without duplicating.
- **Scale does not create it.** GLM's best neuron pair (0.562) is no better than our
  1.45M model's (0.532), and its dense-FFN spectra sit *at* the random-matrix baseline
  while our tiny model is more low-rank than it is.
- **MoE experts do not share a core.** 256 experts trained on disjoint routed streams
  with nothing forcing them together still land at 1.03x chance subspace overlap. The
  router plus load-balancing loss is an effective decorrelation process.

**The resolution of the apparent paradox.** GLM runs usably at ~1.6 bits/weight, which
is real 10x slack — so how can it have no redundancy? Because there are two different
things:

- *precision* redundancy: each weight needs far fewer bits than we give it. Large,
  real, and **already harvested by quantization**.
- *structural* redundancy: two structures compute the same function, store one plus a
  difference. What merge/free/regrow needs. **Not observed anywhere.**

The thesis assumes these are the same. They are not. Quantization is already eating
the slack that exists; consolidation targets a kind that does not appear to be there.

Methodological note: spectral energy **overstates** compressibility. Our lab model
holds 90% of energy in rank 114/192 yet truncating to 128 costs 0.084 nats — the
low-energy directions still carry function. Never read rank@90% as a licence to
truncate.

Untested: activation-based (functional) similarity on GLM, attention heads (30.5% of
the lab model), transient mid-training redundancy, non-MoE frontier models.

### what is actually worth doing next

1. **Point consolidation at the 91.5% it has never touched** — FFN features and
   attention heads inside prelude/core/coda. Needs functional signatures over MLP
   neurons and a merge over MLP rows; a real piece of work, not a config change.
3. **Reuse prelude/coda**, or shrink them, so more than a third of the model is
   subject to the reuse that actually won.

## Layout

```
scrt-lab/
  scrt/
    config.py        # ModelConfig / TrainConfig
    model.py         # BaselineGPT, LoopedGPT, SCRTGPT, ResidualSlotBank
    consolidate.py   # merge / free / regrow
    data.py          # synonym synthetic language
    train.py         # train_arm + run_mvp_comparison
  tests/
  scripts/run_mvp.py
  results/
```

## Config knobs (high-signal)

| Knob | Default | Notes |
|------|---------|--------|
| `n_loops` | 4 | Recurrent depth of shared core |
| `n_slots` | 8 | Fixed residual capacity |
| `lora_rank` | 4 | Rank of each residual adapter |
| `consolidate_every` | 25 | 0 disables consolidation |
| `merge_sim_threshold` | 0.85 | Functional cosine for merge |
| `residual_free_threshold` | 0.05 | Relative energy → free |

## Next experiments (post-MVP)

1. Match baseline param count to SCRT exactly (scale `n_layer` / `n_embd`) for a fair saturation curve  
2. Better merge: shared + explicit residual factors, not just average-and-drop  
3. Apply the same bank idea to attention heads / FFN features  
4. Variable loop count per token (easy vs hard)  
5. Real tokenizer + TinyStories / openwebtext slice

## GPU note

This box has 2×3090. **Default training device is `cuda:1`** (dedicated research card). GPU 0 is left alone unless you override.

```bash
# default — research card
python scripts/run_stepped.py

# explicit override examples
python scripts/run_stepped.py --device cuda:0          # other card
CUDA_VISIBLE_DEVICES=1 python scripts/run_stepped.py --device cuda
```
