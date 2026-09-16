# What this lab found (as of 2026-09-16)

Short version: **the fly's mushroom-body computation, stripped to two ideas, is a real and cheap improvement to a recurrent byte model. The fly's anatomy is not.** Placed at several depths of the stack, the memory brings a Mamba-class recurrence to within 0.02–0.03 bpb of a tuned transformer on code and ahead of it on prose, at equal or fewer parameters; add one short sliding-window attention layer and the same model beats the transformer at 20M on code (0.764 / 0.762 vs 0.781 / 0.789, two seeds each) and on prose (1.307 / 1.308 vs 1.355 / 1.360) and ties it out of domain, with 6 % fewer parameters and the same training time, and the result keeps working at 16× the training window, where a transformer, even with the standard length-extension fix, does not. Full record: [`../notebook.md`](../notebook.md) Runs 0–45.

## The question

Can biological neural design (the fly brain as blueprint) give better or cheaper sequence learning than conventional architectures? Measured on held-out bits per byte (bpb), with matched-size controls.

## What did not work (Runs 1–17, 25)

Every attempt to get a win from fly *organization* tied a fair control or lost:

- Synthetic fly-like topology vs a region-preserving rewire: a wash on WikiText (3.60 vs 3.58) and on binding tasks.
- Cell-type-specific update programs vs one universal program vs one per neuron: a wash.
- Synapse message width (1–4 channels), sensory fan-in (10–50 %), unique Kenyon claws, nonnegative keys, a learned dopamine-like write gate, a delta-rule write: each removed or varied with no loss.
- A frozen 166k-node connectome (nftechie/flm) as a reservoir next to a 1.2B LM: their own no-graph control wins. Not run here, but consistent.

Two implementation lessons cost weeks of runs and are worth stating: (1) a decoder that mean-pools a population cannot read identity, and one parallel update per token leaves the readout two hops behind the input; fix I/O before testing priors. (2) Learning-rate and training-signal fraction each produced a false "dense keys fail" and a false "memory fails at 2048" before being controlled.

## What did work (Runs 21–27)

Two ingredients, both from the mushroom body, both known ideas in ML:

1. **Sparse random expansion as the address.** Project the input through a frozen random matrix to 1024 units, keep the top 4. Two inputs that are the same get the same address; different inputs almost never collide.
2. **Fast synapses.** A per-sequence matrix written by an outer product (address × what came next) and read by the current address. This is linear attention / fast-weight memory; the sparse address is the only change.

Plus two standard fixes: the linear-attention normalizer, and a two-token key (causal conv) so the address is a bigram.

### Retrieval

On a copy task (find a span seen earlier in a window of random bytes), the sparse-key memory learns exact retrieval where learned keys, orthogonal random dense keys, softmax attention and a GRU of the same size learn nothing in the same budget:

| window | spans stored | sparse (4096 units) | softmax attention |
|-------:|-------------:|--------------------:|------------------:|
| 512 | 1 | 94 % | chance |
| 1024 | 32 | 92 % | 53 % |
| 2048 | 16 | 94 % | chance |

At 64 tokens everyone eventually learns it; the sparse key gets there 3–4× sooner and across a 10× learning-rate range. Capacity degrades gracefully and scales with memory width.

### Language

Attached to the best conventional baseline in this lab (a GRU; every attention-style arm loses to it at this size), the memory head improves bpb on every corpus, seed and control (Runs 26–27):

| corpus | GRU 573k | GRU 771k (wider) | learned-key head 836k | **GRU + sparse head 608k** |
|--------|---------:|-----------------:|----------------------:|---------------------------:|
| Python code (dev), 3 seeds | 1.601–1.610 | 1.545–1.560 | 1.518–1.544 | **1.463–1.477** |
| WikiText prose (dev), 3 seeds | 2.079–2.104 | 2.038–2.058 | 2.038 | **2.008–2.019** (output head, 795k) |
| WikiText-2 valid+test (held out) | 2.071 | 2.032 | 2.040 | **2.008** |
| Mixed-language code (held out), 2 seeds | 1.757–1.758 | 1.661–1.677 | — | **1.524–1.543** |

Also held against a learning-rate sweep and 3× training for the GRU (gap unchanged at 0.134), and a leakage control (keys from another sequence → falls back to baseline).

**Not a GRU quirk (Run 28).** The same head on a minGRU (Mamba-class selective recurrence) and on gated linear attention gives the same margins on both seeds and both corpora types (code −0.11…−0.24 bpb, prose −0.02…−0.07), while a learned-key head of the same size gets a third of the code gain and is a net loss on prose:

| backbone | Python code | held-out code | prose |
|---|---:|---:|---:|
| minGRU → + sparse head | 1.62 → **1.48** | 1.77 → **1.53** | 2.08 → **2.04** |
| gated linear attention → + sparse head | 1.66 → **1.53** | 1.73 → **1.55** | 2.10 → **2.07** |
| GRU → + sparse head | 1.60 → **1.46** | 1.76 → **1.52** | 2.08 → **2.01** |

**Does width close the gap? (Run 29.)** A ladder at d = 64, 128, 256, 384 (134k → 3.75M backbone scalars), fixed data and steps, two seeds each:

| backbone | Python code gap by width | held-out code gap | prose gap |
|---|---|---|---|
| minGRU (no memory of its own) | 0.15, 0.14, 0.14, **0.12** | 0.32, 0.23, 0.19, **0.17** | 0.03, 0.04, 0.04, **0.04** |
| gated linear attention (has learned-key fast weights) | 0.12, 0.11, 0.06, 0.12* | 0.20, 0.16, **0.04**, 0.08* | 0.04, 0.04, 0.03, 0.03 |

\* plain GLA is seed-unstable at d=384. On the minGRU the gain is stable across a 28× width range; a 2× wider plain minGRU does not match the headed one. On GLA, which already has a learned-key memory, width lets it learn its own recall on code and the head's gain drops to ~0.05 by ~2M scalars; on prose it keeps its gain everywhere.

**Transformer: the predicted negative (Run 30).** Once a transformer has trained long enough to form attention-based retrieval (9000 steps, code 1.41, repeat-byte loss 0.35), the head makes it worse by 0.1–0.3 bpb. Before that transition (3000 steps, or a lower learning rate) the head helps by ~0.1. The head is for models that lack exact recall.

**Equal training budget vs a transformer (Run 32).** At d=128, 9000 steps, three seeds on code: GRU + sparse head **1.378** vs transformer **1.409** vs minGRU + head 1.41 vs plain GLA 1.43–1.52 (seed-dependent). The transformer retrieves better; the recurrence models everything else better; the sum favours recurrence + head at fewer parameters and constant cost per token.

**Where the gain lands (Run 31).** Binning bytes by the longest earlier match in the window: the head gains 0.3–0.4 bpb on every bin with any match, down to 2-grams (46 % of code bytes), and *costs* 0.06 on the 54 % with none, because its output is added unconditionally. It is purely a retrieval device; a match-strength gate should recover ~0.03 on code and more on prose.

**At scale, on public corpora (Run 34).** d=256, 2048-token windows, 4096-unit head, 8000 steps, two seeds:

| corpus | minGRU | + sparse head | **+ gated head** | transformer (1.5× params) |
|---|---:|---:|---:|---:|
| CPython source | 1.48 | 1.16 | **1.11** | 1.22 |
| enwik8 | 1.85 | 1.68 | **1.63** | 1.84 |
| Linux kernel (never seen) | 3.43 | 2.14 | 2.03 | **2.01** |

The head's gain **grew** with context (code 0.14 → 0.38, prose 0.04 → 0.21) because a longer window holds more repeats. The match-strength gate (Run 33) is worth 0.05 in-domain and 0.1 out of domain.

**Against a properly tuned transformer (Run 35)**, with warmup, cosine decay, weight decay and rotary positions for every arm: the transformer gains 0.2 bpb from the recipe, the recurrences 0.03. Result at d=256 / 2048 tokens: transformer **1.010** vs recurrence + gated head 1.084 on CPython; 1.614 vs **1.602** on enwik8. So: the transformer wins on code by 0.07 and ties on prose. The Run 34 "beats the transformer" line was an artefact of the transformer's plain recipe and is withdrawn. What stands: the head's gain over its own backbone (0.37 code, 0.21 prose) is unchanged by the recipe, and the recurrence + head gets within 0.07 of a tuned transformer at constant cost per token.

**Longer windows and 8× the size (Runs 36–37).** At 4096 tokens the head's gain holds (0.37 on code) and the tuned transformer's lead shrinks to 0.04. At 13–17M parameters (d=512, 8 layers, 260M tokens, kernel source / enwik8, CPython held out):

| corpus | minGRU | + gated head | transformer (1.3× params) |
|---|---:|---:|---:|
| Linux kernel | 1.30 | 0.91 | **0.78** |
| CPython (held out) | 2.42 | 1.68 | **1.46** |
| enwik8 | 1.54 | 1.38 | **1.36** |

The head's gain over its own backbone is size-independent (0.37–0.39 code, 0.17–0.21 prose, 0.74 out of domain). Against a tuned transformer: a replicated 0.02 loss on prose (two seeds each: 1.375 / 1.380 vs 1.360 / 1.355), a loss on code that widens with size (0.13 in-domain, 0.22 held out). (An earlier claim that the recurrence + head trains in ~60 % of the transformer's time compared shared-GPU wall-clocks and is withdrawn; see Run 38.)

**Memory that grows with depth (Run 38).** The code gap grew with size while the memory stayed one head at the top of the stack. Putting a head after every k-th layer (`GRUHybrid(mem_every=k)`, `scripts/run_capacity.sh`) at the 20M kernel setting:

| heads | placement | params | kernel bpb | repeat / non-repeat bytes | train time (shared GPU) |
|---:|---|---:|---:|---|---:|
| 0 | — | 12.9M | 1.295 | 1.03 / 1.44 | 6000 s |
| 1 | top | 13.4M | 0.910 | 0.31 / 1.24 | 7137 s |
| 2 | after layer 4, top | 13.95M | 0.834 | 0.21 / 1.17 | 9553 s |
| **4** | after 2, 4, 6, top | 15.0M | **0.811 / 0.814** (2 seeds) | 0.18 / 1.15 | 10620 s |
| 8 | every layer | 17.15M | 0.802 | 0.17 / 1.14 | 17210 s |
| transformer | attention every layer | 17.09M | **0.781 / 0.789** | 0.16 / 1.12 | 11861 s (6657 s unshared) |

Each doubling of heads is worth about a third of the previous one (0.076, 0.022, 0.010). Four heads is the efficient point: 12 % fewer parameters than the transformer, 0.03 behind on in-domain code (two seeds each side), 0.04 behind out of domain, ahead on prose. Training time is *not* an advantage at 2048 tokens: on an unshared GPU the transformer takes 6657 s and a 4-head arm ~11000 s with this unfused implementation (earlier "60–90 % of the transformer's time" figures compared shared-GPU wall-clocks and are withdrawn); the head arm's cost advantage is that its per-token cost is flat in window length. At exactly matched parameters (8 heads) the code gaps are 0.02 / 0.02 and prose is a 0.01 lead. Where the last 0.02 on code sits (Run 31-style bin sweep): 43 % on bytes with no earlier match (the recurrence is a slightly weaker general model), 29 % on 2–3-gram matches (the memory reads a superposition where attention weighs candidates), and long exact retrieval is within 0.005–0.012 of attention. The extra heads improve non-repeat bytes too (1.24 → 1.15): recall done in the stack frees the recurrence for everything else. Out of domain (kernel-trained, scored on CPython) the gap closes the same way: 1 head 1.68, 2 heads 1.54, 4 heads 1.50, 8 heads 1.48, transformer 1.46. On enwik8: 1 head 1.375 / 1.380, 4 heads **1.351**, 8 heads 1.349, transformer 1.360 / 1.355 — four heads lead on prose at 12 % fewer parameters, and the 4 → 8 step is worth 0.002.

**Slots do not need to grow with the model (Run 39).** 24 small runs varying slot count against width (4× and 16× the width vs fixed 1024) and 6 cells varying it against window at d=256: nothing from 4 writes per slot down to 0.3 leaves seed noise at any width; 8 writes per slot costs 0.02–0.09, 16 costs 0.05, and even at 16 the head keeps 90 % of its gain. Rule: slots ≈ window × active units / 2 (4096 for a 2048 window, which the 20M runs already use). Slots are a frozen buffer: no parameters.

**Context extrapolation (Run 40).** The 20M checkpoints, trained at 2048, evaluated unchanged on longer windows (same windows for every arm at each length):

| eval window | minGRU | + 1 head | + 4 heads | transformer, plain RoPE | transformer, NTK-scaled RoPE |
|---:|---:|---:|---:|---:|---:|
| 2048 | 1.29 | 0.89 | 0.79 | **0.76** | **0.76** |
| 4096 | 1.30 | 0.87 | 0.75 | 1.05 | **0.73** |
| 8192 | 1.34 | 0.92 | **0.76** | 2.06 | 0.81 |
| 16384 | 1.22 | 0.91 | **0.74** | 3.14 | 1.03 |
| 32768 | 1.23 | 1.02 | **0.83** | 3.93 | 1.60 |

The four-head arm is flat to 16× its training length and drifts only at 32k, where the slot load reaches 32 writes per slot. Training the same arm with a slot table sized for the window removes most of that drift (slots add no parameters; they cannot be added after training, because top-k over a larger random table is a sharper address than the trained projections expect):

| eval window | 4096 slots | 32768 slots | 65536 slots | backbone (same windows) | NTK transformer |
|---:|---:|---:|---:|---:|---:|
| 2048 | 0.79 | 0.79 | 0.79 | 1.29 | 0.76 |
| 8192 | 0.76 | 0.74 | 0.74 | 1.34 | 0.81 |
| 32768 | 0.83 | 0.77 | **0.76** | 1.23 | 1.60 |
| 65536 | 0.96 | 0.88 | **0.86** | 1.34 | 2.42 |

With slots ≈ window × active / 2, the gain over the backbone is 0.47–0.50 at 32k and 64k, the same as at 2k, and the absolute rise at 64k matches the backbone's own rise on those windows. A residual ~0.1 of the 8k-window gain is lost past 32k that slots do not recover (a short key written thousands of times reads as a blend of every continuation; recency weighting is the untested lever). So: no hard context limit; a soft one set by slot load, chosen at training time. The plain RoPE transformer fails at 2×. With the standard fix (NTK-aware base scaling at eval time, no retraining) it holds to 2× and wins there, then degrades: the head arm leads by 0.05 at 8k, 0.29 at 16k, 0.77 at 32k. enwik8 shows the same shape (head arm 1.32–1.43 through 32k; NTK transformer 1.36 → 2.40). A transformer *trained* at 8192 (Run 45; 2.6× the training time) scores 0.777 on 8192-byte windows and 0.85 / 0.91 with NTK at 16k / 32k; the local-attention hybrid trained at 2048 scores 0.70 / 0.77 / 0.81 on the same windows. Trained for the length, the transformer still loses at every length.

**Attention within a window, memory beyond it (Runs 41, 43).** A design screen at d=256 (transformer 1.010 on CPython; eleven arms, `results/scr_*.json`) found one thing worth more than seed noise: a single sliding-window attention layer (512 bytes, after layer 4) inside the headed recurrence, −0.077 at matched parameters (0.966), and the only arm to beat the transformer at 2048. Everything else screened (learned recency in the memory 0.013, mixed key widths 0.005, a next-byte count memory 0.004, FFN width 0.019 at matched params) is ≤ 0.02. At the 20M kernel setting the same design (four heads + local 512 + key widths 2,4; `run_capacity.sh ... depth4local`):

| arm | params | kernel | copyable / non | held-out CPython | enwik8 |
|---|---:|---:|---|---:|---:|
| transformer (2 seeds) | 17.09M | 0.781 / 0.789 | 0.160 / 1.117 | 1.460 | 1.360 / 1.355 |
| 4 heads (2 seeds) | 15.01M | 0.811 / 0.814 | 0.182 / 1.151 | 1.502 | 1.351 |
| **4 heads + local attention** (2 seeds) | 16.07M | **0.764 / 0.762** | **0.154 / 1.094** | **1.459 / 1.439** | **1.307 / 1.308** |

By match-length bin (Run 44) the attention layer's gain is on the 34 % of bytes with no earlier match (1.737 → 1.656, 0.05 ahead of the transformer): a 512-byte window is a better local model than the recurrence, which is why the retrieval-side designs (count memory, recency, key widths) gave nothing. The transformer's remaining lead is the 2–4-gram bins, +0.005–0.022 there, ~0.003 weighted. The window saturates at 512 (128 / 256 / 512 / 1024: 0.984 / 0.974 / 0.966 / 0.971 at d=256). Trained at 2048 it still evaluates flat to 32k (0.70–0.81 from 2k to 32k; its RoPE only sees offsets within 512 bytes), where the NTK-scaled transformer is at 1.60. Both seeds agree within 0.005 at every length. On enwik8 the same arm is 1.30–1.31 from 2k to 32k where the NTK transformer goes 1.36 → 2.40. Wall-clock on an unshared GPU at 2048: 6866 s vs the transformer's 6657 s for the same steps, so the earlier 'head arm trains ~1.6× slower' figure (extrapolated from shared runs) is withdrawn.

## How much is that, practically

- Code: −0.14 to −0.23 bpb ≈ **10–15 % lower per-byte perplexity** (10–15 % smaller compressed size). Prose: −0.07 to −0.09 ≈ 5–6 %.
- On the bytes the memory targets (continuations of any earlier repeat, 2-grams up: ~46 % of code bytes): 20–25 % lower perplexity per byte. On bytes with no repeat it costs ~4 % (Run 31).
- Cost: **+4 % parameters per head** at 20M (0.53M each); four heads +16 %. One 4096 × d state matrix per head per sequence; slot count adds no parameters. The index form (`--chunk`) makes the head exact and linear in sequence length. Buying the same gain by widening the GRU costs > 35 % more parameters and still falls short.
- Wall-clock at 20M: on an unshared 3090 the transformer takes 6657 s and the 4-head + local-attention arm 6866 s for the same steps at 2048 tokens (the '~11000 s' figure quoted for the 4-head arm in Run 38 was extrapolated from shared-GPU runs before the sliced top-k and is withdrawn; shared-GPU figures in this file are not comparable). The head arms' cost per token does not grow with window; attention's does, and past 8k the head arm is both cheaper and better.

## Where it would and would not matter

- **Would:** the linear-time model family (Mamba, RWKV, gated linear attention, RNNs) which lacks exact recall, on data with repeats (code, logs, terminal sessions, structured text). A constant-cost-per-token exact-retrieval bolt-on is what those models are missing.
- **Would not:** as a bolt-on to a full-attention transformer. Full attention already retrieves exactly; this head is a cheap approximation of it. Confirmed in Run 30: on a transformer that has formed retrieval, the head costs 0.1–0.3 bpb. The other direction works: a *short* attention window inside the headed recurrence is the design that beats the transformer (Runs 41, 43).
- **Shown to be general (Run 28):** the gain holds on a Mamba-class recurrence and on gated linear attention, not just the GRU.
- **Where it matters most (Run 29):** backbones with no associative memory of their own (selective recurrences). A backbone that already has learned-key fast weights (GLA) grows its own recall on code once wide enough, and the head becomes a small-width and reliability aid there.
- **Would, in the transformer's place, when the input is longer than the training window (Runs 40, 45):** the recurrence + heads generalise to 16× their training length with no positional trick; a RoPE transformer of the same recipe holds to 2× with NTK scaling and loses from 4× on, and one trained at 8192 (2.6× the cost) loses to the hybrid trained at 2048 even at 8192.
- **Settled:** the head's gain over its backbone does not close with steps (Run 32), width (Run 29), recipe (Run 35), context (Run 36) or size (Run 37). The code gap to a transformer *is* retrieval depth (Run 38): heads through the stack close 0.10 of the 0.13; slot count is not the limit (Run 39). Heads saturate at about one per two layers.
- **Unknown:** beyond 17M parameters / 260M tokens of training; a transformer trained at 32k (the 8k-trained one is measured and loses).

## Precision (Run 42)

Everything is fp32. Eval-only on the 20M four-head checkpoint: the memory state at 16 bits and the weights at bf16 are lossless; state at 8 bits costs +0.002 at the training window and +0.055 at 32k (a per-row scale coarsens as rows fill; fixable, untested). Sensitivity to state precision falls with depth: at 8 bits the earliest head costs 0.019, the top head 0.001, so a mixed layout (early heads at 16, late at 8) is within 0.006 of fp32. 6 bits breaks it. Integer weights are untested here.

## What is biological about it, honestly

Sparse expansion (Kenyon cells) and episode-time synaptic writes (KC→MBON plasticity). That is all. The connectome, cell types, sign constraints, claws, dopamine gating and structural plasticity were all tested and none carried computational value here. The fly was a good hint and a bad blueprint.

## Reproduce

```bash
cd /games/ailab/lab/flybrain-lab && uv sync --extra dev && uv run pytest -q
# GRU vs GRU + sparse head on the Python corpus (build corpora first; see scripts/run_sparsela.py and flybrain/data/corpus.py)
uv run python scripts/run_sparsela.py --task code --corpus-file <code.bytes> --device cuda:0 \
  --normalize --key-conv 2 --top-k 4 --n-key 1024 --lr 0.001 --decay 1.0 --d-model 128 --gru-hidden 330 \
  --train-bytes 9000000 --val-bytes 1000000 --steps 3000 --batch 16 --seq-len 512 \
  --arms gru,gru_sparse_pre --run-name repro_code
```

JSON per run in `results/`; checkpoints in `results/ckpt/`.
