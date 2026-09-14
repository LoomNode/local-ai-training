# What this lab found (as of 2026-09-14)

Short version: **the fly's mushroom-body computation, stripped to two ideas, is a real and cheap improvement to a recurrent byte model. The fly's anatomy is not.** Full record: [`../notebook.md`](../notebook.md) Runs 0–34.

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

## How much is that, practically

- Code: −0.14 to −0.23 bpb ≈ **10–15 % lower per-byte perplexity** (10–15 % smaller compressed size). Prose: −0.07 to −0.09 ≈ 5–6 %.
- On the bytes the memory targets (continuations of any earlier repeat, 2-grams up: ~46 % of code bytes): 20–25 % lower perplexity per byte. On bytes with no repeat it costs ~4 % (Run 31).
- Cost: **+6 % parameters**, ≈ +7 % arithmetic per byte in recurrent form, one extra state matrix (48 × 1024 floats) per sequence. Buying the same gain by widening the GRU costs > 35 % more parameters and still falls short.
- Our implementation is an unoptimized Python loop (+50 % wall-clock); the recurrent-form cost is what a fused kernel would pay.

## Where it would and would not matter

- **Would:** the linear-time model family (Mamba, RWKV, gated linear attention, RNNs) which lacks exact recall, on data with repeats (code, logs, terminal sessions, structured text). A constant-cost-per-token exact-retrieval bolt-on is what those models are missing.
- **Would not:** transformers. Full attention already retrieves exactly; this head is a cheap approximation of it. Confirmed in Run 30: on a transformer that has formed retrieval, the head costs 0.1–0.3 bpb.
- **Shown to be general (Run 28):** the gain holds on a Mamba-class recurrence and on gated linear attention, not just the GRU.
- **Where it matters most (Run 29):** backbones with no associative memory of their own (selective recurrences). A backbone that already has learned-key fast weights (GLA) grows its own recall on code once wide enough, and the head becomes a small-width and reliability aid there.
- **Unknown:** behaviour beyond ~3M scalars and 2048 tokens; against a tuned transformer recipe (warmup, weight decay, RoPE). The gap did not close with 3× steps (Run 32) or 28× width (Run 29), and grew with 4× context and data (Run 34).

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
