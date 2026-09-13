# What this lab found (as of 2026-09-13)

Short version: **the fly's mushroom-body computation, stripped to two ideas, is a real and cheap improvement to a recurrent byte model. The fly's anatomy is not.** Full record: [`../notebook.md`](../notebook.md) Runs 0–27.

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

## How much is that, practically

- Code: −0.14 to −0.23 bpb ≈ **10–15 % lower per-byte perplexity** (10–15 % smaller compressed size). Prose: −0.07 to −0.09 ≈ 5–6 %.
- On the bytes the memory targets (exact continuations of an earlier 8-gram: ~20 % of code, ~4 % of prose): 23–45 % lower perplexity. The remaining bytes improve a little too.
- Cost: **+6 % parameters**, ≈ +7 % arithmetic per byte in recurrent form, one extra state matrix (48 × 1024 floats) per sequence. Buying the same gain by widening the GRU costs > 35 % more parameters and still falls short.
- Our implementation is an unoptimized Python loop (+50 % wall-clock); the recurrent-form cost is what a fused kernel would pay.

## Where it would and would not matter

- **Would:** the linear-time model family (Mamba, RWKV, gated linear attention, RNNs) which lacks exact recall, on data with repeats (code, logs, terminal sessions, structured text). A constant-cost-per-token exact-retrieval bolt-on is what those models are missing.
- **Would not:** transformers. Full attention already retrieves exactly; this head is a cheap approximation of it.
- **Shown to be general (Run 28):** the gain holds on a Mamba-class recurrence and on gated linear attention, not just the GRU.
- **Unknown:** whether the gain survives at real scale (bigger models with longer contexts may learn to retrieve on their own, or the head's capacity may bind). Untested: d ≥ 256, > 10 MB, windows > 512 in the hybrid, a transformer backbone.

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
