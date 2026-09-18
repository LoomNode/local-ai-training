# flybrain-lab

A byte-level language model with a **fly-derived architectural prior**, measured honestly against matched conventional baselines.

**Result:** the fly's *anatomy* gave nothing over fair controls. The fly's *computation* — sparse random expansion as an address, plus fast synapses — does. Used as memory heads through the depth of a Mamba-class recurrence, alongside one short sliding-window attention layer, it **beats a tuned RoPE transformer of the same recipe** on code and prose at 16M parameters, by a wider margin at 52M, out of domain, and at every context length out to 16× its training window.

Full record: [`notes/RESULTS.md`](notes/RESULTS.md) (findings), [`notebook.md`](notebook.md) (all 49 runs, including everything that failed), [`notes/HANDOFF.md`](notes/HANDOFF.md) (state and next steps).

## The numbers

Held-out bits per byte, same recipe, same data, same steps for every arm. Training corpus is Linux 6.10 source (200 MB) or enwik8; CPython is never trained on.

| | params | Linux kernel | enwik8 | CPython (held out) |
|---|---:|---:|---:|---:|
| minGRU backbone, 20M class | 12.9M | 1.295 | 1.543 | 2.419 |
| transformer (tuned: warmup, cosine, wd, RoPE) | 17.1M | 0.781 / 0.789 | 1.360 / 1.355 | 1.460 |
| **4 memory heads + 1 local attention layer** | **16.1M** | **0.764 / 0.762** | **1.307 / 1.308** | **1.459 / 1.439** |
| transformer, 50M class | 57.1M | 0.735 | not run | 1.405 |
| **6 memory heads + 1 local attention layer** | **52.5M** | **0.696** | not run | **1.386** |

Two seeds per side at the 20M class, one at 50M. The margin **widens** with scale: 0.02 → 0.039.

**Context.** Trained at 2048 bytes, evaluated unchanged on longer windows (the recurrence has no positional encoding to extend):

| eval window | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|
| transformer, plain RoPE | 0.76 | 1.05 | 2.06 | 3.14 | 3.93 |
| transformer, NTK-scaled at eval | 0.76 | 0.73 | 0.81 | 1.03 | 1.60 |
| transformer *trained* at 8192 (2.6× the compute) | 0.80 | 0.81 | 0.78 | 0.85 | 0.91 |
| **hybrid, 20M** | 0.71 | 0.75 | 0.70 | 0.77 | 0.81 |
| **hybrid, 50M** | **0.64** | **0.69** | **0.64** | **0.72** | **0.75** |

## What the design actually is

1. **Sparse random address.** Project the input through a *frozen* random matrix to 4096 units and keep the top 4. Same input → same address; different inputs almost never collide. (Kenyon cells.)
2. **Fast synapses.** A per-sequence matrix written by an outer product (address × what came next) and read by the current address — linear attention, with the sparse address as the only change. Plus the linear-attention normalizer, a two-byte key, and a match-strength gate. (KC→MBON plasticity.)
3. **One head per two layers.** Retrieval has to happen at several depths, not once at the top. This is what closed the gap to attention on code.
4. **One 512-byte sliding-window attention layer**, mid-stack. Attention inside a short window, sparse-key memory for everything beyond it.

Cost: +4 % parameters per head, no parameters for slots, per-token cost flat in window length.

## What did not work

Kept in the notebook because negative results are the point: synthetic fly topology vs a degree-matched rewire (a wash), cell-type-specific update programs, sign constraints, unique Kenyon claws, a dopamine-like write gate, a delta rule, a 166k-node connectome as a reservoir, and — on the design side — a next-byte count memory, learned recency, wider keys, more or fewer active units, and a second attention layer. Also: bolting this head onto a transformer that has already formed attention retrieval makes it *worse*.

## Reproduce

```bash
uv sync --extra dev
uv run pytest -q                      # 143 tests
# corpora: scripts/build_corpus.py builds the code sets; enwik8 from mattmahoney.net
scripts/run_capacity.sh cuda:0 depth4local 0 16000 linux     # the 20M result
scripts/run_50m.sh cuda:0 hybrid 0                           # the 50M result
scripts/run_50m.sh cuda:0 transformer 0                      # its control
uv run python scripts/cross_eval.py --ckpts results/ckpt/<ckpt>.pt --lens 2048,8192,32768 --chunk 4096 \
  --corpus-file <corpus> --offset <past the training bytes>   # context extrapolation
uv run python scripts/mask_sweep.py --run-name <run> --arms mingru_sparse_post   # gain by match length
```

Checkpoints are not in the repo (4 GB); the run scripts regenerate them, and the two 50M ones may be attached to a release.

## Limits of the claim

- Nothing above 57M parameters or 260M training bytes.
- One seed per side at 50M; no prose run at 50M.
- The memory helps models that *lack* exact recall. It is a loss on a trained transformer.
- Wall-clock is a tie at 2048 bytes, not a win; the advantage is that per-token cost does not grow with the window.
- bf16 autocast is free and 28 % faster; a 16-bit memory state is free; 8-bit costs 0.075 in training.

## Layout

`flybrain/` model and data code · `scripts/` runners · `tests/` 143 tests · `notes/` findings, handoff, directions · `notebook.md` the full run-by-run record · `results/` JSON for every run.

Origin: a ChatGPT handoff (private ChatGPT conversation) asking whether a fly connectome prior helps sequence learning. It does not; two of the mushroom body's computational tricks do.

## License

Apache 2.0, see [`LICENSE`](LICENSE).
