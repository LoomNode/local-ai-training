# One table: learned n-gram memory and in-context memory merged (design, 2026-11-05)

## Why

Qwen3.8-Flash-Next (the Qwen4 preview, config read 2026-11-05) carries three ideas as three separate parts:

1. **N-gram embedding.** A 20M-row hashed table of byte-pair-token bigrams and trigrams, looked up once at layer 2
   and added to the stream: 51B parameters against 6B active. It is learned in training and fixed afterwards: a phrasebook.
2. **Fast-weight memory.** 3 of every 4 layers are Gated DeltaNet, a recurrence over a fast-weight matrix written
   while reading: a notepad.
3. **Sparse addressing.** Qwen Sparse Attention picks micro-blocks, not every token.

This lab's winning model already has 2 and 3 in one part. Each sparse-key memory head (`Block` + `sparse_memory_read`)
addresses a table of 4096 slots through a frozen sparse code (top-4 of 4096) and writes it while reading. That table
starts **empty** for every sequence. Kyle's question: use the ideas together, as one table.

The merge is small in this code. The head's table starts from a **learned** table instead of zeros, and the
n-gram can join the address. Then one table holds what a context usually means (learned) and what it meant earlier in
this text (written), under one sparse address.

## The question

Does one table with a learned prior beat the same parameter budget spent the way Qwen does it, on a separate
n-gram embedding? A secondary question: does an exact n-gram in the address add anything over the content key alone?

## Arms

All arms use the 20M winner (Run 41, `run_capacity.sh depth4local`): d 512, 8 minGRU layers, d_ff 1024, 4 memory heads
of 4096 slots (`--mem-every 2`), key widths 2 and 4, one 512-byte local-attention layer, gate, chunk 256. Same corpus,
seed, steps, schedule and batch in every arm.

| arm | what changes | extra params |
|---|---|---|
| **A** `base` | nothing (the winner as is) | 0 |
| **B** `ngram` | A + a separate Qwen-style hashed n-gram embedding: bigram and trigram tables, 8192 rows each, added after layer 2 through a zero-initialised projection | ~8.65M |
| **C** `prior` | A, and every head's table starts from a learned prior: per slot a value `V0` (d) and a pseudo-count `m0` | ~8.40M |
| **D** `prior+addr` | C, and the head's slot index is shifted by a hash of the last 2 or 3 input bytes (heads alternate 2, 3) | ~8.40M |

B and C/D spend about the same number of table parameters (within 3%). B is Qwen's layout at this scale, and the
control for "it is just more parameters". A is the floor.

## The merged table (arms C and D)

Today, per head and per sequence, with the unit-norm sparse code `q` (query) and `k` (key):

    out_t = gate(m_t) * q_t . S_t / (m_t + 1e-3),   S_t = sum_{s<t} k_{s-1} v_s^T,   m_t = q_t . sum_{s<t} k_{s-1}

**Prior.** Each slot `j` gets learned parameters `V0[j]` (d) and `m0[j] = softplus(r[j])`. The running sums
start from them instead of zeros:

    S_0[j] = m0[j] * V0[j],   Ksum_0[j] = m0[j]

So the learned value counts as `m0[j]` earlier writes of `V0[j]`. Real writes add to it, and the read is a weighted mean
of the prior and what this text wrote. The model learns `m0` per slot: a large `m0` keeps the phrasebook, a small one
lets the notepad take over after a write or two. This is the design's answer to the risk that the two parts fight.
The gate already sees the mass, so a slot with a prior opens the gate even before anything is written.

- Init: `V0 = 0`, `r = 0` (`m0` ≈ 0.69, about one write of weight 0.5 to 0.7). At step 0, reads are shrunk toward zero
  until something is written. That differs from A at step 0 only, and is the price of a symmetric start.
- The prior is shared by the batch and read-only per sequence. Each sequence's writes are its own (today's per-batch
  tables). This is the "shared phrasebook plus private overlay" shape at a toy size: the overlay here is a dense table
  per sequence, because 4096 slots are cheap.
- Decay stays 1.0 (an archive), so the prior never fades within a sequence.

**N-gram address (D).** The sparse code picks `idx_t` (4 slot indices). D shifts them:
`idx'_t = (idx_t + off_h(n-gram_t)) mod 4096`, where `n-gram_t` is the last 2 or 3 input bytes up to and including
`t` (causal, the same bytes the key already sees), and `off_h` is a fixed hash per head. Writes use the same shift, so a
query matches only writes made under the same n-gram *and* similar content. The phrasebook entry is now specific
to the n-gram as well as the content, which is Qwen's indexing, inside the memory's address.

## Not in scope

- Streaming (`init_state` / `stream`) with a prior: refused with a clear error for now.
- A sparse per-sequence overlay over a table too large to copy per sequence (Qwen scale). It only matters if C or D wins.
- Seeds beyond 0, enwik8, 50M. Those come after the early read, if it says so.

## Runs

- Card: one GPU, chosen with Kyle at launch (see below). `--amp` in every arm (Run 47: bf16 is free, the memory stays fp32).
- **Early read:** 4000 steps per arm (warmup 1000, cosine to the end of the 4000), seq 2048, batch 8, Linux kernel
  200 MB, val 2 MB, seed 0. About 30 to 40 minutes per arm on a free card, so about 2.5 h for all four, run in order A, B, C, D.
- Each arm saves a checkpoint (`--save`), which feeds the scoring below.

## What gets reported

Per arm, from the same windows:

1. Validation bits/byte (the run's own eval) and its curve.
2. Held-out CPython bits/byte (`scripts/cross_eval.py`): did the table learn the kernel, or code?
3. Bits/byte per match bin (`scripts/mask_sweep.py`: longest earlier n-gram match 2/3/4/6/8/none). Run 44 found the
   transformer's last lead was in the 2 to 4-gram bins, so that is where a phrasebook should show.
4. Params, step time and peak memory.

## How to read the early result

- At 20M, seed spreads were 0.002 (hybrid) and 0.008 (transformer) at 16k steps. At 4k steps, treat a difference under
  about 0.01 as a tie until a second seed says otherwise.
- **C or D beats B by more than 0.01:** the merge carries its weight. Next: a second seed, then the full 16k schedule.
- **B ≈ C/D, both beat A:** the parameters help but the merge adds nothing at this size.
- **Nothing beats A:** a learned table does not help a 20M byte model on this corpus at 4k steps. Report it and stop
  unless Kyle wants the 16k schedule.
- D vs C answers the address question on its own.

## Where it lives

`/games/ailab/lab/flybrain-lab`, worktree `.worktrees/merged-table`, branch `feat/merged-table` off `791ad91`. The main
checkout's `results/` folder was missing on 2026-11-05 (its tracked JSONs show as deleted, not committed, and the
git-ignored checkpoints are gone). The worktree has its own copy of the tracked results; the main tree is left as found.
