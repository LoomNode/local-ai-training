# Group learning for the fly-brain model (Stages 0–1)

- **Date:** 2026-10-03
- **Status:** design approved by Kyle (plan review, 2026-10-03). Stages 0 and 1 are specified here; Stages 2 and 3 are outlined and get their own specs after each gate.
- **Branch:** `feat/group-learning`. Notebook Runs 50+.

## Goal

Find out whether a group of agents that share one brain and pool their memories makes base training cheaper than a transformer's for the same quality, and end each stage with something Kyle can run.

Stage 0 makes the existing model promptable and gives it agent state. Stage 1 measures whether related files have anything to share, and measures training cost on equal terms for the first time.

## Non-goals

- int8 master weights (slower today, no peak-memory saving).
- A corpus-scale kNN datastore, except as the Gate 1 fallback probe. It works on any architecture, so a win there says nothing about this one.
- Fleet integration (its config names a model that is no longer loaded; its daemon is written to set GPU 1 to 250 W on start).
- A BPE port, a 50M or larger training run, a resume path for interrupted runs, Agent_OS.
- Changing any published claim in `README.md` or `notes/RESULTS.md`. Proposed wording changes go in the Gate 1 report for Kyle to accept.

## Locked decisions (Kyle, 2026-10-02/03)

| Question | Decision |
|---|---|
| What is being built | His own architecture, working: the hybrid in `flybrain/model/hybrid.py` |
| The spine | Base training must cost less than a transformer's for the same quality |
| The idea under test | A group of agents with the same fly-like brain that learn as a group |
| GPUs | Both cards overnight, GPU 1 most of the time |

## Evidence this design rests on

All checked on 2026-10-02/03 against the repo.

- 16M: hybrid 0.764 / 0.762 vs transformer 0.781 / 0.789 bits/byte, 6,866 s vs 6,657 s (fp32, unshared). 50M: 0.696 vs 0.735, timing contaminated by GPU sharing.
- bf16 (`--amp`) was run for the hybrid only (4,919 s, no loss). The transformer never ran under it. Both cards are now capped at 150 W. No existing timing is comparable.
- Plain minGRU 1.295, 4 heads 0.811, 4 heads + local attention 0.764. Local attention with zero heads has a runner (`--arms mingru --local-attn 512`) and no result.
- `GRUHybrid.forward` ignores `state` for the minGRU backbone; `S` and `Ksum` are locals of `sparse_memory_read`; `LocalAttention` has no cache. There is no sampler.
- `corpora/linux_full.bytes` is 220,000,000 bytes: 12,497 files, shuffled with seed 0 (`scripts/build_corpus.py`), each preceded by `\n// ==== <path> ====\n`. Mean file 17.6 kB, median 9.0 kB.
- Training reads bytes [0, 200,000,000). The legacy validation set is [200,100,000, 202,100,000). No saved run or eval has read past 202,100,000. That tail holds 999 file headers (17.9 MB).
- Of those 999 files, 926 have at least one same-directory training file, 692 have at least four, and the median is eight. 178 directories have two or more tail files.
- Slots must be sized at training time (slots ≈ window × active / 2). 65,536 slots hold about 32–64 kB at the trained load.
- Nothing in the lab has carried state across windows or shared memory across sequences.

## Definitions

- **Brain:** the trained weights.
- **Agent:** the brain plus private state: the minGRU carry per layer, the tables `S` and `Ksum` per memory head, and the local-attention cache.
- **Group:** several agents (rows of a batch) reading different byte streams.
- **Commons:** the per-head sum of a group's tables. Same weights give the same address code, so the sum is readable by any agent of that brain.
- **Handoff:** loading a commons into an agent that did not read the material.
- **Transfer ratio:** `(bpb_alone − bpb_handoff) / (bpb_alone − bpb_context)` on the same target bytes and the same related files, reported when the denominator is at least 0.01.

## Stage 0: it runs

### 0.1 Sampler

`scripts/sample.py --ckpt PATH --prompt STR | --prompt-file F --n 400 --temperature 0.8 --top-k 40 --seed 0 --window 2048 --device DEV [--no-stream]`

`--no-stream` recomputes the last `window` bytes with the plain `forward` for every new byte. It needs no model change, ships first, and is the oracle for the streamed path. Output is decoded as UTF-8 with replacement and printed as it is produced. Checkpoints load through `scripts.cross_eval.rebuild`.

### 0.2 Agent state and streaming

New `flybrain/model/state.py`:

- `HeadState(S [B, n_key, d], Ksum [B, n_key], last_idx [B, k], last_val [B, k], conv_tail [B, key_conv−1, d])`
- `AttnCache(k, v, offset)`: unrotated keys and values of the last `window − 1` positions, and the absolute position of the next byte.
- `AgentState(h, heads, attn)` with `new_document()`, which keeps `S` and `Ksum` and drops the carry, the conv tail, the last key and the attention cache.

Changes, all additive (`forward` and every default path stay bit-identical):

- `MinGRULayer.scan(x, h0=None) -> (y, h_last)`: today's body with the first carry taken from `h0`. `forward` returns `scan(x)[0]`.
- `LocalAttention.stream(x, cache) -> (y, cache)`: rotary positions are rebased so angles stay small on long streams; relative offsets inside the window are unchanged.
- `sparse_memory_read(..., state=None)`: starts from `state.S / state.Ksum`, puts `state.last_idx / last_val` in slot 0 of the key shift, returns `(out, state)`. Without `state` the function is unchanged.
- `Block.memory(x, key_src=None, state=None)`: the causal key conv is fed `state.conv_tail` instead of zero padding.
- `GRUHybrid.init_state(batch, device)` and `GRUHybrid.stream(tokens, state) -> (logits, state)`. Supported: `backbone="mingru"`, `where` in `("post", "none")`, `mem_every`, `local_attn`. Anything else raises.

"Exact" means within the repo's test tolerances, not bitwise: fp64 scan chunks align to the segment start and fp32 sums reorder. With `state_bits > 0` streaming is exact only for segments that are multiples of `chunk`.

### 0.3 Group runtime

New `flybrain/group.py`:

- `read(model, streams, state=None, seg=2048) -> (ce, state)`: agents read different streams in segments; returns per-byte cross-entropy and the final state.
- `merge(state) -> commons`: per-head sum of `S` and `Ksum` over agents.
- `preload(state, commons, alpha)`: adds `alpha × commons` to each agent's tables.
- `save_commons(path, commons, model)` and `load_commons(path, model)`: the file carries a hash of the weights that wrote it, and loading under different weights is refused.

## Stage 1: is there anything to share, and what does training cost?

### 1.1 File table and splits

New `flybrain/data/files.py`:

- `file_table(path) -> list[FileEntry(path, start, end)]` from the header lines.
- `related(i, table, lo, hi) -> list[int]`: same directory, inside byte range `[lo, hi)`, excluding `i` and any file with the same content hash.
- `read_region(path, start, end) -> bytes`: a seek and read, not the whole file.
- `fixed_windows(data, seq_len, n) -> (x, y)`: non-overlapping, no RNG.

Splits by file order in the tail: **dev** is the first 500 files (from byte 202,117,357), **test** is the remaining 499 (from byte 211,576,468). Anything tuned is tuned on dev. Test is scored once per reported number.

**Targets** are held-out files with at least four same-directory training files: 353 in dev, 339 in test. The scored bytes are the first 4,096 bytes after the target's header, or the whole file if it is shorter. The median target has 240 kB of same-directory training files and 95 % have at least 32 kB, so every prefix size below is available for nearly all of them.

### 1.2 Census (`scripts/census.py`)

CPU only. For each scored byte and each n in {8, 12, 16}:

- **A:** it continues an n-gram that occurred earlier in the target itself, followed by the same byte.
- **B:** not A, but the n-gram and its continuation occur in the related set.
- **B′:** the same test against a random set of equal size (training files from other directories, seeded).
- **C:** neither.

The related set is the target's same-directory training files in path order, cut at P bytes for P in {8k, 16k, 32k}, the same prefixes 1.3 scores. Choosing *which* related files to read is out of scope; path order is arbitrary and deterministic. A second pass uses same-directory files from the target's own split, to cover material the weights never saw (targets whose directory has at least two other files in that split: about 100 in dev, 83 in test).

Per-byte loss comes from the "alone" condition of 1.3. Reported per bin: share of bytes and mean loss.

**Headroom** `H(X) = share(X) × (loss(X) − loss(A))`, the gain if those bytes became as cheap as self-copyable ones. **The gate number is `H(B) − H(B′)` at n = 8 and P = 32k**: n = 8 is what the lab's `copyable_mask` has always used, and 32 kB is inside the capacity of a 65,536-slot table.

### 1.3 Related reading, zero-shot (`scripts/probe_commons.py --mode context`)

Plain `forward`, no new model code. For each target, one sequence per condition, scoring the target bytes only:

- **alone:** header + target
- **related:** related files (with headers, path order) up to P bytes, then header + target
- **random:** random training files up to P bytes, then header + target
- **related, never trained:** as related, from the target's own split, for the targets listed in 1.2

P is 8k, 16k and 32k for `m20_linux_depth4x64k_s0` (65,536 slots, `--chunk 4096` at eval). `m20_linux_depth4local_s0` and `m50_linux_hybrid_s0` are scored at P of 2k and 4k, their trained load. The 16M transformer is the reference: 2,048 bytes of prefix and the first 2,048 target bytes, NTK-scaled to 4,096. Every model is also reported on that first-2,048 subset so the rows compare.

Reported: bits/byte per condition, and the same split by census bin. The gain should sit on bin B.

### 1.4 Handoff (`scripts/probe_commons.py --mode handoff`)

Needs 0.2 and 0.3. Uses `m20_linux_depth4x64k_s0`, which has no local attention.

- Reader agents, one per related file, each from fresh state, each reading at most 8,192 bytes.
- `merge`, then `preload` into a fresh scoring agent with `alpha` in {0.03, 0.1, 0.3, 1.0}, chosen on dev.
- Also scored: one agent reading the same files in sequence with `new_document()` between them. Summed tables equal sequential reading only for the lowest head, because deeper keys depend on earlier reads; the difference is reported per head.

Reported: transfer ratio on test.

This probe can confirm and cannot refute. No checkpoint was trained to start from a loaded memory, and loaded matches dilute the agent's own through the normaliser.

### 1.5 Yardstick

`scripts/run_sparsela.py`, new flags, all off by default:

| Flag | Effect |
|---|---|
| `--val-fixed N` | Periodic and final evals also score N fixed windows from `--val-offset` |
| `--val-offset B` | Byte offset of that region (the dev split) |
| `--eval-every K` | Eval every K steps (0 keeps `steps // 6`) |
| `--meter` | Sample the card once a second while training |
| `--ckpt-every K` | Save model, optimiser and scheduler every K steps |

New per-arm fields, written only when the flags are on: `train_seconds` (eval and saves excluded), `eval_seconds`, `bytes_seen`, `peak_vram_bytes`, `joules`, `mean_power_w`, `power_limit_w`, `max_temp_c`, `contended`, and `curve_fixed` (step, train_seconds, bpb cold, bpb warm). The existing `seconds` field keeps its meaning.

**Cold** is the mean over the whole window. **Warm** is the mean over its second half.

`scripts/cross_eval.py` gains `--fixed N` (off by default) to score N fixed windows, cold and warm, for final scoring of saved checkpoints.

New `flybrain/meter.py`: a sampler thread over `nvidia-smi` (power draw, power limit, temperature, memory, compute apps) for the physical card the run is on. `joules` is the sum of power × interval. `contended` is true if any other compute process appeared on that card. The run aborts at 85 C. Query fields and the stop follow `local-ai-training/scripts/thermal_guard.py`.

Joules are logged, not headlined: at a fixed cap they are close to 150 W × seconds.

### 1.6 Spine runs

16M, 16,000 steps, the common flags of `scripts/run_20m.sh` (d 512, 8 layers, d_ff 1024, batch 8, seq 2048, lr 6e-4, warmup 1000, cosine, wd 0.1, `--chunk 256`), plus `--amp --meter --val-fixed 128 --val-offset 202117357 --eval-every 1000 --save`. All three on GPU 1, one after another, nothing else on the card.

| Run name | Arm flags |
|---|---|
| `spine16_hybrid_s0` | `--arms mingru_sparse_post --gate --n-key 4096 --mem-every 2 --local-attn 512 --key-convs 2,4` |
| `spine16_tf_s0` | `--arms transformer --rope` |
| `spine16_zeroheads_s0` | `--arms mingru --local-attn 512` |

Final scoring on dev, test and the legacy validation bytes, cold and warm.

- Seed 1 is run for any pair closer than 0.01 bits/byte or 5 % in `train_seconds`.
- If one arm is better in bits/byte but slower than another, the faster arm is rerun with its steps scaled to the same `train_seconds`, with its own cosine schedule.

## Gate 1

Kyle decides, from a notebook write-up with these tables: census shares and headroom, related-reading gains by bin, transfer ratio, spine table.

| `H(B) − H(B′)` at n = 8 | Decision |
|---|---|
| 0.02 bits/byte or more | Stage 2: train a brain with a shared memory |
| less than 0.02 | Exact-match sharing is dead on kernel code. Run the kNN fallback probe before deciding anything else. |

The spine table is reported either way.

## Stages 2 and 3 (outline)

**Stage 2.** Batches become groups: each row reads a different file of one directory. One shared slot table per head; a row sees the other rows' writes from the previous chunk boundary; carried for a few steps and reset per group. Arms at equal seconds, hybrid alone vs hybrid with shared memory, at 1x and 0.5x budgets with their own schedules, two seeds. Success, fixed before the runs: the shared-memory arm at 0.5x the seconds matches the hybrid at 1x on test.

**Stage 3.** Train the winning recipe at the largest budget Kyle wants, on a corpus chosen at Gate 2. Ship the brain with the group runtime.

## Run rules

- GPU 1: `systemctl --user stop qwen-server.service` before a run, start it again when the queue is empty. A request to a running server reloads 19.5 GB and can run the job out of memory (it did on 2026-09-17).
- GPU 0: 22:00–05:00 only.
- The power cap stays at 150 W.
- Check `nvidia-smi` before each launch; refuse a card with another compute process. Kill by explicit PID only.
- Runs are detached `systemd-run --user` units started from a worktree pinned at a commit, logging to `results/`. New run names only; the runner skips a name whose JSON exists.

## Testing

CPU, `uv run pytest -q`. New `tests/test_stream.py`, patterned on `test_chunked_head_matches_parallel_form` and the `LocalAttention.reference` check:

1. Scan carry: T = 70 split at 1 / 31 / 32 / 33 and byte by byte, against one pass, atol 1e-5.
2. `LocalAttention.stream` against `reference`: w = 8, T = 45, segments of 1 / 3 / 8 / 9, and again with a small rebase period.
3. `Block.memory` with state against the parallel form: decay in {1, 0.9}, gate on and off, key_conv in {2, 4}, splits including single positions.
4. Full hybrid (`mem_every=2, local_attn=8, key_convs=(2, 4)`): random segmentations against `forward`, atol 1e-4, and under CPU bf16 autocast.
5. Junction: `new_document()` with emptied tables reproduces a fresh `forward`; the lowest head's table for [A, reset, B] equals `S_A + S_B`.
6. Runner: a tiny CPU run with and without the new flags gives an identical `state_dict` and identical legacy JSON fields.
7. Streamed and recompute samplers give the same greedy bytes.

Plus: `file_table` round-trips a built corpus; `related` never returns the target or a content duplicate; census bins on a hand-built example; handoff with `alpha = 0` equals alone; an empty prefix equals alone; `load_commons` refuses a mismatched weights hash; the meter's joules are within 5 % of mean power × seconds and `contended` trips on a second process.

One regression check on the GPU: the 16M winner re-scored through the new fixed-window path on the legacy validation bytes reads 0.764 within 0.002.

## Risks

- **Streaming exactness.** Rotary positions, the fp64 scan and chunk alignment are each a way to be subtly wrong. Tests 1–5 exist for this.
- **The census is an exact-match proxy.** It can understate what a learned memory would use. That is why a negative result triggers the kNN probe instead of ending the idea.
- **The handoff probe is confirm-only.** Stated above; it is not a gate.
- **Timing noise.** All spine arms run on the same card, alone, with the contention flag recorded.
- **Selection on test.** Dev is for tuning; test is read once per number.

## Noted for later

- A resume path (`--resume`) is needed before any run longer than a GPU 0 night window. `--ckpt-every` only saves.
- Every long-context number in Runs 40–49 crossed into unrelated files, because the corpus is a file shuffle. Whether that changes any wording in `README.md` or `notes/RESULTS.md` is Kyle's call at Gate 1.
