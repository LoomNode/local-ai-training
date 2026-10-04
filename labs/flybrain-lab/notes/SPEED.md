# Speed round: can the fly brain's learning be made cheap?

Opened 2026-10-04 on branch `feat/speed-round`. Kyle, shown the first equal-recipe spine numbers: "i think the training speed is the real win for the fly brain. maybe we need to experiment some more with the designs? we could re review the source material", then: "all 3 sound like worthy experiments. add them to a list AND test all 3".

## Why

Spine runs at 16M parameters, bf16 autocast, 150 W, 16,000 steps (`results/spine16_*_s0.json`, scored fp32 on 1,024 fixed dev windows):

| arm | dev bits/byte (cold) | train seconds | seconds per step |
|---|---:|---:|---:|
| hybrid (4 memory heads + local attention) | 0.785 | 17,586 | 1.10 |
| transformer | 0.807 | 5,901 | 0.37 |
| zero-heads (the backbone + local attention) | 0.830 | 8,189 | 0.51 |

The hybrid learns more per step and costs three times the seconds. The memory heads take 53 % of its time (17,586 - 8,189 s), and the backbone without them is already slower per step than the transformer. The model borrows the fly's address (a frozen sparse random expansion) and uses one-shot writes only as scratch memory inside one window. Everything it keeps after training is learned by backpropagation.

In the fly, the wiring up to the memory layer is random and fixed, and learning changes one layer of synapses by a local rule, in one trial. The question of this round: how much of the learning can move out of backpropagation and into one-pass writes?

## The bar

Bits/byte on the fixed dev windows (`--val-fixed` in the run, `cross_eval.py --fixed` in fp32 afterwards) after equal training seconds on GPU 1 (150 W, bf16, alone on the card), against the transformer with its own schedule. Short budget: 3,600 s per arm. Long budget: the 17,586 s equal-seconds reruns (`spine16_{tf,zeroheads}_eqsec_s0`). Any table built by reading the corpus counts its build seconds.

## The list

| # | Direction | Question | Cheap test | Passes if | Status |
|---|---|---|---|---|---|
| 1 | Cheaper heads, same model | Where do the heads' 0.59 s per step go, and how much of it can go without losing bits? | Profile one step. Time the cuts: heads under bf16, no gradient through the address, 2 heads instead of 4, compiled, chunk size. The best combination gets one 3,600 s run. | it beats the transformer at 3,600 s | timed: 1.01 to 0.40 s/step with `torch.compile` and half precision (transformer 0.35 to 0.28); 3,600 s runs queued |
| 2 | Memory filled by reading the corpus once, and kept | Does a table written in one pass over the training bytes, with no gradient, buy bits that training seconds would otherwise have to buy? | a: existing checkpoints plus the table, mixed by a rule fitted on dev, scored on test (CPU, no training). b: a small network trained with the table's prediction as an input, 3,600 s. c (added after the source re-review): hashed n-gram embedding tables, a fixed address with a learned table, as an extra input, 3,600 s. | a: a gain on test with zero training. b, c: it beats the transformer at 3,600 s | a PASSES (+0.03 to +0.04 on test); b: features built, training switch being built; c not started |
| 3 | Local learning, no backpropagation through the stack | How much is lost when no error signal crosses layers, and is a step any cheaper? | i: backbone frozen at random initialisation, only the readout (and the memory) learn. ii: pairs of layers with their own next-byte loss and no gradient between pairs. iii (added after the source re-review): every other layer frozen at random initialisation, gradients still passing through. 3,600 s each. | within 0.02 bits/byte of the fully trained hybrid at equal seconds | built (`spike/speed-local`), not run |

Expectations written before any of these ran (the controller's, 2026-10-04): 1 helps and cannot reach the transformer by itself, because the backbone without heads is already slower per step; 2 has room for a large win, and the handoff probe (a memory the model was never trained to use gave +0.009 bits/byte on dev) is its first hint; 3 is a negative, as local rules lose quality at scale in the literature.

## Source re-review (2026-10-04)

Two read-only reviews by subagents are kept in `notes/speed-round/`: `lab-record-review.md` (Runs 0-49 condensed: what was tried, what each mechanism costs, what is trained and what is frozen) and `literature-review.md` (63 sources: the mushroom body's learning recipe, fly-derived and related machine learning, the origin brief). The controller read the literature review in full and the record review in part; neither was checked against its sources line by line, and what they mark unverified is unverified.

What they change:

- The fly's cheapness is narrow. One layer of synapses learns, locally, in one to a few trials, on a frozen sparse expansion. Long-term memory needs spaced repetition and is metabolically costly. The fly learns odour valences, not sequences.
- No published fly-derived design shows cheaper language-model training than a tuned backprop baseline at matched quality. Fly wiring gives nothing over controls in two 2026 projects, as in this lab's Runs 1-17. The origin brief proposed a connectome network; the two mechanisms that worked here (the frozen top-k address, the outer-product fast synapses) came from the lab's own runs.
- Row 1. The record holds no profile of a step. The head's fp32 pin is a precaution that was never tested in training. The address gathers whole rows of a projection that has 4 nonzeros per row. The minGRU scan runs in float64 with a Python loop in every layer. TF32, `torch.compile` and a fused optimizer were never tried. In the literature a recurrence with local attention ties a transformer's step time at 2K (Griffin), so a tie per step is the realistic target; the hybrid's lead per step would then count.
- Row 2. A fixed hashed address with a learned table has gains at equal compute in the literature (hash layers, hashed n-gram embedding tables): added as test c. Retrieval memories move cost to inference and storage, so table size and lookup cost are reported with every result.
- Row 3. Backprop-free rules are not cheaper at language-model scale in any source found. The one positive result is partial freezing: frozen random layers interleaved with trained ones cut wall-clock to convergence by up to 22-27 % (Reservoir Transformers): added as test iii.
- Reference. Online context-mixing compressors learn in one pass with local updates and no backpropagation between layers (cmix: 1.17 bits/byte on enwik8, at 18 hours of CPU). A measurement of PPMd, xz and zstd on this corpus goes under Results.

## Results

Filled in as they land. A result here is a spike: the code that produced it is throwaway until a design is chosen.

### Row 1: where a step goes, and what the switches buy (2026-10-04, GPU 1, timing only)

`scripts/profile_step.py` and the switches in `scripts/run_sparsela.py` on branch `spike/speed-heads` (bbd6c30); reports under `results/profile/` there (`pitstop1.log`, `pitstop2.log`, one `.md` per run; not committed). The spine recipe at 16M, batch 8 x 2,048, bf16 autocast, 150 W, the card otherwise empty. Each number is the mean of 15 or 20 training steps after 5 warmup steps, so differences under about 3 % are noise. No run here trained long enough to say anything about bits/byte.

**As the spine runs ran (eager).** Seconds per step: hybrid 1.014 (forward 0.444, backward 0.584), transformer 0.348 (0.120, 0.223), zero-heads 0.489 (0.150, 0.336). The hybrid's forward: memory heads 60 % (table write and read 151 ms, address selection 90 ms), the float64 recurrence 24 %, the rest of the layers 13 %, local attention 2.5 %.

**One switch at a time, eager, hybrid:** `--tf32` 1.016, `--head-amp` (heads under bf16) 0.991, `--sparse-claws` (address from the 4 nonzeros) 1.034, `--detach-state` (no gradient through carried tables) 0.985, `--fused` 1.056, `--scan doubling32` (a doubling scan in float32) 1.361, `--scan doubling` (the same under bf16) 1.272. Nothing helps by more than 3 %, and the doubling scans are slower.

**With `torch.compile` on the training step:**

| arm | switches besides `--compile` | s/step | first step (compile) s |
|---|---|---:|---:|
| transformer | none | 0.276 | 9 |
| transformer | `--tf32 --fused` | 0.276 | 10 |
| zero-heads | none | 0.315 | 41 |
| zero-heads | `--scan doubling32 --tf32 --fused` | 0.320 | 22 |
| zero-heads | `--scan doubling --tf32 --fused` | **0.243** | 22 |
| hybrid | none | 0.522 | 82 |
| hybrid | `--scan doubling32` | 0.535 | 56 |
| hybrid | `--scan doubling` | 0.460 | 56 |
| hybrid | `--head-amp` | 0.452 | 76 |
| hybrid | `--sparse-claws` | 0.497 | 79 |
| hybrid | `--detach-state` (changes the gradient) | 0.479 | 73 |
| hybrid | `--tf32 --head-amp --sparse-claws --scan doubling32 --fused` | 0.500 | 65 |
| hybrid | `--tf32 --head-amp --sparse-claws --scan doubling --fused` | **0.400** | 55 |
| hybrid, 2 heads (`--mem-every 4`) | the `doubling32` combination | 0.415 | 42 |
| hybrid, `--chunk 512` | the `doubling32` combination | 0.477 | 47 |
| hybrid | the `doubling32` combination + `--detach-state` | 0.467 | 50 |

The loss after the measured steps is the same to three decimals with and without each switch that does not change the model (5.135 hybrid, 5.238 zero-heads, 5.285 transformer at 20 steps), and the runner keeps evals and checkpoints on the plain module.

**Reading.** Compiling is the switch that matters: the hybrid goes from 2.9 times the transformer's step time to 1.9 times with compile alone and to 1.45 times (0.400 against 0.276) with the heads and the scan under bf16 as well. Without its heads the backbone is faster per step than the transformer (0.243). The controller's first reading of the eager profile, that the step was bound by kernel time because the card showed 100 % utilisation, was wrong: most of the eager time was many small operations (1,630 multiplies and 1,350 copies per step in the eager run with every switch on) that compile fuses away. In the best compiled hybrid, matrix products, top-k and the batched products of the heads are 60 % of device time.

**Not tested:** whether heads and scan under bf16 hold their quality over a long run (the float32 pin and the float64 closed form were precautions, never tested in training); 2 heads, `--chunk 512` and `--detach-state` together with `--scan doubling`; any compile mode other than the default. The 3,600 s runs (compiled transformer, zero-heads, hybrid, hybrid with 2 heads; `night1.sh` on the spike branch's worktree) answer the first; their step-1,000 scores can be laid next to the spine runs', which share the seed and the warmup.

**Equal seconds, as the spine ran (eager).** The rule-mandated rerun of the transformer for the hybrid's seconds (`spine16_tf_eqsec_s0`: 47,682 steps, 17,622 s) reads 0.718 cold / 0.644 warm on the 128 in-run dev windows, against the hybrid's 0.780 / 0.711 after 17,586 s. Uncompiled, the hybrid's lead per step does not survive equal seconds: it is 0.062 behind. The fp32 scores on the 1,024 windows go into the Stage 1 write-up (Run 53).

### Row 2, test a: a one-pass table added to trained checkpoints (2026-10-04, CPU): passes

`flybrain/corpus_memory.py`, `scripts/build_corpus_memory.py`, `scripts/probe_memory_mix.py`, `results/mem_mix_{dev,test}.json` (commit e70d741; the tables and per-byte losses are under `results/mem/`, not committed).

**The table.** One direct-mapped next-byte table per context order (4, 8, 16 and 32 bytes), 2^27 slots each, 2 GiB in all. A slot holds a 16-bit tag of the context, the byte that followed it, and a confidence counter. It is written in one pass over the 200,000,000 training bytes with no gradient: 22 to 32 s of wall time on one CPU core (two builds, bit-identical tables). The held-out bytes never touch it: 60 of 60 sampled hits at each of orders 8, 16 and 32 were found in the training bytes with the stored byte after them.

**The mix.** p(true byte) = (1 - lambda) x the model's p + lambda x [the table's byte is the true byte], one lambda per (longest order with a hit, confidence class): 16 weights, fitted on dev, applied to test. A hit counts only when its whole context lies inside the model's own 2,048-byte window.

Test, the 1,024 yardstick windows, fp32. The model-alone columns reproduce the committed spine scores to 1.7e-6:

| checkpoint | training seconds | alone, cold | with the table, cold | gain (95 % interval over windows) | alone, warm | with the table, warm | gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 17,586 | 0.7856 | 0.7564 | +0.0292 (+0.0257, +0.0325) | 0.7125 | 0.6878 | +0.0246 (+0.0207, +0.0285) |
| transformer | 5,901 | 0.8098 | 0.7729 | +0.0369 (+0.0332, +0.0405) | 0.7296 | 0.6989 | +0.0307 (+0.0265, +0.0349) |
| zero-heads | 8,189 | 0.8315 | 0.7983 | +0.0332 (+0.0296, +0.0368) | 0.7808 | 0.7504 | +0.0304 (+0.0264, +0.0345) |

Resampling runs of 8 consecutive windows (about one file) instead of single windows widens the cold intervals to (+0.0239, +0.0349), (+0.0309, +0.0433) and (+0.0276, +0.0394). Dev, where the weights were fitted, reads +0.0338, +0.0401 and +0.0368.

Where it comes from (test, cold): orders 16 and 32, 37 % of the bytes, carry about 86 % of the gain; order 4 hits 98.8 % of positions and adds nothing. The quarter of the windows with the most order-32 hits gets about half of the gain, the half with the fewest about a quarter. No held-out file has an exact copy among the training files, so the duplicate-excluded row removes no window.

**Claim:** with no training at all, a table read once from the training bytes lowers each trained checkpoint's bits/byte on held-out files by 0.03 to 0.04. The transformer with the table (0.7729 cold, 0.6989 warm) is ahead of the hybrid alone (0.7856, 0.7125), which took three times the training seconds; the hybrid with the table is at 0.7564. **Not a claim:** that a trained mixer would gain this much or more (the mix is a point mass with 16 weights, and no model here was trained to use it); the third decimal (over the four quarters of the test slab the hybrid's gain is +0.029, +0.024, +0.045, +0.019, and each split is one contiguous 2 MB slab of 110 to 140 files); anything outside this kernel tree; that the table is free at inference (2 GiB, four lookups per byte); why the hybrid gains least (its heads may already carry part of it: untested).

### Reference: classical one-pass compressors on this corpus (2026-10-04, one CPU core)

Marginal bits/byte of the dev and test yardstick regions (2,097,152 bytes each, the 1,024 fixed windows) given the 200,000,000 training bytes: 8 x (size(train + region) - size(train)) / region bytes. The compressor keeps adapting inside the region and reads it in order, so this is not the models' cold 2,048-byte-window protocol. Read it as what one pass with no gradient reaches on this data, not as a row of the spine table.

| tool | seconds to read the training bytes | dev | test | region alone, dev | region alone, test |
|---|---:|---:|---:|---:|---:|
| PPMd, order 6 | 56 | 1.452 | 1.465 | 1.459 | 1.480 |
| PPMd, order 8 | 70 | 1.291 | 1.295 | 1.397 | 1.411 |
| PPMd, order 16 (3.9 GB of model memory) | 45 | 1.138 | 1.128 | 1.383 | 1.396 |
| xz -9e, 512 MiB dictionary | 208 | 1.179 | 1.198 | 1.504 | 1.536 |
| zstd --ultra -22 --long=31 | 246 | 1.264 | 1.282 | 1.570 | 1.585 |

For scale, from `results/spine16_*_s0.json` (128 in-run dev windows): the transformer reads 1.164 after 739 training seconds on the GPU and 0.795 after 5,901 s; the hybrid 0.780 after 17,586 s. One pass of counting in 45 CPU-seconds lands where 12 to 14 GPU-minutes of gradient training land (1.164 at 739 s, 1.067 at 1,111 s), and has no way to improve on the same data. Script: `notes/speed-round/compress_ref.sh` (7-Zip 16.02 PPMd with `mem=3900m`, xz 5.4.1, zstd 1.4.8, single thread, under load from other jobs).
