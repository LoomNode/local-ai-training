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
| 1 | Cheaper heads, same model | Where do the heads' 0.59 s per step go, and how much of it can go without losing bits? | Profile one step. Time the cuts: heads under bf16, no gradient through the address, 2 heads instead of 4, compiled, chunk size. The best combination gets one 3,600 s run. | it beats the transformer at 3,600 s | not started |
| 2 | Memory filled by reading the corpus once, and kept | Does a table written in one pass over the training bytes, with no gradient, buy bits that training seconds would otherwise have to buy? | a: existing checkpoints plus the table, mixed by a rule fitted on dev, scored on test (CPU, no training). b: a small network trained with the table's prediction as an input, 3,600 s. c (added after the source re-review): hashed n-gram embedding tables, a fixed address with a learned table, as an extra input, 3,600 s. | a: a gain on test with zero training. b, c: it beats the transformer at 3,600 s | a building |
| 3 | Local learning, no backpropagation through the stack | How much is lost when no error signal crosses layers, and is a step any cheaper? | i: backbone frozen at random initialisation, only the readout (and the memory) learn. ii: pairs of layers with their own next-byte loss and no gradient between pairs. iii (added after the source re-review): every other layer frozen at random initialisation, gradients still passing through. 3,600 s each. | within 0.02 bits/byte of the fully trained hybrid at equal seconds | building |

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
