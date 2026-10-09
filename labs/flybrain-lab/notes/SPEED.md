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
| 1 | Cheaper heads, same model | Where do the heads' 0.59 s per step go, and how much of it can go without losing bits? | Profile one step. Time the cuts: heads under bf16, no gradient through the address, 2 heads instead of 4, compiled, chunk size. The best combination gets one 3,600 s run. | it beats the transformer at 3,600 s | DOES NOT PASS: 1.01 to 0.40 s/step with `torch.compile` and half precision (transformer 0.35 to 0.28), and the hybrid is still 0.051 behind the transformer at 3,600 s (0.026 before the transformer's own step was fixed) and behind at two hours |
| 2 | Memory filled by reading the corpus once, and kept | Does a table written in one pass over the training bytes, with no gradient, buy bits that training seconds would otherwise have to buy? | a: existing checkpoints plus the table, mixed by a rule fitted on dev, scored on test (CPU, no training). b: a small network trained with the table's prediction as an input, 3,600 s. c (added after the source re-review): hashed n-gram embedding tables, a fixed address with a learned table, as an extra input, 3,600 s. | a: a gain on test with zero training. b, c: it beats the transformer at 3,600 s | a PASSES (+0.03 to +0.04 on test). b PASSES: the table as an input is worth 0.09 at 3,600 s on both networks, and a lookup in the window's own bytes (added during the round) 0.05 on the backbone and about 0.02 on the transformer; with both, the transformer reads 0.691 and the backbone 0.706, against the plain transformer's 0.806. c: 0.019, short of the transformer (0.823) |
| 3 | Local learning, no backpropagation through the stack | How much is lost when no error signal crosses layers, and is a step any cheaper? | i: backbone frozen at random initialisation, only the readout (and the memory) learn. ii: pairs of layers with their own next-byte loss and no gradient between pairs. iii (added after the source re-review): every other layer frozen at random initialisation, gradients still passing through. 3,600 s each. | within 0.02 bits/byte of the fully trained hybrid at equal seconds | DOES NOT PASS: 0.067 (iii), 0.073 (ii) and 0.88 (i) behind the fully trained hybrid at 3,600 s, for a step at most 8 % cheaper |

Expectations written before any of these ran (the controller's, 2026-10-04): 1 helps and cannot reach the transformer by itself, because the backbone without heads is already slower per step; 2 has room for a large win, and the handoff probe (a memory the model was never trained to use gave +0.009 bits/byte on dev) is its first hint; 3 is a negative, as local rules lose quality at scale in the literature.

## State of the round (2026-10-08)

The current summary; the dated sections below are the record. Every number is bits/byte on the same 1,024 fixed dev windows, cold (whole window), one seed, at the base arm's cost (since 2026-10-07 a calibrated step count, not seconds: "The cost measure changes").

| arm | cold | what it says |
|---|---:|---|
| plain transformer, 8x512 (`--rope-table`) | 0.806 | the bar |
| + corpus table (orders 8/16/32) + match lookup, the base arm | 0.691 | the two memories are worth 0.115 |
| + three more orders, template/word/line keys, counters (18 inputs, fused channel) | 0.670 | more keys for the same table: -0.021 at equal cost |
| the same with 2^28 two-way tables | 0.657 | a bigger, better-ruled table: -0.013 at equal steps |
| **4 layers of 512 with the 2^28 tables** | **0.652** | fewer weights, more steps, same table: the best at the hour, 11.5 M parameters |
| **the same at twice the cost: 8 / 6 / 4 layers** | **0.624 / 0.619 / 0.623** | doubling the budget is worth 0.03 at every depth; the depth that wins moves from 4 to 6; 8 catches up to 4 |
| 2 layers with the 2^28 tables / 4 layers with 2^29 tables | 0.663 / 0.650 | the network cannot go below 4 layers (+0.011); the table's size has flattened (-0.007, then -0.002 per doubling) |
| compute only where a cheap gate is unsure: skip 30 % at 4 layers / 50 % at 8 | 0.651 / 0.669 | +0.029 / +0.044 at equal steps: the skipped positions cost twice, the step saved nothing at 4 layers |
| PAQ bit tables added to the 18 inputs | 0.650 | -0.002, the same as a bigger table: under the seed line |
| the fly hybrid (minGRU + heads + local attention) with the 18 inputs | 0.708 | the fly backbone loses 0.038 to the transformer with the same memory |
| 12x384 / 6x768 / window 4096 / GLA, with the 18 inputs | 0.674 / 0.688 / 0.713 / 0.779 | no other backbone beats 8x512 |
| PAQ keys / top-two slots instead of the counters | 0.666 / 0.664 | small (0.004 to 0.006, below the seed line) |

Reading: the knowledge wants to live in the memory, and the network's depth is a knob set by the budget. Every arm that moved weight from the network to the table won, every arm that made the network fancier lost. What is different about this model is not the network, which is a standard transformer, but the memory: a table written in one pass over the corpus by a local write rule (the fly's part), read as inputs with confidence, separable and extendable after training. The depth curve at twice the cost (2026-10-08) settled the shallow question: the hour was talking. Doubling the budget is worth 0.03 at every depth, more than any architecture choice left on the table; 4 layers win an hour, 6 win two hours, and the deeper model gains more per doubling, so the big train's depth follows its budget (and its corpus: the 4-layer arm saw the 200 MB train split four times in two hours). The radical bets of 2026-10-08 (Kyle: "let's test all the radical bets") did not move it: 2 layers lose the hour; computing only where a cheap gate is unsure costs twice what it saves; PAQ bit tables and a 2^29 table are each worth 0.002. Still running overnight: 1 and 0 layers, and learned slots on top of the count table. The open list is `/games/ailab/TODO.md`, the finished one `/games/ailab/DONE.md`.

Where things are: scripts for every arm and table in `notes/speed-round/checks/` (they locate themselves; the box paths `WT`, `SP`, `MEM` at their top are this box's layout); the training runs (curves, steps, bytes seen, parameters, logged seconds) as `results/<run name>.json`; the dev-window scores with per-window values as `results/tables/k_<arm>.json`; the calibration session as `results/tables/calsession_2026-10-07.json`; feature tables under `results/mem/` are rebuilt by the builders, not stored. The model code for the channels is on `feat/speed-round` (`flybrain/model/inputs.py`, `flybrain/data/memfeat.py`, `flybrain/corpus_memory.py`) and the speed switches, the fused channel and `--rope-table` on `spike/speed-int` (3a812b5), which every run since 2026-10-05 is pinned to. Not done: `README.md` and `notes/RESULTS.md` still describe the September state (Kyle's files); nothing is pushed.

## Where the round stands (2026-10-05)

All three rows are measured. Every run of the round, dev and test, fp32, 1,024 windows each, 16M parameters, one hour on GPU 1 unless marked. The test split was read once, after the last run. Sections under Results give each row's detail.

| arm | steps | training seconds | dev cold | dev warm | test cold | test warm | test, against the bar |
|---|---:|---:|---:|---:|---:|---:|---:|
| transformer + match + table | 14,331 | 3,690 + 26.5 | 0.691 | 0.619 | 0.697 | 0.625 | -0.111 |
| backbone + match + table | 13,745 | 3,531 + 26.5 | 0.706 | 0.646 | 0.713 | 0.653 | -0.095 |
| transformer + table | 12,767 | 3,566 + 26.5 | 0.732 | 0.665 | 0.738 | 0.669 | -0.070 |
| backbone + table | 14,093 | 3,536 + 26.5 | 0.745 | 0.702 | 0.750 | 0.707 | -0.058 |
| transformer + match | 14,320 | 3,603 | 0.787 | 0.702 | 0.790 | 0.707 | -0.018 |
| backbone + match | 14,535 | 3,639 | 0.792 | 0.721 | 0.793 | 0.723 | -0.015 |
| **transformer, `--rope-table` (`tf_r`): the bar** | 15,953 | 3,739 | 0.806 | 0.726 | 0.808 | 0.729 |  |
| backbone + n-gram embeddings | 13,471 | 3,627 | 0.823 | 0.772 | 0.823 | 0.773 | +0.015 |
| transformer as first run (`tf_c`) | 12,668 | 3,678 | 0.831 | 0.750 | 0.833 | 0.753 | +0.025 |
| backbone (`zh_c`) | 13,849 | 3,628 | 0.842 | 0.789 | 0.842 | 0.792 | +0.034 |
| hybrid, 2 heads (`hyb2_c`) | 10,028 | 3,511 | 0.850 | 0.780 | 0.849 | 0.781 | +0.041 |
| hybrid, 4 heads (`hyb_c`) | 7,783 | 3,494 | 0.857 | 0.781 | 0.858 | 0.783 | +0.050 |
| hybrid, alternate layers frozen | 8,286 | 3,492 | 0.924 | 0.851 | 0.929 | 0.857 | +0.121 |
| hybrid, local blocks of 2 | 7,995 | 3,610 | 0.930 | 0.853 | 0.937 | 0.859 | +0.129 |
| hybrid, backbone frozen | 8,522 | 3,536 | 1.736 | 1.608 | 1.783 | 1.644 | +0.975 |
| two hours: transformer as first run | 24,735 | 7,326 (GPU 0) | 0.767 | 0.687 | 0.768 | 0.690 |  |
| two hours: hybrid, 4 heads | 16,000 | 7,483 (GPU 0) | 0.787 | 0.713 | 0.786 | 0.713 |  |

Test repeats dev: every arm is within 0.007 of its dev score (the frozen-backbone arm within 0.047) and the order is the same.

1. **Training speed is not where the hybrid wins.** With every switch that leaves the model unchanged, a hybrid step costs 1.9 times a transformer step, and after one hour the hybrid is 0.050 behind on test. The memory heads cost more than they give at this budget: the backbone without them is 0.016 ahead of the hybrid.
2. **What is cheap is memory written without a gradient.** A table written in one pass over the corpus (26.5 CPU-seconds) and fed as an input is worth 0.09 bits/byte on either network. The transformer with table and match lookup after one hour (0.691 dev) is below the plain transformer after 47,682 steps (0.723 dev, `spine16_tf_eqsec_s0`), which is three hours at the corrected step.
3. **A lookup in the window's own bytes does the memory heads' job for nothing.** It gives the backbone 0.05 and puts it 0.065 below the hybrid at 56 % of the step cost. The transformer, whose attention already copies inside the window, gains about 0.02.
4. **Local learning does not pay here.** Withholding backpropagation from part of the stack loses 0.07 to 0.9 and saves at most 8 % of a step.
5. **The baseline needed the same care as the model.** The transformer was computing its rotary tables in every layer; fixing that moved the bar from 0.831 to 0.806 and doubled the hybrid's deficit. Under the same memories the transformer is ahead of the backbone in every pair.

**Recommendation** (the output of a spike; the decision is Kyle's).

- Keep the corpus table and the match lookup as input channels. They are built with tests on `feat/speed-round`. Train with `--compile`, bf16 and `--rope-table` from now on.
- Stop spending training seconds on gradient-trained memory heads, n-gram tables and local learning at this size.
- For the host network the transformer is the better choice on these numbers. The backbone's remaining argument is not speed but streaming: it carries a fixed-size state from call to call, which the group runtime of Stage 0 uses, and the transformer does not.
- The table is a group memory in the sense of the group-learning design: anything that has read a file can write it, with no gradient, and every reader with the same hash can use it. The next question is therefore the Stage 2 question in a cheaper form: does a table extended at inference time with a project's own files, which no training saw, help a table-fed model on a held-out file of that project? Stage 1 measured that related files recover 0.044 bits/byte more than random ones. The probe needs no training: rebuild the features with the related held-out files added (the scored file and its duplicates left out) and score the existing table-fed checkpoints.

**Limits.** One seed and one size (16M parameters). One corpus, kernel code, where exact repeats of 8 to 32 bytes are common. One hour; the table's share shrinks with longer training (test a: 0.03 to 0.04 on 16,000-step checkpoints). The three transformer arms with memories were not rerun with `--rope-table`, nor was the two-hour pair. A table-fed model cannot be prompted yet (`scripts/sample.py` does not compute the channels), and at inference the table is 1.5 GiB with three lookups per byte.

**Where things are.** The channels: `feat/speed-round` (this branch). The speed switches, the local-learning switches and `--rope-table`: `spike/speed-int` (a9a9209), throwaway until a design is chosen. Result files and logs of every run: `results/speed*` on this branch; the timing reports: `results/profile/`. The queue scripts as they ran: `notes/speed-round/queues/` (they name worktrees that need not exist any more). The two one-off checks: `notes/speed-round/checks/`. Checkpoints: `results/ckpt/speed_*` in the main checkout (not committed).

## Tables added after training (2026-10-05 evening, Kyle's question)

Kyle, shown how the table works: "what if more or most of the model worked like that? like each new group of information is a table or memory the model can call on. i was thinking this shifts the size constraint to disk after the reasoning weights are done." Two probes, no training, on the four table-fed checkpoints of the round (transformer and backbone, with and without the match lookup). Scripts in `notes/speed-round/checks/` (`extend_table.py`, `cpython_tables.py`, `score_tables.py`, `analyze_tables.py`); per-window scores in `results/tables/`. Scored on GPU 0 while it was idle (Kyle: "use gpu when available"), fp32, 1,024 windows of 2,048 bytes; intervals are paired over windows.

### Probe 1: the kernel table extended with files the model never saw

The training tables (200 MB of kernel files) extended with 11.6 MB of held-out kernel files: the legacy validation region, the dev files after the yardstick windows, then the window-region files in order, each looked up before it is written, so no position sees its own file or a later one (no file in the region has a copy elsewhere). Two write rules: the build's (a new context takes an occupied slot) and a no-evict rule (a slot that holds another context keeps it). The dev windows then scored with the base features and with the extended ones.

| checkpoint | base | build rule | change | no-evict | change | change on the 406 windows whose directory has material in the extension / on the other 618 (no-evict) |
|---|---:|---:|---:|---:|---:|---|
| transformer + table | 0.7320 | 0.7308 | -0.0013 (-0.0024, -0.0001) | 0.7302 | -0.0018 (-0.0025, -0.0011) | -0.0041 ± 0.0017 / -0.0003 ± 0.0004 |
| backbone + table | 0.7452 | 0.7442 | -0.0010 (-0.0021, +0.0001) | 0.7436 | -0.0016 (-0.0023, -0.0009) | -0.0038 ± 0.0016 / -0.0001 ± 0.0004 |
| transformer + match + table | 0.6905 | 0.6893 | -0.0012 (-0.0023, -0.0002) | 0.6888 | -0.0017 (-0.0024, -0.0010) | -0.0039 ± 0.0016 / -0.0003 ± 0.0004 |
| backbone + match + table | 0.7061 | 0.7050 | -0.0011 (-0.0021, +0.0000) | 0.7045 | -0.0016 (-0.0023, -0.0009) | -0.0036 ± 0.0016 / -0.0001 ± 0.0004 |

Under the build rule the other 618 windows get worse (+0.0011 ± 0.0007 on every checkpoint): the new files evicted as many entries as they added (order 16: 0.9 % of positions gained a hit, 0.9 % lost one). The no-evict rule keeps the gain and drops the loss.

**Reading.** Files added after training are read: where the window's directory has material in the extension the score improves by 0.004, with no training. The effect is small because the base table already holds the whole tree (5.8 % more of the same files adds 0.1 to 0.3 % of hit positions), and the build's replacement rule is wrong for an extension: at 51 to 68 % occupancy it overwrites corpus-wide entries with singletons. A table meant to grow needs a rule that respects confidence, or more slots.

### Probe 2: a kernel-trained model reading a CPython table

The four checkpoints scored on 1,024 windows of `corpora/cpython.bytes` (Python, never seen in training) from byte 40,000,000, with four tables: none (every lookup a miss), the kernel training tables, a table written in one pass over the CPython bytes before the windows (40 MB, 7 s, copies of window files and the straddling file skipped), and the kernel tables extended with the same CPython bytes. On the windows the kernel table answers 49 / 8 / 1 % of positions (orders 8 / 16 / 32), the CPython table 81 / 45 / 17 % (right 69 / 79 / 89 %).

| checkpoint | no table | kernel table | CPython table | kernel + CPython | CPython table minus none |
|---|---:|---:|---:|---:|---:|
| transformer + table | 1.433 / 1.296 | 1.390 / 1.254 | 1.058 / 0.973 | 1.059 / 0.973 | -0.375 (-0.390, -0.359) |
| backbone + table | 1.490 / 1.408 | 1.454 / 1.372 | 1.108 / 1.057 | 1.110 / 1.059 | -0.382 (-0.397, -0.366) |
| transformer + match + table | 1.343 / 1.189 | 1.305 / 1.152 | 1.002 / 0.906 | 1.003 / 0.906 | -0.341 (-0.356, -0.326) |
| backbone + match + table | 1.373 / 1.237 | 1.348 / 1.213 | 1.041 / 0.956 | 1.043 / 0.957 | -0.332 (-0.347, -0.317) |

The plain checkpoints on the same windows: transformer (`tf_r`) 1.396 / 1.256, `tf_c` 1.416 / 1.276, backbone 1.503 / 1.418, hybrid 1.460 / 1.324.

**Reading.** A model trained once to read *a* table reads a table of another language. With a CPython table that took 7 s to write, the kernel-trained transformer goes from 1.39 to 1.06 bits/byte on Python, 0.34 below the best plain model, with no gradient step on Python. The kernel table on Python is worth 0.04; the old entries in a merged table neither help nor hurt. Without any table the table-trained models are 0.017 worse than their plain twins: they lean on the table, but not much. The match lookup alone is worth 0.09 on Python (1.433 to 1.343 with no table) against about 0.02 on the kernel: this Python has more repeats inside a window. How far that is from a model trained on Python: two reference runs on GPU 0 (`results/cpy_tf_mt_r.json`, `cpy_tf_r.json`; the transformer trained on the same 40 MB of CPython for 15,953 steps, about six passes over it, with match + table and plain). Plain, trained on Python: 0.831 / 0.768. With match + the CPython table, trained on Python: 0.725 / 0.661. So the kernel-trained transformer handed the CPython table (1.002 / 0.906) recovers about 70 % of the gap between no training on Python (1.396) and an hour of it (0.831), and the Python-trained model with the same table is a further 0.28 ahead. On Python as on the kernel, the table and match lookup are worth 0.11 to a model trained with them.

**Claim:** the table channel transfers: a model trained with one corpus's table uses a table of a corpus it never saw, and the gain (0.33 to 0.38 bits/byte on Python) is an order of magnitude larger than anything in the round that cost training time. Files added to an existing table after training are read too (0.004 where they are related), provided the write rule does not evict. **Not a claim:** that a table replaces training on the new corpus: it recovers about 70 % of what an hour of training on it gives a plain model, and a model trained on the corpus reads the same table to 0.28 better; the size on text with fewer exact repeats than CPython's test suite; anything about tables keyed by the model's state rather than by bytes (the next thing to build).

## What the fly's memory circuit suggested, tested (2026-10-05 night)

The read of FlyWire v783 is in `notes/speed-round/fly-circuit.md`; it names three mechanisms the table lacks. The first was tested the same night, and the test turned up two things about the table itself. Scripts `checks/loss_pass.py`, `checks/gated_tables.py`; scores `results/tables/g_*.json`; GPU 0 while idle; the two transformer table-fed checkpoints of the round (trained with the 2^27 tables), dev windows, cold bits/byte.

### Write what surprised you: fails

The fly writes a memory only when the key meets a computed signal; here the signal is the plain transformer's own loss at each training position (`loss_tf_r.npy`: mean 0.768 bits; 26 % of positions at or above 0.5 bit, 20 % at or above 1 bit, 14 % at 2). Tables written only at those positions, against tables written everywhere, at three sizes:

| table | positions written | transformer + table | transformer + match + table |
|---|---:|---:|---:|
| every position, 2^27 slots (the round's table) | 100 % | 0.732 | 0.691 |
| loss >= 0.5 bit, 2^27 | 26 % | 0.789 | 0.734 |
| loss >= 1 bit, 2^27 | 20 % | 0.820 | 0.758 |
| every position, 2^25 | 100 % | 0.767 | 0.723 |
| loss >= 1 bit, 2^25 | 20 % | 0.829 | 0.768 |
| every position, 2^24 | 100 % | 0.790 | 0.744 |
| loss >= 1 bit, 2^24 | 20 % | 0.842 | 0.780 |
| loss >= 2 bits, 2^24 | 14 % | 0.863 | 0.797 |

Gating keeps the contexts whose continuation is unpredictable, and their stored byte is right 40 to 70 % of the time instead of 66 to 90 %. A gated table is worse than an ungated one of the same size everywhere. The table's worth is in confirming predictable continuations with high confidence, not in covering the weights' failures; the fly's rule fits one bit of valence per odour, not a next byte. (The checkpoints were trained with the ungated table's statistics, which also counts against the gated ones; the gap is too large for that to be the reason.)

### The table's size and its write rule: both pay

| table | transformer + table | transformer + match + table | hits at order 32 (right) |
|---|---:|---:|---:|
| 2^24 slots, the build's rule | 0.790 | 0.744 | 4.8 % (0.90) |
| 2^24, confirmed entries stay | 0.774 | 0.730 | 6.4 % (0.92) |
| 2^25, the build's rule | 0.767 | 0.723 | 6.5 % (0.90) |
| 2^25, confirmed entries stay | 0.757 | 0.714 | 7.8 % (0.92) |
| 2^27, the build's rule (the round's table) | 0.732 | 0.691 | 10.4 % (0.91) |
| 2^27, confirmed entries stay | 0.729 | 0.688 | 11.0 % (0.91) |
| 2^28, the build's rule | 0.724 | 0.683 | 11.8 % (0.91) |
| 2^29, the build's rule | 0.720 | 0.680 | 12.7 % (0.91) |

Two more rows, same weights (`checks/assoc_tables.py`): a **two-way table**, where a context may sit in either of two buddy slots and a newcomer replaces the weaker one unless both are confirmed, and the biggest direct-mapped tables.

| table | transformer + table | transformer + match + table |
|---|---:|---:|
| 2^24, two-way | 0.770 | 0.726 |
| 2^25, two-way | 0.751 | 0.709 |
| 2^27, two-way | 0.723 | 0.683 |
| 2^29, confirmed entries stay | 0.719 | 0.679 |
| 2^30, the build's rule | 0.718 | 0.678 |

The two-way table at 2^27 scores what the direct-mapped table scores at 2^28 (0.723 against 0.724): a doubling for free. Past 2^28 the curve flattens for these weights (0.724, 0.720, 0.718 at 2^28, 2^29, 2^30), which were trained against a 2^27 table and its hit statistics.

"Confirmed entries stay": a slot whose entry has been confirmed at least once (confidence 2 or more) is never taken by a different context. It is the first half of the fly's compartments (a memory that repetition has confirmed is kept) and it is free: -0.016 at 2^24, -0.010 at 2^25, -0.003 at 2^27, and by construction it removes the eviction loss of probe 1. Size: every doubling of the slots is still worth something, 2^27 to 2^28 -0.008, to 2^29 -0.012, with the same weights and no training; at 2^29 the three tables are 6 GiB. Where that curve ends was not measured.

**Claim:** with fixed weights, more table is a better model (0.732 to 0.718 from 8x the slots, flattening), a better write rule is a better model (0.732 to 0.729 at the same size, 0.790 to 0.774 at an eighth the size), and a two-way table is worth a doubling (0.732 to 0.723 at 2^27); writing only where the weights fail is worse at every size. **Not a claim:** anything beyond 2^30; that a model trained with the bigger or better-ruled table would not gain more (likely, untested); that gating by a different signal (the table's own disagreement, the readout's confidence) would also fail.

## What else repeats: other keys for the same table (2026-10-06, GPU 0)

Kyle's question ("is there anything else that repeats that we could try to compute once?"): the byte table looks up the exact last 8, 16 or 32 bytes. Other things that repeat in code can be looked up by the same machinery with a different key, computed once over the corpus on the CPU and served through the same input channel as extra pseudo-orders (`checks/key_tables.py`, `checks/counter_tables.py`; one pass each, 27 minutes for the keys, 3 for the counters). The arms retrain the round's best recipe (transformer + match lookup + 2^27 byte table, with `--rope-table`) at the same step count, 14,331 steps = the hour's steps of that recipe, so the only difference between rows is the extra inputs (`checks/keys_gpu0.sh`, `checks/counters_gpu0.sh`, `checks/all_gpu0.sh`; scores `results/tables/k_*.json`; dev windows, bits/byte).

| arm | extra inputs | parameters | train s | cold | warm | cold vs base (95 %) |
|---|---|---:|---:|---:|---:|---:|
| base | none (orders 8, 16, 32) | 17.90 M | 3,307 | 0.691 | 0.620 | |
| orders | byte orders 4, 12, 64 | 18.30 M | 3,382 | 0.675 | 0.605 | -0.017 (-0.018, -0.016), better on 86 % of windows |
| keys | template 16/32, words 2/3, lines | 18.58 M | 3,384 | 0.673 | 0.605 | -0.018 (-0.020, -0.016), better on 80 % |
| counters | column, indent, previous indent, brace depth, paren depth, in-string, in-comment | 18.85 M | 3,483 | 0.688 | 0.619 | -0.003 (-0.004, -0.002), better on 61 % |
| all | the three rows above together, 18 inputs | 19.93 M | 3,683 | 0.665 | 0.599 | -0.026 (-0.028, -0.024), better on 90 % |

Do they stack? Partly: -0.026 together against -0.038 if the three gains were independent. The combined arm beats the orders arm by 0.009 (interval ±0.002, better on 66 % of windows) and the keys arm by 0.008 (±0.001, 72 %), so the keys and the extra orders catch different things with some overlap.

The keys: **template** = the last 16 or 32 symbols with every identifier run replaced by one placeholder (whitespace kept) plus the current partial identifier, so `if (x < 0) return -EINVAL;` is found under any `x`; **words** = the last 2 or 3 tokens by content with whitespace dropped; **lines** = the previous full line plus the current line so far. Keys and orders are indistinguishable from each other (keys - orders -0.001, interval (-0.003, +0.000)). The **counters** are not lookups: each is a function of the bytes before the position (a small state machine for strings and comments, depths reset at file headers), fed as a value through the same channel (pred = 128 + the value, confidence 1 everywhere, so the channel's output bonus lands on byte codes that are never the answer). They are worth 0.003: the weights learn to keep them on their own.

What the keys add before any training (`results/tables/coverage_*.txt`, dev positions): the byte table hits 80 % of positions and is right at 71 % of its hits. The template-16 key hits 57 %, right 66 %, and covers 6 % of positions the byte table misses, right at 60 % of those; words 64 %, right 53 %, covering 6 % more at only 32 %; lines 18 %, right 79 %, covering 0.3 % more. Where a key and the byte table disagree, the byte table is right more often (35 % against 19 % for the template key). Order 4 hits 99 % of positions and covers 19 % the longer orders miss, right at 44 % there; order 64 hits 4 % and wins its disagreements with the shorter orders (77 % against 10 %), but those are 0.2 % of positions. So the gain is not from new correct hints alone: the model gets as much from three more byte orders (one of them a nearly complete "last continuation of this 4-gram" over the corpus) as from the three structural keys. It is not from the channel having more parameters either: the counters add seven channels and 0.95 M parameters for 0.003. What the extra orders and keys bring is corpus-wide statistics the weights have not learned in an hour (the last continuation of every 4-gram, with a count), an external n-gram model read through the channel; the counters bring nothing the weights do not already compute.

**Claim:** at equal steps, three more byte orders or the three structural keys each take the best arm from 0.691 to about 0.674 (-0.017, intervals of ±0.002), with 2 % more training time and 0.4 to 0.7 M more parameters; the counters are worth 0.003 for 5 % more time; all of them together reach 0.665 (-0.026), two thirds of the sum of the parts, at 11 % more time per step. **Not a claim:** which of keys and orders is better; that 0.665 holds at longer budgets, where the weights catch up on the n-gram statistics (at equal seconds it holds as 0.670: next section); anything about use time, where the keys cost a hash each per byte; anything about seeds (one each; training noise here is typically 0.003 to 0.005).

### The same inputs at equal seconds, through a fused channel, and from 2^28 two-way tables (2026-10-07, GPU 1)

Three more arms, same recipe and dev windows (`checks/fused_gpu1b.sh`, `checks/gpu1_queue.sh`, `checks/gpu1_queue3.sh`, `checks/assoc_all.py`). The **fused channel** (`--fused-channels`, `spike/speed-int` 3a812b5, `tests/test_fused_channel.py`) is the input channel without its Python loop over orders: the same parameters in one byte table, one class table and one bonus layer, read with one `embedding_bag` and one gather, the same numbers to float rounding and the same gradients. In a 400-step calibration on a quiet card it costs 0.2345 s per step against 0.2565 unfused and 0.2308 for the base arm, so the 18 inputs cost 1.6 % per step instead of 11 %. The **equal-seconds arm** got the steps that calibration buys in the base's 3,307 s: 13,253. The **2^28 two-way arm** reads all eleven lookup orders from two-way tables of 2^28 slots (the rule of "The table's size and its write rule": a context may sit in either of two buddy slots), built in one leak-free pass in 172 s with 11.8 GB of RAM, 1 GiB per order on disk.

| arm | inputs | tables | steps | train s | cold | warm | cold vs base (95 %) | vs the combined arm |
|---|---|---|---:|---:|---:|---:|---:|---:|
| base | orders 8, 16, 32 + match | 2^27 | 14,331 | 3,307 | 0.691 | 0.620 | | |
| all (above) | 18 inputs | 2^27 | 14,331 | 3,683 | 0.665 | 0.599 | -0.026 | |
| all, fused channel | 18 inputs | 2^27 | 14,331 | 3,765 * | 0.665 | 0.598 | -0.027 | -0.001 (-0.001, -0.000) |
| all, fused, equal seconds | 18 inputs | 2^27 | 13,253 | 3,486 * | 0.670 | 0.604 | -0.021 (-0.023, -0.019), better on 82 % | +0.005 (±0.001) |
| all, fused, 2^28 two-way | 18 inputs | 2^28 two-way | 14,331 | 3,780 * | 0.657 | 0.592 | -0.034 (-0.037, -0.032), better on 92 % | -0.008 (-0.009, -0.007), better on 71 % |

\* Contended timings: the box was building tables (12 GB of RAM, one core) and running a second arm on GPU 0 during all three, so each step took 0.263 s instead of the calibrated 0.2345, and the equal-seconds arm overran its 3,307 s by 5 %. The step-time saving of the fused channel is the calibration's number; these hour-runs do not show it, and they are not a timing measurement.

At equal steps the fused channel is the unfused model (-0.001, within noise), as built. At equal seconds the combined inputs are worth -0.021 on the round's bar (0.670; held to exactly 3,307 s the arm would have had 5 % fewer steps, worth about 0.003 by the step-count arm, so about 0.672). The 2^28 two-way tables add -0.008 on top, trained: the same -0.008 that "The table's size and its write rule" measured with fixed weights for a doubling, so the weights do not learn to use the bigger table any better than they already did. Held-out hit rates per order, 2^27 direct-mapped against 2^28 two-way (`results/tables/stats_feat_all_27.json`, `stats_feat_all28.json`): order 8 0.779 to 0.799, 12 0.515 to 0.557, 16 0.340 to 0.385, 32 0.123 to 0.152, 64 0.053 to 0.070, template-16 0.553 to 0.585, template-32 0.265 to 0.303, words-2 0.614 to 0.637, words-3 0.453 to 0.486, lines 0.189 to 0.213; right-given-hit up about one point everywhere; order 4 is saturated at 0.99 either way. The table-side and input-side gains stack: 0.691 to 0.665 to 0.657, -0.034 together, better on 92 % of windows; against the plain transformer's 0.806, -0.149.

Run record for the day, all three lost runs my own doing: the guard yielded GPU 0 to Kyle's game twice (05:31, the first equal-seconds attempt; 08:52, the first 2^28 attempt three minutes in; `results/tables/guard.log`); the first fused equal-steps run died of a bus error at 06:24 because my rebuild of the key tables rewrote feature files it had memory-mapped (`fused_gpu1.log`; the builders now refuse to write a file any process maps); and a watcher that waited on `pgrep -f` matched its own shell, so the 2^28 build never started and GPU 1 sat idle from 07:24 to 08:45.

**Claim:** the 18 inputs are worth -0.021 at equal seconds (0.670, interval ±0.002, better on 82 % of windows); the fused channel is the same model at a cheaper step (calibration -8.6 %); 2^28 two-way tables for all eleven lookup orders add -0.008 trained, the same as with fixed weights, for 0.657 in all. **Not a claim:** any timing from today's hour-runs (contended); that the gains hold at longer budgets or on another corpus; anything about seeds; that bigger tables than 2^28 add more when trained (the fixed-weights curve flattened past 2^28).

## The cost measure changes: calibrated steps (2026-10-07)

Kyle, after the contended day above: "i think we should stop using literal time passage as a metric then. it seems like congestion is gonna prevent good reads." The bar is still bits/byte at the base arm's cost, but cost is no longer an arm's own clock. It is the arm's step time **relative to the base's**, both measured back to back in one quiet calibration session (400 steps each, `--val-fixed 8 --eval-every 100`, nothing else of ours on the box, base first and last; pace = seconds per step over steps 100 to 400, so compile and lazy-init overhead drop out), and an arm's budget is a step count: 14,331 x base pace / arm pace (`checks/calsession.sh`, `checks/budget_ratio.py`). The hour-run's wall clock is logged and never a result. The session ran 12:00 to 12:34 on GPU 1 (`results/tables/calsession.log`, `calsession_2026-10-07.json`); Kyle's game ended partway through and the base paced 0.2286 s/step before it and 0.2292 after (drift +0.3 %), so GPU 1's step time does not see the other card: the morning's 12 % was my table builds and the second arm.

| arm | s/step | cost | steps at the base's cost |
|---|---:|---:|---:|
| base: 8x512, orders 8/16/32 + match | 0.2286 | 1.00 | 14,331 |
| combined: 18 inputs, fused channel | 0.2416 | 1.06 | 13,560 |
| 12x384, combined inputs | 0.2492 | 1.09 | 13,147 |
| 6x768 | 0.3424 | 1.50 | 9,566 |
| 8x512, window 4096 | 0.6110 | 2.67 | 5,361 |
| the fly hybrid (minGRU + heads + local attention) with the combined inputs | 0.4649 | 2.03 | 7,046 |
| GLA with the combined inputs | 0.3612 | 1.58 | 9,068 |
| 8x512, PAQ keys instead of the counters | 0.2497 | 1.09 | 13,116 |
| 8x512, top-two tables (22 inputs) | 0.2520 | 1.10 | 12,997 |
| 4x512, combined inputs | 0.1312 | 0.57 | 24,960 |
| 6x512 | 0.1854 | 0.81 | 17,672 |
| 4x512, 2^28 two-way tables | 0.1281 | 0.56 | 25,562 |

### Backbones at the base's cost, all with the combined inputs (2026-10-07, both cards)

The lock question: is the transformer the right host for the memory? Five backbones, each fed the 18 inputs through the fused channel, each at the step count above (`checks/tier1_common.sh`, `tier1_gpu1.sh`, `tier1_gpu0.sh`; the two cards share one list through claim files, GPU 0 under the guard; the 12x384 arm ran before the session with 12,487 steps from the old seconds budget, 5 % under its calibrated 13,147). The reference is the combined arm at equal seconds (0.670, 13,253 steps; the session would give it 13,560). Dev windows, bits/byte; the interval is the paired 95 % on the window differences.

| arm | parameters | steps | cold | warm | vs the combined arm (0.670) |
|---|---:|---:|---:|---:|---:|
| 8x512 (the combined arm) | 19.93 M | 13,253 | 0.670 | 0.604 | |
| 12x384 | 16.54 M | 12,487 | 0.674 | 0.607 | +0.004 (±0.001), better on 39 % |
| 6x768 | 33.03 M | 9,566 | 0.688 | 0.621 | +0.018, better on 8 % |
| 8x512, window 4096, scored at 2048 | 19.93 M | 5,361 | 0.713 | 0.641 | +0.043, better on 0 % |
| the same, scored at its own 4096 on the same 2 MB | | | 0.669 | 0.632 * | -0.001 against 0.670 regrouped to 4096-byte spans, better on 44 % |
| the fly hybrid | 18.91 M | 7,046 | 0.708 | 0.645 | +0.038 (±0.001), better on 1 % |
| GLA | 16.26 M | 9,068 | 0.779 | 0.716 | +0.108, better on 0 % |

\* warm at 4096 is the second 2,048 bytes of each span, not the same bytes as the other rows' warm halves.

Nothing beats 8x512. Narrower and deeper ties (+0.004 at 5 % fewer steps than its budget; the rerun would close most of that, not flip it); wider and shallower loses 0.018 with 65 % more parameters; a 4096 window costs 2.7x per step and at a third of the steps ties on the same bytes when given its long context and loses 0.043 at 2048. The fly hybrid with the same inputs loses 0.038, better on 1 % of windows: the memory heads, the minGRU and the local-attention layer cost 2.03x per step and do not pay for it at this budget, as in Row 1. GLA (`--arms gla`) loses 0.108, with half the bytes seen; it has had no tuning here and is not a fair measure of GLA, only of GLA as the lab runs it.

### The idea arms: PAQ-style keys, top-two slots, fewer layers (2026-10-07, both cards)

Kyle's ranking of the four ideas of the day: "4 sounds the most interesting. i like 3 too. 1 and 2 are good but not exciting." Idea 4 is PAQ's context set as table keys and a table that keeps two candidates per slot; idea 3 is fewer layers with the table doing more of the work. Arms (`checks/ideas_common.sh`, `ideas_gpu1.sh`, `ideas_gpu0.sh`; tables `checks/paq_keys.py`, `checks/top2_tables.py`):

- **PAQ keys** (`feat_paq`): the eleven lookup orders plus seven keys from the PAQ family, replacing the seven counters: two *indirect* contexts (the byte that followed the previous occurrence of the last 1 or 2 bytes, hashed with those bytes), three *sparse* contexts (bytes -2,-3,-4; -3..-6; -2,-4,-6,-8), two *record* contexts for the line above (the byte above and its neighbours; the column with the byte above and the byte to the left). Each key hits at nearly every position and is right at 17 to 32 % of them.
- **Top-two** (`feat_top2`): the eleven lookup orders from tables whose slot keeps two candidate bytes with counts (the second promoted when its count passes the first's), read as 22 inputs. A second candidate is present at 59 % of order-8 hits and is right at 23 % of the positions where the first is wrong.
- **4 and 6 layers** of 512 with the combined inputs (2^27 tables), and **4 layers with the 2^28 two-way tables**.

| arm | parameters | steps | bytes seen | cold | warm | vs the combined arm (0.670) |
|---|---:|---:|---:|---:|---:|---:|
| 8x512, combined, 2^27 | 19.93 M | 13,253 | 217 M | 0.670 | 0.604 | |
| 8x512, combined, 2^28 two-way, equal steps (above) | 19.93 M | 14,331 | 235 M | 0.657 | 0.592 | -0.013 (cost 1.06, not at budget) |
| PAQ keys | 19.93 M | 13,116 | 215 M | 0.666 | 0.602 | -0.004 (±0.001), better on 57 % |
| top-two | 20.47 M | 12,997 | 213 M | 0.664 | 0.599 | -0.006 (±0.001), better on 63 % |
| 4x512 | 11.52 M | 24,960 | 409 M | 0.659 | 0.593 | -0.012 (±0.001), better on 84 % |
| 6x512 | 15.73 M | 17,672 | 290 M | 0.657 | 0.591 | -0.013 (±0.001), better on 86 % |
| 4x512, 2^28 two-way | 11.52 M | 25,562 | 419 M | **0.652** | **0.587** | -0.018 (±0.001), better on 90 % |

Idea 4 is small: the PAQ keys are worth 0.004 more than the counters they replace, the second candidate 0.006, both under the 0.01 line this note treats as needing a second seed before it is believed. Idea 3 is the result of the day: at the base's cost, 4 or 6 layers beat 8 (-0.012 and -0.013; 4 against 6 is +0.002, a tie), and 4 layers with the 2^28 tables reach **0.652**, the round's best number, with 11.5 M parameters against 19.9 M, 1.4 GiB of VRAM against 2.65, and -0.005 (±0.001, better on 69 %) against the 8-layer 2^28 arm that cost 6 % more than the budget. The table-size gain at 4 layers is -0.007 (4x512 2^27 to 2^28), the same -0.007 to -0.008 as at 8 layers and with fixed weights: three measurements, one number, so the two gains are additive and the weights' depth does not change what the table is worth. Against the plain transformer (0.806) the 4-layer arm is -0.154 at the same cost and 60 % of the parameters.

Why the shallow arm wins is the open question. It sees 419 MB in the hour (2.1 passes over the train split) against 217 MB, so part of it is that the 8-layer model is step-starved at one hour, and that reverses with budget in the usual way; part may be real, the table standing in for layers that only learn n-gram statistics. The two are told apart by the depth curve at twice the cost, queued 2026-10-08 on both cards (`checks/scale_common.sh`, `scale_gpu1.sh`, `scale_gpu0.sh`: 8, 6 and 4 layers, all with the 2^28 tables, at 27,120 / 35,344 / 51,124 steps = twice the session budgets). If 8 layers catch up or pass, depth is a budget question and the big train picks by its budget; if 4 still win, the lock takes the shallow model. (Answered the same day, "The depth curve at twice the cost" below: 8 caught up to 4 and 6 passed both; a budget question.)

**Claim:** at the base's calibrated cost, the 8x512 transformer with the combined inputs beats every other backbone tried (12x384 tie at +0.004, 6x768 +0.018, window 4096 +0.043 at 2048 and -0.001 on the same bytes at 4096, the fly hybrid +0.038, GLA +0.108); fewer layers beat it: 4 or 6 layers -0.012 to -0.013, and 4 layers with the 2^28 two-way tables 0.652 (-0.018, better on 90 % of windows, 11.5 M parameters); the table's size is worth the same -0.007 at every depth. The PAQ keys (-0.004) and the top-two slots (-0.006) are small. **Not a claim:** that fewer layers win at any larger budget (the 2x curve is running); anything about seeds (one each; the two small gains are below the seed line); that GLA or the hybrid cannot be tuned to do better; any timing from the hour-runs (3,299 to 3,848 s logged, cards shared).

### The radical bets (2026-10-08): designs and what was measured before the arms ran

Kyle, shown the backbone and idea results: "2 sounds like a positive result right? i think we need to try more radical ideas", then "let's test all the radical bets next". The reading they start from: the knowledge wants to live in the memory and the network wants to be small. Four bets, each at the base's calibrated cost on GPU 1 (GPU 0 was Kyle's that day), all with the 2^28 two-way tables and the 4-layer arm (0.652) as the reference; the depth curve at twice the cost (8 / 6 / 4 layers, `checks/scale_common.sh`) ran first to tell whether the shallow win is the hour talking. Calibration sessions 2 and 3 (`checks/calsession2.sh`, `calsession3.sh`) budget the arms. Results are in the next subsection as they land.

**A. How small can the network go** (`checks/a_common.sh`, `a_gpu1.sh`): 2, 1 and 0 layers of 512 with the combined inputs (`l2b`, `l1b`, `l0b`; 0 layers = byte embedding + channels, a norm and the output head, a mixer of the table's answers), and 4 layers with 2^29 two-way tables (`l4c`; `checks/assoc_all.py 29`, 197 s, 4.6 GB on disk). The 2^29 tables barely move the held-out hit rates over 2^28 (order 8: 0.799 to 0.800, 16: 0.385 to 0.391, 32: 0.152 to 0.158, lines 0.213 to 0.218; `results/tables/assoc_all29.log`): at 2^28 the tables are no longer the limit, the corpus is.

**B. Learned slots** (`l4n` in the same queue): the round already tested a learned hashed table as row 2c (`--ngram-emb`, 0.019 alone on the backbone, the weakest channel), so B is tested as an addition to the counted table: 4 layers, 2^28 tables, plus `--ngram-emb 4,8,12,16,32 --ngram-rows 2097152 --ngram-dim 32 --ngram-sparse` (335 M sparse parameters updated only where touched).

**C. Compute only where the memory is unsure** (`spike/speed-skip` 3d9687f, `--skip-frac`; `checks/c_common.sh`, `c_gpu1.sh`; `tests/test_skip.py`). The design had to change before it was built. The first idea was to gate by the table's confidence; measured on the 4-layer checkpoint's per-position losses on the dev windows (`checks/skip_ceiling.py`, `results/tables/skip_ceiling_l4b.json`), the 33 % of positions where the table is surest (by P(right | order, class) from train bytes) still hold 18 % of the model's loss, 0.35 bits/byte there, and the table's byte is right at only 84 % of them: a model would lose a lot by trusting the table there. Gating by a model's own predictive entropy is another matter: the 50 % lowest-entropy positions by the 4-layer model's own entropy hold 1.2 % of its loss (0.015 bits/byte there), the 30 % lowest 0.1 %. A cheap model's entropy transfers only partly (`checks/gate_transfer.py`, `gate_transfer_l0.json`): a 0-layer mixer trained 4,000 steps (1.33 bits/byte) puts 1.7 % of the big model's loss in its 30 % lowest-entropy positions and 7.4 % in its 50 % lowest. So C is BLT's construction with the lab's pieces: a 0-layer head over the same inputs predicts every position and its entropy is the gate (threshold = a running EMA of the batch's skip-fraction quantile, frozen at eval; causal, no top-k over the window); the stack runs on the kept positions only, at their original rotary positions, each kept row also receiving the mean input of the run skipped before it through a zero-initialised projection; a skipped position carries the last kept position's output plus a zero-initialised projection of its own input; the gate head has its own next-byte loss. With nothing skipped the model is exactly the plain model (tested). Arms: 4 layers skipping 30 % and 50 % (`c30`, `c50`), and 8 layers skipping 50 % (`c50l8`): the saved compute spent on depth, about a 4-layer step on every position. Expectation written before the runs: small either way; at this scale training is step-bound (4 layers beat 8 at equal cost), so the compute C saves buys 0.01 to 0.02 per doubling of steps, and the carried-state path has to be nearly as good as the stack on the skipped half.

**E. PAQ-style bit tables** (`checks/paq_bits.py`, `paq_bits2.py`, built by a subagent; `results/tables/stats_feat_paqbits.json`, `stats_feat_paqbits2.json`; two builds of 210 and 118 s). Slots hold two counts per (context, bit-prefix) node with PAQ's nonstationary update; a lookup walks the byte's bit tree (a width-4 beam) and hands over the top two bytes with probability bins. The first build, without a slot check, is what PAQ does and it fails here as a feature: each context spreads over eight nodes and nothing detects a foreign slot, so at 2^28 the tables fill to 59 % (order 8) and 99 % (orders 32, 64) and the false hits swamp the signal: order 8 hits 92 % of held-out positions and is right at 48 % (the count table: 80 % and 67 %), order 16 hits 97 % and is right at 19 % (38 % and 79 %). The second build adds a 16-bit check per slot (a foreign or empty slot anywhere on the walk is a miss): false hits drop to zero, the candidates become sharp (order 8 right at 76 %, order 16 at 89 %; bin p >= 0.9 right at 95 to 99 %), and coverage collapses (order 8 hits 35 %, order 16 9 %), because evicting on every foreign tag throws away 7 to 75 % of the writes. PAQ's mechanism needs about eight times the slots of a byte-level table for the same coverage; at 2^28 that is a third of the count table's reach. So E's arm (`e_mt_r`) tests the sharper distribution as an *addition*: the 18 inputs plus the bit tables' two candidates for orders 4, 8, template-16 and words-2 (`results/mem/feat_all28_paq`, 26 inputs).

### The depth curve at twice the cost (2026-10-08, GPU 1): depth is a budget knob

The question left by the idea arms: is the shallow win the hour talking? The three depths, all with the combined inputs and the 2^28 two-way tables, each at exactly twice its calibrated hour budget (`checks/scale_common.sh`, `scale_gpu1.sh`; 27,120 / 35,344 / 51,124 steps; the logged clocks 7,332 to 7,434 s, within 1.4 % of each other, so the calibration held; the 8-layer arm shared the card for a few minutes with a smoke test, which touches the clock and not the steps). Runs `results/l8x2_mt_r.json`, `l6x2_mt_r.json`, `l4x2_mt_r.json`; scores `results/tables/k_l8x2_mt.json`, `k_l6x2_mt.json`, `k_l4x2_mt.json`.

| arm | parameters | steps | bytes seen | cold | warm | vs itself at 1x | vs 4 layers at 2x |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8x512 | 19.93 M | 27,120 | 444 M | 0.624 | 0.560 | -0.033 (±0.001) from 0.657 | +0.0015 (±0.0004), 4 layers better on 57 % |
| 6x512 | 15.73 M | 35,344 | 579 M | **0.619** | **0.555** | -0.038 (±0.001) from 0.657 (that arm had 2^27 tables; -0.031 net of the table's -0.007) | **-0.0038 (±0.0004), better on 61 %** |
| 4x512 | 11.52 M | 51,124 | 838 M | 0.623 | 0.558 | -0.029 (±0.001) from 0.652 | |

Six against eight at 2x: -0.0053 (±0.0004), better on 68 %.

Three things. First, doubling the budget is worth 0.03 at every depth, more than any single architecture decision in the round after the table itself (the table inputs 0.115, the extra keys 0.021, the table's size 0.013, the depth 0.005): the models are nowhere near done learning at an hour, and the big train's budget will matter more than anything still being argued. Second, the ranking moves with the budget, in the usual direction: at the hour, 4 and 6 layers tie and 8 is behind; at two hours, 6 leads, 8 has closed to a tie with 4. The shallow win at 1x was mostly step starvation, and "the network wants to be small" was half right: fewer layers buy more steps, and more steps are worth more than more layers *at these budgets*. Depth is a knob set by the budget, and the curve says 4 for an hour, 6 for two, and 8 or more if the budget keeps growing, because the deeper model's gain per doubling is the larger (0.033 against 0.029). Third, the corpus is the next wall: the 4-layer arm saw its 200 MB train split 4.2 times in two hours, the 8-layer 2.2 times; the deeper model gets more from each byte, and a big train on this corpus will be choosing between repeating the data and going deeper, which also favours depth. The curves on the 128 fixed windows (`curve_fixed`) are still falling at the last evaluation before the cosine tail at every depth.

**Claim:** at twice the hour budget, 6 layers of 512 with the combined inputs and 2^28 two-way tables reach 0.619 cold / 0.555 warm, -0.004 (±0.0004, 61 % of windows) against 4 layers and -0.005 (68 %) against 8; every depth gains 0.029 to 0.033 from the doubled budget; the depth that wins moves from 4 (hour) to 6 (two hours). **Not a claim:** which depth wins at four hours or more (not run; the trend says 6 or 8); anything about seeds (one each; the 6-against-4 gap is above the paired error but under the 0.01 line this note uses for a single seed); that the 2x arms are the best use of two hours (a 2^29 table or a wider model at 2x was not run; the table's size is worth -0.007 at 1x and was not re-measured at 2x).

### Bet A, first pass (2026-10-08, GPU 1): 2 layers lose the hour, the table's size has flattened

Calibration session 2 (`results/tables/calsession2_2026-10-08.json`; base 0.2254 s/step at the start, 0.2202 at the end, drift -2.3 %, under the 3 % line; Kyle's game was on GPU 0 and the box at load 8 to 11 throughout, which the within-session ratio cancels) budgeted the five arms; two ran (`checks/a_gpu1.sh`, runs `results/l2b_mt_r.json`, `l4c_mt_r.json`, scores `k_l2b_mt.json`, `k_l4c_mt.json`).

| arm | parameters | steps | bytes seen | cold | warm | vs 4 layers, 2^28 (0.652) |
|---|---:|---:|---:|---:|---:|---:|
| 2x512, 2^28 two-way | 7.3 M | 43,751 | 717 M | 0.663 | 0.598 | +0.011 (±0.0004), worse on 81 % |
| 4x512, 2^29 two-way | 11.5 M | 26,813 | 439 M | **0.650** | **0.586** | -0.002 (±0.0004), better on 57 % |

Two layers lose the hour (+0.011, and +0.006 against 8 layers), so the hour's depth curve is 2: 0.663, 4: 0.652, 6: about 0.650, 8: 0.657, and the optimum sits at 4 to 6 even at the smallest budget: the network cannot go that small. The 2^29 tables are worth -0.002 at 4 layers, after -0.007 for 2^27 to 2^28: the table's size has flattened, as the held-out hit rates said it would (order 8: 0.799 to 0.800), and the corpus is what the table is short of now, not slots. The table-size steps are 2^26 to 2^27 -0.008 (fixed weights), 2^27 to 2^28 -0.007, 2^28 to 2^29 -0.002.

The other three arms did not run in the first pass, both failures mine: the 1- and 0-layer arms were refused by `budget_ratio.py`'s plausibility cap (60,000 steps; their calibrated budgets are 71,202 and 165,309 steps at paces of 0.0454 and 0.0195 s/step), and the learned-slots arm's calibration died twice because `--ngram-sparse` cannot be compiled (inductor: "Cannot access storage of SparseTensorImpl"; the session's `cal()` kept only the yardstick line, so the error surfaced only in the journal). Second pass queued behind the C/E chain (`checks/a2_gpu1.sh`, unit `tables-a2-gpu1`): the cap is an argument now (200,000 for this queue), the n-gram table updates densely (the dense table compiles; the row is still exempt from weight decay), and `l4n` gets a mini-session of its own (base, arm, base, drift under 3 %) because the quiet session's calibration of it is the one that died.

**Claim:** at the hour, 2 layers of 512 with the combined inputs and 2^28 tables are +0.011 (±0.0004, 81 % of windows) behind 4 layers; 2^29 two-way tables are worth -0.002 (±0.0004) at 4 layers. **Not a claim:** anything about 1 and 0 layers or learned slots on top (second pass running); that 2^29 is worth nothing at a larger budget (not measured); seeds (one each).

### Bets C and E (2026-10-08, GPU 1): skipping the stack costs twice what it saves, and the PAQ bit tables add 0.002

Calibration session 3 (`results/tables/calsession3_2026-10-08.json`; base 0.2240 s/step at the start, 0.2268 at the end, drift +1.2 %, the box at load 2 to 5) budgeted the four arms; three ran (`checks/c_gpu1.sh`; the 4-layer 50 % arm was refused by the same 60,000-step cap that cost bet A two arms, and is not rerun: see below). Runs `results/c30_mt_r.json`, `c50l8_mt_r.json`, `e_mt_r.json`; scores `k_c30_mt.json`, `k_c50l8_mt.json`, `k_e_mt.json`; the split by kept and skipped positions `skip_where_c30.json`, `skip_where_c50l8.json` (`checks/skip_where.py`, the first 256 dev windows).

**The calibration was wrong for the skip arms, and the protocol learned something.** The quiet session gave `c30` 0.0645 s/step (ratio 0.288 to the base, 49,760 steps) and `c50l8` 0.0668 (48,036 steps). In their runs they paced 0.1435 and 0.195 s/step from the first evaluation to the last, 2.2 and 2.9 times the calibration. The reason is in the calibration runs' own curves: their pace rose from 0.049 to 0.096 s/step between steps 200 and 400, and their threshold stood at 2.4 to 3.2 nats at step 400 against 0.17 to 0.51 at the end of training. Early in training the gate's entropies fall fast and the threshold, an EMA with momentum 0.99, lags above them, so far more than the target fraction is skipped and the steps are cheap; the 400-step session measured that regime, not the one the hour is spent in. The rule for arms whose compute depends on their own state (anything with a learned gate): the pace from the calibration must be checked against the pace at the run's first evaluation, and a difference over 10 % voids the budget. Both skip arms therefore got about twice the steps the protocol meant to give them, and the honest comparisons are at equal steps against the plain arms that exist: `c30` (49,760 steps, 0.1435 s/step, 7,058 s) against the plain 4-layer arm at 2x (51,124 steps, 0.145 s/step, 7,394 s), the same real cost; `c50l8` (48,036 steps) against the plain 8-layer at 2x (27,120 steps), which had 0.56 of its steps.

| arm | parameters | steps | bytes seen | cold | warm | against |
|---|---:|---:|---:|---:|---:|---|
| 4x512, skip 30 % (`c30`) | 12.18 M | 49,760 | 815 M | 0.651 | 0.589 | plain 4x512 at 51,124 steps 0.623: **+0.029**; at 25,562 steps (0.652): -0.0004, a tie at twice the steps |
| 8x512, skip 50 % (`c50l8`) | 20.59 M | 48,036 | 787 M | 0.669 | 0.607 | plain 8x512 at 27,120 steps 0.624: **+0.044** with 1.8x the steps |
| 4x512 + PAQ bit tables, 26 inputs (`e`) | 12.61 M | 24,268 | 398 M | 0.650 | 0.585 | 4x512 with 18 inputs at 25,562 steps 0.652: -0.002 (±0.0004), better on 55 % |

Where C loses, on the first 256 dev windows, at the frozen thresholds (the model skips 35 % of dev positions at the 30 % setting and 55 % at the 50 % setting; dev files are not the training distribution): the skipped positions cost the skip model about twice what the full stack pays there, 0.068 against 0.035 bits/byte at 30 % (the share of all loss sitting in skipped positions doubles, 2.0 to 3.8 %) and 0.184 against 0.101 at 50 %. At 30 % that is +0.012 of the +0.027 gap on these windows, the exact number the ceiling measurement predicted before the run (cheap head 0.079 against big 0.038 there), and the other +0.016 is on the kept positions: the stack on a gapped subsequence learns less per step than the stack on every position (0.917 against 0.893 at the same steps), which no ceiling measurement could have shown. At 50 % the whole gap is the skipped positions (+0.083 there, 55 % of positions). And the saving was not there to pay for it: at 4 layers a step with 30 % skipped costs the same as a plain step (0.1435 against 0.145 s; the selection, the gathers, the pooled run and the gate head cost what the saved 30 % of a 4-layer stack cost), and at 8 layers skipping 50 % saves 29 %. C fails on both counts, mechanism and price, and the 4-layer 50 % arm would have landed between the two it is bracketed by.

E is what the hit rates said: the sharper distribution of the bit tables, added for four of the keys, is worth -0.002, the same as a 2^29 table, under the seed line.

**Claim:** at equal steps (and equal clock), a 4-layer model that skips the stack at the 30 % of positions where a 0-layer gate is surest is +0.029 behind the plain 4-layer model (0.651 against 0.623), and an 8-layer model skipping 50 % is +0.044 behind the plain 8-layer model with 0.56 of the steps; the skipped positions cost about twice what the stack pays there, and the kept positions learn less per step; the step saved nothing at 4 layers and 29 % at 8. PAQ bit tables added to the 18 inputs are worth -0.002 (±0.0004). The 400-step calibration misjudges an arm whose compute depends on its own state by 2 to 3x. **Not a claim:** that no gating can work (a gate trained for the skipped positions' loss, a wider carried path, or a top-k selection could do better; BLT's patches are inference-time wins first); anything about the 4-layer 50 % arm (not run); seeds (one each).

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

`scripts/profile_step.py` and the switches in `scripts/run_sparsela.py` on branch `spike/speed-heads` (bbd6c30); reports under `results/profile/` (`pitstop1.log`, `pitstop2.log`, one `.md` per run). The spine recipe at 16M, batch 8 x 2,048, bf16 autocast, 150 W, the card otherwise empty. Each number is the mean of 15 or 20 training steps after 5 warmup steps, so differences under about 3 % are noise. No run here trained long enough to say anything about bits/byte.

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

### Row 1 at equal seconds, compiled: one hour and two hours (2026-10-04/05): does not pass

`night1.sh` on the worktree of `spike/speed-heads` (bbd6c30), GPU 1, bf16 autocast, 150 W, no other compute process on the card (`contended=False` in every run). Each arm: a 400-step calibration, then one run whose step count fills 3,600 training seconds (`budget.py`: pace from steps 100 to 400, drift factor 1.06), then the checkpoint scored in fp32 on the 1,024 dev windows by `cross_eval.py`, which rebuilds the plain module. Cold is the whole 2,048-byte window, warm its second half. Results and logs: `results/speed_<arm>_3600_s0.json`, `speed_score_dev_<arm>.json`, `speed_night1.log`.

| arm | switches besides `--tf32 --fused --compile` | steps | training seconds | s/step | dev cold | dev warm |
|---|---|---:|---:|---:|---:|---:|
| transformer as first run (`tf_c`) | none | 12,668 | 3,678 | 0.290 | 0.831 | 0.750 |
| transformer, the corrected bar (`tf_r`, run the next morning from a9a9209) | `--rope-table` | 15,953 | 3,739 | 0.234 | 0.806 | 0.726 |
| backbone without heads (`zh_c`) | `--scan doubling` | 13,849 | 3,628 | 0.262 | 0.842 | 0.789 |
| hybrid, 2 heads (`hyb2_c`) | `--scan doubling --head-amp --sparse-claws` | 10,028 | 3,511 | 0.350 | 0.850 | 0.780 |
| hybrid, 4 heads (`hyb_c`) | `--scan doubling --head-amp --sparse-claws` | 7,783 | 3,494 | 0.449 | 0.857 | 0.781 |

The step counts exclude compile time: the compiler's on-disk cache was warm from the timing runs (cold, the first step costs 9 s for the transformer, 22 to 41 s for the backbone, 55 to 82 s for the hybrid).

**Two hours.** GPU 0 in its night window, with step counts instead of a clock, because the two cards do not run at the same pace: 16,000 hybrid steps are 7,182 GPU-1 seconds at the one-hour run's 0.449 s/step, and the transformer does 24,735 steps in that time at its 0.290. `speed_tf_c_2h_s0`: 0.767 cold / 0.687 warm. `speed_hyb_c_2h_s0`: 0.787 / 0.713.

**Half precision holds.** The compiled hybrid with heads and scan under bf16 reads 0.787 / 0.713 at 16,000 steps; the spine hybrid (heads in fp32, scan in float64, eager) read 0.785 / 0.711 at the same step count, seed and schedule. At steps 1,000 and 2,000 the compiled runs match the spine runs within 0.007 (hybrid 1.355 against 1.356, backbone 1.365 against 1.366, transformer 1.532 against 1.539 at step 1,000, on the 128 in-run windows).

**Reading.** Against the transformer as first run, a compiled hybrid step costs 1.55 times a transformer step, and the hybrid is 0.026 behind at one hour and 0.020 behind at two. That transformer was on a slow step: with its rotary tables computed once (`tf_r`; see "The transformer's rotary tables" below) a hybrid step costs 1.9 times a transformer step and the hybrid is 0.051 behind at one hour. The two-hour pair was not rerun; at the corrected pace the transformer's 7,182 seconds are about 30,600 steps, not 24,735, so the 0.020 understates that gap as well. The hybrid's lead per step (0.022 at 16,000 steps in the spine table) does not pay for its step. The backbone without heads ends 0.036 behind the corrected bar, on a step that is no longer the cheapest (0.262 s against 0.234). At one hour the heads cost more than they give on the whole window (0.857 and 0.850 against the backbone's 0.842) and help only late in it (0.781 and 0.780 warm against 0.789).

**Claim:** with every switch that leaves the model unchanged, the hybrid does not beat the transformer at equal training seconds at 16M parameters, at one hour or at two. **Not a claim:** anything about 50M parameters or budgets beyond two hours; that these are the last switches (`--detach-state`, 2 heads at two hours, other compile modes are untested at length); the third decimal (one seed). `tf_r` ran 3.9 % over its seconds (the drift factor undershot), worth about 0.004.

### Row 2, tests b and c: cheap memories as inputs, one hour each (2026-10-05, GPU 1): passes

Three input channels on `feat/speed-round` (abc4a00, 91f4d46, a5fabd1, b80fc79; `flybrain/model/inputs.py`, `flybrain/data/memfeat.py`), merged with the speed switches on `spike/speed-int` (782e0bc) and run from pinned worktrees by `night2.sh`, `night4.sh` and `night5.sh` under the protocol of the section above. Each channel adds input embeddings that start at zero, and a bonus on the output logit of the byte the memory names, so the model starts as the plain model.

- **Corpus table** (`--mem-feat DIR --mem-orders 8,16,32`, +407,052 parameters). The one-pass table of test a, as an input: at every position, the byte the table predicts from the 8, 16 and 32 bytes before it, with a confidence class. `build_corpus_memory.py --features` writes the features for all 220,000,000 bytes in 25.4 s (one pass over the training bytes, one CPU core) plus 1.1 s for the held-out bytes. A training position's lookup is made before the 1 MiB chunk that holds it is written, so it never reads its own continuation; a held-out position reads the finished table. The 26.5 s count against the budget: these arms train for 3,574 s. Held out, the table answers at 78 % / 34 % / 12 % of positions (orders 8 / 16 / 32) and its byte is right 66 % / 78 % / 85 % of those times.
- **Match lookup** (`--match-feat 8,16,32`, +403,977 parameters). Computed inside the forward pass from the window's own bytes: at every position, the byte that followed the most recent earlier occurrence of the same 8-, 16- or 32-byte context in this window, with a count class (none, 1, 2 to 3, 4 or more). A hash and a sort; no corpus, no storage, no gradient through it.
- **n-gram embeddings** (`--ngram-emb 2,3,4,6,8`, +71,336,256 parameters). Hashed embedding tables (262,144 rows of 64 per order) of the last 2 to 8 bytes, added to the input: a fixed address with a table learned by backpropagation.

Dev, fp32, the 1,024 windows. The last column is against the corrected bar, the transformer with its rotary tables computed once (`tf_r`, see the next section); every other transformer arm ran without that switch.

| arm | steps | training seconds (+ table build) | dev cold | dev warm | against the corrected bar |
|---|---:|---:|---:|---:|---:|
| transformer, `--rope-table` (`tf_r`), the corrected bar | 15,953 | 3,739 | 0.806 | 0.726 | |
| transformer as first run (`tf_c`) | 12,668 | 3,678 | 0.831 | 0.750 | +0.025 |
| transformer + table | 12,767 | 3,566 + 26.5 | 0.732 | 0.665 | -0.074 |
| transformer + match | 14,320 | 3,603 | 0.787 | 0.702 | -0.019 |
| transformer + match + table | 14,331 | 3,690 + 26.5 | 0.691 | 0.619 | -0.115 |
| backbone (`zh_c`) | 13,849 | 3,628 | 0.842 | 0.789 | +0.036 |
| backbone + table | 14,093 | 3,536 + 26.5 | 0.745 | 0.702 | -0.061 |
| backbone + match | 14,535 | 3,639 | 0.792 | 0.721 | -0.014 |
| backbone + match + table | 13,745 | 3,531 + 26.5 | 0.706 | 0.646 | -0.100 |
| backbone + n-gram embeddings | 13,471 | 3,627 | 0.823 | 0.772 | +0.017 |
| hybrid, 4 heads (`hyb_c`), for reference | 7,783 | 3,494 | 0.857 | 0.781 | +0.051 |

**What each memory is worth, against the same network without it.** The table: -0.099 on the transformer (both runs on the slow step, 12,767 against 12,668 steps) and -0.097 on the backbone. The match lookup: -0.050 on the backbone; on the transformer -0.019 against the corrected bar, from a run with 10 % fewer steps than that bar, so somewhat more at equal steps. Both: -0.136 on the backbone; -0.115 on the transformer, again on 10 % fewer steps. The n-gram tables: -0.019 on the backbone.

**What each costs.** The table and the match lookup cost no measurable step time (calibrated s/step: backbone 0.245 in the evening and 0.235 the next morning, with table 0.238, with match 0.232, with both 0.244). The n-gram tables cost step time (0.251 calibrated, against 0.232 to 0.245 for the other backbone arms), 71M parameters and 1.1 GiB more peak memory (5.85 against 4.78 GiB). The table needs its feature files in training (6 bytes per corpus byte, 1.3 GB) and at inference the table itself (0.5 GiB per order) with three lookups per byte.

**The table reaches before the window.** A training or scoring window's first positions get lookups whose context starts before the window, bytes the model itself cannot see. With those lookups set to a miss (`notes/speed-round/checks/mask_prewindow.py`, the first 128 dev windows), transformer + table goes from 0.7554 to 0.7599 and backbone + table from 0.7672 to 0.7716: 0.0045 of the gain, all of it in positions 0 to 30. The corrected gains are 0.094 and 0.093. With every lookup set to a miss the same two checkpoints read 0.927 and 0.910, about 0.10 and 0.08 worse than the plain models: a model trained with the table leans on it.

**Where the match lookup works.** On the bytes `copyable_mask` marks as continuing a string already seen in the window (about a third), the backbone goes from 0.257 to 0.150 with the lookup, past the transformer's 0.170 and the hybrid's 0.187; the transformer goes from 0.170 to 0.135. On the other bytes the change is small (backbone 1.127 to 1.106, transformer 1.117 to 1.105). The lookup does the job the memory heads were built for, and helps the backbone more than the transformer, whose attention already copies inside the window.

**Reading.** The memories are worth more than the choice of network. With the same memories the transformer and the backbone are 0.005 to 0.015 apart, the transformer ahead each time although its memory arms ran on 10 to 20 % fewer steps than its own best step buys; the table and the match lookup are worth 0.02 to 0.14. Table and match lookup overlap little (0.097 and 0.050 alone, 0.136 together on the backbone): one carries the corpus, the other the window. Every arm with a table or a match lookup beats the corrected bar, by 0.014 (backbone + match) to 0.115 (transformer + match + table). Backbone + match beats the hybrid's hour by 0.065 at 56 % of its step cost. The n-gram tables are the weakest (0.019), the only channel that costs step time and parameters, and the only one that does not reach the bar.

**Claim:** at one hour and 16M parameters, a table read once from the corpus (26.5 CPU-seconds) fed as an input buys 0.09 bits/byte on both networks, and a lookup in the window's own bytes buys 0.05 on the backbone and about 0.02 on the transformer, at no step cost. **Not a claim:** that the gains hold at longer budgets (test a's zero-training gain on the 16,000-step checkpoints was 0.03 to 0.04, so some of the 0.09 is what longer training would have learned anyway); that the backbone is the better network under these memories (the transformer is ahead in every pair, and the transformer with both ran 3 % over its budget); the second decimal of the match gain on the transformer (its match arms ran without `--rope-table`); anything outside this kernel tree, where exact repeats of 8 to 32 bytes are common; that a table-fed model can be prompted (`scripts/sample.py` does not compute the channels yet).

### The transformer's rotary tables: why one arm ran faster, and the corrected bar (2026-10-05)

The transformer with the match lookup got 14,320 steps in its hour where the plain transformer had got 12,668. The first reading, committed here in 7f608b5 and told to Kyle, was that a game on the other card had slowed the plain run. That was wrong. A 400-step pace check of the plain transformer at 07:34, with the other card idle, gave 0.2686 s/step against 0.2681 the evening before. (The backbone's check gave 0.2346 against 0.2452, 4.5 % faster than in the evening; that one is unexplained.)

The cause is in the model code. `rope()` computes the rotary angles and their cosine and sine from scratch, for q and for k, in every layer, and the compiler fuses that into each rotation kernel: 16 forward and 16 backward kernels of about 1.1 ms, 36 ms of a 255 ms step. With the match channel in the graph the compiler emits one forward kernel per layer at 0.5 ms for the same work (12 ms a step); which of its heuristics flips was not looked for. Under the 150 W cap, less work in a step also shortens every other kernel. `notes/speed-round/checks/probe.py` runs the runner's own training step with the variants alternated on GPU 1 inside the same minutes, 40 steps each (the card warms from round to round):

| transformer, compiled | ms/step, rounds 1 to 3 |
|---|---:|
| plain | 249, 261, 268 |
| plain, `--rope-table` | 213, 218, 225 |
| with match lookup | 227, 233, 240 |
| with match lookup, `--rope-table` | 219, 224, 231 |

`--rope-table` (`spike/speed-int`, a9a9209) keeps the two tables as buffers that are computed once and not saved with the weights. Logits and gradients are bit-identical in eager mode (`tests/test_rope_table.py`), and the losses after the timed steps agree to three decimals. It is off by default.

**The corrected bar.** `tf_r` (`night7.sh`, the worktree pinned at a9a9209): the plain transformer with `--rope-table`, calibrated at 0.213 s/step and budgeted like every other arm. 15,953 steps in 3,739 training seconds (0.234 s/step over the run; the drift factor undershot, so it ran 3.9 % over its 3,600 s, worth about 0.004): 0.806 cold / 0.726 warm on the 1,024 dev windows, against `tf_c`'s 0.831 / 0.750. Its step-1,000 score matches `tf_c`'s (1.536 against 1.532).

What follows from it:

- Every "against the bar" number of this round is now read against 0.806, not 0.831.
- The transformer arms with the match lookup had two thirds of the speed-up by accident (14,320 and 14,331 steps); `--rope-table` would give them about 4 % more. Transformer + table ran on the slow step (12,767 steps). None of the three was rerun, so each understates what the transformer does with that memory.
- The backbone's one local-attention layer has the same recomputation (`rope_at`, two rotations a step, about 2 % of its step). Not changed.
- Row 1: the hybrid is 0.051 behind at one hour. A pace check of the hybrid the same morning gave 0.429 s/step against 0.436 the evening before, so its own hour stands.

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

### Row 3: local learning, one hour each (2026-10-05, GPU 1): does not pass

`night3.sh` on the worktree pinned at 613575d (`spike/speed-int`): the protocol and the switches of `hyb_c`, the compiled hybrid with 4 heads, plus one flag per arm. `hyb_c` is the reference; a pace check the same morning put its step within 2 % of the evening's (0.429 against 0.436 s calibrated).

| arm | flag | trained parameters | steps | training seconds | s/step | dev cold | dev warm | against the hybrid |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| the hybrid, fully trained (`hyb_c`) | | 16,066,824 | 7,783 | 3,494 | 0.449 | 0.857 | 0.781 | |
| ii: blocks of two layers, each with its own next-byte loss, no gradient between blocks | `--local-blocks 2` | 16,463,880 | 7,995 | 3,610 | 0.452 | 0.930 | 0.853 | +0.073 |
| iii: every other layer frozen at its random initialisation | `--freeze-layers 1,3,5,7` | 8,156,418 | 8,286 | 3,492 | 0.422 | 0.924 | 0.851 | +0.067 |
| i: the backbone frozen at random, only readout, embedding and memory heads learn | `--train-only readout,embed,heads` | 2,399,496 | 8,522 | 3,536 | 0.415 | 1.736 | 1.608 | +0.879 |

**Reading.** None comes within 0.02. The step is hardly cheaper, 0 to 8 % over the run: the heads' forward and the recurrence dominate it, a frozen layer still passes the gradient to the layers below, and the blocks of ii each run their own backward pass and three extra readouts. So the loss is a loss per step: at step 7,000 on the 128 in-run windows the three read 0.923 (ii), 0.923 (iii) and 1.795 (i) against the hybrid's 0.852. In ii the blocks' own training losses at the end are 1.21, 1.00, 0.98 and 0.98 from the first block to the last: with no error signal from above, the third and fourth blocks add 0.02 to what the second reaches. Arm i is the nearest thing here to the fly's recipe (fixed random wiring, one learned layer of memory) and it is the furthest from a language model.

**Claim:** at one hour and 16M parameters, withholding backpropagation from part of the hybrid's stack costs 0.07 bits/byte (local blocks, alternate frozen layers) to 0.88 (frozen backbone) and saves at most 8 % of the step. **Not a claim:** anything at longer budgets (the frozen arms' published gain is time to convergence, and one hour is far from convergence); the same switches on the transformer or on the backbone without heads, where a layer is a larger share of the step (not run); other local rules (none here uses a non-gradient update).

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
