# Arms B, C and D: a small student, a decoder on the table, an Engram module (2026-11-06)

The question from the report (`reports/Memory architectures for a local model.md`): can a Qwen3.5 student that shares the 27B's vocabulary carry Kyle's material in its weights, and is the 27B a better teacher for that than his own text? And two memory arms that bolt onto an existing model: a small decoder trained to imitate the n-gram table and mixed at the output (Arm C, the Memory Decoder recipe), and a hashed 2/3-gram embedding module added after layer 2 (Arm D, the Engram recipe).

Everything here is scored on the honest split from the Arm A correction: train and build on files and his sessions dated before 2026-09-01, score the 150 held-out windows of his own turns and the 250 dialogue windows dated on or after it. The bar is the 27B's own numbers on those windows: **1.0732** on his turns with two retrieved chunks, **1.0001** with file headers as of the date, and 0.8326 / 0.8337 on the dialogue. Bits per byte, lower is better; "all windows" numbers are reported in `results/tables/armB/post_cutoff_bpb.txt` but are contaminated for every trained arm (its training set holds the pre-cutoff sessions).

## What ran

- **Training set** (`checks/armB/make_train.py`): his files and his pre-cutoff sessions cut into 6,000-byte segments, each prefixed with the retriever's memory (two chunks with headers, as of the segment's date, own file excluded), plus 30 % replay (FineWeb-edu and CPython); 6,171 sequences, 10.8 M loss tokens (his files 6.9 M, sessions 0.6 M, replay 3.3 M). Loss only on the segment, never on the memory prefix.
- **Teacher pass** (`checks/armB/teacher_pass.py`): the Q4 27B on every sequence, K = 16 i.i.d. samples per loss position from its full distribution (sampled-target distillation, unbiased), 10,780,961 positions, 4.2 h on one 3090. The teacher's own nats per token on the material: his files 1.47, his sessions 2.08, replay 1.96.
- **Students** (`checks/armB/train_lora.py`): LoRA r = 64 on all linear modules, embeddings frozen, one epoch (297 steps of 32,768 loss tokens), lr 1e-4 cosine, grad checkpointing. Loss = α · KD + (1 − α) · CE where KD = −mean_j log q(v_j) over the 16 sampled teacher tokens and CE is the true next token. 2B epoch ≈ 80 min at ~2,050 loss tok/s; 4B ≈ 3 h at ~950.
- **Scoring** (`checks/armB/student_score.py`): the student with the merged adapter, BOS + memory + first half, scored on the second half, three conditions: cold, two chunks (`mem`), headers + as-of (`hdrasof`); then the split table mixed on top (`armA/mix.py --since`), as for the 27B.
- **Arm C** (`checks/armC/`): per-position targets from the table (longest order with count ≥ 1, own document and later dates excluded, top 16 renormalised; 6.4 M of his 7.5 M positions have a target, the true token holds 0.32 of the table's mass on average). The 0.8B student trains on 0.5 · KL(table) + 0.5 · CE on his material only; at score time its per-token distribution is mixed into the 27B's by a gate λ per decoder-entropy bucket, fitted on the other set (`checks/armC/mix_decoder.py`).
- **Arm D** (`checks/armD/engram.py`): hashed 2- and 3-grams, 4 heads per order, 500,009 rows × 64 dims, zero-initialised up-projection, depthwise causal conv, sigmoid gate on the hidden state, added after decoder layer 2 of the 2B and trained with the adapter at lr 1e-3, Arm B's data and the winning α.

## The numbers (post-cutoff windows; his turns / the dialogue)

| model | cold | two chunks | headers + as-of | + split table (two chunks) |
|---|---|---|---|---|
| 27B Q4 (the bar) | — | **1.0732** / 0.8326 | **1.0001** / 0.8337 | 1.0504 |
| 2B untrained | 1.2598 / 1.1628 | 1.2488 / 1.1570 | 1.2508 / 1.1621 | |
| 4B untrained | 1.2284 / 1.1024 | 1.1858 / 1.0702 | 1.1790 / 1.0653 | |
| **2B, plain CE (α = 0)** | 1.1832 / 1.1104 | **1.1763** / 1.1024 | 1.1767 / 1.1053 | 1.1700 (−0.006) |
| 2B, half KD (α = 0.5) | 1.1917 / 1.1037 | 1.1833 / 1.0957 | 1.1841 / 1.0985 | |
| 2B, pure KD (α = 1) | 1.2377 / 1.1275 | 1.2206 / 1.1120 | 1.2216 / 1.1147 | 1.2035 (−0.017) |
| 2B + Engram, α = 0 (Arm D) | 1.1817 / 1.1102 | 1.1746 / 1.1018 | 1.1753 / 1.1051 | 1.1687 (−0.006) |
| 4B, α = 0 | killed at step 215/297 (Kyle 2026-11-06 10:05: "just free the gpus"); no checkpoint, no score | | | |
| 0.8B decoder alone (Arm C) | 1.3910 / 1.3286 | 1.4154 / 1.3535 | 1.4185 / 1.3614 | |

Arm C, the decoder mixed into the 27B (`results/tables/armC/`): two chunks **1.0732 → 1.0402** on his turns (gate fitted on the dialogue; five-fold within 1.0259; the raw table gave 1.0504 on the same windows), headers + as-of 1.0001 → 0.9899 (the table: 0.9885); the dialogue 0.8326 → 0.8344, nothing. Mixed into the 2B students instead of the 27B: −0.001 to −0.009. The fitted gate puts λ ≈ 0.8 on the lowest-entropy bucket (decoder entropy < 0.5 nats) and ≈ 0.02 everywhere else.

## What it means

- **The teacher hurts.** More of the 27B's distribution in the loss gives a worse student, monotonically: 1.176 (none) → 1.183 (half) → 1.221 (all) on his turns with memory, and the same order on the dialogue. The 27B is uncertain on his material (1.47 nats on files, 2.08 on sessions): matching its spread spends the 2B's capacity on the teacher's doubt instead of on his text. The distillation pilot of October saw the opposite at slice scale; the difference is the data: here the student gets the real text, and the real text is the better target. The literature agrees for large teacher gaps and high-entropy targets (the continual-learning note in `reports/Next levers for a local assistant.md`).
- **One epoch of his own text is worth as much as retrieval is to the 27B.** The 2B gains 0.077 cold and 0.073 with memory from a single epoch, the same size as the 27B's gain from two retrieved chunks (1.169 → 1.073 on all windows). But the gains do not stack much: the trained 2B barely uses its memory prefix (1.183 → 1.176), and the table on top gives it 0.006 against the 27B's 0.023, because the weights already hold the verbatim material the table and the chunks would supply. The pure-KD student, which absorbed less of the text, gets three times as much from the table.
- **The gap to the 27B stays.** 0.10 bits per byte on his turns and 0.27 on the dialogue after training, and the dialogue gap is the one that matters for an assistant: the student learned him, not the assistant's craft. A 2B that knows him is about as good as a 27B that does not (1.176 vs the 27B's cold 1.169 on all windows), which is the honest summary of what a personal fine-tune buys.
- **The Memory Decoder is the best memory arm on the plain condition.** A 0.8B trained to imitate the table beats the raw table (0.033 vs 0.023 on his turns), because its gate is cleaner: it fires almost only where the decoder is near-certain. With headers and as-of in the prompt the two are equal (0.010 vs 0.012), and on the dialogue neither does anything. It costs a 0.8B forward pass per token alongside the 27B; the table costs a hash lookup.
- **The Engram module adds nothing** on top of the trained weights (−0.002, within noise). The hashed n-gram embedding and the LoRA learn the same thing from the same text; the module would need to be the only place the text lives to earn its rows.

## Open

- The 4B at α = 0 was killed at step 215 of 297 to free the GPUs for the app work (no mid-run checkpoint); whether size closes any of the 0.10 / 0.27 gap stays open. Its dev CE at step 50 was 1.315 against the 2B's 1.413 at the same step, so it was ahead.
- A learning curve by week (table or weights built from everything up to week k, scored on week k + 1) would say whether use keeps paying; the time split makes it cheap.
- The decoder's gain is on the plain condition; the product runs with headers and as-of, where it ties the table. Deployable only with a logit hook on the server.

Runs and scores: `/vault/ailab/flybrain-mem/armB/{data,teacher,runs,scores}/`, `armC/{targets,mix}/`; summaries copied to `results/tables/armB/`, `armC/`. Chains and markers in the session scratchpad (`chain_gpu1e.sh`, `chain_gpu0b.sh`); post-cutoff summariser `post.py`.
