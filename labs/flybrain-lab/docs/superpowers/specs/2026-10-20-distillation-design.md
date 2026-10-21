# Distillation from Qwen into the reader — design (2026-10-20)

**Status.** Approved by Kyle 2026-10-20 evening ("All that is good. Let's sweep 3 ... Watch disc space"); plan: `docs/superpowers/plans/2026-10-20-distillation.md`.

**Decision this serves.** Kyle, 2026-10-20 evening (ruling R130): nothing in the model #2 menu moves the needle; the memory is the product,
and distillation is the one lever left on the reader itself ("the distillation sounds worth it to me for sure"). This spec is the
distillation half. It is item 7 of the brainstorm ("a big model's predictions computed once as the training target").

**Goal.** Train the in-house reader (model #1's recipe: the 12×512 byte transformer fed by the count tables) on a big local model's
predictions instead of, or beside, the text's own next byte, and measure on the same bar as everything else: bits per byte, cold and
warm, on the mix's dev windows, Kyle's held-out turns and the dialogue. The question the pilot answers: **is a byte of teacher
distribution worth more to the reader than a byte of text, and by how much?** If it is worth several bytes of text, the reader can be
trained on a few tens of megabytes of distilled prose and read Kyle's text better than it does after a gigabyte; if it is worth about
one, the lever is dead and the reader stays a text-trained model.

## What exists

- The reader and its runner: `scripts/run_sparsela.py`; the loss is `F.cross_entropy(logits, y)` over 256 byte classes at every
  position of a 2,048-byte window (line 372); the table inputs come from `flybrain/data/memfeat.py`'s `FeatureStore`, memory-mapped
  per-position arrays (`pred_n*.npy`, `conf_n*.npy`) read by corpus position (`windows(starts, seq_len)`).
- The prose corpus mixB (`/games/ailab/corpora/talkB.bytes`, 1.34 GB: Stack Exchange 1 GB, Wikipedia 250 MB, oasst2, docs) with its
  2^28 byte tables (`talkB-tables/feat_all28`), and the hour reader trained on it (`k_talk_mixB.json`: cold 1.8026 on his turns, 1.7654
  on the dialogue).
- The teacher, Qwen3.8-27B as the yardstick scorer (`qwen_memory.py`, llama-cpp-python, the UD-Q4_K_XL GGUF, `logits_all=True`),
  reading at about **0.9 KB/s on GPU 1 at the 150 W cap** (the memory runs: 5.5 s a window of 4.9 KB). Its vocabulary has 248,320
  tokens of 1 to 32 bytes; the reader predicts bytes.
- Kyle's world (`/games/ailab/corpora/kyle/world/`, 0600): the assistant side of his transcripts and his own notes, leak-checked
  against the held-out sets at 64 bytes; it was trained on by the world fine-tune, never scored. His held-out turns and the dialogue
  are never trained on, never distilled over, never printed.

## The teacher pass (`notes/model2/checks/distill_teacher.py`, new)

Qwen reads a slice of the corpus in 6 KB windows that overlap by 1 KB (the first KB of each window is context only, not emitted, so
every emitted position has at least 1 KB of context). At every token position it keeps the **top K = 512 tokens** of the next-token
distribution by probability (on the host, from the full logits; the dropped mass is recorded). From those the next-**byte**
distribution at every byte position is derived:

- **At a token boundary** (the byte that starts the next token): P(b) = Σ P(t) over kept tokens t whose first byte is b. Exact up to
  the dropped mass.
- **Inside a token** (the text's next token τ = β1..βm, and β1..βj have been seen, 1 ≤ j < m): P(b) ∝ Σ P(t) over kept tokens t with
  t[:j] = β1..βj, len(t) > j and t[j] = b. Tokens that *end* at j are dropped (their next byte would come from a next-token
  distribution the pass does not have); the rest renormalised. This is the usual byte marginalisation of a token model, approximate
  only in that second clause.
- **Fallback**: if no kept token matches the prefix, the position gets the hard target alone (recorded as a count; expected rare,
  since the text's own token is almost always in the top 512).

Stored per byte position of the slice: the **top 8 bytes and their probabilities** (`teacher_ids.npy [N, 8] uint8`, `teacher_p.npy
[N, 8] float16`; 24 bytes a position), plus `teacher.json` (the slice's corpus ranges, the teacher, K, the dropped-mass and fallback
counts). Positions outside the slice hold zeros. 24 bytes a position is 0.75 GB for a 32 MB slice; `/games` has room. The pass is
resumable by window (it is the kind of job the GPU 0 guard interrupts), runs under `CUDA_VISIBLE_DEVICES`, and prints counts only.

**The throughput truth.** At 0.9 KB/s the 27B distils about **3 MB an hour**; ten GPU-hours is 32 MB. The hour reader reads some
250 MB of text in its hour, the 12-layer reader a gigabyte. So the distilled slice is lapped many times over, and the pilot is
designed around that (below). Two ways to widen it, measured in P0 before anything else: a faster teacher (gpt-oss-20b MXFP4, an MoE
with 3.6B active parameters, is on /vault; several times the speed, a weaker and differently-trained teacher), and the 27B through
HF 4-bit (`inspect/Qwen3.8-27B-bnb-4bit`) with the top-K taken on the GPU, which removes the 248k-float host copy per token.
The yardstick model is the default teacher; the MoE is the fallback if the slice would otherwise be too small to matter.

**The slice.** 32 MB for the pilot, chosen to look like Kyle's use rather than the corpus's average: 16 MB of oasst2 dialogue and
Stack Exchange answers from mixB's train region (never its dev windows, never the tail region of `talkB.bytes`, which holds the
held-out text), and 16 MB from his world: every assistant reply (4.3 MB, all there is) and then his own notes, docs and scripts to
the budget (the files the world fine-tune trained on; never his turns). The corpus ranges are fixed in `distill/layout.json` and the
same slice feeds every arm.

## The student (`scripts/run_sparsela.py`, `flybrain/data/memfeat.py`)

- `TeacherStore(dir)` beside `FeatureStore`: the two arrays memory-mapped, `windows(starts, seq_len)` → `(ids, p)` of shape
  `[B, T, 8]`, zeros outside the slice.
- The loss at each position: `(1 − α)·CE(y) + α·soft`, with `soft = −Σ_k p_k · log_softmax(logits)[ids_k]` over the stored 8
  (p renormalised to the stored mass, so a position whose teacher mass is all on the true byte reduces to CE). Where `p` is all zero
  (outside the slice) the soft term is zero and the hard CE carries the position alone. `--teacher DIR --teacher-alpha α`
  (default 0.5); `--sample-from slice|corpus|mixed` decides where windows are drawn (`mixed`: half the batch from the slice).
- Nothing else in the recipe changes: the same tables, the same depth, the same schedule, the calibrated steps of one quiet session.

## The pilot (GPU 1 after the sweep; each arm the hour recipe, 4 layers, at equal steps)

| arm | windows from | loss | what it isolates |
|---|---|---|---|
| a. text, the corpus | mixB train | hard | the baseline (re-run in the session for the clocks; `k_talk_mixB` says where it lands) |
| b. text, the slice | the 32 MB slice, lapped | hard | the slice's text alone: the control for "a small slice lapped" |
| c. teacher, the slice (α = 0.5) | the slice, lapped | soft | **the lever**: b against c is the worth of the teacher's byte over the text's byte |
| d. mixed (the best α of c) | half corpus, half slice | hard + soft | the practical recipe: a against d is the gain the reader would ship with |
| c at α = 0.2, 0.8, 1.0 | the slice, lapped | soft | the weight's direction (Kyle: sweep it); α = 1.0 is the teacher alone |

Scored cold and warm on the mix's dev, his turns and the dialogue. **Verdict rule**: the lever is alive if c beats b by more than
0.03 on his turns or the dialogue (b and c share every byte of text; only the target differs) and d beats a. Then the second step is
the slice at 100 MB and the 12-layer reader; if it is dead, the reader stays text-trained and the memory track has GPU 1.

## Costs and order

- P0, throughput (15 min of GPU 1 or GPU 0 under the guard): each teacher candidate on 1 MB of mixB, KB/s and the dropped mass at
  K = 512. Decides the teacher and the slice size.
- P1, the pass: about 10 GPU-hours for 32 MB with the 27B (2 to 3 with the MoE). Runs under the guard on GPU 0 at night, or on
  GPU 1 after the sweep; resumable either way.
- P2, the code: `distill_teacher.py` (the marginalisation is CPU code, unit-tested against the vocabulary with `vocab_only=True`
  before any GPU time), `TeacherStore`, the loss and the flags, a smoke on the CPU. About a day.
- P3, the seven arms (a, b, c at four weights, d): seven hours of GPU 1, one session. Disk is checked before every step (Kyle: "Watch disc space"); mixB's saved tables and the teacher store go on /vault.

## Not in this spec

Distilling the big model into a mid-size generator ("shrinking the generator", Kyle: later); token-level distillation into the
token reader (the tokeniser differs from the teacher's; byte marginalisation covers both); the memory track (the sweep, the reranker).

## Open for Kyle's review

1. The teacher: the yardstick 27B by default, the MoE only if P0 says the 27B's slice is too small to matter. Agree?
2. The slice's second half from his world's assistant replies (trained on before, by the fine-tune; never his own turns). Agree?
3. α: swept at 0.2, 0.5, 0.8 and 1.0 on the slice arm (Kyle, 2026-10-20: "Let's sweep 3"); the mixed arm takes the best.
