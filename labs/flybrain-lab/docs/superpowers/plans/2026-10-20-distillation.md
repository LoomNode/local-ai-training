# Distillation pilot — implementation plan (2026-10-20)

**Spec:** `docs/superpowers/specs/2026-10-20-distillation-design.md`, approved by Kyle 2026-10-20 evening with two changes: sweep the
teacher weight α (0.2, 0.5, 0.8, 1.0) instead of fixing it, and watch disk space. Executed natively (me), tasks in order; the GPU
tasks queue behind the sweep (unit `talk-qwen-sweep-gpu1`, done about 2026-10-21 evening).

**Goal:** measure whether a byte of Qwen's next-byte distribution is worth more to the reader than a byte of text (arm c against arm b),
and what the mixed recipe gains over the corpus baseline (d against a).

## Global constraints

- Kyle's held-out text is never trained on, never distilled over, never printed: `heldout.bytes`, and the tail region of
  `/games/ailab/corpora/talkB.bytes` from `tail_offset` 1,339,982,312 (its layout.json). The slice is drawn from mixB's train region
  `[0, 1,207,223,865)` and from his world (`corpora/kyle/world/`: the replies, 4.3 MB, and his own files; never his turns); every slice
  byte range is re-checked for 64-byte spans of `heldout.bytes`. Everything derived from his world is mode 0600.
- Disk: /games has 65 GB free; new files over 1 GB go on /vault (268 GB free) with a symlink; `df` is checked before each big write
  and a step stops under 20 GB free on its disk. mixB's key hashes (54 GB, transient) are deleted after the table pass.
- GPU 1 only after the sweep; 150 W; CPU jobs `nice -n 19 taskset -c 0-5,8-13`; the arms at equal steps in one session (the
  calibrated-steps rule); commits as LoomNode, never push; TODO.md lean, DONE.md gets the result.

## Tasks

### T1. mixB's saved tables (CPU, ~1 h; `/vault/ailab/flybrain-mem/talkB/t28`, 16 GB)
The slice's table inputs come from the frozen tables through `use_features.text_features`; mixB kept only its features. Steps:
`env CORPUS=talkB.bytes TRAIN_END=1207223865 MEM=talkB-tables OUT=/vault/.../talkB/final/feat_keys JOBS=4 key_tables.py --keys-only`;
`env TABLES_ONLY=1 KEYS=... COUNTERS=talkB-tables/feat_counters assoc_all.py 28 /vault/.../talkB/t28`; delete the keys.
Test: `use_features.py --check /vault/.../talkB/t28 talkB-tables/feat_all28` (the dev offset, 1,024 windows): 0 differences.

### T2. The slice (`notes/model2/checks/distill_slice.py`; `/games/ailab/corpora/distill.bytes`, 32 MB, 0600; `distill/layout.json`)
16 MB from mixB's train region (oasst2 documents first, then Stack Exchange, by the splits manifest, from the start of the train region),
16 MB from the world (the 516 reply files, 4.3 MB, then his files in sorted order to the budget). Each document is leak-checked at 64 bytes
against `heldout.bytes`; counts only are printed. layout.json: the ranges `[start, end, source, origin]`, sha256 of the slice.
Then the features: `distill_features.py` (or a `--features` mode): `Tables(t28)` + `text_features(slice, ALL18)` → `distill-tables/feat_all28/pred_n*.npy, conf_n*.npy` (uint8 [S]) + `features.json` in FeatureStore's format (corpus, corpus_bytes = S, train_end = S, orders, chunk, alignment, table_digest). Rows: `text_features` returns S + 1 rows (row p = the inputs for byte p); the first S are stored.
Test: `FeatureStore(distill-tables/feat_all28, ALL18).check_corpus(distill.bytes)` passes; `fixed(0, 4, 2048)` shapes `[4, 2048, 18]`.

### T3. The byte marginalisation (`notes/model2/checks/distill_bytes.py`, CPU)
`class ByteVocab(llm)`: `tok = np.zeros([V, 32], uint8)`, `lens [V]` from `detokenize([i])`. `kept(ids, probs)` → a view `[K, 32]`.
`next_bytes(tok_k, len_k, p_k, prefix) -> (ids8, p8, mass)`: `m = (len_k > j) & (tok_k[:, :j] == prefix).all(1)` with `j = len(prefix)`;
`w = np.bincount(tok_k[m, j], weights=p_k[m], minlength=256)`; `mass = w.sum()`; top 8 of `w` (ids uint8, p float16 normalised to
`mass`, i.e. summing to 1); `mass = 0` means no kept token matched (the caller writes the hard target: ids = [true byte], p = [1]).
Tests (`pytest`, a toy vocabulary of 6 tokens): the boundary case sums first bytes exactly; the prefix case drops tokens ending at j
and renormalises; the fallback; and on the real vocabulary (`vocab_only=True`) a synthetic distribution over 512 random tokens gives
p8 summing to 1 and ids distinct.

### T4. The teacher pass (`notes/model2/checks/distill_teacher.py`, GPU; `/vault/ailab/flybrain-mem/distill-teacher/`, 0.8 GB, symlinked from `/games/ailab/corpora/distill-teacher`)
`--slice distill.bytes --out DIR --model GGUF [--n-ctx 6144 --window 6144 --overlap 1024 --top-k 512 --bench MB --limit N --resume]`.
Per window `[s, s + W)`: tokens with byte offsets (cumulative `len(detokenize([t]))`, asserted to sum to the window's length);
`llm.eval(toks)`; `scores [n_tok, V]` → `torch.topk(512)` and the row logsumexp on the CPU (float32) → kept probabilities and the
dropped mass per row; for each token position i (predicting token i + 1 of bytes β1..βm) and each j in 0..m−1, position
`s + offset(i + 1) + j` gets `next_bytes(kept_i, prefix = β1..βj)`; the first `overlap` bytes of a window (context) are not emitted.
Writes `teacher_ids.npy [S, 8] uint8`, `teacher_p.npy [S, 8] float16` as memmaps, `progress.json` (windows done; resume), `teacher.json`
(the teacher, K, mean dropped mass, fallback count, KB/s). `--bench 1` prints KB/s and the dropped mass and writes nothing.
Test on the CPU before GPU time: `--limit 2 --model <GGUF> --cpu` on 2 windows with the vocabulary (slow but complete): the stored
rows of a window re-read agree with a direct recomputation; the hard-target byte is in ids8 for > 99 % of positions.

### T5. The runner (`scripts/run_sparsela.py`, `flybrain/data/memfeat.py`, `flybrain/data/text.py`)
- `TeacherStore(dir)`: the two memmaps; `windows(starts, seq_len) -> (ids [B, T, 8] uint8, p [B, T, 8] float16)` at positions
  `starts + t + 1` (the alignment of `FeatureStore.windows`).
- `StackedFeatureStore([(dir, lo, hi), ...])`: position `p` in segment k reads `dir_k` at `p − lo_k + base_k`; `size = Σ (hi − lo)`;
  `train_end = size`; `windows()` splits each window at segment boundaries (a window never crosses one: the sampler avoids it).
- `batch_starts_ranges(ranges, weights, seq_len, batch, seed)`: per window, pick a range by weight, then a uniform start inside it
  with `seq_len + 1` to spare; the same generator and seed convention as `batch_starts`. Flags: `--teacher DIR`, `--teacher-alpha α`,
  `--sample-ranges "lo:hi:w,lo:hi:w"`, `--mem-feat-stack "DIR:lo:hi,DIR:lo:hi"` (replaces `--mem-feat` for the mixed arm; the
  corpus file is the concatenation).
- The loss: `ce` per position; `soft = −(p · log_softmax(logits).gather(ids)).sum(−1)`; where `mass == 0` (no teacher row) `soft = ce`;
  `loss = ((1 − α) · ce + α · soft).mean()`. Recorded in the run's json: `teacher: {dir, alpha, rows}`.
- Test: a CPU smoke (`--device cpu --n-layers 1 --d-model 64 --steps 3 --batch 2` on `distill.bytes` with `--teacher`, then with the
  stacked store on a 2-segment toy): runs, the loss is finite, `teacher` appears in the json; α = 0 reproduces the plain loss to 1e-6.

### T6. The pilot chain (`notes/model2/checks/talk/talk_distill_gpu.sh`; results in `results/tables/distill/`)
Waits for `results/tables/qwen-sweep/sweep.done`. P0: `distill_teacher.py --bench 1` for the 27B UD-Q4_K_XL and gpt-oss-20b
MXFP4 (KB/s, dropped mass) → `bench.txt`. P1: the pass with the chosen teacher (the 27B unless its KB/s puts 32 MB past 14 h).
P3, the hour recipe (`talk_hour_gpu.sh`'s COMMON + TF + the byte flags, 25,562 steps, one session):
`a_text` (talkB, train region), `b_slice` (distill.bytes, hard), `c_a02 c_a05 c_a08 c_a10` (distill.bytes, α), `d_mix` (talkD = talkB[:train_end] + slice; ranges 0.5 / 0.5; the best α of c).
Scoring as the hour's: the mix's dev (talkB's dev offset), his turns and the dialogue (the tail windows) with `score_tables.py`;
`distill.txt` holds the lines; `df` checked before each arm. Then the table (cold / warm, every arm, paired per window against a).

### T7. The write-up
MODEL2.md "Distillation: the pilot"; DONE.md; TODO.md; memory; the verdict by the spec's rule (c − b < −0.03 on his turns or the
dialogue, and d < a).

## Review focus (the five ways this breaks a person using it)
1. A slice byte range inside the tail region or a leaking document: the slice builder's 64-byte check and the range assertion.
2. Token byte offsets not summing to the window (a multi-byte character split at a window edge): windows cut at the slice's document
   boundaries where possible, and the assertion fails loud, never silently shifting positions.
3. The teacher rows misaligned by one against the features (`p` vs `p + 1`): the T4 test re-reads rows and the T5 α = 0 identity.
4. A mixed-arm window crossing the talkB / slice boundary: the range sampler keeps `seq_len + 1` inside one range.
5. Disk: the key hashes (54 GB) left behind, or the pass writing to /games: the T1 delete step and the symlink are checked by `ls`.
