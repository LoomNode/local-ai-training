# Model #2 on tokens: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (native; the tasks share data formats and the session's context). Steps use checkbox (`- [ ]`) syntax.

**Goal:** the first number of model #2: an hour arm of the reader on 32k tokens against the same arm on bytes, on the package corpus's first GB, scored in bits/byte on the 613 clean dev windows.

**Architecture:** a token stream beside the byte corpus (ids, byte lengths, byte offsets); the table builders generalised to token positions (n-gram orders on ids, keys and counters read at token starts); the reader with a configurable vocabulary, a shared-embedding channel for the count orders and a value channel for the counters; a scorer that maps the byte windows to token spans and divides bits by bytes.

**Tech stack:** Python, numpy, HF `tokenizers` (in the lab venv), torch; the lab's `flybrain` package in a new worktree `.worktrees/tok` (branch `spike/tokens` from `c5825fc`, the big train's code); the checks in the speed worktree under `notes/model2/checks/`.

**Spec:** `docs/superpowers/specs/2026-10-16-model2-tokens-design.md` (speed worktree, commit 2d021b7).

## Global constraints

- The lab interpreter is `/games/ailab/lab/flybrain-lab/.venv/bin/python`; installs via `uv pip install --python <that>`; tests with `.venv/bin/python -m pytest`.
- CPU jobs run `nice -n 19 taskset -c 0-5,8-13`, `CUDA_VISIBLE_DEVICES=""`, and as transient user units (`systemd-run --user --unit <name> -p MemoryMax=...`) with a log under `results/tables/tok/`.
- Never rewrite a `.npy` a running process has mapped (`table_common.refuse_if_in_use`); the cooldown jobs map the byte features until this evening.
- GPU 1 is busy until ~17:00 (the final cooldown), GPU 0 until ~10:00 (the snapshot) and then only under the guard through `gpu0_slot.sh`.
- Corpus files: `/games/ailab/corpora/pkgs.bytes` (5,478,327,694 bytes; `pkgs/layout.json`: train_end = dev_offset = 5,283,527,978, dev_bytes 81,607,390). Token files go to `/games/ailab/corpora/pkgs-tok/` (on `/games`, 131 GB free; the full-corpus token features too).
- Commit as LoomNode with the trailer; never push; feature dirs and corpora are not committed.

## Review focus

1. A token's feature must describe the state before the token (prefix-honest): a feature at token i that depends on bytes inside token i is a leak (R107). Pinned by the prefix recompute test in Task 3.
2. The byte window to token span map must not drop or double-count bytes: the sum of `tok_len` over the span equals the window's bytes to within one token. Pinned in Task 2.
3. `train_end_tok` must fall at a file boundary or after the last train byte, never inside the dev region. Pinned in Task 1.
4. Invalid UTF-8 bytes must round-trip (the stream decodes back to the corpus exactly). Pinned in Task 1.
5. The value channel (counters) must never add a logit bonus on token ids 0..127. Pinned in Task 4.

---

## Task 1: the token stream (`notes/model2/checks/tokenize_corpus.py`, test `tests/test_tokens.py::test_stream_roundtrip`)

Files: create `notes/model2/checks/tok_common.py` (shared helpers), `notes/model2/checks/tokenize_corpus.py`; test in the speed worktree `tests/test_tokens.py`.

Interfaces produced:
- `tok_common.byte_token_ids(tok) -> np.ndarray[256] int32`: the id of each single-byte token (GPT-2 byte-to-unicode map; checked against `pre_tokenizers.ByteLevel.alphabet()`).
- `tok_common.id_byte_len(tok) -> np.ndarray[V] uint8`: each id's byte length (decode the token string through the inverse map; a merged token's length is the sum of its bytes; every length is 1..255, asserted).
- `tok_common.encode_file(tok, data: bytes, byte_ids) -> np.ndarray uint16`: decode with `errors="surrogateescape"`, split the str on lone surrogates (`[\udc80-\udcff]`), encode the valid segments (`tok.encode(seg, add_special_tokens=False).ids`) and emit the byte id for each invalid byte, in order. Round trip: `b"".join(token bytes) == data`.
- `tok_common.file_units(corpus) -> list[(lo, hi)]`: `[0, cut_1), [cut_1, cut_2), ..., [cut_k, N)` from `table_common.file_cuts` (a unit starts at a header's "/").
- The stream files in `--out DIR`: `tokens.u16` (little-endian uint16, one id per token), `tok_len.u8`, `tok_off.u64` (first byte offset of each token, plus one sentinel = N), `layout.json` = the byte layout's fields plus `n_tokens`, `train_end_tok` (the index of the first token whose offset >= train_end; asserted equal to a file unit's first token), `dev_tok` [lo, hi), `test_tok` [lo, hi), `tokenizer` (path, sha256 of the json), `vocab`.
- `tok_common.TokenStream(dir)`: memmaps the three arrays; `.span(a, b) -> (t0, t1)` the tokens whose first byte lies in [a, b) (`searchsorted(tok_off, a, "left")`, `searchsorted(tok_off, b, "left")`); `.bytes_in(t0, t1) = int(tok_len[t0:t1].sum())`.

Steps:
- [ ] Write `tests/test_tokens.py::test_stream_roundtrip`: build a 300 KB synthetic corpus in `tmp_path` with three files (headers `\n\n// ==== a.c ====\n`), text with tabs, CRLF, a run of 0xFF bytes and a truncated multibyte sequence; train a 1k tokenizer on it with `tokenizer_train.py --max-bytes`; run `tokenize_corpus.py`; assert the decoded stream equals the corpus byte for byte, `tok_off[-1] == N`, `tok_len.sum() == N`, `train_end_tok` is the first token of the dev file, and no token spans a file boundary (`tok_off` contains every unit start).
- [ ] Run it: FAIL (no module).
- [ ] Implement `tok_common.py` and `tokenize_corpus.py` (args `--corpus`, `--layout`, `--tokenizer`, `--out`, `--jobs 8`, `--chunk 64MB`; a fork pool over chunks of whole files, each worker encodes its files with `encode_batch` and writes ids to `tokens.partial.u16` at a precomputed token offset — two passes: pass 1 counts tokens per chunk (returns counts), pass 2 writes; or one pass writing per-chunk `.u16` parts concatenated at the end; choose the parts approach: simpler and exact).
- [ ] Run the test: PASS. Commit.
- [ ] Run on the corpus as unit `tok-stream` (log `results/tables/tok/tokenize_corpus.log`), ~15 min; then `tok_common.py --check`: decode 50 random file units and compare to the corpus; print n_tokens, bytes/token overall, train/dev/test token counts.

## Task 2: the dev windows on tokens (`tok_common.dev_spans`, test `test_dev_spans`)

- [ ] Test: for the synthetic corpus, byte windows of 64 from the dev offset: `dev_spans(stream, offset, T, idx)` returns (t0, t1) per window with `bytes_in(t0, t1)` within one token length of 64 and consecutive windows' spans adjacent.
- [ ] Implement: `dev_spans(stream, offset, T, indices) -> list[(t0, t1)]` = `[stream.span(offset + i*T, offset + (i+1)*T) for i in indices]`.
- [ ] On the real stream: the 613 clean dev windows; print the total bytes (expect 1,255,424 ± 613 tokens' worth) and the token count. Commit.

## Task 3: the tables on tokens (`notes/model2/checks/tok_tables.py`, test `test_tok_tables_prefix_honest`)

Interfaces produced: a feature dir `<out>/` with `features.json` {kind: "features", corpus (the byte corpus), stream (dir), n_tokens, train_end_tok, orders: [...], value_orders: [1001..1007], log2_slots, two_way: true, slot_dtype, stats}, and per order `pred_n<o>.npy` (uint16, (n_tokens,)) and `conf_n<o>.npy` (uint8). Alignment: feature[i] is the prediction for token i from tokens before it (count orders) or the state at token i's first byte (counters, keys); i < train_end_tok: lookup before its chunk's write; i >= train_end_tok: the frozen table.

Orders: n-gram 1, 2, 3, 4, 6, 8, 12, 16 (ids; `context_hashes(tokens_u64, lo, hi, orders)`); keys 116, 132, 202, 203, 301 (hash arrays from `key_tables.compute_keys` on the chunk's bytes with the halo, read at token starts: `keys[kid][tok_off[i] - ctx]`); counters 1001–1007 from `counter_tables.counters` likewise (`values[kid][tok_off[i] - lo]`, clipped to 0..127, stored in pred, conf = 1; the string/comment state carried across chunks as `counter_tables.main` does: chunks are whole files; a chunk entered in a non-code state is recomputed in order).

Slot: `SLOT_TOK = np.dtype([("tag", "<u2"), ("tok", "<u2"), ("conf", "u1"), ("pad", "u1")])`; `write`/`lookup` as `assoc_all.py` with `tok` in place of `byte`.

- [ ] Test: a synthetic stream (Task 1's corpus); build at log2 = 12 with CHUNK = 500 tokens; (a) leak-free: rebuild with train_end_tok halved; features at positions below the smaller train_end agree exactly between the two builds; (b) prefix-honest: for 20 random tokens i, recompute every order's input from tokens[:i] / bytes[:tok_off[i]] alone (keys and counters from `compute_keys` / `counters` on the prefix bytes; n-grams by `hash_context` on the id prefix) and look them up in the final tables: pred/conf equal the stored features for i >= train_end_tok; (c) counters stored as values: `pred_n1001` = column at the token's first byte.
- [ ] Implement `tok_tables.py` (env/args: STREAM, CORPUS, OUT, LOG2, CHUNK (tokens, 2^18), JOBS; the key/counter pass chunked by whole files with workers writing per-token key arrays `key_n<kid>.npy` (uint64, n_tokens) and counter arrays (uint8) first — 8 B × 5 keys × 1.7 G = 68 GB for the full corpus: too much; instead compute keys per chunk inside the table pass, single process, chunk = a group of whole files of ~2^18 tokens (~800 KB of bytes; `compute_keys` on 800 KB + halo takes ~0.3 s; 6,500 chunks for 5.3 GB → 35 min: acceptable, no key files at all).
- [ ] Test PASS. Commit.
- [ ] Build the hour-arm corpus pair: `corpora/pkgs1g.bytes` = `pkgs.bytes[:1e9] + pkgs.bytes[dev_offset:dev_offset + dev_bytes]` with its `pkgs1g/layout.json` (train_end 1e9, dev_offset 1e9), its token stream (Task 1 on it), its byte features `/games/ailab/corpora/pkgs1g-tables/feat_all28` via `pkgs_tables.sh` (env CORPUS/MEM/LAYOUT), and its token features `/games/ailab/corpora/pkgs1g-tok/feat28` via `tok_tables.py` at 2^28. Units `pkgs1g-bytes` and `pkgs1g-tok`, CPU, ~1 h each.

## Task 4: the reader on tokens (worktree `.worktrees/tok`, branch `spike/tokens`)

Files: modify `flybrain/data/text.py` (`load_tokens`), `flybrain/data/memfeat.py` (uint16 pred, `n_tokens`, value orders), `flybrain/model/inputs.py` (`SharedFeatureChannel`), `flybrain/model/hybrid.py` (vocab from args; the value channel; `--vocab`), `scripts/run_sparsela.py` (`--vocab`, `--tokens`, the 256s, bits/byte of the fixed windows via `--tok-len`), `scripts/cross_eval.py::rebuild` (vocab, channels from the recorded args). Tests in that worktree: `tests/test_tokens_reader.py`.

Interfaces:
- `load_tokens(path, max_tokens=None, offset=0) -> Tensor uint16` (np.fromfile, like `load_bytes`).
- `FeatureStore(dir, orders)`: pred arrays may be uint8 or uint16; `size` = `n_tokens` if present else `corpus_bytes`; `check_corpus(path)` accepts a file of `size` bytes or `2 * size` bytes (the u16 stream); `.value_orders` (set) from `features.json`.
- `SharedFeatureChannel(d_model, d_out, n_orders, n_classes, embed: nn.Embedding)`: `embed(pred, cls)` = `embed(pred) * gate[o] * (cls > 0) + class_emb[o][cls]` summed over orders (gate: a learned scalar per order, init 0; class_emb zeros); `add_bonus` as `FusedFeatureChannel`'s (per-order, per-class bonus on the predicted token's logit).
- `GRUHybrid(..., vocab_size=V, mem_orders=..., mem_value_orders=(1001,...), shared_table=True)`: `self.table` = Shared channel over the count orders, `self.values` = `FusedFeatureChannel(d_model, out_dim, n_value_orders, 2, vocab=128)` with its bonus never applied; forward splits `mem_pred`/`mem_conf` columns by a registered index buffer.
- `run_sparsela.py`: `--vocab V` (default 256), `--tokens` (the corpus file is a uint16 stream; `--train-bytes` and `--val-offset` are then token counts, named as they are), `--tok-len FILE` (with `--val-fixed`: the fixed rows also carry `bpb_bytes` = bits over the windows' byte lengths from `tok_len`; the curve's `bpb` field becomes bits per token in token mode, with `unit: "token"` recorded), `--mem-value-orders` read from the store by default.

- [ ] Tests: (1) `SharedFeatureChannel` with gate = 1 and a one-hot embedding reproduces `FeatureChannel.embed` on bytes; zero-init leaves the model at the plain model; (2) a `GRUHybrid` with vocab 1000, 3 count orders and 2 value orders runs forward on random ids, logits shape [B, T, 1000], and the value channel adds no bonus (logits with value conf = 0 vs 1 differ only through the embedding path: check by zeroing the value embeddings); (3) `train_loss` with vocab 1000 (no hard-coded 256); (4) `FeatureStore` on a uint16 feature dir.
- [ ] Implement; tests PASS; `pytest tests/ -x -q` in the tok worktree stays green (the byte path untouched: `--vocab 256` default, `shared_table` only in token mode).
- [ ] Commit on `spike/tokens`.

## Task 5: the scorer on tokens (`notes/model2/checks/score_tokens.py`, test `test_score_tokens_bytes`)

- [ ] Implement: args `--stream DIR --corpus BYTES --offset --T 2048 --windows FILE --n 1024 --device --out-dir --warm-bytes 2048 name=ckpt@featdir`; for each window span (Task 2) the cold pass on `tokens[t0:t1]` with features at `t0+1 ..` (FeatureStore.at); the warm pass on `tokens[span(a - 2048, b)]` scoring only the window's tokens; per window bits = Σ CE/ln2 over the scored tokens; `bpb = Σ bits / Σ bytes` over all windows (byte-weighted) and also the mean of per-window bits/byte; JSON like `score_tables.py`'s plus `bpt`, `bytes`, `tokens`.
- [ ] Test on the synthetic corpus with a 20-step model from Task 4 (CPU): the per-window bytes equal `bytes_in`, the sum over windows of bits equals the direct loss over the concatenated spans.
- [ ] Commit.

## Task 6: the hour arms (`notes/model2/checks/tok_hour_gpu.sh`)

- [ ] The byte arm `h1g_bytes_r`: 4 layers, `$TFI`-like flags on `corpora/pkgs1g.bytes` with `--train-bytes 1000000000 --val-offset 1000000000 --val-fixed 128 --mem-feat .../pkgs1g-tables/feat_all28 --mem-orders $ALL18 --match-feat 8,16,32`, steps 25,562 (the 4-layer hour of the speed round; the clock logged). Scored by `score_tables.py` on the 613 clean dev windows of `pkgs1g.bytes` (dev offset 1e9; the same window indices).
- [ ] The token arm `h1g_tok_r`: `--tokens --vocab 32768 --corpus-file pkgs1g-tok/tokens.u16 --train-bytes <train_end_tok> --val-offset <train_end_tok> --tok-len ... --mem-feat pkgs1g-tok/feat28 --mem-orders 1,2,3,4,6,8,12,16,116,132,202,203,301,1001,...,1007 --match-feat 2,4,8`, the same 25,562 steps; scored by `score_tokens.py` on the same byte windows.
- [ ] The scaled token arm `h1g_tok_eq_r` at steps = 25,562 × (byte s/step ÷ token s/step) from the two arms' logged paces (the calibrated-steps rule).
- [ ] Run through `gpu0_slot.sh` on GPU 0 (after the snapshot) or on GPU 1 after 17:00, whichever frees first; verdict line `tokens vs bytes, clean dev: bytes X, tokens Y (equal steps), tokens Z (equal compute)` into `results/tables/tok/hour.log`.
- [ ] Record in `notes/MODEL2.md` (new, the round's record), DONE/TODO, memory, ledger; commit the JSONs.
