# Model #2: the reader on tokens — design

Date: 2026-10-16. Round: model #2 (after the speed round and model #1; records `notes/SPEED.md`, ledger R110/R111). Status: the design was locked in chat on 2026-10-15/16; this is its written form, for Kyle to correct.

## Intent

Kyle (2026-10-15): "i'm still wanting to train a local model at this point. we can still test on qwen but model #2 will be another in house model." Then: "ok 32k it is"; the package corpus first, with "we aren't going to get an intelligence if it doesn't learn how to communicate with me" (the corpus after this one is English prose and dialogue, not more code); "28 is fine i think. we could test smaller maybe"; "yeah we should sweep"; the stop rule as #1's, by outcome at each doubling.

Model #2 is model #1's recipe with one structural change: the reader works in 32k BPE tokens instead of bytes. Everything that carried #1 stays: the hashed count tables (two-way buddy slots, confirm/decay/replace, the leak-free build), the in-window match lookup, the directory tier (later, use layer), the warmup-stable-decay schedule with long cooldowns, the stop by outcome, the bits/byte bar on the same clean dev windows and the same held-out streams, Qwen3.8-27B as the yardstick (0.212 / 0.394 bits/byte on Voxelheim / lab Python).

Why tokens (results/tables/tok/, ledger R110 and its correction): on the 613 clean package dev windows a 32k byte-level BPE reads 3.18 bytes per token (Qwen's 151k: 3.64). The 12×512 reader costs about 50 M MACs per position plus 512×V for the output; per byte that is 50.5 M on bytes, 21.1 M at 32k, 35.2 M at Qwen's vocabulary. At equal step time the token reader sees three times the text, its 2048-position window covers 6.5 KB instead of 2 KB, and the table does one lookup per token instead of per byte (a 2^28 table is as full on 16 GB of text as on 5 GB today).

Success, in order: (1) an hour arm on tokens beats the matching hour arm on bytes on the clean dev windows in bits/byte; (2) the sweep picks the shape; (3) the big train on both cards, stopped by outcome, yields model #2 and its final numbers against #1 and Qwen on the held-out streams.

## The token stream

- **Tokenizer:** `corpora/pkgs-tok/bpe32k.json`, HF `tokenizers`, byte-level BPE (GPT-2 byte alphabet and regex pre-tokenizer, no special tokens, no prefix space), trained on the package corpus's train split (`layout.json` `train_end` = 5,283,527,978) by `notes/model2/checks/tokenizer_train.py`. Every one of the 256 bytes is a token of its own, so every byte string has a tokenization.
- **The stream:** `corpora/pkgs-tok/tokens.u16` (one uint16 id per token over the whole corpus, ~1.7 G tokens), `tok_len.u8` (each token's byte length, from the fixed per-id byte length `id_len[32768]`), and `tok_off.u64` (each token's first byte offset, with a sentinel equal to the corpus length). Every corpus byte belongs to exactly one token: text is tokenized per file (the corpus's file headers `\n\n// ==== ` are cut points, so no token spans two files; inside a file, 16 MB chunks cut at newlines); bytes that are not valid UTF-8 are emitted as their single-byte alphabet token, so the stream round-trips the corpus exactly (`tokens -> bytes == corpus`, the acceptance test).
- **Layout in tokens:** `corpora/pkgs-tok/layout.json` adds `train_end_tok` (the first token at or after `train_end`), the dev/test ranges in tokens, and names the byte layout it was cut from.
- **The dev windows for the bar:** a clean dev byte window [a, a+2048) is scored as the tokens whose first byte lies in it; the window's byte count is the sum of their lengths (2048 give or take a token). Cold = those tokens alone. Warm = preceded by the tokens of the previous 2048 bytes, the same reach in bytes as #1's warm. bits/byte = the summed token bits over the summed bytes. The same rule scores the held-out streams (Voxelheim, lab Python) once the use layer exists on tokens.

## The tables on tokens

A position is a token. Its context inputs, the same three families as #1's 18:

1. **Token n-grams:** orders 1, 2, 3, 4, 6, 8, 12, 16 tokens (`corpus_memory.context_hashes` runs unchanged on uint16).
2. **The byte-defined keys** 116/132 (template of the last 16/32 bytes), 202/203 (the last 2/3 words), 301 (the previous line and the current line so far): computed by the existing byte code chunk by chunk and read at each token's first byte, never materialised in full (the per-byte key files were 220 GB).
3. **The counters** 1001–1007 (column, indent, previous indent, brace and paren depth, in-string, in-comment): the byte state machine's value at the token's first byte, stored as the value itself (no 128 offset; `features.json` lists them as `value_orders`).

Stored symbol = the next token id. Slot = tag u2 | token u2 | conf u1 | pad u1 (6 bytes; #1's slot was 4). Build as #1's `assoc_all.py` (two-way buddies, lookup before write per chunk, positions at or after `train_end_tok` read the frozen table), with `--tokens`. Features: `pred_n<o>.npy` uint16 and `conf_n<o>.npy` uint8, one entry per token; `features.json` carries `n_tokens`, `train_end_tok`, `value_orders`, and the tokenizer's path and hash.

Table sizes: 2^28 to train on (Kyle); the trained reader is also scored through 2^26, 2^27 and 2^30 tables of the same stream ("we could test smaller maybe"; ship smaller is the test).

## The reader on tokens

- Token embedding 32,768 × 512, tied with the output; windows of 2048 tokens; batch 8; the corpus memmapped as uint16 (the 5 GB train split no longer fits the loader's in-RAM copy of bytes).
- **The fused channel:** each count order's predicted token is embedded with the reader's own tied token embedding, scaled by a per-order learned gate, plus a per-order confidence-class embedding (5 classes as today). No per-order vocab tables (18 × 32k × 512 would be 295 M parameters beside a 25 M transformer). Value orders (the counters) keep their own small embeddings (128 × 512 each).
- **Match features** on tokens: orders 2, 4, 8 tokens (#1's 8/16/32 bytes); the hashing already works on ids.
- Loss: cross-entropy over 32k; the runner reports bits/token, and bits/byte per window through `tok_len`; `score_tables.py --tokens` writes the dev-window numbers under the rule above.
- Unchanged: RoPE, 12 layers and 512 wide to start (the sweep tries 8 and 768), warmup-stable-decay, `--val-fixed`, checkpoints, the stop machinery (`big_finish.sh`, `cooljob.sh`, the queue on GPU 0 under the guard).

## Tests and runs, in order

1. **Unit tests** (`tests/test_tokens.py`, lab interpreter): the stream round-trips a corpus sample exactly and `tok_off` matches; invalid UTF-8 bytes become single-byte tokens; the dev-window token selection sums to 2048 ± one token; a feature build on a 2 MB synthetic corpus is prefix-honest (recomputing position p's inputs from tokens [0, p] gives the same row; R107's lesson) and leak-free across `train_end_tok`; the loss reshape and the bits/byte arithmetic at vocab 32k.
2. **The first hour arm, tokens against bytes** on the package corpus's first 1 GB (the hour arms' split; features for 1 GB are ~17 GB per tokenizer variant, the full corpus ~90 GB): a 4-layer byte arm and a 4-layer token arm at equal steps, the per-step clock ratio logged (the token arm's softmax is 32k wide; expect about +30 % a step at 4 layers), and a second token arm at the steps scaled by that ratio (the calibrated-steps rule) for the equal-compute point. Scored on the clean dev windows under the rule above. Decides whether #2 goes on tokens.
3. **The sweep** (hour arms, same split): vocabulary 16k / 32k / 64k (each its own tokenizer, stream and 2^28 features), width 512 / 768, depth 8 / 12, the names input per token (R111), the table at 2^26 / 2^27 / 2^28 at score time. One change per arm against the 32k base.
4. **The big train:** the chosen shape on the full corpus's features; GPU 1 trains, GPU 0 cools the doubling snapshots at 10 % under the guard; a check at each doubling of steps on the clean dev windows; stop when a doubling buys under 0.005 (Kyle calls it); a two-week cap on GPU 1. One pass over the train split is ~107k steps (16k tokens a step), so the data wall is expected between 400k and 800k steps; the English corpus is the work item that gates going further.

## Not in this spec

The use layer on tokens (`index_project.py`, `stream_inputs.py`, `sample_tables.py`, `~/write.sh`): pass two, after the first number. The English prose and dialogue corpus: its own decision and spec. Qwen as a teacher (would need Qwen's tokenizer, trimmed): not planned.

## Risks and how they are caught

- The shared-embedding channel is a second change riding with tokens. If the token arm loses, a byte arm with the shared channel isolates it before tokens are blamed.
- Disk: the full corpus's token features at 18 orders are ~90 GB (`/vault` has 57 GB free, `/games` 131 GB). The sweep runs on the 1 GB split; the full build happens once, for the chosen shape, on `/games`, or on `/vault` after Kyle's cleanup.
- RAM at use time: 2^28 at 6-byte slots is 1.5 GB per order, 2^30 is 6 GB per order; the "ship smaller" scoring in the sweep is what answers the RAM question on tokens.
- The counters and keys are evaluated at token starts, so a token containing a newline or a quote changes state inside itself; the feature describes the state before the token, which is the prefix-honest definition. The unit test pins it.
