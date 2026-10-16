# Model #2: the reader on tokens

The record of the model #2 round, in the order things happened (the speed round and model #1 are in `SPEED.md`). Design:
`docs/superpowers/specs/2026-10-16-model2-tokens-design.md`; plan: `docs/superpowers/plans/2026-10-16-model2-tokens.md`; code:
the checks under `notes/model2/checks/`, the reader in the worktree `.worktrees/tok` (branch `spike/tokens`, from the big train's
`c5825fc`); tests `tests/test_tokens.py` (speed worktree) and `tests/test_tokens_reader.py` (tok worktree).

## The lock (2026-10-15 evening, Kyle)

Model #2 is trained here ("model #2 will be another in house model"; Qwen stays the yardstick). It is model #1's recipe with one
structural change: 32k BPE tokens instead of bytes ("ok 32k it is"). The package corpus first; the corpus after it is English prose and
dialogue, not more code ("we aren't going to get an intelligence if it doesn't learn how to communicate with me"), with its own held-out
bar before it goes in (a side agent's point: every verdict here is bits/byte on code windows). Train on 2^28, test smaller tables at
use ("28 is fine i think. we could test smaller maybe"). A sweep before the big train ("yeah we should sweep"). The stop rule as #1's:
by outcome at each doubling, Kyle calls it; a two-week cap.

## The vocabulary (2026-10-15/16, `results/tables/tok/`)

Why 32k and not Qwen's 151k: four byte-level BPE tokenizers trained on 100 MB of the package corpus and Qwen3.8-27B's tokenizer, over
the 613 clean package dev windows (1,255,424 bytes) and the two held-out streams:

| vocabulary | bytes/token dev | vox | lab | output layer, M MACs/token | reader compute per byte, M MACs (12x512) |
|---|---|---|---|---|---|
| 256 (bytes, #1) | 1.00 | 1.00 | 1.00 | 0.1 | 50.5 |
| 8k | 2.91 | 2.66 | 2.63 | 4.2 | 18.7 |
| 16k | 3.07 | 2.79 | 2.81 | 8.4 | 19.1 |
| 32k | 3.18 | 2.90 | 2.93 | 16.8 | 21.1 |
| 64k | 3.24 | 3.01 | 3.00 | 33.6 | 25.9 |
| Qwen 151k | 3.64 | 3.21 | 3.36 | 77.8 | 35.2 |

Each doubling of the vocabulary buys less text per token; at 512 wide the output layer at 151k would outweigh the transformer, and
1.3 billion training tokens would touch most of its rows a handful of times. The best vocabulary grows with the model: tens of thousands
for ours, hundreds of thousands for a 27B. Qwen's tokenizer only if Qwen becomes a teacher (then trimmed to the corpus's tokens).
(The first run of the probe read the windows at the kernel corpus's offset inside the package train split: 8k 2.77, 16k 2.93, 32k 3.04,
64k 3.11, Qwen 3.32; same ordering, same call.)

The tokenizer: `corpora/pkgs-tok/bpe32k.json`, trained on the full 5.28 GB train split in 292 s (`checks/tokenizer_train.py`): 3.19
bytes/token on the clean dev windows, 2.95 on Voxelheim and the lab's Python.

## The stream, the tables, the reader (2026-10-16 morning)

- **The stream** (`checks/tok_common.py`, `tokenize_corpus.py`): the corpus tokenized per file unit (no token spans two files), invalid
  UTF-8 bytes as single-byte tokens, so the stream decodes back to the corpus byte for byte (tested). `corpora/pkgs-tok/`: 1,828,428,626
  tokens over 5,478,327,694 bytes (2.996 bytes/token over the whole corpus, headers and all), train_end_tok 1,761,921,874, 509,810 file
  units, 484 s. A clean dev byte window is scored as the tokens whose first byte lies in it (`TokenStream.span`), bits over bytes.
- **The tables** (`checks/tok_tables.py`): a position is a token; its inputs the n-gram orders 1, 2, 3, 4, 6, 8, 12, 16 (ids), the byte
  keys 116/132/202/203/301 read at the token's first byte (`key_tables.compute_keys` per chunk with its halo), the counters 1001–1007 at the
  token's first byte stored as values (`value_orders`); the stored symbol is the next token (slot tag u2 | tok u2 | conf u1 | pad: 6 B).
  Tests: every stored input of a held-out token equals the same input recomputed from the prefix alone and looked up in the final tables
  (prefix-honest, R107); another dev file leaves every training feature unchanged (leak-free).
- **The reader** (tok worktree, commit fcc2180): `--tokens --vocab 32768` (the corpus file is the uint16 stream), `--shared-table` (the
  table's and the match's predicted token embedded with the model's own token embedding, a learned per-order gate, no per-order 32k
  tables), the counters through their own small value channel with no output bonus, uint16 features, bits per byte of the fixed windows
  through `--tok-len`. The byte path is untouched: the 701 existing tests pass; `--vocab 256` default. `checks/score_tokens.py` scores a
  checkpoint on the byte windows (cold: the window's tokens; warm: the 2048 bytes before it read first).

## The first number: tokens against bytes at the hour (2026-10-16)

`checks/hour_corpus.sh` built the pair: `corpora/pkgs1g.bytes` = the package corpus's first 999,975,026 bytes (cut at a file header) +
its dev region (the same bytes as the bar's windows), its token stream, its byte features (2^28, the pipeline of `pkgs_tables.sh`) and its
token features (2^28). `checks/tok_hour_gpu.sh` (unit `h1g-gpu0` through the GPU 0 slot, after the build): three 4-layer arms, `h1g_bytes_r`
(the 18 inputs, match 8/16/32) and `h1g_tok_r` (the 20 token inputs, match 2/4/8, shared channel) at the same 25,562 steps, then
`h1g_tok_eq_r` at the steps that cost what the byte arm cost (the clocks' ratio); scored on the 613 clean dev windows
(`results/tables/tok/k_h1g_{bytes,tok,tok_eq}.json`, verdict line in `tok/hour.log`).

The verdict (16:22; `results/tables/tok/hour.log`, `k_h1g_{bytes,tok,tok_eq}.json`; the run JSONs in the tok worktree's `results/`):

| arm | steps | clock | clean dev cold | warm | bits/token |
|---|---|---|---|---|---|
| bytes (`h1g_bytes_r`) | 25,562 | 3,992 s (0.156 s/step) | **0.8100** | **0.7101** | |
| tokens at equal steps (`h1g_tok_r`) | 25,562 | 10,133 s (0.396 s/step) | 0.8316 | 0.7041 | 2.656 |
| tokens at equal compute (`h1g_tok_eq_r`) | 10,070 | 3,981 s | 0.9590 | 0.8307 | 3.063 |

**Tokens do not pay at the hour.** At equal steps, with three times the text per step, the token arm ties (+0.022 cold, -0.006 warm).
At equal compute it loses by 0.15. The reason is the clock: a step of the token model costs **2.54x** a byte step under the 150 W cap
(the card's SM clock sat at 240 MHz against the byte arm's higher clocks: the 32k output layer is dense work that the power cap
prices above its MAC count), not the 1.3x the vocabulary probe's MAC count predicted. So the token arm sees 3.0x the text for 2.54x
the cost, 1.18x more text per joule, and has a 32k softmax to learn on top; at the hour that is a wash at best. The raw fixed windows
at the train/dev boundary (128 windows, leaky ones included) read the same way: bytes 0.772/0.685, tokens 0.749/0.698 at equal steps.

Two things the number does not settle. (1) Whether the shared-embedding channel, not the tokens, is the handicap (the spec's risk note):
the control `h1g_bytes_sh_r` (`checks/tok_hour_ctrl.sh`: the byte arm with `--shared-table`, same steps) answers that; if it lands near
0.8100 the channel is innocent. (2) The scale: the gap could close with depth and steps (the output layer is a fixed cost per step while
the transformer's share grows with depth; at 12 layers the 32k layer is 10 % of the MACs, not 33 %), but a lever that loses an hour by
0.15 at equal compute is not where #2's budget goes first. The sweep on vocabulary (16k/64k streams built, on /vault) waits for the
control and Kyle's call.

**The control (17:33; `results/tables/tok/hour_ctrl.log`, `k_h1g_bytes_sh.json`; the run `h1g_bytes_sh_r.json` in the tok worktree).** The
byte arm with the shared channel (`checks/tok_hour_ctrl.sh`, unit `h1g-ctrl` through the slot; the same 25,562 steps, 3,767 s, the plain
byte arm's clock): **0.8216** cold / 0.7209 warm, +0.0115 / +0.0108 against the plain byte arm's 0.8100 / 0.7101. The channel costs about
0.012 at the hour, twice the hour's seed noise (0.006), so it is a real handicap and the token arm carries it. If it costs tokens what it
costs bytes, the token arm corrected reads about 0.820 / 0.693 at equal steps (0.010 behind cold, 0.017 ahead warm: still a tie) and about
0.947 / 0.820 at equal compute (still 0.14 behind). The channel explains half of the equal-steps gap and none of the equal-compute loss;
the clock does. **The verdict stands: tokens are not #2's first lever** (R114). What the channel costs and why: on bytes a per-order table
is 256 rows and free, so embedding the predicted byte with the model's own embedding and one gate per order throws away a free degree of
freedom; on 32k tokens the per-order tables would be hundreds of millions of parameters at 512 wide (13 token-valued inputs x 32,768 x 512),
which is why the channel exists. The honest version of the token arm is a smaller vocabulary with per-order tables (8k: a quarter of the
rows, and a quarter of the output layer that set the clock), so the vocabulary sweep's first question is whether 8k tokens pay at equal
compute, not whether 16k or 64k beat 32k. It runs only if Kyle wants the lever tested further; the budget's case is bytes and depth.
