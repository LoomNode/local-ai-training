# Arm A: a gated token n-gram table mixed into Qwen's logits (2026-11-05)

The first arm of the research note `/games/ailab/reports/Memory architectures for a local model.md` (Kyle, 2026-11-05: "let's do some
research first", then "run arm A on gpu 1"). The literature's output-side tables (kNN-LM, Infini-gram, NEST) cut 3-9% of the loss when
the datastore overlaps the test text, and the lab's own byte tables carried most of the small reader's gain; the question was whether a
table over Kyle's world does the same on top of the 27B that already reads two retrieved chunks (1.0545 on his turns, 0.7488 on the
dialogue, the 2026-10-19 memory condition).

## What ran

- **The table** (`notes/model2/checks/armA/ngram_index.py`): every world file (`corpora/kyle/world/files`, 2,197 files, 9,737,285 Qwen
  tokens) tokenised with the 27B's own tokenizer, each position dated by its file's creation date (`asof_dates.json`), rolling hashes of every
  n-gram ending at each position for n = 1..13, each order sorted by (hash, date). A query is two `searchsorted`s. 1.4 GB on disk, 59 s to
  build on the CPU. Self-check: a world file's own text matches at every order with the true next token at 0.95.
- **The scorer** (`qwen_tokens.py`): `qwen_memory.py`'s memory condition, unchanged prompts (the span retriever's top two chunks, or the
  sweep's `choices.json` with `--headers`), keeping every scored token's log-probability. It reproduces the 2026-10-19 per-window numbers
  to four decimals on every window (1.0545 / 0.7488). 1.5-2.3 s a window on GPU 1; 852 windows in 25 min.
- **The mixture** (`mix.py`): at every scored token, `p = lam * p_table + (1 - lam) * p_qwen`, with `p_table = count(suffix + token) /
  count(suffix)` at the longest context suffix the world holds (`n_eff`, 1..12; 0 = unseen), and `lam` a function of the bucket
  (`n_eff` bin x count bin, 45 buckets), fitted by maximum likelihood on one set and scored on the other, plus five-fold within a set and
  the in-sample fit as the ceiling. Counts are **as of the window's session** (files created before it; the lab's hindsight rule) unless
  marked "all files". Bits per byte over the second halves, the usual bar.

## The numbers (the 2-chunk memory, `results/tables/armA/base/`)

| | his turns (340) | the dialogue (512) |
|---|---|---|
| Qwen + 2 chunks | 1.0545 | 0.7488 |
| + table, gate fitted on the other set, as of | **1.0531** (-0.0014) | 0.7491 (+0.0004) |
| + table, five-fold within the set, as of | 1.0507 (-0.0038) | 0.7488 (+0.0000) |
| + table, in-sample ceiling, as of | 1.0502 | 0.7483 |
| + table, all files (no as-of), other set's gate | 1.0479 (-0.0067) | 0.7478 (-0.0010) |
| + table, all files, five-fold | 1.0447 (-0.0098) | 0.7471 (-0.0017) |
| fixed lam 0.05 / 0.1 / 0.2, as of | 1.0589 / 1.0732 / 1.1079 | 0.7591 / 0.7735 / 0.8065 |

Counting a suffix as seen at 2 occurrences instead of 1 (`--cmin 2`) moves nothing (1.0529 / 0.7494).

**Where the bits are.** The world has seen 95% of the scored tokens' contexts, but the longest match averages 1.9 tokens and the table's
probability of the true token is 0.08 at that length; the fitted gate puts lam at 0.00-0.01 for every order up to 5. The gain lives at
orders 6 and up, 0.4-0.6 probability of the true token and lam 0.04-0.17, which is 2,342 of 99,970 tokens on his turns (1,894 of 135,000
on the dialogue). Those tokens carry -138 of the -505 bits on his turns; the rest comes from order 2 at lam 0.00-0.01 (a hair of mass
on the right token across 30k tokens). On the dialogue the order-1 bucket costs +192 bits (the gate, fitted on his turns, is slightly wrong
for it) and wipes the rest out. Without the as-of rule the long matches double (3,413 tokens) and carry -701 bits, still 0.007 bits/byte.

By date: windows from 2026-09 on (150 of his, 250 dialogue) see most of the world and gain -0.0017 / +0.0010; earlier windows -0.0013 /
-0.0002. The as-of rule is not what limits this arm; the overlap is.

## What it means

The arm fails its bar by a wide margin: -0.001 to -0.004 bits/byte against a literature expectation of -0.03 to -0.06 (3-6% of the
loss). The published gains came from datastores that contain the test text's own kind of repetition (Enron's mailbox, Law-MT's
boilerplate, Wikipedia with a Wikipedia store); Drozdov 2022's rule, gain tracks n-gram overlap, holds here in the other direction. Kyle's
world, after the leak filter (no file holding a 64-byte span of the held-out text) and the as-of rule, holds his turns' contexts for
two tokens on average and a 6-token context 2% of the time. The retrieved chunks in the prompt already give the 27B the few long matches
there are (the 1.169 -> 1.0545 drop), and the table adds a sliver on top of them. A larger world (more of his files, the code) would raise
the overlap, but the lab's 2026-10-21 hindsight check says that gain would be a ceiling too.

Retire the output-side table on the 27B as a lever for this corpus. The table's value stays where the lab found it: inside a small reader
that has no other way to see the world. Arms B (the distilled Qwen3.5 student), C (Memory Decoder) and D (an Engram-style table in the
student) are untouched by this result, and B is next.

## The headers + as-of memory (the sweep's best arm, 0.9659 / 0.7502; `results/tables/armA/hdrasof/`)

The same table on top of the better memory (the scorer again reproduces the sweep's per-window numbers):

| | his turns (340) | the dialogue (512) |
|---|---|---|
| Qwen + headers, as of | 0.9659 | 0.7502 |
| + table, gate fitted on the other set, as of | **0.9653** (-0.0006) | 0.7508 (+0.0006) |
| + table, five-fold within the set, as of | 0.9651 (-0.0008) | 0.7502 (-0.0001) |
| + table, all files (no as-of), other set's gate | 0.9627 (-0.0033) | 0.7490 (-0.0012) |
| + table, all files, five-fold | 0.9615 (-0.0044) | 0.7484 (-0.0018) |

Smaller still: the better memory already puts the long matches in the prompt, so the fitted gate goes to zero below order 5 and the
orders 5+ carry -199 bits on his turns in total. The as-of ceiling without dates is the same 0.003-0.004 as before. Arm A is closed on
both memories.

## Correction (2026-11-06): the table never held his own writing; with his earlier sessions it wins 0.023

A side agent's note, relayed by Kyle while Arm B was starting: the world (`corpora/kyle/world/files`) is his md, txt and sh files, and
`kyle_heldout.py` put every kept turn of his into the held-out set (5,813 turns of 26,564 seen, the rest dropped as duplicates, pastes
over 2,000 bytes or tagged blocks), so the table above never saw a line he typed. The papers' large gains came from stores holding the
test author's own text. The honest way to put his writing into the table is a time split, since all of his turns are test data:

- **Session dates** (`notes/model2/checks/armB/session_dates.py` -> `corpora/kyle/session_dates.json`, 0600): every session document in
  `kyle.bytes` (371) and `dialogue.bytes` (121) carries a header; its date is `asof_dates.json`'s date of any window wholly inside it, else
  the header (claude and gemini ISO, codex the first ten digits of a zero-padded millisecond epoch, grok the session file's mtime).
  370 + 121 dated, and on the 122 sessions where both sources exist they agree on every one.
- **The split** at 2026-09-01: the table holds the 1,097 world files created before it (7.38M tokens) and, in the second variant, the 296
  held-out sessions dated before it (0.60M tokens; `ngram_index.py build --before --sessions-before`); the score is on the windows
  dated on or after it (`mix.py --since`): 150 of his turns (43,456 tokens), 250 dialogue windows (69,599). The same as-of rule applies
  within the table; the gate is fitted on the other set as before.

| post-2026-09-01 windows | his turns (150) | the dialogue (250) |
|---|---|---|
| Qwen + two chunks | 1.0732 | 0.8326 |
| + table of earlier files | 1.0730 (-0.0002); five-fold 1.0696 | 0.8347 (+0.0021); five-fold 0.8325 |
| + table of earlier files **and his earlier sessions** | **1.0504 (-0.0227)**; five-fold 1.0388 (-0.0343); in-sample 1.0372 | 0.8381 (+0.0055); five-fold 0.8319 |
| Qwen + headers, as of | 1.0001 | 0.8337 |
| + table of earlier files | 1.0002 (+0.0000); five-fold 1.0001 | 0.8353 (+0.0016); five-fold 0.8338 |
| + table of earlier files and his earlier sessions | **0.9885 (-0.0116)**; five-fold 0.9813 (-0.0188); in-sample 0.9802 | 0.8379 (+0.0041); five-fold 0.8332 |

With his sessions in the table the mean matched order on his turns goes from 1.9 to 2.5 and the table's probability of the true token
from 0.09 to 0.16; the gain comes from every order from 2 up (order 8-9: 612 tokens, p 0.82, -592 bits). The dialogue, mostly the
assistant's side, gains nothing, so the gate fitted on it is a conservative one for his turns (the five-fold number within his turns is the
in-set fit). The 0.023 is the literature's range from a zero-parameter table behind the 27B, honestly dated. Results in
`results/tables/armA/split/`, run dirs under `/vault/ailab/flybrain-mem/armA/split/`. Arm A is reopened as a deployable piece: the open
questions are the gate's stability week to week and the update path as sessions accrue. Arm B's student trains on the same split.
