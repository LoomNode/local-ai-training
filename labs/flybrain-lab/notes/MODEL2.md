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
At equal compute it loses by 0.15. The reason is the step's cost, and that is arithmetic, not the power cap: a step of the token model costs **2.54x** a byte step, and by the
runner's own count (`rflops`, the weight matmuls per position) the 4-layer token step is 2.95x the byte step (48.0 M against 16.25 M flops:
the 32k output layer alone, 33.6 M, is twice a 4-layer transformer), so under the cap the token arm got 16 % *more* flops per second than
the byte arm. (The write-up of 16:22 said the cap priced the output layer above its MAC count, which had predicted 1.3x; that 1.3x is the
12-layer ratio from the vocabulary table, misapplied to the 4-layer test. Corrected the same evening when a side agent asked whether the
verdict depends on the cap: it does not; a re-timing at a higher cap would move the ratio toward 2.95x, not 1.3x.) So the token arm sees
3.0x the text for 2.54x the cost, 1.18x more text per joule, and has a 32k softmax to learn on top; at the hour that is a wash at best. At
12 layers the same count gives 1.66x (the output layer is 40 % of the step, not two thirds), so tokens would get 1.9x the text per joule
there; what that is worth on the bar is unmeasured. The raw fixed windows
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

## The second scale point: 12 layers at four hours (2026-10-16 evening, both cards)

Kyle, on the hour verdict and the control: "let's keep testing both as we scale for now. i'm just thinking our setup may change things."
So bytes and tokens both stay in the race, and the pair is re-run at each scale point of #2 rather than settled at 4 layers. The reason
it could turn: the token step's extra cost is the fixed 32k output layer, two thirds of a 4-layer step and 40 % of a 12-layer one (the
runner's flop count), so at the big train's depth tokens get 1.9x the text per joule instead of 1.18x; whether the reader converts that
text is the open question (at 4 layers and equal steps it did not).

The point: 12 layers, the big train's shape, at four times the hour's budget (BUDGET_S = 4 x 3,992 s, the hour pair's byte clock), on
the same first GB with the same features and inputs as the hour pair. `checks/tok_l12_cal.sh` (unit `l12g-cal`, GPU 1, 22:02) times 600
steps of each 12-layer arm on the quiet card and sets each arm's steps to BUDGET_S / its pace (the calibrated-steps rule), then checks
that a token arm resumes from a `--ckpt-every` checkpoint. `checks/tok_l12_gpu.sh` runs the arms with checkpoints every 2,000 steps and an
automatic resume (`bigtrain.sh`'s pattern, so a game on GPU 0 costs at most 2,000 steps): `l12g_tok_r` on GPU 1 (unit `l12g-gpu1`),
`l12g_bytes_r` then the control `l12g_bytes_sh_r` on GPU 0 through the slot (unit `l12g-gpu0`, orchestrator `l12g-gpu0-orch`, after the
calibration). Each arm is scored on the 613 clean dev windows as it finishes (`results/tables/tok/k_l12g_{bytes,tok,bytes_sh}.json`); the
verdict lines land in `tok/l12g_verdict.txt`. About four and a half hours of each card once the calibration is in (the byte arms about
nine hours of GPU 0, more with games). Not run: an equal-steps token arm at this depth (6.6 hours for a quality-only number) and the
vocabulary axis (8k with per-order tables); either follows if this point is close.

**The verdict (03:22; `results/tables/tok/l12g_verdict.txt`, `k_l12g_{bytes,tok}.json`; the run JSONs `l12g_{bytes,tok}_r.json` in the tok
worktree's `results/`).** The calibration (`tok/l12g_cal.log`, 22:02-22:14 on the quiet GPU 1): 12-layer paces 0.353 s/step for bytes and
0.626 for tokens, a ratio of **1.77x** (the flop count said 1.66x; 2.54x at 4 layers), so 45,237 byte steps against 25,518 token steps for
the 15,968 s budget; the token resume from a step checkpoint checked. The models: bytes 28.3 M trainable scalars (model #1's size), tokens
59.4 M, of which 31 M are the two 32,768 x 512 vocabulary matrices (the embedding and the untied output layer). Both arms ran without an
interruption (no game): the token arm on GPU 1 at 0.620 s/step (15,823 s), the byte arm on GPU 0 at 0.398 s/step (17,988 s, 13 % over its
calibrated pace: GPU 0 drives the desktop; the steps are the metric, the clock is only logged).

| arm | steps | clock | clean dev cold | warm | bits/token | passes over the GB |
|---|---|---|---|---|---|---|
| bytes, 12 layers (`l12g_bytes_r`, 28.3 M) | 45,237 | 17,988 s (GPU 0) | **0.7234** | **0.6250** | | 0.74 |
| tokens at equal compute, 12 layers (`l12g_tok_r`, 59.4 M) | 25,518 | 15,823 s (GPU 1) | 0.7843 | 0.6575 | 2.505 | 1.26 |
| the hour, 4 layers: bytes | 25,562 | 3,992 s | 0.8100 | 0.7101 | | 0.42 |
| the hour, 4 layers: tokens at equal compute | 10,070 | 3,981 s | 0.9590 | 0.8307 | 3.063 | 0.49 |

**Tokens still lose at equal compute, by 0.061 cold / 0.033 warm, down from 0.149 / 0.121 at the hour.** The scale step (three times the
depth, four times the budget) bought the byte arm 0.087 and the token arm 0.175: the token arm gained twice as much, as the arithmetic said
it would. The cost ratio fell from 2.54x to 1.77x, so the token arm read 1.69x the text of the byte arm (1.255 GB against 0.741 GB) rather
than 1.18x, and the 32k softmax that was two thirds of a 4-layer step is 40 % of a 12-layer one. What it did not do is cross. Kyle's "our
setup may change things" is half right so far: the gap closes with scale; it has not closed.

Two cautions on the number. (1) The output layer's share is now fixed: 12 layers is #2's depth, so any further closing has to come from
the budget axis alone (the token arm's gain per doubling against the byte arm's), and that is a trend the next point measures, not one the
arithmetic predicts. (2) The first GB is a wall for tokens before it is for bytes: the equal-compute token arm already read the corpus 1.26
times (bytes 0.74), so the next scale point on this corpus would be measuring repetition; it belongs on the full package corpus (5.3 GB;
the full token stream `pkgs-tok` exists, its 2^28 features do not yet) or on the English corpus with R112's held-out set. The raw fixed
windows at the boundary (128, leaky ones included) read bytes 0.6884 / 0.6023, tokens 0.7004 / 0.6499, a gap of 0.012 cold where the clean
bar says 0.061: the token arm, a second pass in, has seen more of the leaky windows' twins, which is the leak check doing its job.

Running after the verdict: the shared-channel control at this depth (`l12g_bytes_sh_r`, GPU 0 from 03:22, the morning) and, on the idle
GPU 1 from 03:27, the equal-steps token arm `l12g_tok_eq_r` (unit `l12g-eq-gpu1`; the token arm at the byte arm's 45,237 steps, 1.77x the
compute, about 7.9 hours; quality only, and at 45k steps it reads the GB 2.2 times). It answers the question the control cannot: how much
of the token arm's per-step lead survives depth on the clean bar (at the hour it was a tie), which is the ceiling on what a cheaper output
layer (8k with per-order tables) could buy at this point. Each line lands in `tok/l12g_verdict.txt` as its arm finishes.

**The control at depth (08:26; `k_l12g_bytes_sh.json`; the run `l12g_bytes_sh_r.json` in the tok worktree; GPU 0 03:22-08:26, no game).**
The byte arm with the shared channel at 12 layers and the same 45,237 steps: **0.7278** cold / 0.6303 warm against the plain byte arm's
0.7234 / 0.6250, +0.0044 / +0.0052. At the hour the channel cost 0.012, twice the seed noise; at depth it costs 0.005, under the 12-layer
seed noise (the 12-layer seed pair at 4x on the old corpus differed by 0.008). So the channel's handicap shrinks with the model, as the
output layer's share does: the free degree of freedom the 256-row per-order tables give a 4-layer reader matters less once 12 layers can
shape the embedded prediction themselves (the channel also drops those tables, 2.75 M parameters: 25.6 M against 28.3 M). For the token
arm the correction is therefore small: at equal compute tokens corrected read about 0.780 / 0.652, still 0.057 / 0.027 behind bytes. The
equal-compute gap at this point is the step's cost and nothing else; the equal-steps arm (about 11:30) says what that cost buys.

**The equal-steps arm at depth (11:23; `k_l12g_tok_eq.json`; the run `l12g_tok_eq_r.json` in the tok worktree; GPU 1 03:27-11:23).** The
token arm at the byte arm's 45,237 steps (1.57x the clock, 2.2 passes over the GB): **0.7356** cold / **0.6095** warm against bytes'
0.7234 / 0.6250, +0.012 cold / -0.016 warm, 2.349 bits/token. The same tie as at the hour (+0.022 / -0.006 there): three times the text
per step buys tokens nothing cold and 0.016 warm, and depth did not change that (corrected for the channel's 0.004, +0.008 / -0.020). So
the ceiling on what a cheaper output layer could buy at this point is about 0.02 warm and nothing cold, and only if the layer were free;
the 8k per-order-table arm cannot beat bytes here by more than that. **There is no code hybrid worth building at this scale**; the hybrid
is the role split Kyle proposed the same morning (bytes for code, the prose side measured on its own text), and the token arm's case rests
on prose, where it is unmeasured.

| 12 layers, clean dev (613 windows) | steps | clock | cold | warm | bits/token |
|---|---|---|---|---|---|
| bytes | 45,237 | 17,988 s | **0.7234** | 0.6250 | |
| tokens at equal compute | 25,518 | 15,823 s | 0.7843 | 0.6575 | 2.505 |
| tokens at equal steps | 45,237 | 28,296 s | 0.7356 | **0.6095** | 2.349 |
| bytes with the shared channel (control) | 45,237 | 18,070 s | 0.7278 | 0.6303 | |

**Kyle, on the point and the control (2026-10-17 morning): "so far i say we train both ways for now. one isn't clearly better so i feel
like there could be nuance or hybrid aproaches we could try down the line."** R118: both arms go to the next scale point. The fact beside
it: at equal compute bytes lead at both points so far, and the 0.061 at this one is about a doubling and a half of budget on the package
corpus (0.041 a doubling); what is open is the trend, which halved the gap in one scale step. The next point goes on the full package
corpus (the token arm already laps the first GB), which needs the full stream's 2^28 token features first (`tok_tables.py` on `pkgs-tok`,
1.83 G tokens: about 115 GB on /vault at the first GB's 21 GB per 333 M tokens, hours of six cores); the proposal is 16x the hour, about
16 hours an arm, where the token arm reads the corpus 0.9 times and the byte arm 0.56. The hybrids, for later and in the order I would try
them: the byte reader over token-keyed count tables (the knowledge lives in the table; a token key spans more text per order than a byte
key, and the reader's step stays a byte step); token input with a byte-rate output head (the trunk at the token rate, a small decoder at
the byte rate, so the step costs like tokens and the output like bytes: patching, in the literature's terms); and the 8k vocabulary with
per-order tables, which the equal-steps arm bounds. None is built now.

## The prose corpus (2026-10-17, `notes/model2/checks/talk/`, data on `/vault/ailab/corpora-src/talk/`)

Kyle, the morning after the 12-layer point: "the sources is probs the most important decision we are gonna make for a while. this has to
be right or at least close from the git go". The discussion settled the principle before the list: **training text is chosen by register,
reference text by content.** The reader's habits come from what it trains on, so the training corpus has to sound like Kyle and the
questions he asks; a text the model should *know* (the Bible, the inventors' works, man pages, a project's docs) goes into a table, where
it costs no training compute, does not tilt the reader's register, and can be loaded or dropped by context (the project-index mechanism,
0.23-0.28 on a project's own files). On "coding knowledge from a table": knowledge yes (the table is the playbook; the corpus table and
the project index already are one), skill no (composition is the reader's, and the consolidation runs showed the weights hold almost
nothing without the table). R119: the sources, each measured on Kyle's held-out set before mixing.

**Kyle's held-out sets** (R112, R120; `kyle_heldout.py`, run by Kyle as `~/heldout.sh` because the auto-mode classifier refuses my own
reads of the chat stores; `/games/ailab/corpora/kyle/`, mode 0600, never trained on, not committed). His own typed turns from every
assistant transcript store he has: Claude Code on goblin, wyvern, fairy and genie (the other hosts' stores pulled to /vault by rsync),
the grok CLI (goblin, fairy), codex (goblin, wyvern) and the Gemini CLI (goblin); Antigravity left nothing on this box. One document per
session; sidechains, tool results, pasted blocks and system-injected text dropped; exact repeats removed; a user turn over 2,000 bytes is a
paste, an attached file or injected context, not typing (his median turn is 50-110 bytes; in the Claude store the cap removes 5 messages
holding 243 KB, four of them tag-filled, one a 177 KB paste) and is dropped; secrets scrubbed by pattern (433 redactions). Two sets: his
turns (`kyle.bytes`: 371 sessions, 698 KB = **340 windows**) and the dialogue (`dialogue.bytes`: the same sessions with both sides in time
order, "User:" / "Assistant:", the assistant side whichever model answered; drawn with a seed across stores to a cap of **2,047 windows**:
codex 1.98 MB, Claude 1.92 MB, grok 0.29 MB), back to back in `heldout.bytes`, the dev region of every scoring corpus (`heldout.json` has the
regions and the per-store counts). A side agent's point, taken before anything was scored: his turns alone measure *reading* him, not
*answering* him, and a source of reply text could rank low on that bar and still be what the model needs most; so every source gets both
numbers and the proportions weigh both. The first cut (goblin's Claude transcripts only) was 161 windows of turns; the stores took it to 340.

**The sources** (the workers' `extract_<src>.py`, one file per source in the corpus format, a sidecar `<src>.json` with the URLs, the
license and the counts):

| source | what | size | license |
|---|---|---|---|
| `stackexchange` | unix, askubuntu, serverfault, superuser, gamedev threads (the archive.org dump; the newest threads to each site's budget; question + answers by score) | 1.04 GB (450,477) | CC BY-SA 4.0 |
| `ubuntu-irc` | #ubuntu day logs from irclogs.ubuntu.com, joins/parts dropped, "<nick> message" | 300 MB budget | public logs |
| `oasst2` | OpenAssistant conversations, English trees, "User:" / "Assistant:" turns | 37.8 MB (5,401 trees) | Apache-2.0 |
| `wikipedia` | English Wikipedia 2023-11, articles with 4+ technical terms (35,379 of 2.09 M seen) | 500 MB | CC BY-SA 4.0 |
| `wikipedia-simple` | all of Simple English Wikipedia 2023-11 (300+ chars) | 266 MB (170,450) | CC BY-SA 4.0 |
| `docs` | Godot docs, PyTorch docs, Cosmos SDK docs, Celestia docs, the Arch wiki (English pages) | 52 MB (4,438) | CC BY 3.0 / BSD / Apache-2.0 / GFDL |

**The measurement** (`talk_tables.sh`, unit `talk-tables`, CPU; `talk_readers.sh`, unit `talk-readers-gpu1`): per source, a scoring corpus
of its budgeted documents as train with `heldout.bytes` as the dev region (`build_corpus_talk.py --dev-file`), its 2^28 byte tables
over that file, then the hour reader (4 layers, the hour pair's byte arm, 25,562 steps; its in-run curve on Kyle's turns) trained on that
source alone and scored cold and warm on Kyle's turns (`results/tables/talk/k_talk_<src>.json`) and on the dialogue (`k_talk_<src>_dlg.json`);
lines in `talk/scores.txt`. The baseline on both sets: model #1 with the code corpus's t28 tables (`k_{kyle,dialogue}_model1_t28.json`),
i.e. where the code model stands on Kyle's chat. The mix
(`build_corpus_talk.py`: documents shuffled across sources, 5 % of each source's documents held out for dev and for test, every split cut
on a document header) is built from the sources that paid, in proportion; then its 32k BPE, its first-GB slice and the pair at the hour.

**The numbers** (`results/tables/talk/scores.txt`; cold bits/byte, warm in parentheses; "passes" = the hour reader's 419 MB of training
text over the source's train bytes):

| source | train bytes (passes) | his turns (340 windows) | the dialogue (2,047 windows) |
|---|---|---|---|
| model #1 + its t28 tables (12 layers, 960k steps of code: the baseline) | — | 1.8273 | 1.7469 |
| `stackexchange` | 1.00 GB (0.42) | **1.8255** (1.729) | 1.8442 (1.773) |
| `oasst2` | 37.8 MB (11) | 2.1195 (2.013) | 1.9414 (1.873) |
| `docs` | 52 MB (8.0) | 2.1350 (2.030) | 1.9956 (1.929) |
| `wikipedia` | 500 MB (0.84) | 2.1401 (2.012) | 2.0210 (1.935) |
| `ubuntu-irc` | 300 MB (1.4) | 2.2438 (2.136) | 2.1884 (2.116) |
| `wikipedia-simple` | 266 MB (1.6) | 2.2494 (2.128) | 2.1033 (2.018) |

Register beats volume. Stack Exchange, the one source written in the register he types in (technical questions and their answers), is
0.3 ahead of every other prose source on his turns, and an hour reader trained on it alone reads his turns as well as model #1 does with
thirty times the budget and the big code table. The two "conversational" sources are last: Ubuntu IRC is chat, but idle chat, and Simple
Wikipedia is simple, but neither is technical. The two small sources, oasst2 (the only source of assistant *answers*) and docs (what the
answers quote), rank by the dialogue well above their rank by his turns, the side agent's point made concrete. Both drop out: IRC and
Simple Wikipedia.

**The mixes** (`talk_tables.sh`'s `MIX`, the same recipe per source; the small sources whole, the budgets are document selections):

| mix | recipe | train | his turns | the dialogue |
|---|---|---|---|---|
| `mixA` | stackexchange 1 GB + oasst2 + docs | 1.09 GB | 1.8023 (1.708) | 1.7747 (1.705) |
| `mixB` | mixA + wikipedia 250 MB | 1.34 GB | 1.8026 (1.709) | **1.7654** (1.697) |
| `mixC` | code 500 MB (the package corpus) + stackexchange 500 MB + oasst2 + docs | 1.09 GB | **1.8006** (1.700) | 1.7697 (1.696) |
| `mixD` | mixB + code 500 MB (the union) | 1.84 GB | 1.8019 (1.707) | 1.7714 (1.701) |

The small sources pay far above their weight: oasst2 + docs are 8 % of mixA and move Stack Exchange alone by 0.023 on his turns and
0.070 on the dialogue, and mixA beats model #1 on *both* sets at the hour. The three mixes tie on his turns (0.002 apart; the hour's seed
noise is about 0.005). On the dialogue the Wikipedia share (B) and the code share (C) each edge A, by 0.009 and 0.005 cold, and C is the
best warm reader on both sets; neither hurts his turns. The fourth mix, the union of the two, lands between them on the dialogue
(1.7714: B's and C's edges over A were not additive, they were noise-sized), and on his turns the four mixes span 0.002. **The hour
cannot separate B, C and D**; what separates them is the long train, which laps the corpus (the 12-layer curves' wall was passes over
the text: model #1's 800k steps are 13 GB of training text, ten passes over mixB and seven over mixD), so the corpus is the union, mixD:
every source that paid, nothing that hurt, the most bytes. Ruling R124. The code
baseline alone (`talk_pkgs1g_r`, the first GB of the package corpus under the same recipe) is still running: GPU 0's slot yields to games
and the talk readers carry no checkpoints, so each of today's three games restarted its hour.
