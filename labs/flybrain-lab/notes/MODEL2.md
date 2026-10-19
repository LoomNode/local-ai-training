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
| `pkgs1g` (the code corpus's first GB: the code baseline, run after the mixes) | 1.00 GB (0.42) | 2.0920 (1.961) | 2.0248 (1.926) |
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
baseline alone (`talk_pkgs1g_r`, the first GB of the package corpus under the same recipe; it ran last, on GPU 1, after GPU 0's slot lost
its hour to games three times) reads his turns at 2.0920 and the dialogue at 2.0248. On his turns that is ahead of every prose source
but Stack Exchange (his turns are full of paths, flags and code) and 0.27 behind it; on the dialogue it is behind oasst2 as well. Model #1
is this recipe at 12 layers, 37x the steps and the big table, and it is 0.27 / 0.28 ahead of it: the long train is worth that much on his
chat, and an hour of Stack Exchange is worth the same on his turns. Code in the mix bought 0.002 (mixC over mixA): the code reader's
knowledge of his turns is the part the prose already covers.

### The build (2026-10-17 23:31 to 2026-10-18 01:09, `talk_corpus.sh`, unit `talk-build`, CPU only)

`build_corpus_talk.py --dedup --tail-file` wrote `/games/ailab/corpora/talk.bytes`: 1,844,874,429 bytes, 508,446 documents, split by
document at a 5 % held-out share: train 1,656,781,807 bytes (457,164 documents: Stack Exchange 900 MB, code 450 MB, Wikipedia 226 MB,
docs 47 MB, oasst2 34 MB), dev 91.4 MB at 1,656,781,807 (1,024 windows, the mix's own bar), test 91.8 MB, and Kyle's two held-out sets
as a tail region of 4.9 MB at 1,839,982,152, so the corpus's own tables cover them (the leak lesson in reverse: the tables must see the
windows' *context*, never their text; the readers' train bytes stop at train_end). The file is mode 0600 for that tail and is never
committed. The new `--dedup` (the first 1,024 bytes of a document's body, hashed) dropped 6,263 bundled copies from the code share and
none from the prose: the package corpus's duplicates are the vendored libraries found on 2026-10-11, the prose sources came clean.

The 32k BPE trained on the corpus merged Markdown rulers and runs of spaces into tokens of up to 512 bytes, which the stream's one-byte
token-length field (`tok_len.u8`) cannot hold, so the first stream died ("token byte lengths must lie in 1..255, got 1..512").
`tokenizer_train.py` now caps a token at 255 bytes (`--max-token-bytes`); the retrained tokenizer's longest token is 197 bytes (the stream's longest emitted token 197)
(the code tokenizer's is 80) and the bad one is kept beside it as `bpe32k.bad512.json`. The stream: 540,835,604 tokens over the 1.84 GB,
3.41 bytes a token (the code corpus's 3.4 again; prose and code tokenize alike here because the merges are byte-level and the
Stack Exchange share is half code), train_end_tok 485,712,500.

Timings and sizes (the Ryzen under the game): corpus 15 s, tokenizer 73 s, stream 143 s, keys 538 s, counters 156 s, byte assoc tables
1,622 s, token tables 3,139 s; byte `feat_all28` 38 GB + `feat_counters` 25 GB on `/games`, token `feat28` 31 GB on `/vault`
(`corpora/talk-tok` is a symlink to it), 421 GB free on `/vault`, 212 GB on `/games`. The pair at the hour on it (`talk_hour_gpu.sh`, unit
`talk-hour-gpu1`: `th_bytes_r`, `th_tok_r` at equal steps, `th_tok_eq_r` at equal compute, each scored on the mix's dev, his turns and
the dialogue) took GPU 1 at 01:13.

### The pair at the hour on the prose corpus (2026-10-18 01:13 to 06:11, GPU 1, `talk_hour_gpu.sh`, unit `talk-hour-gpu1`)

The hour recipe on `talk.bytes` with its own 2^28 tables, three arms, each scored cold (warm) on three sets: the mix's dev windows
(1,024 held-out documents' windows at dev_offset: the corpus's own bar, half Stack Exchange, a quarter code), Kyle's turns (340) and the
dialogue (2,047). Runs `th_*_r.json` in the tok worktree, scores `talk/k_th_*.json`, the lines in `talk/hour.txt`.

| arm | steps | clock | the mix's dev | his turns | the dialogue |
|---|---|---|---|---|---|
| bytes (`th_bytes_r`) | 25,562 | 3,721 s | 0.9828 (0.9287) | 1.8026 (1.7057) | 1.7718 (1.7005) |
| tokens at equal steps (`th_tok_r`) | 25,562 | 9,813 s | **0.9426** (0.8784) | **1.7910** (1.6002) | **1.7656** (1.6670) |
| tokens at equal compute (`th_tok_eq_r`) | 9,692 | 3,707 s | 1.0470 (0.9827) | 1.9008 (1.7056) | 1.8584 (1.7644) |

Three things. **The build reproduces the mix**: the byte arm reads his turns at 1.8026 and the dialogue at 1.7718 where `talk_mixD_r`
(the same recipe on the mix before the dedup and the tail region) read 1.8019 / 1.7714; the 6,263 dropped copies and the tables'
coverage of the tail changed nothing at the hour. **Tokens pay on prose.** On the code corpus at the hour tokens lost by 0.149 at equal
compute and by 0.022 at equal steps (clean dev); on the prose corpus they lose by 0.064 at equal compute and *win* by 0.040 at equal steps
on the mix's dev, a swing of 0.06 per step in tokens' favour, and the step ratio is the same (2.64x here, 2.54x on code: the 32k output
layer's cost does not depend on the text). On his two sets the cold picture is the same in shape and smaller (equal steps −0.012 / −0.006,
equal compute +0.098 / +0.087). **Warm, tokens read his turns much better than bytes.** With 2,048 bytes of context the equal-steps token
arm reads his turns at 1.600 against the byte arm's 1.706 (−0.106; the dialogue −0.034), and the equal-compute token arm, 0.098 behind
cold, *ties* bytes warm (1.7056 vs 1.7057). The byte reader's window is 2 KB of text and the warm context fills it; the token reader's is
2,048 tokens, 7 KB, so the window and its context both fit, and a chat is the case where the context (the turns before) is worth the most.
This is the context-length argument for tokens made on his own text, and it is the first set where the token reader is ahead of the byte
reader at equal compute by any measure.

Where this leaves R118 (both ways as we scale): on code the gap closed with depth (0.149 at the hour, 0.061 at 12 layers and 4x) without
crossing; on prose it starts at 0.064 and the equal-steps arm is already ahead, so the 12-layer point should cross. That point runs next
(`talk_l12_gpu.sh`, unit `talk-l12-gpu1`, GPU 1 from 06:30: `tl12_bytes_r` at 45,237 steps, `tl12_tok_eqc_r` at 25,518 (equal compute,
the code point's calibrated steps: the per-step work is the same model with the same inputs, clocks logged), `tl12_tok_eqs_r` at 45,237
(equal steps, 1.77x the compute); about 17 hours, the lines in `talk/l12.txt`).

## The weights' pile: the parameters' jobs, abstracted (2026-10-18, `notes/model2/checks/weight_pile.py`, `checks/talk/`)

Kyle, after the corpus and the pair: "we are just trying to make things do too much at once", then "what jobs do the parameters do that
we can pull out and substitute something else with? like for the tables it was pulling memory out". Tokens, the mixes, the 370M plan
and the ratchet were parked (the ratchet buys memory, not speed: its forward dequantises; a lever for a 1B model, not this one), and the
12-layer pair on the prose corpus was stopped after its byte arm (`tl12_bytes_r`, 45,237 steps: the mix's dev 0.8871, Kyle's turns
1.6911, the dialogue 1.6535; against the hour's 0.9828 / 1.8026 / 1.7718, four times the budget and three times the depth bought
0.096 / 0.112 / 0.118; the token arms wait at their checkpoints).

### The pile

`dump_window.py` scores every byte of a window with the reader's memories on and with each withdrawn; `weight_pile.py` credits a byte
to the memory whose withdrawal costs it the most (at least 0.25 bit), else to the weights, and sorts the weights' bytes by kind. The
hour reader `th_bytes_r` on 16 windows of Kyle's turns, 16 of the dialogue, 8 of the mix's dev (`results/tables/talk/weight_pile.txt`;
counts only, the dumps of his text live under `corpora/kyle/dumps`, mode 0600). Kyle's turns:

| kind of byte | share of bytes | share of bits | mean bits | the weights' share of them, their mean, top-3 right |
|---|---|---|---|---|
| the first letter of a word | 17 % | 37 % | 3.75 | a third, 4.57, 34 % |
| a later letter of a word | 55 % | 40 % | 1.30 | a third, 2.14, 81 % |
| punctuation and digits | 10 % | 15 % | 2.6 / 3.0 | half, 4.9 / 4.5 |
| spaces and newlines | 17 % | 7 % | 0.64 / 1.39 | solved |

The table is credited with 60 % of the bytes and the weights with 31 %, and the two halves cost about the same bits. Three readings. The
job is choosing the next word: a first letter costs 3.75 bits on average, 4.6 where the weights own it with the truth in the top three
a third of the time, and 3.3 even where the table is credited: the book narrows the choice, it does not make it. Spelling is in the
book: once a word has started the table carries it, and where the weights take over they are near (top three 81 %). And the book is in
the wrong dialect: the table's bytes cost 1.50 on his turns against 0.85 on the corpus's own dev and on code (`l4b`, the kernel hour
reader, whose weights' bytes cost 0.45 with 93 % under a bit: on code the weights' job is small and done; on prose it is large and
undone). The abstractions that followed, in the order the pile ranks them: the right book (his own text in the memory), a dictionary
(a context-free key on the current word's letters), the live vocabulary (the words seen so far in the window), a skeleton at prose
scale (the template keys 116/132, which replace identifiers with a placeholder, were already among the eighteen inputs; the new ones
are 4 and 8 symbols). Kyle: "test all 4".

### The four tests

Each against `th_bytes_r` (the hour recipe, ALL18, the match 8/16/32), cold on Kyle's turns / the dialogue / the mix's dev; the
details and the lines in `results/tables/talk/jobs.txt` and `book.log`.

| test | what | his turns | the dialogue | the mix's dev |
|---|---|---|---|---|
| the right book, his files | 16.7 MB of his notes, docs, scripts and DONE.md indexed into a copy of the saved tables (`index_project.py`), read by the unchanged reader through `sample_tables.py`: **no training** | **−0.049** | **−0.044** | — |
| the right book, his world | the files plus the assistant replies of the 572 sessions outside the dialogue set, 19.7 MB | −0.118 | −0.086 | — |
| the dictionary | keys 200 (the current partial identifier alone) and 201 (with the last token), trained in | −0.012 | −0.012 | −0.009 |
| the live vocabulary | in-window matches 900 (the word prefix) and 902 (the last two words + the prefix) beside the match 8/16/32 | −0.011 (warm −0.015) | −0.009 | −0.005 |
| the skeleton at prose scale | the template keys 104 and 108 (4 / 8 symbols, identifiers replaced) | −0.007 | −0.011 | −0.008 |

The book is the result. Every one of Kyle's turns is in the held-out turns set, so none can go into the memory; the world was built
from the other half of his world (`kyle_world.py`, run by Kyle: the classifier keeps his stores his), with every file holding a 64-byte
span of the held-out sets dropped (347 of them: the dialogue sessions' replies and ~/TODO.md). The reader read the held-out windows
through the saved tables alone (1.8040 on his turns, within 0.0014 of the stored features: the use layer's path is exact) and through
the same tables with the world written in. The world's 0.118 on his turns is not claimable: the replies answer the very turns being
scored, a leak of topic if not of text (the dialogue's 0.086 is clean, its sessions were excluded whole). The files-only book is the
honest number, 0.049 on his turns at zero training, five times either key family and twenty-five times the whole mix search; on the
dialogue the files and the replies are each worth about 0.04. The three key families each bought a hundredth: real (the hour's seed
noise is 0.005 and the three sets agree) and not levers. The dictionary did what a lexicon should, spelling 1.30 to 1.28 a letter on his
turns and the first letter unchanged at 3.75 (`weight_pile_dict.txt`); the live vocabulary takes bytes from both the table and the
weights without moving either kind's mean (`weight_pile_vocab.txt`). **What is in the book matters; how it is indexed hardly does.**

### The wall and the leads

The memory covers recall, a fifth of the bits on his text; the other four fifths are the next-word choice, and no key for the same text
moves them. Leads ranked by expected gain per day: his world in the weights (a fine-tune on the 16.7 MB; register alone was worth 0.26
between Stack Exchange and code); a language model as the reader with the lab's memory as its world (Qwen on his sets with and without
the index's text in front of it: the one lead that can halve the number, and the test of whether the memory is worth anything to a
model that can already talk); a longer window (warm, tokens read his turns 0.1 better than bytes because their window held the earlier
turns); the reply side in the mix; distillation; scale. Then Kyle's autocomplete idea: the model only has to decide the word, most of
the compute goes to finishing words (55 % of the bytes, 81 % solved); two smaller models, a decider per word and a speller per letter: the
Byte Latent Transformer's shape and the old word+character LMs'. The token arm already measured the decider's per-step edge (0.04 on the
corpus, 0.1 warm on his turns); the unknown is whether the speller can be shallow. Queued (R127, `talk_leads_gpu.sh`, unit
`talk-leads-gpu1`, lines in `talk/leads.txt`): `th_l0_r` and `th_l1_r` (0 and 1 layers, dumped by kind: what depth buys at word starts
against inside words), `th_w4k_r` (a 4,096-byte window at the same bytes a step, compared on the last 1,024 bytes of 4,096-byte windows of
his turns), `th_world_r` (`th_bytes_r`'s weights fine-tuned 2,000 steps on the world with inputs through the corpus tables alone,
`world_features.py`; the runner's new `--init-from`).

**The probes and the leads, in (2026-10-19 00:06; `talk/leads.txt`, `weight_pile_thl0.txt`, `_thl1.txt`, `_thwo.txt`).** The split is
dead: with no network a later letter of a word costs 1.87 and a first letter 4.39 (against 1.30 / 3.75 at four layers), with one layer
1.55 / 4.06, so the depth from one layer to four earns 0.14 bits a byte inside words and 0.05 at word starts. Finishing a word needs the
same context as choosing it; a speller cannot be shallow unless it inherits the decider's state, which is the Byte Latent Transformer
proper, a two-to-three day build for an uncertain gain. Not built; the autocomplete as a product shape stands. Context is a lever: the
4,096-byte reader (`th_w4k_r`, the same bytes a step) reads the last 1,024 bytes of 4,096-byte windows of his turns at 1.480 against the
2 KB reader's 1.589 with 1,024 bytes of context (−0.109), the same tenth the token arm showed warm; on the standard 2 KB windows it is
0.015 worse. The register lands in the weights, but only the register that is in the text: `th_bytes_r` fine-tuned 2,000 steps on his
world's files (five minutes, lr 1e-4, inputs through the tables alone) reads the dialogue 0.066 better and his turns no better (the
files are in the assistant's voice, not his), and the corpus 0.095 worse (no replay). With replay (`th_world2_r`: the world followed by a 32 MB slice of the corpus's train
region, 3,000 steps, seven minutes) his turns read 0.037 better cold and the dialogue 0.063, the corpus 0.008 worse: the knowledge his
turns need is kept and the register is added, a lever the size of the book and stackable with it (weights against memory) and with
the long window (context). Next: the three combined, at use time.

### The three levers combined (2026-10-19 06:49, `talk_combo_gpu.sh`, `talk/combo.txt`)

The 4 KB reader fine-tuned with replay on his world (`th_combo_r`, 3,000 steps at lr 1e-4) and read through the tables with his files
indexed in; every model on the same bytes, the last 1,024 bytes of each 4,096-byte window of his turns (170 windows) and of every eighth
of the dialogue's (128), the held-out file's inputs computed through the saved tables alone and through the book by the use layer's path:

| on the last 1,024 bytes of 4 KB windows | his turns | the dialogue |
|---|---|---|
| the hour reader, 1 KB of context | 1.7477 | 1.7146 |
| the 4 KB reader, 3 KB of context | 1.6542 (−0.094) | 1.6708 (−0.044) |
| + the book | 1.6251 (−0.123) | 1.6379 (−0.077) |
| + the fine-tune instead | 1.6310 (−0.117) | 1.6206 (−0.094) |
| **all three** | **1.6108 (−0.137)** | **1.5995 (−0.115)** |

They stack at about three quarters of additive. Context, memory and weights are three places, and each lever lives in its own: 0.137
bits a byte on his turns, 8 % of the bits, for an hour arm, seven minutes of fine-tuning and an index built on the CPU, against the
0.112 that four times the budget and three times the depth bought. On the standard 2 KB windows `th_combo_r` reads 1.0013 / 1.7738 /
1.7222 (the mix's dev / his turns / the dialogue).
