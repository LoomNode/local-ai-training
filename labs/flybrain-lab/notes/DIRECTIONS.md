# Step back: what failed, what to try

Date: 2026-09-12. Not a training run. Evidence is `notebook.md` Runs 0–7 plus a **no-train** lag probe in this file.

V0’s hypothesis was: a fly-like *organizational prior* (sparse modular graph, typed GRUs, E/I signs, sensory/readout bottlenecks) should beat degree-matched shuffle at next-byte prediction, and compete with a matched-budget GRU. After seven locked WikiText slices at N=128, **that hypothesis has not been tested**, because the graph is barely on the path from current byte to logits.

## What the numbers actually say

Locked protocol: N=128, mean head, d=4, 200 Adam steps, WikiText-2 first 80k bytes (72k/8k), unigram **4.564**, bigram **3.729**. Scaling gate was ~4.3.

| What we turned | Fly val | Shuffle val | vs unigram | Fly≈shuffle? |
|----------------|--------:|------------:|:-----------|:-------------|
| Flatten head (77% decoder) | 4.522 | **4.471** | slight | shuffle *better* |
| State dim 1 / 4 / 8 | 4.69 / 4.60 / 4.59 | overlay | no | yes |
| Message scalar / 2 / 4 / projected | 4.600–4.601 | overlay | no | yes |
| Sensory 10 / 12.5 / 25 / 50% | 4.599–4.591 | overlay | no | yes |
| Cell universal / type / unique | 4.600 / 4.600 / 4.596 | overlay | no | yes |
| Decode sensory (Run 8) | 4.601 | **4.421** | shuffle only | no |
| Same-step skip (Run 9) | 4.594 | 4.601 | no | yes |
| Ordered hops (Run 10) | 4.597 | 4.597 | no | yes |
| Flatten + ordered WikiText (Run 13) | **3.586** | **3.621** dest. | **yes** (≪ 4.564) | nearly (fly −0.035) |
| Region-preserving shuffle (Run 14) | 3.600 | **3.577** region | **yes** | **yes, shuffle ≥ fly** |

Matched GRU *does* beat unigram when `numel()` lands on a decent shape (4.08 at 6.2k, 3.50 flatten). So the budget and the corpus can learn sequence. The graph arms converge to the unigram table (kiss 4.56 around step 150, bounce to 4.60). That is not “needs more steps.”

V1–V3 are plumbing. They do not bear on bits/byte.

## Ruled out (on *this* motif graph, *this* I/O, *this* 1-pass-per-token schedule)

These are the same experiment in costume. Each keeps disjoint sensory/readout, one message-pass per byte, and decode from readout.

- Scalar synapses as the sole bottleneck (Run 4).
- Too few sensory neurons (Run 5).
- θ too shared / too unique / not typed (Runs 6–7).
- Mean vs flatten as the *graph* story (flatten’s win is the 16 640-scalar decoder; shuffle still wins).
- “Just scale N” as the next move. More copies of a 2-hop lag still cannot see the current byte.
- FlyWire / BANC. A real olfactory graph is longer-path, not shorter.

Not ruled out: every other neuron family, every other task, every other I/O schedule.

## The failure that the knobs never touched

At N=128 seed 0:

| Fact | Fly | Shuffle |
|------|-----|---------|
| `sensory ∩ readout` | empty | empty |
| Sensory in-degree | **0** | (shuffled) |
| Sensory out | 48 edges, **all → Kenyon** | 12 of them land on readout |
| Min hops sensory → readout | **2** (15/16 readout at 2, one at 3) | **1** (10/16 at 1) |
| Self-loops | 0 | 0 |
| Message-passes per token | 1 | 1 |

`step()` builds messages from **old** \(H\), adds the current byte only onto `sensory_idx`, then GRU-updates everyone in parallel, then reads `readout_idx`. Sensory neurons have no incoming synapses, so they *are* a recurrent encoder of the current byte (GRU hidden-to-hidden). Nobody reads them.

Consequence for next-byte CE at time \(t\) (predict \(x_{t+1}\) from \(x_{\le t}\)):

| Population after `step(x_t)` | Earliest byte it can contain |
|------------------------------|------------------------------|
| Sensory | \(x_t\) (same step) |
| Kenyon | \(x_{t-1}\) |
| Readout (fly) | \(x_{t-2}\) |
| Readout (shuffle, lucky edges) | \(x_{t-1}\) |

A bigram needs \(x_t\). Fly logits are at best a delayed trigram-or-worse, which on this corpus is numerically unigram. Shuffle’s accidental sensory→readout edges are why flatten shuffle beat fly — the control is **not** “same I/O, rewired motifs.” Destination shuffle *creates* the shortcut the fly motif forbids.

### No-train lag probe (2026-09-12)

Untrained `FlyLanguageModel`, seq len 8, RMS change in a population at the last step when one earlier token is flipped.

| lag | fly d(sensory) | fly d(kenyon) | fly d(readout) | shuffle d(readout) |
|----:|---------------:|--------------:|---------------:|-------------------:|
| 0 (current) | 0.145 | **0** | **0** | **0** |
| 1 | 0.052 | 0.0009 | **0** | 0.0015 |
| 2 | 0.029 | 0.0007 | 0.00001 | 0.0021 |

Zeros are structural (no path), not small random weights. Training cannot put \(x_t\) in fly readout without changing I/O or the inner schedule. The 0.00001 at lag 2 is why 200 steps can kiss unigram and then bounce: a vanishing delayed cue, then noise.

**Claim:** Runs 2–7 did not test topology-as-prior. They tested “1-pass bottlenecked graph whose decoder cannot see the current byte.”  
**Not a claim:** fly wiring is useless; GRUCells cannot do language; N=128 is too small.

## Why V3 and plasticity are the wrong next page

The sandbox loop decodes `'�'` because the LM is a unigram. Growing edges (V1) or slots (V2) does not create a same-step path from byte to readout. Do not spend cycles on playbook terminals until logits move with \(x_t\).

## Ranked tests

Keep N=128, no FlyWire, honest `numel()`, sparse COO, three arms, same WikiText protocol **unless a row names a different task**. Stop a line when it beats unigram *and* shows a fly/shuffle split; that is the first time the original hypothesis is on the table.

### Tier 0 — cheap diagnosis (do these before another 200-step WikiText knob)

| # | Test | What it asks | How | Cheap because | Pass / fail |
|---|------|--------------|-----|---------------|-------------|
| 0a | **Read the sensory population** | Is the byte even in the net? | `--decode-pop sensory`; graph unchanged. **Done** Run 8 (`wiki_mean_d4_read_sensory.json`): fly **4.601** (unigram), shuffle **4.421**, GRU **3.863**. | One flag, no new graph | Fly still ~4.56 ⇒ lag was **not sufficient**. Shuffle under unigram ⇒ decoded pop needs **byte + mixing**. Next is 1a/1c, not another cell knob. |
| 0b | **Trained lag probe** | Did training amplify the lag-2 path? | Save a checkpoint from 0a or a 50-step fly; repeat the token-flip table. | Seconds | If readout lag-0 stays 0, architecture is unchanged. If lag-2 grows and val still ~4.56, delayed cue is too weak for WikiText. |
| 0c | **Synthetic bigram** | Can *any* path learn P(next\|current) when the data is only that? | Toy corpus with a planted bigram, vocab 16, 200 steps. Arms: fly, shuffle, GRU, plus 0a. | Tiny | If GRU and 0a succeed and fly-readout fails, the lag diagnosis is closed. |

Do **not** start 0a and a new message-kind in the same slice.

### Tier 1 — put the graph on the gradient path (still WikiText, still N=128)

These are different hypotheses. Run **one** at a time. Prefer 1a or 1c first; they are the smallest diffs.

| # | Test | What it asks | Implementation sketch | What would change our mind |
|---|------|--------------|----------------------|----------------------------|
| 1a | **Same-step sensory inject into readout** | HANDOFF skip: is the graph usable once \(x_t\) is visible? | `--skip sensory_readout`. **Done** Run 9 (`wiki_mean_d4_skip.json`): fly **4.594**, shuffle **4.601**. Lag-0 exists; skip |embed| ~4× |graph incoming|, so mixing is drowned. | Fly still ~4.56. Byte-on-path is not enough under scalar-sum + mean. Next is 1c (ordered hops, no embed-on-readout), not 1d. |
| 1b | **K inner graph steps per token** | Motifs need hops *inside* a byte, not across bytes. | For each token: inject once, then `K ∈ {2,3,4}` `step`s without a new byte (or a `propagate()` that does not re-inject). Decode after K. Report K and extra BPTT depth. | Fly at K=2–3 beats unigram; K=1 (current) does not. Cost is K× compute; stay at N=128. |
| 1c | **Ordered update** | Parallel Jacobi update is the delay. | `--update ordered`. **Done** Run 10 (`wiki_mean_d4_ordered.json`): fly/shuffle **4.597**, GRU **4.169**. Lag-0 through the graph is real; WikiText still unigram. | Path exists; mean-head still ~4.56. Next is **2a** or a wider readout of the 16 neurons, not 1d / FlyWire / nftechie/flm. |
| 1d | **Region-preserving / type-aware shuffle** | Current shuffle is a bad control. | `--shuffle region`. **Done** Run 14 (`wiki_flatten_d4_ordered_region.json`): fly **3.600**, region-shuffle **3.577**, GRU **3.460**. | Motifs do not beat a fair rewire on this slice. Do not scale N. |

1a preserves the V0 bottleneck story (bytes still do not broadcast to all N). 0a does not (decoder reads the input population). If the scientific claim is “bottlenecked graph,” 1a/1b/1c are the real tests; 0a is only a stethoscope.

### Tier 2 — is WikiText the wrong first task?

Olfactory expansion (Kenyon) is a pattern-separation prior, not a language prior. Even with I/O fixed, next-byte WikiText may hide a weak prior in unigram noise.

| # | Test | Why | Gate |
|---|------|-----|------|
| 2a | **Copy / delayed identity** | Cue then τ silence. **Done** Run 11 mean (`delay_n128_ordered.json`): fly chance at τ=0. **Done** Run 12 flatten (`delay_n128_ordered_flatten.json`): fly/shuffle/GRU **1.0** at τ=0..4. | Mean was the copy bug. Flatten copies; topology still a wash. Next language test: WikiText flatten+ordered. |
| 2b | **Sparse binding / odd-one-out** | Run 16 unplanted 3-claw **negative**. **Run 17** unique-PN 4-claw AND: fly **0.219/0.210** > shuffle **0.188/0.181**. | Duplicate claws can no longer fake coincidence. |
| 2c | **Character vocab 27** | 256-way unigram is a huge floor; letters+space make sequence stats louder. | Same protocol, different corpus. Do not treat as WikiText. |

Run 2a if Tier 1 WikiText still sits on unigram after a same-step path exists. That isolates “cannot learn English” from “cannot learn a delay the graph can represent.”

### Tier 3 — neuron / synapse family (only after a path exists)

Do not re-run these on the current 2-hop lag; they would reproduce Runs 4–7.

- Residual update \(H \leftarrow H + \mathrm{GRU}(\cdot)\) (deep-graph vanishing).
- Drop sign-lock (allow \(p\) to change sign) as an ablation, not as biology.
- LSTMCell / linear RNN / LRU at matched `numel()`.
- Attention or weighted readout over the readout *population* (mean may cancel the few neurons that have the cue). Flatten already hinted this, but flatten also exploded the decoder — use a 16-d attention mix into `Linear(4,256)`, not `Linear(64,256)`.
- Longer train / lr sweep (0.001 vs 0.01). Only after 1a/1c: current 0.01 already *arrived* at unigram by step 50.

### Tier 4 — gated (do not open)

| Gate | Closed until |
|------|----------------|
| N ≈ 512 then 1k | N=128 graph < ~4.3 **and** fly ≠ shuffle. Run 13/14 meet the bits gate; region-shuffle ≥ fly, so N stays closed. |
| FlyWire / BANC subset | That, plus synthetic motifs boringly solid (loader already exists) |
| V1 prune/grow as science | Graph on-path and a metric that could move with rewiring (2a/2b, not unigram WikiText) |
| V2 extra slots as science | Same |
| V3 as an agent | Decode is not `'�'`; sandbox stays allowlist, not bash |
| Dense \(N\times N\) | Never; sparsity is the prior |

## Suggested order (one slice per session)

1. **0a** sensory-mean readout. **Done (Run 8):** fly 4.601; shuffle 4.421.  
2. **1a** same-step skip. **Done (Run 9):** fly 4.594 / shuffle 4.601; skip drowns graph.  
3. **1c** ordered update. **Done (Run 10):** fly/shuffle 4.597. Same-token motif path is not enough for WikiText mean-head.  
4. **2a** delayed identity. **Done (Run 11 mean chance; Run 12 flatten 1.0).**  
5. WikiText flatten + ordered. **Done (Run 13):** fly **3.586** vs unigram 4.564.  
6. **1d** type-aware shuffle on flatten+ordered WikiText. **Done (Run 14):** shuffle 3.577 ≥ fly 3.600.  
7. **2b** sparse binding flatten. **Done (Run 15).** Small-head MB **Done (Run 16):** fly > shuffle on two seeds.  
8. Stolen block as a recurrent LM layer (no connectome). **Done (Run 19):** mixer 3.36 vs dense 3.28 vs GRU 3.16 at 600 steps; 34% fewer FLOPs, not better bits.  
9. [nftechie/flm](FLM.md) is a different program (frozen 1.2B + frozen MaleCNS). Do not start it as a V0 run.

JSON naming if/when run: `wiki_mean_d4_read_sensory.json`, `wiki_mean_d4_skip.json`, `wiki_mean_d4_ksteps.json`, `wiki_mean_d4_ordered.json`. Record in `notebook.md`; do not treat this file as a results table.

## Anti-patterns for the next coding agent

- Another cell-kind or message-kind at N=128 with **mean** readout.
- Scaling N “to give hops more room” while fly≈shuffle.
- Calling flatten 4.522 a fly win; Run 13 3.586 is the unigram beat, still not a motif win.
- Reading GRU val gaps as architecture when `matched_gru` re-sized the baseline.
- Destination-shuffle as a topology control while it adds sensory→readout edges the fly graph lacks.
- V3 playbook files as evidence of a coding brain.
