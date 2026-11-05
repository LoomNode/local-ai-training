# flybrain-lab notebook

Sticky across sessions. Raw dumps live in [`results/`](results/). This file is the **interpreted record**: protocol, numbers, what we will and will not claim.

Handoff: Fly Brain Text Generation (private ChatGPT conversation).  
Sibling delayed-identity lab: `../biointel-lab` (not part of this repository).

## Protocol (locked unless a row says otherwise)

| Knob | Value |
|------|--------|
| Graph | `synthetic_flylike` (motifs, not FlyWire), seed 0; shuffle seed 1 |
| N | 128 unless named tiny |
| State dim | 4 unless swept |
| Synapse (V0) | sign-locked \(w=\mathrm{sign}\cdot\mathrm{softplus}(p)\); message kind swept (scalar / vector2 / vector / projected) |
| Train | Adam lr=0.01, 200 steps, seq=64, batch=8, CPU |
| Corpus | first 80 000 bytes of `/games/ailab/lab/scrt-lab/data/wiki.train.raw` |
| Split | prefix 90 % train (72 000) / suffix 10 % val (8 000) |
| Metric | held-out bits/byte = CE nats / ln 2, 4 val batches |
| Controls | degree-matched shuffle; unique-weight GRU matched on `numel()` |

Floors on that exact split (add-one MLE):

| Predictor | val bpb |
|-----------|--------:|
| uniform (8 bit) | 8.000 |
| unigram | **4.564** |
| bigram | **3.729** |

A model that cannot beat **4.564** is not using sequential structure.

Reproduce:

```bash
cd /games/ailab/lab/flybrain-lab
uv sync --extra dev
uv run pytest -q
uv run python scripts/run_tiny.py
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --seq-len 64 --batch 8 --train-bytes 80000
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --readout mean --state-dim {1,4,8} --run-name wiki_mean_d{1,4,8}
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --readout mean --state-dim 4 --message vector --run-name wiki_mean_d4_vector
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --seq-len 64 --batch 8 --train-bytes 80000 --readout mean --state-dim 4 --message vector2 --run-name wiki_mean_d4_vector2
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --seq-len 64 --batch 8 --train-bytes 80000 --readout mean --state-dim 4 --message projected --run-name wiki_mean_d4_projected
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --seq-len 64 --batch 8 --train-bytes 80000 --readout mean --state-dim 4 --n-sensory {13,32,64} --run-name wiki_mean_d4_sens{13,32,64}
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --seq-len 64 --batch 8 --train-bytes 80000 --readout mean --state-dim 4 --cell unique --run-name wiki_mean_d4_unique
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --seq-len 64 --batch 8 --train-bytes 80000 --readout mean --state-dim 4 --cell universal --run-name wiki_mean_d4_universal
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --seq-len 64 --batch 8 --train-bytes 80000 --readout mean --state-dim 4 --decode-pop sensory --run-name wiki_mean_d4_read_sensory
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --seq-len 64 --batch 8 --train-bytes 80000 --readout mean --state-dim 4 --skip sensory_readout --run-name wiki_mean_d4_skip
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --seq-len 64 --batch 8 --train-bytes 80000 --readout mean --state-dim 4 --update ordered --run-name wiki_mean_d4_ordered
uv run python scripts/run_delay.py --device cpu --n 128 --steps 200 --batch 32 --update ordered --delays 0,1,2,4 --n-classes 16 --run-name delay_n128_ordered
uv run python scripts/run_delay.py --device cpu --n 128 --steps 200 --batch 32 --readout flatten --update ordered --delays 0,1,2,4 --n-classes 16 --run-name delay_n128_ordered_flatten
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --seq-len 64 --batch 8 --train-bytes 80000 --readout flatten --state-dim 4 --update ordered --run-name wiki_flatten_d4_ordered
uv run python scripts/run_eval.py --device cpu --n 128 --steps 200 --seq-len 64 --batch 8 --train-bytes 80000 --readout flatten --state-dim 4 --update ordered --shuffle region --run-name wiki_flatten_d4_ordered_region
uv run python scripts/run_binding.py --device cpu --n 128 --steps 200 --batch 32 --readout flatten --update ordered --shuffle region --n-classes 16 --n-features 16 --k 3 --run-name bind_n128_ordered_region
uv run python scripts/run_binding.py --device cpu --n 128 --steps 400 --batch 32 --lr 0.05 --readout emit --update ordered --shuffle region --inject local --n-classes 16 --n-features 16 --k 3 --n-distractors 5 --n-readout 16 --n-kenyon 80 --claws 3 --kenyon-top-k 20 --kenyon-theta 2.5 --seed 0 --run-name bind_mb_n128_trained
uv run python scripts/run_binding.py --device cpu --n 128 --steps 400 --batch 32 --lr 0.05 --readout emit --update ordered --shuffle region --inject local --n-classes 16 --n-features 16 --k 3 --n-distractors 5 --n-readout 16 --n-kenyon 80 --claws 3 --kenyon-top-k 20 --kenyon-theta 2.5 --seed 1 --run-name bind_mb_n128_trained_s1
uv run python scripts/run_binding.py --device cpu --n 128 --steps 400 --batch 32 --lr 0.05 --readout emit --update ordered --shuffle region --inject local --n-classes 16 --n-features 12 --k 3 --n-distractors 5 --n-sensory 12 --n-readout 16 --n-kenyon 80 --claws 4 --kenyon-top-k 8 --kenyon-theta 3.5 --seed 0 --run-name bind_mb_n128_uniqueand
uv run python scripts/run_binding.py --device cpu --n 128 --steps 400 --batch 32 --lr 0.05 --readout emit --update ordered --shuffle region --inject local --n-classes 16 --n-features 12 --k 3 --n-distractors 5 --n-sensory 12 --n-readout 16 --n-kenyon 80 --claws 4 --kenyon-top-k 8 --kenyon-theta 3.5 --seed 1 --run-name bind_mb_n128_uniqueand_s1
uv run python scripts/run_mixer.py --seed 0 --run-name mixer_bind_s0
uv run python scripts/run_mixer.py --seed 1 --run-name mixer_bind_s1
uv run python scripts/run_mixer.py --n-features 48 --n-classes 32 --k 4 --n-distractors 8 --n-expand 256 --claws 5 --top-k 16 --n-read 32 --theta 2.5 --seed 0 --run-name mixer_bind_highd_s0
uv run python scripts/run_mixer.py --n-features 48 --n-classes 32 --k 4 --n-distractors 8 --n-expand 256 --claws 5 --top-k 16 --n-read 32 --theta 2.5 --seed 1 --run-name mixer_bind_highd_s1
```

---

## Run 0 — plumbing (`results/tiny_cpu.json`)

Repeating sentence `"the cat sat on the mat…"`, **train loss only**, N=64, 40 steps, seq=32.

| Arm | scalars | state | train bpb | greedy |
|-----|--------:|------:|----------:|--------|
| fly | 11087 | 256 | 8.20 → 3.21 | `"the cat                         "` |
| shuffle | 11087 | 256 | 8.05 → 3.23 | same space-collapse |
| gru | 11100 | 26 | 8.02 → 1.97 | `"the cat t t t t …"` |

**Claim:** the loop `bytes → graph → loss → backward` runs.  
**Not a claim:** topology, language, or efficiency.

---

## Run 1 — WikiText flatten head (`results/wiki_cpu_n128.json`)

V0 spec readout: `Linear(n_readout × 4, 256)`. N=128, 200 steps.

| Arm | val bpb | train bpb | scalars | decoder | synapse | program |
|-----|--------:|----------:|--------:|--------:|--------:|--------:|
| fly | 4.522 | 4.442 | 21566 | 16640 | 478 | 352 |
| shuffle | 4.471 | 4.382 | 21566 | 16640 | 478 | 352 |
| gru | **3.502** | 3.534 | 21568 | — | — | — |

Fly plateaued by step ~50 (val 4.52 and stayed). Greedy fly: `"The   ae…"`. GRU: `"The and the and the…"`.

**Claim:** flatten V0 is **77 % decoder**. Fly ≈ unigram. Shuffle slightly *better* than fly (noise). GRU beats bigram.  
**Not a claim:** fly wiring helps language.

Also logged in the same JSON: mean-head three-arm at d=4 (see Run 2).

---

## Run 2 — mean-pool head, state-dim sweep

`Linear(state_dim, 256)` so the graph, not the head, holds the budget.  
JSON: `results/wiki_mean_d{1,4,8}.json`.

| d | fly val | shuffle val | gru val | fly scalars |
|--:|--------:|------------:|--------:|------------:|
| 1 | 4.692 | 4.715 | 4.292 | 5138 |
| 4 | 4.600 | 4.600 | **4.080** | 6206 |
| 8 | 4.591 | 4.590 | 4.432 | 7966 |

Fly/shuffle never beat unigram **4.564**. 1→4→8 is a ~0.1 bit bump, still above the table. Topology is a wash. GRU at 6.2k beats unigram but not bigram. GRU at 8.0k is *worse* than at 6.2k: `matched_gru` minimizes `|numel−fly|`, not quality — do not read that row as “wider GRU fails.”

Greedy fly is spaces. GRU d=4: `"The the the the…"`.

**Claim:** scalar-synapse type-GRUCells on this synthetic graph are unigram-ish predictors. Extra state dimensions and extra neurons-of-this-kind are the wrong next knob.  
**Not a claim:** biological topology is useless in general.

---

## Run 3 — vector messages (`results/wiki_mean_d4_vector.json`)

Same protocol as Run 2 d=4, but each synapse carries the full 4-d state. Still **one** trainable \(p_{ij}\) per edge. Type cell is `GRUCell(4,4)`. Sensory is a rank-1 mix into all four channels.

| Arm | val bpb | train bpb | scalars |
|-----|--------:|----------:|--------:|
| fly | 4.601 | 4.547 | 6338 |
| shuffle | 4.601 | 4.546 | 6338 |
| gru | **4.070** | 4.075 | 6322 |

Fly val curve: 8.05 → 4.56 (step 150, unigram) → **4.60** (step 200). Shuffle overlays it. GRU still descending at 200 (8.15 → 4.07).

**Claim:** widening the *message* does not beat unigram and does not create a fly/shuffle split. The starved scalar channel was not the only problem.  
**Not a claim:** no possible fly-derived net can model bytes.

Tests after this change: **26 passed** (`uv run pytest -q`).

---

## Run 4 — message-width set (vector2 + projected 4-d synapses)

Same locked protocol as Run 2 d=4 (`--readout mean --n 128 --steps 200 --seq-len 64 --batch 8 --train-bytes 80000`). JSON: `results/wiki_mean_d4_vector2.json`, `results/wiki_mean_d4_projected.json`. Scalar and 4-d×scalar-\(p\) rows are reused from Runs 2–3.

| kind | fly val | shuffle val | gru val | fly scalars | notes |
|------|--------:|------------:|--------:|------------:|-------|
| scalar (Run 2) | 4.600 | 4.600 | **4.080** | 6206 | one emit→channel, one \(p\) per edge |
| vector2 | 4.601 | 4.602 | 4.553 | 6240 | \(w\odot h_{:2}\), one \(p\) per edge |
| vector (Run 3) | 4.601 | 4.601 | **4.070** | 6338 | \(w\odot h\), one \(p\) per edge |
| projected | 4.601 | 4.603 | **4.217** | 7772 | \(w\in\mathbb{R}^{4}\) per edge, sign broadcast |

Unigram **4.564**. Bigram **3.729**. Graph arms never go comfortably below unigram (scaling gate was ~4.3). Fly and shuffle overlay at 4.60 on every kind. Fly curves all kiss 4.56 around step 150 and bounce to 4.60 by 200.

GRU rows are **not** a message-kind effect: `matched_gru` re-sizes the baseline to fly `numel()`. Vector2’s 6234-scalar GRU is a worse shape (4.55, barely unigram); the 6322-scalar GRU at vector is 4.07. Do not read “2-d messages hurt the GRU.”

**Decision-tree branch:** richer messages **did not help**. Scalar communication was not the binding constraint on this motif graph + type-GRUCell. Topology still a wash. **Do not scale N.** Next branch in the tree (not this slice) is the sensory interface / fan-in, then the neuron-update abstraction if that also fails.

**Claim:** at N=128 mean-head, widening the synapse from 1 to 2 to 4 channels, and giving each edge its own 4-d sign-locked \(w\), leaves the graph at unigram.  
**Not a claim:** synapses never need more than a scalar; FlyWire would not change this; unique-per-neuron \(F\) is ruled out.

Tests after this change: **32 passed** (`uv run pytest -q` twice).

---

## What we are *not* documenting as science

- ChatGPT architecture essays (source of V0 constraints, not evidence).
- Samples that are spaces or `"the the the"`.
- CPU vs CUDA wall time at N=128 (Python timestep loop; CUDA was slower).

## Run 5 — sensory fraction (`results/wiki_mean_d4_sens{13,32,64}.json`)

Locked mean-head d=4 scalar protocol, N=128. Default V0 is `n_sensory = max(4, N//8) = 16` (Run 2). Sweep ~10% / 25% / 50% of N.

| n_sensory | frac | fly val | shuffle val | gru val | fly scalars |
|----------:|-----:|--------:|------------:|--------:|------------:|
| 13 | 10% | 4.599 | 4.598 | 4.323 | 5444 |
| 16 | 12.5% (Run 2) | 4.600 | 4.600 | **4.080** | 6206 |
| 32 | 25% | 4.599 | 4.600 | 4.239 | 10270 |
| 64 | 50% | 4.591 | 4.599 | **3.604** | 18398 |

Unigram **4.564**. Graph arms stay at 4.59–4.60. The 50% fly row is 0.009 above unigram — not a sequence-learning win, not a fly/shuffle split. GRU improves at 64 sensory because the embedding is `256×64` and `matched_gru` copies that budget; that is not a sensory-fraction effect on the graph.

**Decision-tree branch:** sensory bandwidth **did not help** the graph. Message width (Run 4) and input fraction (Run 5) both leave fly≈shuffle≈unigram. Next is the **neuron/update primitive** (unique-per-neuron cells vs shared type-GRUCell), still N=128, still not FlyWire.

**Claim:** at N=128, feeding 10–50% of neurons as sensory does not get the type-GRUCell graph off unigram.  
**Not a claim:** I/O can never matter; FlyWire sensory regions would not change this.

---

## Run 6 — unique-per-neuron cells vs type-GRUCell (`results/wiki_mean_d4_unique.json`)

Same locked mean-head d=4 scalar protocol, N=128. Type-shared is Run 2 (`wiki_mean_d4.json`). Unique path is one vectorized GRU per neuron (`einsum` over N, **not** 128 Python `GRUCell`s).

| cell | fly val | shuffle val | gru val | fly scalars | program |
|------|--------:|------------:|--------:|------------:|--------:|
| type (Run 2) | 4.600 | 4.600 | **4.080** | 6206 | 4 shared GRUCells |
| unique | 4.596 | 4.600 | **3.578** | 17118 | N unique GRUs |

Unigram **4.564**. Unique fly kisses 4.554 at step 150 and ends **4.596**. Shuffle 4.600. No topology split. GRU is better only because it inherited the larger unique budget (~17k vs ~6k).

**Decision-tree branch:** unique \(F_i\) **did not** get the graph off unigram. Shared type-programs were not the thing starving sequence learning. With message width, sensory fraction, and unique cells all sitting at ~4.56, the failure is the **graph + this neuron family + this I/O**, not “θ is too shared.” **Do not scale N. Do not FlyWire.**

**Claim:** giving every neuron its own GRU still yields a unigram predictor on this motif graph.  
**Not a claim:** no recurrent unit can work; the graph is getting a usable sequential signal that F is too small to use.

---

## Run 7 — typed vs universal update (`results/wiki_mean_d4_universal.json`)

Same locked mean-head d=4 scalar protocol, N=128. Universal = **one** shared `GRUCell` + one emit for every neuron. Typed = four type-specific programs (Run 2). Unique = per-neuron (Run 6).

| cell | fly val | shuffle val | gru val | fly scalars | programs |
|------|--------:|------------:|--------:|------------:|---------:|
| universal | 4.600 | 4.601 | 4.239 | 5942 | 1 |
| type (Run 2) | 4.600 | 4.600 | **4.080** | 6206 | 4 |
| unique (Run 6) | 4.596 | 4.600 | **3.578** | 17118 | N |

Unigram **4.564**. Typed and universal overlay at 4.600. Unique is 4.596. Cell-type diversity does **not** move bits/byte. GRU differences track `matched_gru` budget, not typing.

**Decision-tree branch:** typing the update rule does not help (and does not hurt) on this graph. Combined with Runs 4–6, diversity of \(F\) is not the unigram stall. **Do not scale N.**

**Claim:** one universal GRU vs four type-GRUs is a wash at N=128 mean-head.  
**Not a claim:** cell types could not matter on a real connectome or a task that uses them.

---

## Run 8 — decode from sensory (`results/wiki_mean_d4_read_sensory.json`)

Stethoscope from [`notes/DIRECTIONS.md`](notes/DIRECTIONS.md) §0a. Same locked mean-head d=4 scalar type-GRU protocol, N=128. Graph unchanged. Decoder pools **sensory** instead of readout. Mean head still `Linear(4, 256)` (6206 scalars, same as Run 2).

| decode | fly val | shuffle val | gru val | fly scalars |
|--------|--------:|------------:|--------:|------------:|
| readout (Run 2) | 4.600 | 4.600 | **4.080** | 6206 |
| sensory | 4.601 | **4.421** | **3.863** | 6206 |

Unigram **4.564**. Bigram **3.729**. Scaling gate ~4.3.

Fly sensory-mean overlays readout-mean: kiss 4.557 at step 150, bounce to **4.601**. Greedy is spaces. So the population that **structurally contains \(x_t\)** is still a unigram predictor under a mean head.

Shuffle sensory-mean is the first mean-head *graph* arm under unigram (Δ = −0.143), still descending at step 200, greedy `"The a a a…"`. Not through the gate. **Shuffle wins** — opposite of the fly-prior bet.

Why the split can exist here and not on readout: fly sensory in-degree is **0** (byte only, 4-d GRU hidden). Destination shuffle dumps edges *into* sensory, so shuffle’s decoded population has **byte + graph mixing**. Fly readout has mixing without \(x_t\); fly sensory has \(x_t\) without mixing. Neither is enough. Shuffle sensory accidentally has both.

GRU 3.863 is a matched-budget run, not a decode-pop effect (6184 vs 6206 scalars).

**Decision-tree branch:** I/O lag was **not sufficient**. Seeing the current byte in the decoded population does not get the *fly* graph off unigram. Next is putting \(x_t\) onto a population that already has recurrent mixing (**1a** same-step skip into readout, or **1c** ordered update), still N=128. Do not scale N. Do not treat shuffle 4.421 as a fly-wiring win.

**Claim:** mean-pooling the 16 in-degree-0 sensory GRUs is a unigram predictor; the same head on shuffled sensory (incoming synapses + byte) beats unigram.  
**Not a claim:** the graph cannot work; flatten-sensory would also fail; 4.421 is a topology success for the fly motif.

Tests after this change: **56 passed** (`uv run pytest -q`).

---

## Run 9 — same-step sensory→readout skip (`results/wiki_mean_d4_skip.json`)

DIRECTIONS §1a. Locked mean-head d=4 scalar type-GRU, N=128. Decoder still reads **readout**. In `step()`, the current-byte embedding is `scatter_add`’d onto readout incoming as well as sensory. No extra scalars (6206, same as Run 2). No new edges.

Lag unit test: fly readout RMS at lag-0 is 0 without skip, >0 with skip.

| I/O | fly val | shuffle val | gru val | fly scalars |
|-----|--------:|------------:|--------:|------------:|
| readout, no skip (Run 2) | 4.600 | 4.600 | **4.080** | 6206 |
| decode sensory (Run 8) | 4.601 | **4.421** | **3.863** | 6206 |
| skip onto readout | 4.594 | 4.601 | 4.447 | 6206 |

Unigram **4.564**. Fly kisses 4.558 at step 150, ends **4.594**, greedy spaces. Shuffle overlays at 4.601 (the Run 8 shuffle win is gone). GRU 4.447 is a matched-budget roll, not a skip effect.

At init, skip |embed| on readout is ~4× |graph incoming| (0.90 vs 0.23 after 16 steps; readout in-degree ~8.8). The byte is on the path but **drowns** the graph on the shared scalar channel. Both arms become “mean of 16 byte-dominated GRUs,” i.e. Run 8 fly.

**Decision-tree branch:** putting \(x_t\) on readout in the same step does **not** get fly off unigram. Byte-on-path was necessary; it is not sufficient under mean + scalar sum. **Do not scale N.** Next distinct I/O schedule is **1c** ordered update (hops finish inside the token *without* adding embed to readout). Optional: scaled skip so graph isn’t drowned. WikiText delayed-identity (**2a**) if 1c also sits on unigram.

**Claim:** a same-step skip onto this mean-head scalar graph is still a unigram predictor.  
**Not a claim:** no skip can work; separate skip vs synapse channels would fail; ordered inner hops are ruled out.

Tests after this change: **61 passed** (`uv run pytest -q`).

---

## Run 10 — ordered hops (`results/wiki_mean_d4_ordered.json`)

DIRECTIONS §1c. Locked mean-head d=4 scalar type-GRU, N=128, **no skip**. Each token: inject \(x_t\) into sensory only, then update sensory → Kenyon → core → DA → readout, recomputing messages from already-updated sources. Decoder still mean-pools readout. Same 6206 scalars as Run 2. Unit test: lag-0 readout RMS is 0 under parallel, >0 under ordered (path through the graph, not an embed dump).

| I/O | fly val | shuffle val | gru val | fly scalars |
|-----|--------:|------------:|--------:|------------:|
| parallel (Run 2) | 4.600 | 4.600 | **4.080** | 6206 |
| decode sensory (Run 8) | 4.601 | **4.421** | **3.863** | 6206 |
| skip (Run 9) | 4.594 | 4.601 | 4.447 | 6206 |
| ordered hops | 4.597 | 4.597 | **4.169** | 6206 |

Unigram **4.564**. Fly kisses 4.551 at step 150, ends **4.597**, greedy spaces. Shuffle overlays. GRU **4.169** still beats unigram (matched 6184 scalars).

**Decision-tree branch:** same-token hops through the motif **do not** get this mean-head graph off unigram. Schedule was a real bug; fixing it is not enough. **Do not scale N. Do not load MaleCNS/FlyWire.** Next if we stay on V0: delayed identity (**2a**) or a wider readout of the 16 neurons, not another hop schedule.

**Claim:** ordered sensory→Kenyon→readout at N=128 mean-head is still a unigram predictor.  
**Not a claim:** no graph recurrence can work; a pretrained LM + frozen connectome reservoir (see [`notes/FLM.md`](notes/FLM.md)) is the same experiment.

Tests after this change: **66 passed** (`uv run pytest -q`).

---

## Run 11 — delayed identity (`results/delay_n128_ordered.json`)

DIRECTIONS §2a. Cue token in `{0..15}`, then τ silence tokens (`255`), predict the cue at the last step. N=128, mean head, d=4, scalar type-GRU, **ordered hops** (so τ=0 is not the 2-hop lag). Chance **1/16 = 0.0625**. Same 6206 scalars as WikiText.

| τ | fly acc | shuffle acc | gru acc | fly val bpb |
|--:|--------:|------------:|--------:|------------:|
| 0 | 0.062 | 0.062 | **1.000** | 4.026 |
| 1 | 0.062 | 0.062 | **0.887** | 4.002 |
| 2 | 0.062 | 0.062 | **1.000** | 4.035 |
| 4 | 0.062 | 0.062 | 0.137 | 4.035 |

log₂(16) = 4. Fly val bpb sits on that table: it learned “answer is one of 16 classes,” not which class. Probes collapse to class 9. GRU copies at τ=0 and τ=2; τ=4 is only weakly above chance at this budget/200 steps.

**Decision-tree branch:** even **immediate copy** fails for fly/shuffle. The same-token graph path exists (Run 10 unit test) but the 4-d mean of 16 neurons does not carry 16-way identity. WikiText unigram was not “too hard a language task.” **Do not scale N.** Next if staying: wider readout of those 16 neurons (flatten/attend), not MaleCNS.

**Claim:** ordered-hop fly at N=128 cannot copy a 16-way cue at τ=0.  
**Not a claim:** no recurrent graph can copy; GRU delay-4 is a memory limit.

Tests after this change: **71 passed** (`uv run pytest -q`).

---

## Run 12 — flatten delayed identity (`results/delay_n128_ordered_flatten.json`)

Same task as Run 11 (16-way cue, τ silence, last-step predict), N=128, ordered hops, **flatten** the 16 readout neurons: `Linear(64, 256)` not `Linear(4, 256)`. Decoder is 16 640 / 21 566 scalars (**77 %**). Chance 0.0625.

| τ | fly acc | shuffle acc | gru acc | fly val bpb |
|--:|--------:|------------:|--------:|------------:|
| 0 | **1.000** | **1.000** | **1.000** | 0.011 |
| 1 | **1.000** | **1.000** | **1.000** | 0.006 |
| 2 | **1.000** | **1.000** | **1.000** | 0.009 |
| 4 | **1.000** | **1.000** | **1.000** | 0.005 |

Probes match cue→cue on all four classes. Mean-head Run 11 was chance at every τ. Flatten+ordered copies through τ=4.

Fly **=** shuffle **=** GRU. Topology still a wash. GRU’s τ=4 jump vs Run 11 is the larger matched budget, not a graph effect.

**Decision-tree branch:** the 4-d mean was the copy failure. Per-neuron readout states carry the cue. That does **not** yet say fly wiring helps, and it does **not** say WikiText is solved (Run 1 flatten was delayed I/O). Next language slice: WikiText **flatten + ordered**. Still N=128. Not MaleCNS.

**Claim:** flattening the 16×4 readout with ordered hops copies 16-way delayed identity at N=128.  
**Not a claim:** the fly motif beats shuffle; the 77 % decoder is doing sequence modeling rather than reading a code the graph already built.

Tests after this change: **72 passed** (`uv run pytest -q`).

---

## Run 13 — WikiText flatten + ordered (`results/wiki_flatten_d4_ordered.json`)

Locked WikiText protocol, N=128, flatten `Linear(64, 256)`, ordered hops, no skip. Same 21566 scalars as Run 1 (decoder still 77 %). Unigram **4.564**, bigram **3.729**, old scaling gate ~4.3.

| I/O + head | fly val | shuffle val | gru val |
|------------|--------:|------------:|--------:|
| flatten, parallel (Run 1) | 4.522 | 4.471 | **3.502** |
| mean, ordered (Run 10) | 4.597 | 4.597 | 4.169 |
| flatten + ordered | **3.586** | **3.621** | **3.572** |

Fly sits on unigram until ~step 100 (val 4.54), then breaks: 3.94 → 3.63 → 3.59. Still slightly descending at 200. Greedy is `"The the the…"`, not English — V0 success was bits/byte, not readable text.

Fly **beats unigram by 0.98 bpb** and is under bigram. Shuffle and GRU land in the same band. Fly minus shuffle = −0.035 (right sign, one seed, not a topology claim). GRU 3.572 is a matched 21568-scalar run; Run 1 GRU was 3.502 on the same budget with parallel (GRU ignores `update`).

**Decision-tree branch:** the WikiText stall was mean-pool + delayed I/O, not “graphs cannot do bytes.” Flatten+ordered is the first graph arm through the ~4.3 gate. **Do not scale N** until fly ≠ shuffle on a fair control (type-aware shuffle, 1d). Decoder is still 77 %.

**Claim:** N=128 fly with flatten + ordered hops beats unigram and bigram on this WikiText slice.  
**Not a claim:** fly motifs beat shuffle; the model writes English; 77 % decoder is innocent.

Tests: **72 passed** (no new code; flag combination).

---

## Run 14 — type-aware shuffle (`results/wiki_flatten_d4_ordered_region.json`)

DIRECTIONS §1d. Same flatten + ordered WikiText protocol as Run 13. Shuffle permutes destinations **inside each (src-region, dst-region) class**, so sensory still only hits Kenyon (no destination-shuffle shortcuts). Degrees, signs, contact counts, I/O sets preserved. Unit test: destination shuffle at this seed adds sensory→readout edges; region shuffle adds none.

| Control | fly val | shuffle val | gru val |
|---------|--------:|------------:|--------:|
| destination shuffle (Run 13) | **3.586** | 3.621 | 3.572 |
| region-preserving shuffle | 3.600 | **3.577** | 3.460 |

Unigram **4.564**. Bigram **3.729**. Both graph arms still break through unigram after ~step 100. Fly minus region-shuffle = **+0.023** (shuffle slightly better). One seed; noise vs Run 13 fly 3.586.

**Decision-tree branch:** with a working I/O+head, **motifs do not beat a fair rewire**. The destination-shuffle “fly win” of −0.035 was not topology. **Do not scale N.** Decoder still 77 %. Next if staying: more seeds, a smaller head, or a Kenyon-like binding task (2b) — not MaleCNS.

**Claim:** region-preserving shuffle learns this WikiText slice as well as the fly motif.  
**Not a claim:** no graph can help any task; 0.023 is a significant motif *loss*.

Tests after this change: **73 passed** (`uv run pytest -q`).

---

## Run 15 — sparse binding (`results/bind_n128_ordered_region.json`)

DIRECTIONS §2b. 16 overlapping 3-hot patterns on 16 features; present the three feature tokens in random order; predict class. Flatten + ordered, N=128, **region-preserving** shuffle. Chance **0.0625**. Same 21566-scalar flatten head as Run 13.

| Arm | val acc | val bpb | step to acc=1.0 |
|-----|--------:|--------:|----------------:|
| fly | **1.000** | 0.125 | ~175 |
| region shuffle | **1.000** | 0.013 | ~125 |
| gru | **1.000** | 0.012 | ~50 |

Probes cue-class → class on all four checks. Shuffle reached perfect **earlier** than fly. GRU first.

**Decision-tree branch:** Kenyon-like conjunctions are learnable with this I/O+head, and **motifs still do not help**. The 64-d flatten decoder saturates the task for any mixer. **Do not scale N.** A motif test needs a **small** head (mean, or `Linear(16, 16)`) so the graph has to do the binding.

**Claim:** flatten+ordered copies 16-way 3-hot sets; region shuffle matches accuracy and is faster.  
**Not a claim:** expansion cannot help under a tiny readout; 0.125 vs 0.013 bpb is a motif *loss*.

Tests after this change: **76 passed** (`uv run pytest -q`).

---

## Run 16 — trained small-head mushroom body (`results/bind_mb_n128_trained.json`, `results/bind_mb_n128_trained_s1.json`)

N=128, **not** FlyWire. 16 PNs / 80 Kenyon / 16 MBONs. Kenyon claws are random unique 3-sets. Graph seed is `task_seed + 10007` so Kenyon[i] is **not** class pattern i (same-seed RNG was planting 16/16; tests lock that). Local bag inject, coincidence θ=2.5, WTA top-20, trained emit `Linear(16, 16)` (decoder in=16, 1218 scalars). Chance **0.0625**.

| Seed | fly acc | region-shuffle acc | gru acc |
|-----:|--------:|-------------------:|--------:|
| 0 | 0.241 | **0.286** | 0.271 |
| 1 | 0.277 | **0.313** | 0.225 |

Fly > chance on both seeds. Fly **does not** beat region-preserving shuffle. An earlier 0.47 vs 0.28 split was the planted-claw RNG bug, not a motif win.

**Claim:** small-head MB binding is learnable above chance at N=128; unique claws do not beat a fair rewire.  
**Not a claim:** fly motifs help this task; WikiText topology win; FlyWire.

Tests after this change: **83 passed** (`uv run pytest -q`).

---

## Run 17 — unique-PN coincidence (`results/bind_mb_n128_uniqueand.json`, `results/bind_mb_n128_uniqueand_s1.json`)

Same unplanted graph seed offset. Kenyon incoming counts each PN **once** (shuffle duplicate claws cannot 3×-count one glomerulus). 12 PNs, 80 Kenyon with **4 unique claws**, θ=3.5 (need 4 distinct active PNs), WTA top-8, 16-way 3-hot + 5 distractors, trained emit `Linear(16, 16)` (1318 scalars). Chance **0.0625**. Region shuffle still cannot add sensory→readout.

| Seed | fly acc | region-shuffle acc | gru acc |
|-----:|--------:|-------------------:|--------:|
| 0 | **0.219** | 0.188 | 0.242 |
| 1 | **0.210** | 0.181 | 0.237 |

Both seeds: fly > chance and fly > shuffle. GRU is slightly ahead of fly; the bar is the fair rewire, not GRU.

**Claim:** unique-claw AND + sparse WTA beats region-preserving shuffle on this binding slice.  
**Not a claim:** WikiText motif win; FlyWire; a large language model.

Tests after this change: **84 passed** (`uv run pytest -q`).

---

## Run 18 — coincidence mixer, no connectome (`results/mixer_bind_s0.json`, `mixer_bind_s1.json`, `mixer_bind_highd_s0.json`, `mixer_bind_highd_s1.json`)

Steal the computation, not the fly graph: frozen sparse expand → count unique hits → ReLU(z−θ) → top‑k → Linear(D→m→C). Matched-budget bag MLP and GRU. Expand seed offset so rows are not class patterns.

**12 features, 16 classes, 3-hot + 5 distractors** (same slice as Run 17). Expand 80, claws 4, θ=3.5, top‑8, m=16. Chance 0.0625. Trainable mixer **1568**.

| Seed | mixer | mlp | gru |
|-----:|------:|----:|----:|
| 0 | 0.250 | **0.253** | **0.253** |
| 1 | 0.231 | **0.265** | 0.238 |

**48 features, 32 classes, 4-hot + 8 distractors.** Expand 256, claws 5, θ=2.5, top‑16, m=32. Chance 0.031. Trainable **9280**.

| Seed | mixer | mlp | gru |
|-----:|------:|----:|----:|
| 0 | 0.547 | **0.854** | 0.724 |
| 1 | 0.564 | **0.865** | 0.727 |

Mixer ≫ chance. Dense bag MLP wins: it sees the whole multi-hot. The expand–AND stack is a hash, not a better classifier of a short bit vector.

**Claim:** the stolen block is a real, small-head model and learns binding above chance.  
**Not a claim:** it beats MLP/GRU at matched `numel()`; it is a language model; FlyWire.

Tests after this change: **89 passed** (`uv run pytest -q`).

---

## Run 19 — coincidence block as a recurrent byte LM (`results/seqmix_wiki_s{0,1,2}.json`, `seqmix_wiki_s{0,1,2}_600.json`, ablations `seqmix_wiki_{topk0,D1024,claws8,theta0}_s{0,1}_600.json`)

The stolen block placed in a residual recurrent stream, no connectome (`flybrain/model/seqmixer.py`, `scripts/run_seqmixer.py`):

```
u   = embed(x_t) + U h_{t-1}                 d=24
z   = E u        E frozen 0/±1, 4 claws/row, D=256
a   = TopK(ReLU(z − θ), 16)                  θ learned per unit
h_t = tanh(u + W R a)                        R: 256→24, W: 24→24
```

Controls at matched trainable scalars (~20.1k): same skeleton with a dense GELU FFN (`d_ff` picked by `matched_dense_ffn`), and `matched_gru`. Locked WikiText slice (72k/8k, seq 64, batch 8, lr 0.01). Unigram **4.564**, bigram **3.729**. Init seeded per arm.

| Steps | Seed | mixer | dense FFN | GRU |
|------:|-----:|------:|----------:|----:|
| 200 | 0 | **3.729** | 3.817 | 3.605 |
| 200 | 1 | 3.806 | 3.623 | 3.651 |
| 200 | 2 | 3.671 | 3.628 | 3.575 |
| 600 | 0 | 3.408 | 3.269 | **3.210** |
| 600 | 1 | 3.370 | 3.259 | **3.179** |
| 600 | 2 | 3.302 | 3.298 | **3.096** |

Means at 600: mixer **3.36**, dense **3.28**, GRU **3.16**. FLOPs/step (batch 1): mixer **17.9k**, dense 27.2k, GRU 24.1k.

Ablations (mixer only, 600 steps, seeds 0/1): top‑k off 4.567 / 3.411 (one seed never leaves unigram); D=1024 m=6 top‑64 3.387 / 3.444; 8 claws 3.420 / 3.707; θ init 0 3.389 / 3.368. None beats the dense control.

**Claim:** expand→AND→top‑k→thin→residual is a working LM layer: under unigram and bigram on every seed, 34% fewer FLOPs than the dense FFN at equal scalars, learns about as fast at 200 steps.  
**Not a claim:** better bits/byte than a dense FFN or a GRU at matched scalars (it is ~0.08 and ~0.2 bpb worse at 600 steps). The FLOP saving does not close that gap on this task.

Tests after this change: **95 passed** (`uv run pytest -q`).

---

## Run 20 — FLOP-matched controls at d=48 (`results/seqmix_flops_d48_*.json`)

Same block as Run 19, wider: d=48, D=1024, 4 claws, top‑64, m=48. Controls matched on **FLOPs/step** (`flop_matched_dense_ffn`, `flop_matched_gru`), which gives the dense arms ~38k scalars vs the mixer's ~80k (the frozen expand is free in scalars; the D×m read is not). lr 0.01 diverges for every arm at this width (mixer stuck at unigram on one seed, dense 3.8 at 1500 steps); lr 0.003 is best for all three arms, lr 0.001 favours the mixer (3.16 vs dense 3.44 / GRU 3.36, 2 seeds).

**lr 0.003, 600 steps, ~50k FLOPs/step**

| Seed | mixer (80k) | dense FFN (38k) | GRU (38k) |
|-----:|------------:|----------------:|----------:|
| 0 | 3.105 | 3.114 | **3.082** |
| 1 | 3.091 | 3.155 | **3.056** |
| 2 | **3.081** | 3.225 | 3.099 |
| mean | 3.09 | 3.16 | 3.08 |

**Both matched** (mixer m=8 → 36.8k scalars / 41k FLOPs; dense 33.3k / 41k; GRU 33.6k / 42k), lr 0.003, 600 steps

| Seed | mixer | dense FFN | GRU |
|-----:|------:|----------:|----:|
| 0 | **3.064** | 3.275 | 3.130 |
| 1 | 3.150 | 3.194 | **3.103** |
| 2 | 3.112 | 3.165 | **3.090** |
| mean | 3.11 | 3.21 | 3.11 |

**1500 steps**, lr 0.003, FLOP-matched m=48: mixer 2.987 / 2.973 vs dense 2.873 / 2.870 vs GRU **2.873 / 2.823** (seeds 0/1). Mixer train bpb on seed 0 ends above its val (3.26 vs 2.99): late training is noisy.

Top‑k at lr 0.003 (seed 0): k=0 3.263, k=32 **3.096**, k=64 3.105, k=128 3.127, k=256 3.296. Sparsity helps; too little or too much hurts.

Wall-clock per step is 2–3× the dense arm in this Python loop (top‑k and the per-token loop), unrelated to the FLOP count.

**Claim:** at equal FLOPs, and at equal FLOPs **and** scalars, the coincidence block beats a dense GELU FFN on every seed at 600 steps (~0.07–0.1 bpb) and ties a GRU. It learns faster at low lr. This is the first control the stolen block beats on text.  
**Not a claim:** it beats a GRU; it stays ahead of the dense FFN at 1500 steps (it does not: 2.98 vs 2.87); that this is a fly result — no connectome, no cell types, the ingredients are sparse ±1 expand, threshold, top‑k, thin read.

### Run 20b — seed count (11 seeds at 600 steps, 5 at 1500; both scalars and FLOPs matched, m=8, lr 0.003)

| Steps | n | mixer | dense FFN | GRU | mixer−dense (paired) | mixer−GRU (paired) |
|------:|--:|------:|----------:|----:|---------------------:|-------------------:|
| 600 | 11 | 3.127 ± 0.034 | 3.168 ± 0.044 | 3.110 ± 0.027 | **−0.041**, se 0.020, t −2.0, wins 9/11 | +0.017, se 0.017, wins 4/11 |
| 1500 | 5 | 2.990 ± 0.028 | 2.948 ± 0.030 | 2.917 ± 0.028 | +0.042, se 0.025, wins 1/5 | +0.074, se 0.023, t +3.2, wins 1/5 |

The 3-seed 0.1 gap shrank to 0.04 with 11 seeds: about two standard errors, 9/11 wins. Real but small, and it is an *early-training* effect. By 1500 steps both dense arms are ahead and the GRU lead is clear. Not noise; not a durable win.

Tests after this change: **96 passed** (`uv run pytest -q`).

---

## Run 21 — fast weights on the sparse code (`results/fw_copy_s{0,1,2}.json`, `fw_copy_dense1024_s{0,1,2}.json`, `fw_copy_sparse_s0_1500.json`, `fw_copy_dense1024_s0_1500.json`, `fw_wiki_s{0,1,2}.json`)

The KC→MBON idea, no connectome (`flybrain/model/fastweight.py`, `scripts/run_fastweight.py`). Coincidence block from Run 20 (d=48, D=1024, 4 claws, top‑64, m=48) plus a per-sequence Hebbian matrix `F [m, D]`, starting at 0:

```
key_t = a_t / ‖a_t‖                           a_t = TopK(ReLU(E u_t − θ))
r_t   = R a_t + F key_t                       retrieve what followed this code before
h_t   = tanh(u_t + W r_t)
F    ← 0.95 F + σ(G h_t) · (V u_t) ⊗ key_{t−1}   learned modulator gates the write
```

Gradients flow through the writes (BPTT, seq 64). Arms at ~67k FLOPs/byte: **sparse+FW**, sparse without FW (Run 20 block), **dense+FW** with an all-active ReLU(Gaussian) code sized for FLOP parity (D=74), and a FLOP-matched GRU. Extra width control: dense+FW at **D=1024** (433k FLOPs, 6.5× the sparse arm).

**In-context copy.** 64 uniform random bytes with a random 16-byte span repeated later. Scored only on bytes 2–16 of the second occurrence, so the floor is **8.0 bpb / 0.4 % acc** and nothing but in-sequence memory can score. 600 steps, batch 32, lr 0.003.

| Seed | sparse+FW | sparse no FW | dense+FW (D=74) | dense+FW (D=1024) | GRU |
|-----:|----------:|-------------:|----------------:|------------------:|----:|
| 0 | **63 %** (4.71 bpb) | 0.3 % | 18 % | 0.3 % | 0.5 % |
| 1 | **72 %** (3.80) | 0.3 % | 0.9 % | 0.3 % | 0.5 % |
| 2 | **73 %** (3.89) | 0.3 % | 27 % | 0.3 % | 0.4 % |

1500 steps, seed 0: sparse+FW **91 %** (1.48 bpb); dense+FW D=1024 **0.3 %** (8.02). The sparse arm's curve is 0 → 1 → 8 → 30 → 51 → 65 % over 600 steps and still rising; the wide dense arm never leaves chance.

**WikiText** (same arms, locked slice, 600 steps): sparse+FW 3.126 / 3.008 / 3.096 vs sparse no FW 3.105 / 3.091 / 3.081 vs dense+FW 3.114 / 3.127 / 3.099 vs GRU 3.046 / 3.027 / 2.976. Fast weights cost nothing on text at this length and buy nothing; GRU still best on plain bpb.

**Why the dense arm fails:** Hebbian outer-product memory needs keys that do not overlap. A top‑64-of-1024 code gives near-orthogonal keys; an all-active ReLU code gives keys with high pairwise overlap, so every write interferes with every read. That is the textbook reason for Kenyon-cell sparseness, and it shows up here as chance vs 91 %.

**Claim:** with a Hebbian fast-weight store, the sparse coincidence code is what makes in-sequence memory work: it beats an all-active code at equal FLOPs and at 6.5× the FLOPs, and beats a FLOP-matched GRU that cannot copy at all. First result in this lab where the biological design principle wins a fair control by a wide margin, on the learning-speed / in-context axis, not asymptotic bpb.  
**Not a claim:** better WikiText bpb than a GRU (it is not); a comparison against attention (softmax attention copies this trivially; the claim is about outer-product memory); a connectome result; that the modulator is doing anything a constant could not (untested).

Tests after this change: **101 passed** (`uv run pytest -q`).

---

## Run 22 — gated linear attention baseline, gate ablation, key sparsity (`results/fw_copy_gla*.json`, `fw_copy_gauss*.json`, `fw_copy_constgate*.json`, `fw_copy_topk*_s0.json`, `fw_copy_*_kemb_*.json`, `fw_wiki_gla*.json`)

Same fast-weight state and write timing as Run 21; only the key changes. `code=learned` is a gated-linear-attention arm: learned dense key **and** query projections (elu+1 or identity feature map, L2-normalized), FLOP-matched (key dim 59) or wide (256, 1024). `code=dense --feature identity` is a frozen signed-Gaussian key: near-orthogonal but all-active. Copy task as Run 21 (floor 8.0 bpb / 0.4 %).

**Copy accuracy, 600 steps unless noted**

| Key | FLOPs | seed 0 | seed 1 | seed 2 | 1500 steps |
|-----|------:|-------:|-------:|-------:|-----------:|
| sparse top‑64 (Run 21) | 67k | **63 %** | **72 %** | **73 %** | **91 %** |
| GLA elu+1, matched D=59 | 67k | 0.3 % | 0.3 % | 0.3 % | 0.3 % |
| GLA elu+1, D=256 | 162k | 0.3 % | | | |
| GLA elu+1, D=1024 | 532k | 0.3 % | 0.3 % | 0.3 % | 0.3 % |
| GLA identity, matched D=59 | 67k | 1.4 % | 15 % | | |
| GLA identity, D=1024 | 532k | 0.5 % | 0.3 % | | 0.4 % |
| GLA identity, D=1024, lr 0.001 / 0.0003 | 532k | 0.6 % / 0.4 % | | | 0.2 % |
| Gaussian identity, matched D=74 | 67k | 27 % | 26 % | | |
| Gaussian identity, D=1024 | 434k | 0.4 % | 0.5 % | | |
| Gaussian identity, D=1024, lr 0.001 | 434k | 8 % | | | |

Keys from the **embedding only** (exactly the same key for the same byte, no recurrent context): sparse **71 % / 78 %**; Gaussian D=1024 0.8 % / 0.6 %; GLA identity D=1024 0.4 % / 0.3 %. So context drift is not why dense keys fail. Near-orthogonality is not enough either (Gaussian D=1024 has cosine ≈ 0.03 and fails; sparse top‑64 codes share ~4 of 64 units and work).

**Gate ablation** (sparse arm): constant gate 0.5 → 0.3 % / 2 %; constant gate **1.0 → 69 % / 71 %**, same as the learned gate (63–73 %). The learned modulator is **not** doing anything a constant write strength does not. Earlier "modulator essential" reading was gate magnitude.

**Key sparsity sweep** (sparse arm, seed 0, active units of 1024): k=8 **84 %**, 16 83 %, 32 82 %, 64 63 %, 128 **0.3 %**, 256 0.3 %, 512 0.3 %. A cliff between 6 % and 12 % active. Sparser is better and cheaper (k=8: 51k FLOPs).

**Sparse arm at lr 0.001:** 21 % at 600 steps (slower, not broken).

**WikiText** GLA elu+1 matched: 3.175 / 3.228 / 3.160 (sparse+FW 3.13 / 3.01 / 3.10; GRU 3.05 / 3.03 / 2.98). GLA D=1024: 3.976 (diverged at lr 0.003).

**Claim:** on Hebbian/outer-product in-context memory, a frozen sparse top‑k code beats learned dense keys (gated linear attention with key and query) at equal FLOPs, at 8× the FLOPs, at three learning rates, with either feature map, and with context-free keys. The working regime is ≤ ~6 % active units; k=8 is best. The write gate can be a constant. The sparse block is not worse on WikiText than the GLA arm.  
**Not a claim:** we know *why* dense keys fail here (not orthogonality, not context noise, not lr — open); that a properly tuned GLA/DeltaNet implementation with delta-rule writes, chunked training, and a longer schedule would also fail (this is one 64-byte task and a BPTT toy); that this beats softmax attention.

Open mechanism candidates for a later slice: sparse nonnegative keys make retrieval a near-exact-match lookup with few interfering terms; and gradient through `F` touches only k units per write, so credit assignment does not entangle every stored pair. Test: top‑k of a *Gaussian* projection (no claws) and *signed* sparse keys.

Tests after this change: **103 passed** (`uv run pytest -q`).

---

## Run 23 — mechanism, parallel form, and length 512 (`results/fw_copy_{gausstopk,signed,delta}*.json`, `results/sla_*.json`)

### Mechanism (recurrent `FastWeightLM`, copy at 64, 600 steps, k=8 of 1024)

| Key | seed 0 | seed 1 |
|-----|-------:|-------:|
| claws ±1, top‑8 (Run 22) | 84 % | 85 % |
| **Gaussian projection, top‑8** | **91 %** | **91 %** |
| claws, **signed** top‑8 | 83 % | 82 % |
| claws top‑8, **delta-rule** write | 85 % | 85 % |
| Gaussian projection, top‑64 | 92 % | |

Claws are not needed (a Gaussian projection with top‑k is as good or better). Nonnegativity is not needed. The delta rule buys nothing at this length. **Top‑k sparsity is the only ingredient that survived every test.**

### Parallel form (`flybrain/model/sparsela.py`, `scripts/run_sparsela.py`)

Same memory written as a causal T×T product: `out_t = Σ_{s<t} decay^{t-1-s} (q_t·key_{s-1}) v_s`, 2 residual blocks (memory + FFN), d=64, no recurrent state, constant gate. Key kinds: sparse claws top‑k, Gaussian top‑k, dense Gaussian, learned key+query (GLA), softmax attention (ceiling, given the same key_{s-1} shift), plus a GRU. 85k–100k scalars each. At sequence 64 it reproduces the recurrent result (sparse 90 %) in 9 s on a 3090 instead of 5 min on CPU.

### Learning rate was part of the earlier "dense fails" (copy at 64, 1500 steps)

| Key | lr 0.003 | lr 0.001 (s0 / s1) | lr 0.0003 | 10k steps |
|-----|---------:|-------------------:|----------:|----------:|
| sparse top‑16 | **90 %** | **92 / 92** | **92 %** | 92 % |
| Gaussian top‑16 | | 92 / 92 | | |
| dense Gaussian | 69 % | 92 / 92 | | |
| learned (GLA) | 0.4 % | 48 / 91 | 0.7 % | 92 % (lr 0.001), 0.4 % (lr 0.003) |
| softmax | | 93 / 92 | 0.7 % | 92 % |

**Step at which copy first exceeds 90 % (lr 0.001, two seeds):** sparse/Gaussian top‑k **250**; dense Gaussian 500; softmax 750; learned keys 750–1500. Everyone ends at 91–93 %. At lr 0.0003 sparse is at 90 % by step 500 while softmax and learned have not left chance by 1500; at lr 0.003 the dense arms break and sparse still works.

**Claim (revised from Run 22):** at sequence 64 the sparse key is a **learning-speed and robustness** advantage (3–4× fewer steps than softmax attention, works across a 10× lr range), not a capability gap. The Run 21/22 "dense at chance" rows were at lr 0.003, which is outside the dense arms' window.

### Length 512 (copy, span 32, uniform random bytes, 1500 steps, lr 0.001, no decay)

| Key | 1024 units, k=16 | 1024, k=4 | 8192, k=16 | 8192, k=4 | 8192, k=32 | 10k steps (1024, k=4) |
|-----|-----------------:|----------:|-----------:|----------:|-----------:|---------------------:|
| sparse claws | 2.6 % / 0.4 % | **58 %** | 57 % | 57 % | 57 % | 58 % |
| Gaussian top‑k | 52 % / 25 % | 57 % | 57 % | 57 % | 57 % | 58 % |
| dense Gaussian | 0.4 % / 0.4 % | | | | | |
| learned (GLA) | 0.3 % / 0.3 % | | | | | 0.4 % |
| softmax (with shift) | 0.4 % | | | | | 0.3 % |
| GRU (h=96) | 0.3 % | | | | | |

At 512 the learned-key and softmax arms never learn copy, even in 10k steps; the GRU cannot. Sparse keys do, once the active fraction is low enough (k=4 of 1024 at 1500 steps; k=16 needs 8192 units). Decay 0.98 (window ~50) kills it, as expected for gaps up to 450.

The **57 % plateau** is not capacity: it is the same at 1024/4, 8192/4, 8192/16, 8192/32 and after 10k steps. It is the ambiguity of a one-token key: in 512 uniform bytes each value recurs ~2× in the window, so "the byte before" points at ~2 stored values. At 64 that ambiguity is rare, hence 92 %. Fix under test: a short causal conv so the key sees the last 2–3 tokens.

### WikiText at 512 (2 MB train, 200 kB held out, 1500 steps, lr 0.001; unigram 4.532, bigram 3.525)

| Arm | val bpb | first quarter | last quarter |
|-----|--------:|--------------:|-------------:|
| GRU (h=96, 88k) | **2.586** | 2.617 | 2.562 |
| learned (GLA) | 3.239 | 3.222 | 3.233 |
| dense Gaussian | 3.318 | 3.311 | 3.310 |
| Gaussian top‑16 | 3.380 | 3.379 | 3.371 |
| sparse top‑16 | 3.387 | 3.388 | 3.376 |
| softmax | 3.480 | 3.458 | 3.488 |

Text is a clear negative for every attention-style arm at this size: the GRU is 0.65 bpb better, and no arm uses the long context (first vs last quarter flat). The sparse key is the second-worst text model here.

**Claim:** sparse top‑k keys give a fast-weight memory that learns in-context retrieval 3–4× faster than softmax attention at 64, across a 10× lr range, and still learns at 512 where attention, GLA and a GRU of this size do not. Capacity scales with units/active-fraction; the cheap direction is fewer active units.  
**Not a claim:** it helps text (it does not, at 64 or 512, at this size); that softmax attention cannot copy at 512 with more layers/steps/positional encodings (this is a 2-block, 100k-scalar model); that this is fly anatomy (claws, nonnegativity and the DA gate were all removed without loss).

Tests after this change: **108 passed** (`uv run pytest -q`).

---

## Run 24 — the 512 plateau, fixed (`results/sla_copy512_norm*.json`, `sla_copy512_fair_*.json`, `sla_copy64_fair_s0.json`, `sla_wiki512_fair_*.json`)

Two things were wrong with the length-512 setup, neither of them capacity:

1. **One-token keys are ambiguous at 512.** In 512 uniform bytes each value recurs ~2× in the window, so "the byte before" points at ~2 stored values. A causal depthwise conv (kernel 2) before the key makes it a bigram key. Keys still match exactly (probe: true-pair cosine 1.000, random 0.013–0.027, 4 nonzeros).
2. **Nonnegative keys accumulate coherent cross-talk.** Without a normalizer the memory output at 512 has norm ~8–10 vs signal ~1; a bigram key doubles the random cosine and tips it over, so *every* conv arm sat at train loss 8.0 at three learning rates. The linear-attention normalizer (divide by total match weight) fixes it. Signed keys alone do not.

| Sparse, 1024 units, k=4, 1500 steps, lr 0.001, copy@512 | acc |
|---|---:|
| no normalizer, 1-token key (Run 23) | 58 % |
| no normalizer, 2-token key (learned or frozen conv, lr 0.003/0.001/0.0003) | 0.4 % |
| normalizer, 1-token key | 56 % |
| **normalizer, 2-token key** | **94.4 %** |
| normalizer, signed 2-token key | 94.4 % |

**Fair setting** (normalizer + 2-token key for every arm), copy@512, seed 0: sparse **94 %**, Gaussian top‑k **91 %**, dense Gaussian 0.3 %, learned (GLA) 0.3 %, softmax 0.3 %, GRU 0.3 %. Seed 1: sparse **92 %**, Gaussian top‑k **93 %**, all four controls 0.3–0.4 %.

**Fair setting, copy@64** (step at which acc first exceeds 90 %): sparse **250** (ends 96 %), Gaussian top‑k 500 (96 %), dense Gaussian ~1250 (92 %), softmax 1000 (**99 %**), learned keys never (0.3 % at 1500). Softmax has the higher ceiling; sparse gets to 95 % four times sooner.

**Fair setting, WikiText@512** (2 MB / 200 kB, 1500 steps; unigram 4.532, bigram 3.525):

| Arm | val bpb | first quarter | last quarter | Run 23 (no norm, 1-token) |
|-----|--------:|--------------:|-------------:|--------------------------:|
| GRU (h=96) | **2.586** | 2.617 | 2.562 | 2.586 |
| learned (GLA) | 2.886 | 2.939 | 2.855 | 3.239 |
| softmax | 2.925 | 2.995 | 2.888 | 3.480 |
| dense Gaussian | 2.973 | 3.071 | 2.928 | 3.318 |
| Gaussian top‑4 | 3.139 | 3.325 | 3.032 | 3.380 |
| sparse top‑4 | 3.160 | 3.346 | 3.047 | 3.387 |

The normalizer + bigram key improved every attention arm by 0.3–0.5 bpb and, for the first time, the last quarter of a window is predicted better than the first (context is being used). But the **text ordering is the reverse of the copy ordering**: learned keys best, sparse worst among attention arms, and the GRU still 0.3 bpb ahead of everything. Decay 0.98 does not change it (sparse 3.195, learned 2.884).

**Claim:** with a bigram key and a normalizer, a sparse top‑k key does exact in-context retrieval at 512 (94 %) where learned keys, softmax, dense random keys and a GRU of the same size learn nothing in this budget; at 64 it learns 4× faster than softmax but caps lower (96 vs 99 %).  
**Not a claim:** it helps text. On WikiText the sparse key is the weakest attention arm by 0.27 bpb and every attention arm trails a GRU. Exact-match retrieval is not what byte-level text prediction at this size wants; soft/semantic matching (learned keys) is.

**Where that leaves the line:** sparse keys are a cheap, fast, robust *exact* memory. The honest product-shaped test is a mixed model (learned-key heads for text plus a sparse-key head for retrieval) on a task that needs both, e.g. text with in-document repeats, code, or a terminal transcript, at a size where attention beats a GRU. Not a byte-LM replacement.

Tests after this change: **107 passed** (`uv run pytest -q`).

---

## Run 25 — the six follow-up tests (`results/sla_big_*.json`, `sla_needle*.json`, `sla_cap1024_*.json`, `cross_wiki_to_code.json`)

Setup for the text runs: 4 blocks, d=128, d_ff=256, learned-key dim 64, sparse head 1024 units / top‑4, normalizer, bigram key conv, lr 0.001, 3000 steps, batch 16, seq 512, 9 MB train / 1 MB held out. ~470–600k scalars per arm; GRU hidden 330 (573k). `mixed` = one learned-key head + one sparse-key head per block. Scoring splits bytes into **copyable** (target continues an 8-gram seen earlier in the window, followed by the same byte there) and the rest. A first `mixed` pass had a 64-unit sparse head by mistake (`sla_big_*_s0.json` mixed rows); the corrected rows are `sla_big_*_mixed_s0.json`.

### 1+2. Repeat-rich data, mixed head, larger size

| Corpus (copyable frac) | GRU | softmax | **mixed** | learned | tied | sparse only |
|---|---:|---:|---:|---:|---:|---:|
| WikiText (4 %) all bytes | **2.079** | 2.443 | 2.335 | 2.418 | 2.466 | 2.908 |
| WikiText copyable | 1.759 | 2.227 | **1.430** | 2.169 | 2.188 | 1.884 |
| Python code (20 %) all bytes | **1.602** | 1.820 | 1.832 | 1.972 | 2.019 | 2.558 |
| Python code copyable | 0.836 | **0.509** | 0.595 | 0.895 | 0.847 | 0.706 |
| Python code non-copyable | **1.792** | 2.146 | 2.138 | 2.239 | 2.310 | 3.017 |
| synthetic terminal (28 %) | **0.095** | 0.115 | 0.126 | 0.123 | 0.138 | 0.235 |

- The sparse head does what it is for: on repeat bytes, mixed beats learned-only by 0.74 bpb on prose and 0.30 on code, and sparse-only beats learned-only on repeat bytes on both corpora while being much worse elsewhere.
- Mixed is the best fast-weight arm overall on both real corpora (2.335 vs learned 2.418; 1.832 vs 1.972) and matches softmax on code, in recurrent form.
- **The GRU still wins every corpus overall**, by 0.2–0.3 bpb, at this size and step count. Softmax attention wins the copyable bytes on code (0.51). The synthetic terminal corpus is too regular to separate anyone (everything ≈ 0.1 bpb) and is uninformative.
- `tied` (trainable Gaussian key, q=k) is not better than `learned` on any corpus: the sparse advantage is not an initialization effect that a dense key inherits.

### 3. Needle / long context (single span, uniform random bytes, 1024-unit sparse memory unless noted)

| seq | span | signal frac | sparse | softmax | learned |
|----:|-----:|-----------:|-------:|--------:|--------:|
| 512 | 32 | 6 % | **94 %** (Run 24) | 0.3 % | 0.3 % |
| 1024 | 16 | 1.5 % | 0.4 % (also 4096 units) | 0.2 % | |
| 1024 | 64 | 6 % | **80 %** | 0.3 % | |
| 2048 | 16 | 0.7 % | 0.7 % | 0.5 % | 0.3 % |
| 2048 | 64 | 3 % | 0.4 % (also 4096 units) | 0.4 % | |
| 4096 | 16 | 0.4 % | 0.4 % | 0.2 % | 0.2 % |

Sparse retrieval works at 1024 when the scored fraction is ~6 %, and softmax/learned never learn at any of these lengths in 1500 steps. At 2048 nothing has worked yet: not span 64, not 4096 units. Whether that is the memory (cross-talk grows with writes) or the learning signal (≤3 % of positions) is **not separated** by these runs. 16k-unit and signed-key runs at 2048 were queued (see addendum).

### 4. Capacity (seq 1024, span 16, `n_spans` written then recalled)

n=1: 0.4 % (1024 units) / 0.6 % (4096 units); n=4: 1.3 %. The single 64-byte span at the same length gets 80 %, so this is not raw capacity; several short spans at 1024 fail for a reason these runs do not isolate (short-span keys carry 1.5 % signal each; the multi-span layout is untested at 512 where single spans work). Treat the capacity curve as **not yet measured**.

### 5. Cross-corpus, no gradient (WikiText-trained → Python code)

| arm | 512 ctx | 2048 ctx |
|-----|--------:|---------:|
| sparse | 7.01 | 6.80 |
| mixed | 7.72 | 7.62 |
| learned | 8.06 | 8.40 |
| softmax | 8.40 | 8.31 |
| GRU | 10.08 | 9.83 |

Byte models trained on prose are catastrophically overconfident on code (worse than uniform coding). A 4× longer window recovers ≤0.2 bpb for any arm. **No in-context domain adaptation** from any of these at this size; the evaluator is verified (wiki GRU on wiki: 2.10).

### 6. Warm start

`tied` (Gaussian-initialised trainable key, q=k) ≈ `learned` everywhere (2.47 vs 2.42 wiki; 2.02 vs 1.97 code). The frozen sparse key's copy advantage does not transfer to a dense key by initialisation.

**Claims:** a sparse-key head added to learned-key linear attention improves repeat-byte prediction on real prose and code by a wide margin and improves overall bpb modestly, at recurrent cost; sparse retrieval works to 1024 tokens where softmax and learned keys learn nothing in this budget.  
**Not claims:** anything here beats a GRU on bytes at ~500k scalars (it does not, on any corpus); the memory scales past 1024 (untested cleanly); in-context adaptation to a new domain (none observed); capacity curve (not measured).

### Addendum — capacity and length, measured (`results/sla_cap1024_*.json`, `sla_cap2048_*.json`)

Once the scored fraction is ≥ ~6–12 % of positions (several spans per window), the sparse memory learns at every length tried and degrades gracefully with load; width buys margin. Softmax attention only learns when ~half the positions carry signal.

| seq | spans × span | scored frac | sparse 1024 units | sparse 4096 units | softmax |
|----:|-------------:|-----------:|------------------:|------------------:|--------:|
| 1024 | 1 × 16 | 1.5 % | 0.4 % | 0.6 % | 0.2 % |
| 1024 | 1 × 64 | 6 % | **80 %** | | 0.3 % |
| 1024 | 4 × 16 | 6 % | 1.3 % | **93 %** | 0.3 % |
| 1024 | 8 × 16 | 12 % | **88 %** | **93 %** | 0.3 % |
| 1024 | 16 × 16 | 23 % | **87 %** | **93 %** | 0.4 % |
| 1024 | 32 × 16 | 47 % | **81 %** | **92 %** | 53 % |
| 2048 | 1 × 16 / 1 × 64 | 0.7 / 3 % | 0.7 / 0.4 % | — / 0.5 % | 0.5 % |
| 2048 | 8 × 32 | 12 % | 45 % | **94.5 %** | 0.4 % |
| 2048 | 16 × 32 | 24 % | | **94 %** | |

Reading: the earlier "fails at 2048" was two things at once — too little training signal (single short span) and, at 2048 with 1024 units, genuine cross-talk (45 % → 94.5 % by going to 4096 units). At 1024 tokens a 1024-unit / 4-active memory holds 32 spans (~500 stored pairs) at 81 %; 4096 units holds them at 92 %. Softmax attention at this size needs ~half the window to be retrievable before it learns anything.

**Claim:** a 4096-unit, 4-active sparse memory does exact in-context retrieval at 2048 tokens (94.5 %) where softmax attention of the same size learns nothing; capacity degrades gracefully and scales with width.  
**Not a claim:** the memory helps text prediction (Run 25 main table stands); softmax cannot learn these tasks with more steps, positional encodings, or more heads.

Tests after this change: **109 passed** (`uv run pytest -q`).

---

## Run 26 — sparse memory head on the GRU (`results/hyb_*.json`, `flybrain/model/hybrid.py`)

The GRU wins bytes at this size, so attach the exact-match memory to it. `gru_sparse_pre`: memory keyed on embeddings (bigram conv), read added to the GRU input. `gru_sparse_post`: memory keyed on GRU states, read added to the GRU output before the decoder. `gru_learned_post`: same placement with a learned key+query head (is it the sparse key or just an extra layer?). `gru400`: plain GRU widened to the post arms' parameter count. Same protocol as Run 25 (d=128, 9 MB, 3000 steps, seq 512, lr 0.001).

**Python code (20 % copyable), two seeds**

| Arm | scalars | all s0 / s1 | copyable s0 / s1 | non-copyable s0 / s1 |
|-----|--------:|------------:|-----------------:|---------------------:|
| GRU h=330 | 573k | 1.601 / 1.603 | 0.833 / 0.846 | 1.791 / 1.791 |
| GRU h=400 (param-matched) | 771k | 1.545 / 1.558 | 0.802 / 0.806 | 1.729 / 1.745 |
| **GRU + sparse head, input** | 608k | **1.463 / 1.477** | 0.507 / 0.519 | **1.700 / 1.715** |
| GRU + sparse head, output | 795k | 1.480 / 1.487 | **0.486 / 0.477** | 1.727 / 1.738 |
| GRU + learned-key head, output | 836k | 1.518 / 1.544 | 0.630 / 0.656 | 1.739 / 1.764 |

**WikiText (4 % copyable), seed 0**

| Arm | all | copyable | non-copyable |
|-----|----:|---------:|-------------:|
| GRU h=330 | 2.079 | 1.760 | 2.091 |
| GRU h=400 | 2.038 | 1.695 | 2.051 |
| GRU + sparse head, input | 2.050 | 1.553 | 2.069 |
| **GRU + sparse head, output** | **2.009** | **1.323** | **2.036** |
| GRU + learned-key head, output | 2.038 | 1.678 | 2.053 |

- On code, a sparse head on the GRU input costs +6 % scalars and takes 0.14 bpb off the whole corpus (both seeds), with repeat bytes at 0.51 vs 0.83. A GRU 35 % wider gets only 0.05 of that. The learned-key head at 4× the added scalars gets half the gain. The non-copyable bytes improve too (1.79 → 1.70): the retrieved context helps beyond exact repeats.
- On prose the effect is smaller but present: the output-placed sparse head is the best arm (2.009 vs GRU 2.079 vs wide GRU 2.038), and it beats the learned-key head at the same placement.
- Placement: input-side is the cheaper win on code; output-side (keys from GRU state) wins on prose and on repeat bytes.

**Claim:** adding the sparse-key fast-weight memory to a GRU beats a plain GRU, a parameter-matched wider GRU, and the same head with learned keys, on both a code corpus and prose, at ≤ +6–39 % scalars, with the effect concentrated on repeat bytes and spilling over to the rest. This is the first configuration in this lab that beats the best conventional baseline on bits per byte.  
**Not a claim:** competitive with real LMs (500k-scalar byte models); the gain survives at scale or with a transformer backbone (untested); anything about the fly beyond "sparse expansion + fast synapses" (Runs 21–24).

Tests after this change: **110 passed** (`uv run pytest -q`).

---

## Run 27 — confirming Run 26 (`results/conf_*.json`)

Everything that could have made Run 26 a fluke, run once each. Protocol as Run 26 unless stated.

**Seeds.** Code, hybrid (sparse head on input) − GRU: −0.138, −0.126, −0.140 (seeds 0/1/2). Prose, hybrid (sparse head on output) − GRU: −0.070, −0.085, −0.079. No overlap on any seed. Parameter-matched wide GRU (h=400, 771k): behind the 608k hybrid on all three code seeds (1.545/1.558/1.560 vs 1.463/1.477/1.470) and all three prose seeds (2.038/2.049/2.058 vs 2.009/2.019/2.008).

**GRU tuning.** lr 0.0005 → 1.689; 0.001 → 1.601; 0.002 → 1.589. Best GRU is still 0.12 behind. 9000 steps: GRU 1.511, hybrid 1.377 — the gap (0.134) is unchanged at 3× the training, and both are still improving.

**Leakage control** (`--shuffle-keys`: keys and queries come from a different sequence in the batch, so retrieval is noise). Code: 1.613 / repeat bytes 0.86, i.e. exactly plain GRU (1.601 / 0.83). Prose (output placement, +220k scalars): 2.043 / 1.69, i.e. exactly the wide GRU and the learned-key head (2.038 / ~1.7); the real head is 2.009 / 1.32. The extra projections are worth ~0.04 on prose and nothing on code; retrieval is the rest. Nothing leaks.

**Held-out prose** (WikiText-2 valid+test articles, never used): GRU 2.071, wide GRU 2.032, learned-key head 2.040, sparse-on-input 2.048, **sparse-on-output 2.008**; repeat bytes 1.73 → 1.35.

**Held-out code** (1 173 mixed C++/TS/JS/C/GDScript files, 22–26 % repeat bytes; a first attempt at this corpus was a single 10 MB hex dump and was discarded): seed 0 GRU 1.757, wide GRU 1.661, **hybrid 1.524**; repeat bytes 1.30 → 0.72. Seed 1: GRU 1.758, wide GRU 1.677, **hybrid 1.543**; repeat bytes 1.28 → 0.74.

**Claim (confirmed):** a sparse-key fast-weight memory head on a GRU beats the GRU, a 35 % wider GRU, a learned-key head at the same placement, a better-tuned GRU, and a 3×-longer-trained GRU, on three seeds per corpus, on the development corpora and on two held-out corpora, with a leakage control that collapses to baseline. The gain is 0.07–0.09 bpb on prose and 0.13–0.23 on code, concentrated on repeat bytes and spilling over to the rest.  
**Not a claim:** anything beyond ~600k-scalar byte models at 512-token windows; a transformer backbone; real-LM scale.

Tests after this change: **111 passed** (`uv run pytest -q`).

---

## Run 28 — the head on linear-time backbones people use (`results/bb_*.json`, `flybrain/model/hybrid.py`)

Is the Run 26 gain a GRU quirk? Two more backbones, same size class, same protocol (d=128, 4 layers, 9 MB, 3000 steps, seq 512), each plain / + sparse head (input, output) / + learned-key head (seed 0):

- **minGRU** (selective diagonal recurrence, the Mamba class: `h_t = (1−z_t) h_{t−1} + z_t W x_t`, data-dependent gate, chunked closed-form scan in float64). 463k scalars.
- **Gated linear attention** (learned key/query, data-dependent scalar decay per position, normalizer, bigram conv). 530k. With the decay gate this backbone is far stronger than the ungated `learned` arm of Runs 23–25 (code 1.66 vs 1.97).

| backbone | corpus | plain (s0 / s1) | + sparse, input | **+ sparse, output** | + learned-key head |
|---|---|---:|---:|---:|---:|
| minGRU | Python code | 1.620 / 1.628 | 1.503 / 1.519 | **1.480 / 1.499** | 1.548 |
| minGRU | held-out code | 1.770 / 1.765 | 1.568 / 1.580 | **1.534 / 1.544** | 1.616 |
| minGRU | WikiText | 2.079 / 2.087 | 2.060 / 2.060 | **2.042 / 2.045** | 2.087 |
| GLA | Python code | 1.664 / 1.660 | **1.525 / 1.551** | 1.555 / 1.556 | 1.593 |
| GLA | held-out code | 1.770 / 1.696 | 1.567 / 1.565 | **1.563 / 1.535** | 1.710 |
| GLA | WikiText | 2.091 / 2.106 | 2.082 / 2.087 | **2.063 / 2.082** | 2.101 |
| GRU (Run 26–27) | Python code | 1.601 / 1.603 | **1.463 / 1.477** | 1.480 / 1.487 | 1.518 / 1.544 |
| GRU | held-out code | 1.757 / 1.758 | **1.524 / 1.543** | | |
| GRU | WikiText | 2.079 / 2.104 | 2.050 | **2.009 / 2.019** | 2.038 |

Repeat bytes (copyable), output head vs plain: minGRU code 0.53 vs 0.91, held-out code 0.69 vs 1.33, prose 1.37 vs 1.75; GLA code 0.54 vs 0.86, held-out 0.70 vs 1.06, prose 1.40 vs 1.76.

- On all three backbones, both corpora types, both seeds: the sparse head wins. Code −0.11…−0.14 (dev), −0.13…−0.24 (held-out), prose −0.02…−0.07.
- The learned-key head at the same placement gets a third to a half of the code gain and is a **net loss on prose** for both new backbones (2.087 vs 2.079; 2.101 vs 2.091). Same head, same cost, different key: the sparse key is the ingredient.
- Placement: output (keys from backbone state) is best on minGRU and on prose everywhere; input is best for GLA on Python code. Either works.
- Backbone ranking without the head: GRU ≈ minGRU > GLA on code; all three ≈ 2.08–2.10 on prose. With the head, GRU+sparse remains the best model in the lab on every corpus.

**Claim:** the sparse-key fast-weight head is not a GRU quirk. It improves a Mamba-class selective recurrence and a gated-linear-attention model by the same margins it improved the GRU, on development and held-out corpora, two seeds each, while a learned-key head of the same size does not. It is a general add-on for the linear-time family.  
**Not a claim:** transformer backbones (untested, and full attention already retrieves); scale beyond ~600k scalars / 512 tokens.

Tests after this change: **114 passed** (`uv run pytest -q`).

---

## Run 29 — width ladder: does the gain shrink as the backbone grows? (`results/ladder_*.json`, `scripts/run_ladder.sh`, `scripts/ladder_summary.py`)

The decisive question for the head: do wider models grow their own exact recall, making the head a small-model trick? Run 28 protocol (4 layers, d_ff = 2d, 9 MB, 3000 steps, seq 512, lr 0.001) at d = 64, 128, 256, 384, plain vs + sparse head at the output, two seeds each, three corpora. The head is fixed at 1024 units / top-4 (≈135k scalars at d=384), so its share of the model falls from 7 % to 4 % up the ladder. `uv run python scripts/ladder_summary.py` prints the tables.

**minGRU** (no memory mechanism of its own):

| d | backbone params | Python code: plain → +head (gap) | held-out code (gap) | prose (gap) | repeat bytes, code: plain → +head |
|---:|---:|---|---|---|---|
| 64 | 134k | 1.887 → 1.733 (**0.15**) | 2.164 → 1.847 (**0.32**) | 2.261 → 2.230 (0.03) | 1.11 → 0.63 |
| 128 | 463k | 1.626 → 1.484 (**0.14**) | 1.767 → 1.537 (**0.23**) | 2.083 → 2.043 (0.04) | 0.90 → 0.53 |
| 256 | 1.71M | 1.467 → 1.330 (**0.14**) | 1.506 → 1.316 (**0.19**) | 1.950 → 1.912 (0.04) | 0.79 → 0.46 |
| 384 | 3.75M | 1.402 → 1.277 (**0.12**) | 1.415 → 1.247 (**0.17**) | 1.909 → 1.870 (0.04) | 0.73 → 0.43 |

Seeds agree to ±0.01 everywhere on minGRU (per-seed values in the JSON).

**Gated linear attention** (already a learned-key fast-weight memory):

| d | backbone params | Python code (gap) | held-out code (gap) | prose (gap) | repeat bytes, code: plain → +head |
|---:|---:|---|---|---|---|
| 64 | 167k | 1.885 → 1.771 (0.12) | 2.057 → 1.857 (0.20) | 2.273 → 2.234 (0.04) | 1.03 → 0.64 |
| 128 | 530k | 1.656 → 1.551 (0.11) | 1.719 → 1.559 (0.16) | 2.102 → 2.066 (0.04) | 0.86 → 0.54 |
| 256 | 1.85M | 1.439 → 1.375 (0.06) | 1.390 → 1.354 (**0.04**) | 1.996 → 1.968 (0.03) | 0.63 → 0.48 |
| 384 | 3.95M | 1.462 → 1.347 (0.12) | 1.374 → 1.296 (0.08) | 1.961 → 1.931 (0.03) | 0.71 → 0.45 |

GLA plain is seed-sensitive at width: code d=256 plain 1.393 / 1.485, d=384 plain 1.506 / 1.418 (worse than d=256 in one seed), while the head arms sit at 1.37 / 1.38 and 1.38 / 1.31. The wide plain GLA is not training cleanly at this lr/step budget, so the GLA d=384 row is partly an optimization story.

Reading:

- **On the minGRU the gap does not close.** Across a 28× increase in backbone size the code gap moves 0.15 → 0.12 and the prose gap stays at 0.04; on held-out code it narrows 0.32 → 0.17 but is still the largest effect in the table. Repeat-byte loss with the head at d=64 (0.63) is better than the plain backbone reaches at d=384 (0.73). Width does not buy the minGRU exact recall; the head does, at 4 % of the parameters.
- **On GLA the gap does close, on code, at d≥256.** A wide GLA learns its own recall (repeat bytes 0.63 at d=256, near the head's 0.48), and the head's gain drops to 0.04–0.06 on the two code corpora. On prose the plain GLA never builds it and the head keeps its 0.03. This is the expected contrast: GLA *is* a fast-weight memory with learned keys, and given enough width its learned keys get selective enough to retrieve; a selective diagonal recurrence has no such mechanism to grow into.
- **The head makes GLA reliable.** Where the plain GLA's retrieval is seed-dependent (1.39 vs 1.49), the head arms agree to 0.01.
- Cost of the head at d=384: +8 % params, +6 % arithmetic per byte. Buying the same gain with width fails: minGRU d=256 + head (1.85M scalars) reaches 1.330 on code and 1.316 on held-out code; the plain minGRU at d=384 (3.75M, more than 2× the parameters and 2× the arithmetic) reaches only 1.402 / 1.415.

**Claim:** for a backbone without its own associative memory (selective recurrences, the Mamba class), the sparse-key head's gain is stable across a 28× width range at fixed data and steps: −0.12…−0.17 bpb on code, −0.04 on prose, at 4–7 % of the parameters. For a backbone that already has a learned-key fast-weight memory (GLA), the head is a small-width and a reliability aid on code, and its gain falls to ~0.05 by ~2M parameters.  
**Not a claim:** behaviour beyond 4M parameters, longer training, or seq > 512; whether the minGRU gap eventually closes with data rather than width (this ladder holds steps and data fixed).

---

## Run 30 — transformer as the negative control (`results/bb_transformer_*.json`, `TransformerBackbone` in `flybrain/model/hybrid.py`)

Prediction: full attention already retrieves exactly, so the head should give a transformer nothing. Pre-LN causal transformer, 4 layers, d=128, 4 heads, learned positions, 727k scalars (same class as minGRU 463k / GLA 530k), Run 28 protocol, plus longer runs on Python code.

**3000 steps, lr 0.001** (the ladder protocol), two seeds:

| corpus | plain | + sparse, input | + sparse, output | + learned-key |
|---|---|---|---|---|
| Python code | 1.971 / 1.956 | 1.872 / 1.943 | 2.018 / 1.921 | 1.971 / 1.938 |
| held-out code | 1.929 / 2.235 | 1.966 / 1.998 | 2.186 / 2.356 | 2.139 / 2.082 |
| prose | 2.474 / 2.592 | 2.438 / 2.489 | 2.488 / 2.451 | 2.689 / 2.690 |

Undertrained and seed-unstable (plain swings 0.3 on held-out code; repeat-byte loss 0.94–1.72, i.e. no retrieval formed), far behind every linear-time backbone at this budget (1.62–1.66 code, 2.08 prose). Not a usable control on its own.

**9000 steps on Python code**, two seeds each:

| lr | plain (repeat bytes) | + sparse, input | + sparse, output |
|---|---|---|---|
| 0.001 | **1.412 / 1.409** (0.37 / 0.34) | 1.517 / 1.539 | 1.735 / 1.692 |
| 0.0003 | 1.953 / 1.939 (1.08 / 1.03) | 1.866 / 1.934 | 1.877 / 1.877 |

- At lr 0.001 × 9000 steps the transformer forms its own retrieval (repeat-byte loss 0.34–0.37, lower than any head arm on any backbone reaches) and lands at 1.41, the best code number in the lab at d=128. **With the head it is worse by 0.11–0.33.** Once attention retrieves, the head is interference.
- At lr 0.0003 × 9000 steps retrieval never forms (repeat bytes 1.03–1.08) and the head helps by 0.06–0.09, as it did at 3000 steps. The transformer's outcome is bimodal around the induction-head transition, and the head only helps before it.
- The 3000-step transformer's failure was late induction-head formation, not capacity: same model, 3× steps, 1.97 → 1.41.

**Claim:** on a transformer that has learned attention-based retrieval, the sparse-key head gives nothing and costs 0.1–0.3 bpb; the head's gain is specific to models that lack exact recall. The linear-time family is where it belongs.  
**Not a claim:** that the head beats a well-trained transformer anywhere (it does not; at 9000 steps the plain transformer 1.41 is ahead of minGRU + head at 3000 steps 1.48 — different step budgets, not compared at equal training); anything about transformer scale or long contexts.

Tests after Runs 29–30: **118 passed**.

---

## Run 31 — where the gain lands: bytes binned by longest earlier match (`scripts/mask_sweep.py`, `results/mask_mingru_code_s0_masksweep.json`)

Runs 26–30 split bytes by an 8-gram copyable mask, and about half the head's code gain fell on the "non-copyable" side. Rebin every validation byte by the longest earlier-in-window n-gram (ending at t, followed there by the same next byte) at n ∈ {2, 3, 4, 6, 8}, and score a saved minGRU pair (d=128, Python code, seed 0, 1.621 vs 1.481):

| longest match | share of bytes | minGRU | + sparse head | gain |
|---|---:|---:|---:|---:|
| none (no 2-gram repeat) | 53.6 % | 2.202 | 2.260 | **−0.057** |
| 2-gram | 7.4 % | 1.328 | 0.915 | +0.41 |
| 3-gram | 5.3 % | 0.997 | 0.568 | +0.43 |
| 4–5-gram | 8.2 % | 0.839 | 0.532 | +0.31 |
| 6–7-gram | 5.7 % | 0.741 | 0.419 | +0.32 |
| ≥8-gram | 19.9 % | 0.906 | 0.533 | +0.37 |

Weighted sum reproduces the total gap (0.140). Reading:

- **The "non-copyable" gain was short repeats.** Every bin with any earlier match gains 0.3–0.4 bpb, down to 2-grams; the 8-gram mask had only counted 20 % of bytes as retrievable when 46 % have some repeat the bigram-keyed memory can use.
- **On genuinely novel bytes the head costs 0.06 bpb** (54 % of bytes → −0.03 of the total). The head's output is added unconditionally; on a byte with nothing to retrieve it is noise. So the head is purely a retrieval device: it does not free backbone capacity, and it has a small interference cost that a match-strength gate (zero the head's output when the normalizer's mass is small) should remove. Worth ~0.03 bpb on code, more on prose where 96 % of bytes have no 8-gram match.

**Claim:** the head's gain is entirely on bytes with an earlier match in the window, at every match length from 2 up; it costs 0.06 bpb on bytes with none.  
**Not a claim:** that a gate recovers the 0.03 (untested).

---

## Run 32 — equal training budget: recurrence + head vs transformer vs plain GLA (`results/eq9k_code_s{1,2}.json`, `conf_code_9k.json`, `bb_transformer_long_code_s{0,1}.json`)

Same corpus (Python code), d=128, 9000 steps, lr 0.001, seq 512, so the comparison is at equal steps and data.

| model | scalars | seeds | bpb | repeat bytes (≥8) | other bytes |
|---|---:|---:|---|---:|---:|
| transformer (4 heads, learned positions) | 727k | 3 | 1.412 / 1.409 / 1.405 → **1.409** | 0.35 | 1.67 |
| **GRU + sparse head** (input) | 608k | 3 | 1.377 / 1.378 / 1.380 → **1.378** | 0.47 | 1.60 |
| minGRU + sparse head (output) | 498k | 2 | 1.394 / 1.426 → 1.410 | 0.52 | 1.63 |
| GLA, plain | 530k | 2 | 1.518 / 1.432 → 1.475 | 0.60–0.77 | 1.64–1.70 |
| GRU, plain (Run 27) | 573k | 1 | 1.511 | 0.77 | 1.69 |

- **GRU + head beats the transformer by 0.03 on all three seeds** (seed spread 0.003 on both). The transformer retrieves better (0.35 vs 0.47 on repeat bytes); the recurrence is better on everything else (1.60 vs 1.67); the sum favours the recurrence + head, at fewer parameters and constant cost per token.
- minGRU + head ties the transformer (1.41 vs 1.41; one seed above, one below).
- **Plain GLA at d=128 has not converged to its headed version at 3× steps** (1.475 vs 1.55 for GLA + head at 3000 steps; 1.43–1.52 seed-dependent). The Run 29 catch-up was a d=256 phenomenon; at d=128 the learned keys are still finding recall at 9000 steps, and whether they do is seed-dependent.

**Claim:** at this size and budget a linear-time recurrence with the sparse-key head matches or beats a transformer of the same width at equal training, on code, three seeds. A plain gated-linear-attention model at the same width does not reach either within 3× the steps.  
**Not a claim:** anything at larger scale, longer contexts, or with a tuned transformer recipe (warmup, weight decay, more heads); the transformer here is the plain recipe every other arm got.

---

## Run 33 — match-strength gate on the head (`--gate`, `results/gate_mingru_code_s0*.json`)

Run 31 showed the head costs 0.06 bpb on bytes with no earlier match because its output is added unconditionally. Gate the head's output by `g = mass / (mass + softplus(τ))`, where `mass` is the pre-normalizer weight sum (how strongly the current key matched anything in memory) and τ is a learned scalar. One arm, minGRU d=128, Python code, seed 0, same protocol as Run 31:

| longest match | share | minGRU | + head | **+ gated head** |
|---|---:|---:|---:|---:|
| none | 53.6 % | 2.202 | 2.260 | 2.245 |
| 2-gram | 7.4 % | 1.328 | 0.915 | **0.870** |
| 3-gram | 5.3 % | 0.997 | 0.568 | **0.491** |
| 4–5 | 8.2 % | 0.839 | 0.532 | **0.459** |
| 6–7 | 5.7 % | 0.741 | 0.419 | **0.330** |
| ≥8 | 19.9 % | 0.906 | 0.533 | **0.431** |
| **total** | | 1.621 | 1.481 | **1.435** |

- The gate recovers only 0.015 of the 0.06 unmatched cost, not the predicted 0.03. The prediction was wrong about where the gain would come from.
- It improves **every matched bin by 0.05–0.10**. Cause: the normalizer divides the read by the match mass, so a weak, ambiguous match was amplified to the same magnitude as a clean one; the gate restores magnitude ∝ match strength, which lets the decoder trust strong matches and discount weak ones.
- Net: gap over the plain backbone 0.14 → **0.19** (+30 %), one extra scalar parameter.

**Claim:** gating the head by match mass improves it by 0.046 bpb on code at d=128 (one seed), mostly on matched bytes.  
**Not a claim:** multi-seed or other-corpus confirmation (the scale run carries gated and ungated arms).

---

## Run 34 — scale: d=256, 2048-token windows, public corpora (`scripts/run_scale.sh`, `results/scale_*.json`, `results/cross_cpython_to_linux.json`)

Everything before this was ≤ 4M scalars, 512-token windows, corpora nobody else has. This run: 4 layers, d=256, d_ff=512, head 4096 units / top-4, seq 2048, batch 8, 8000 steps (~130M tokens, ~3 passes of CPython, 1.4 of enwik8), lr 0.001, two seeds. Arms: plain minGRU (1.71M), minGRU + sparse head (1.85M), minGRU + gated head (Run 33; 1.85M), transformer (4 heads, learned positions; 2.76M, of which 0.5M is the position table).

Corpora: **CPython 3.12.6** source (`.c/.h/.py`, 2 KB–200 KB files, hex-filtered, seeded shuffle; 40 MB train / 2 MB val), **enwik8** (first 90 MB train, val from 90.1 MB — the standard byte-level LM benchmark), and the **Linux 6.10** kernel (`kernel/ mm/ fs/ net/ lib/ include/linux/`, 20 MB) as the never-seen code corpus, evaluated with the CPython checkpoints.

| corpus | minGRU | + sparse head | **+ gated head** | transformer |
|---|---:|---:|---:|---:|
| CPython (in-domain), s0 / s1 | 1.486 / 1.480 | 1.162 / 1.157 | **1.110 / 1.108** | 1.212 / 1.220 |
| enwik8, s0 / s1 | 1.839 / 1.851 | 1.671 / 1.680 | **1.625 / 1.630** | 1.842 / 1.834 |
| Linux kernel (held out), s0 / s1 | 3.448 / 3.417 | 2.173 / 2.099 | 2.033 / 2.018 | **1.975** / 2.038 |

Repeat-byte (≥8-gram) share at 2048 tokens: CPython 36.5 %, enwik8 11.2 %. Repeat-byte loss, CPython: minGRU 1.11 → head 0.48 → gated 0.40; transformer 0.37. Non-repeat: 1.71 → 1.55 → 1.52; transformer 1.69. enwik8 repeat: 1.26 → 0.58 → 0.47; transformer 0.53. Non-repeat: 1.91 → 1.81 → 1.77; transformer 2.00.

Reading:

- **The head's gain grew with context.** Gap over the plain minGRU: code 0.14 at 512 tokens (Run 29) → **0.32** (ungated) / **0.38** (gated) at 2048; prose 0.04 → **0.17 / 0.21**. A longer window holds more repeats (code 20 → 37 %, prose 4 → 11 %), and every one of them is a byte the head predicts at ~0.4–0.5 bits instead of 1.1–1.3. On enwik8 that is the difference between 1.84 and 1.63.
- **Recurrence + gated head beats the transformer at equal training, both seeds, both corpora, with 2/3 of the parameters**: code by 0.10, prose by 0.21. The split is the same as at 512 tokens: the transformer retrieves a little better (0.37 vs 0.40 on repeat bytes), the recurrence models the rest better (1.52 vs 1.69), and at 2048 tokens the attention costs 16× more per window than at 512 while the head's cost is unchanged.
- **Out of domain, the gated head ties the transformer** (2.03 / 2.02 vs 1.98 / 2.04). The kernel is tab-indented C and every model is 0.5–0.7 bits worse on it; the plain minGRU cannot adapt inside the window at all (first eighth 3.6, last eighth 3.4), the memory arms and the transformer all do (2.7 → 1.9). In-context adaptation is what the head buys a recurrence.
- **The gate replicates** on both corpora and both seeds: −0.05 on code, −0.045 on prose, and −0.1 on the held-out kernel, where it matters most.
- enwik8 1.63 at 1.85M parameters and 130M training tokens is a small-model number (published large transformers reach ~1.0; small LSTMs from 2016 sit ~1.4 with far more training), but it is on the standard benchmark and comparable.

**Claim:** at 4× the width, data and context of the earlier runs, the sparse-key fast-weight head's gain on a Mamba-class recurrence grows rather than shrinks, and the gated head beats a same-recipe transformer of larger size at equal training on code and on enwik8, and ties it on held-out code. Two seeds, seed spread ≤ 0.01 on every arm.  
**Not a claim:** anything against a tuned transformer recipe (warmup, weight decay, RoPE, more heads); beyond 3M scalars; beyond 2048 tokens.

Tests: **123 passed**.

---

## Run 35 — tuned recipe: does the equal-budget win survive a proper transformer? (`scripts/run_tuned.sh`, `results/tuned_*.json`)

Run 34 compared every arm under one plain recipe (Adam, constant lr, learned positions). The fair objection: transformers depend on their recipe more than recurrences do. Same setting as Run 34 (d=256, seq 2048, 4096-unit head, 8000 steps) with warmup 500, cosine decay to 10 %, AdamW weight decay 0.1 on matrices, and rotary positions for the transformer (`--rope`, replacing the learned position table; 2.24M scalars). Every arm gets the same schedule and decay. Seed 0.

| corpus | model | plain recipe (Run 34) | **tuned recipe** | recipe gain |
|---|---|---:|---:|---:|
| CPython | minGRU | 1.486 | 1.450 | 0.04 |
| CPython | minGRU + gated head | 1.110 | 1.084 | 0.03 |
| CPython | transformer | 1.212 | **1.010** | **0.20** |
| enwik8 | minGRU | 1.839 | 1.814 | 0.03 |
| enwik8 | minGRU + gated head | 1.625 | **1.602** | 0.02 |
| enwik8 | transformer | 1.842 | 1.614 | **0.23** |

- **The recipe is worth 0.2 bpb to the transformer and 0.03 to the recurrences.** The Run 34 "beats the transformer" margin was mostly the transformer's untuned recipe. That result is withdrawn as a claim about transformers.
- **Code: the tuned transformer wins by 0.07** (1.010 vs 1.084). Prose: the head still edges it by 0.012, within seed noise. Call it a tie on prose and a transformer win on code.
- **The head's gain over its own backbone is untouched**: 0.37 on code, 0.21 on prose, same as under the plain recipe. What the recipe changes is the transformer, not the head.
- Cost side, for the record: at this shape the recurrence + head runs a step in about 130 ms against about 190 ms for the transformer (Run 36 will measure this on an idle GPU), and its state is constant in the window length.

**Claim:** with a tuned recipe, a same-recipe transformer beats the recurrence + head on code by 0.07 and ties it on enwik8 at 2048 tokens, 8000 steps. The head's 0.2–0.4 bpb gain on a Mamba-class recurrence is independent of the recipe.  
**Withdrawn:** Run 34's "beats a transformer at equal training" as a general statement; it holds only against an untuned transformer.  
**Not a claim:** anything about seeds (one so far), longer contexts (Run 36), or larger models (Run 37).

---

## Run 36 — 4096-token windows (`scripts/run_ctx4096.sh`, `results/ctx4096_*.json`)

Run 35 setting (tuned recipe) with seq 4096 and batch 4, so tokens per step are unchanged; index-form head (`--chunk 256`). CPython, seed 0. Repeat-byte share rises to 42 % (37 % at 2048, 20 % at 512).

| model | 2048 (Run 35) | **4096** | change |
|---|---:|---:|---:|
| minGRU | 1.450 | 1.408 | −0.04 |
| minGRU + gated head | 1.084 | **1.042** | −0.04 |
| transformer (rotary) | 1.010 | 1.006 | −0.004 |

- The head's gain over its backbone holds at **0.37** at both lengths. The transformer's lead over the recurrence + head shrinks from 0.07 to **0.04**: doubling the window gave the transformer nothing (RoPE at 4096 with 8000 steps of training), while the recurrence + head keeps improving with the repeat share.
- Cost, from this shared-GPU run: the head arm's 8000 steps took 2200 s, the transformer's 3435 s; attention's window cost grew 4× from 2048 while the head's did not.

enwik8 at 4096 (repeat share 16 %, vs 11 % at 2048):

| model | 2048 (Run 35) | **4096** |
|---|---:|---:|
| minGRU | 1.814 | 1.823 |
| minGRU + gated head | 1.602 | **1.556** |
| transformer (rotary) | 1.614 | 1.821 |

The transformer **lost 0.2** when the window doubled: repeat-byte loss 0.59 → 1.20, i.e. it never formed retrieval at 4096 on prose in 8000 steps of batch 4 (half as many sequences as at 2048; sparser repeat signal than code, where the same transformer at 4096 was fine). The head arm improved by 0.05 and its gain over the backbone rose to 0.27. This is the Run 30 bimodality again: attention's retrieval is something it has to learn, and at long context on sparse-repeat data it can fail to within the budget; the sparse head retrieves from step zero.

**Claim:** at 4096 tokens the recurrence + gated head is within 0.04 of a tuned transformer on code, and the head's 0.37 gain over its backbone is independent of window length between 2048 and 4096. On enwik8 at 4096 the transformer failed to learn retrieval within the budget and the head arm beats it by 0.27; that is a training-fragility result, not a capacity one.  
**Not a claim:** seeds; that the transformer would not recover with more steps or a larger batch.

---

## Run 37 — 20M class (`scripts/run_20m.sh`, `results/m20_*.json`, `results/cross_m20_linux_to_cpython.json`)

d=512, 8 layers, d_ff 1024 (minGRU 12.9M, + head 13.4M, transformer 17.1M), seq 2048, batch 8, 16000 steps (~260M tokens), tuned recipe (warmup 1000, cosine, wd 0.1, RoPE), index-form head, lr 6e-4. Trained on the Linux 6.10 kernel (`.c/.h`, 200 MB; CPython held out) and on enwik8. Seed 0.

| corpus | minGRU | + gated head | transformer | head − backbone | transformer − head |
|---|---:|---:|---:|---:|---:|
| Linux kernel (in-domain) | 1.295 | 0.910 | **0.781** | **0.385** | 0.13 |
| CPython (held out, kernel-trained) | 2.419 | 1.683 | **1.460** | 0.74 | 0.22 |
| enwik8 | 1.543 | 1.375 | **1.360** | 0.17 | 0.015 |

Repeat / non-repeat bytes, kernel: minGRU 1.03 / 1.44, head 0.31 / 1.24, transformer 0.16 / 1.12. enwik8: 0.88 / 1.63, 0.36 / 1.50, 0.32 / 1.49.

- **The head's gain over its backbone does not shrink with size**: 0.385 on code at 13M vs 0.37 at 1.85M; 0.17 on prose vs 0.21. On held-out code it is 0.74, because the plain recurrence cannot adapt in-window and the head can.
- **The transformer pulls ahead on code at this size**: 0.13 in-domain (0.07 at 1.85M), 0.22 out of domain (a tie at 1.85M). It now leads on non-repeat bytes as well as on retrieval (1.12 vs 1.24), i.e. attention at 17M is doing more than exact recall, and a fixed 4096-unit top-4 memory is not growing with it.
- **Prose is a tie**: 1.360 vs 1.375, with the transformer 27 % larger. enwik8 1.36–1.38 at 13–17M parameters and 260M tokens is in the range of the small LSTM/transformer results on this benchmark (e.g. ~1.3–1.4 for 2016–2018 small models); the large published models (~1.0) train 10–100× longer.
- Wall-clock at this size (shared GPUs, so indicative): head arm 7137 s, plain minGRU 6000 s, transformer 11861 s for the same steps. The recurrence + head trains in 60 % of the transformer's time at 2048 tokens.

**Claim:** the sparse-key head's gain on a Mamba-class recurrence is stable from 0.5M to 13M parameters (0.37–0.39 bpb on code, 0.17–0.21 on prose) and largest out of domain. Against a tuned transformer at equal training the recurrence + head ties on prose and loses on code, and the code gap widens with model size (0.04–0.07 → 0.13 in-domain, 0 → 0.22 held-out).  
**Seed 1 on enwik8 (2026-09-14 23:30, `results/m20_enwik8_gate_s1.json`, `results/m20_enwik8_s1.json`):** head arm 1.380 (seed 0: 1.375), transformer 1.355 (seed 0: 1.360). Spread 0.005 on both arms; the 0.02 transformer lead on prose is replicated, at 27 % more parameters. Transformer wall-clock on the shared GPU: 17540 s.

**Not a claim:** kernel seeds (one); larger sizes. Whether a multi-head memory closes the code gap: see Run 38.

---

## Run 38 — does the memory need to grow with the model? (`scripts/run_capacity.sh`, `--mem-every`) — queued 2026-09-14

Run 37 grew the backbone 26x while the head stayed at one head of 4096 slots, and the code gap to the transformer grew with size (0.07 → 0.13). Two kinds of memory capacity, tested separately, at the Run 37 kernel setting (d=512, 8 layers, seq 2048, 16k steps, tuned recipe, gate, index form):

| arm | memory | params | isolates |
|---|---|---:|---|
| `m20_linux_gate_s0` (Run 37) | 1 head x 4096 | 13.4M | baseline: 0.910 |
| `m20_linux_slots16k_s0` | 1 head x 16384 | 13.4M | slots (should scale with the window; 4096 already >= 2048) |
| `m20_linux_depth4_s0` | 4 heads x 4096 after layers 2/4/6/8 | 15.0M | heads in depth (multi-hop retrieval, the thing attention has at every layer) |
| `m20_linux_depth4x16k_s0` | 4 heads x 16384 | 15.1M | both (the 2x2 corner; queued behind depth4 on GPU 0) |

**19:50 — slots16k and the 4x16k corner cancelled before starting.** Run 39's width ladder showed the memory is a wash at <=4 writes/slot and gains <=0.015 from halving the load below 2, across a 6x range of widths; the 20M head sits at 2 writes/slot, so going to 0.5 is predictable. depth4 is the only Run 38 arm that runs. The corner comes back only if depth4 moves the code number.

Both stay under the transformer's 17.1M, so no feed-forward trimming. Interleaved heads (`GRUHybrid(mem_every=2)`) get their own random address (seed offset) and the same gate/normalizer/bigram key/index form; `mem_every=0` is bit-identical to before (tests: count, causality, distinct keys, chunked == parallel, rebuild). Scaling rule under test: heads ∝ depth, slots ∝ window — the way transformer retrieval scales.

Slot pressure is real, not a control: 4-of-4096 active with 2048 writes per window is ~2 writes per slot, ~8 superposed values under a query; Run 36's gain growing with window is the only (indirect) evidence it is not the limit. Prediction: depth moves more than slots; the corner says whether they add.

**21:50 — depth4 result (`results/m20_linux_depth4_s0.json`), one seed:**

| 20M, kernel code | bpb | repeat / non-repeat | params | wall-clock (shared GPU) |
|---|---:|---|---:|---:|
| minGRU | 1.295 | 1.03 / 1.44 | 12.9M | 6000 s |
| + 1 head (Run 37) | 0.910 | 0.31 / 1.24 | 13.4M | 7137 s |
| + 4 heads in depth | **0.811** | 0.18 / 1.15 | 15.0M | 10620 s |
| transformer | 0.781 | 0.16 / 1.12 | 17.1M | 11861 s |

The gap to the transformer goes from 0.13 to 0.03 with the head arm 12 % smaller. Repeat bytes 0.31 → 0.18 (transformer 0.16), and non-repeat bytes also improve 1.24 → 1.15: the extra heads are not only better recall, they let the backbone spend its state on something else. This is the retrieval-depth hypothesis confirmed at one point. **Seed 1 (2026-09-15 02:30, `results/m20_linux_depth4_s1.json`): 0.814**, repeat / non-repeat 0.18 / 1.16. Replicated to 0.003.

**22:43 — depth8 launched on GPU 0** (`--mem-every 1`, a head after every layer, 17.15M vs the transformer's 17.09M: parameter-matched to 0.3 %). Then depth8 on enwik8 (does the prose tie become a win?). depth4 kernel seed 1 queued on GPU 1 behind the Run 39b window cells.

**03:31 — depth8 kernel (`results/m20_linux_depth8_s0.json`): 0.802**, repeat / non-repeat 0.17 / 1.14, 17.15M params, 17210 s on the shared GPU. Heads slope so far (kernel, 20M): 1 head 0.910 → 4 heads 0.811 / 0.814 → 8 heads 0.802; transformer 0.781. The first three extra heads are worth 0.10, the next four 0.01: retrieval depth saturates around 4 heads at 8 layers. At exactly matched parameters the transformer keeps a 0.02 lead in-domain at the training window; the head arm leads by 0.3 at 4096 and by 2+ beyond (Run 40). Wall-clock: 8 heads cost more per step than 4 (17210 vs 10620 s, both shared), and at 8 heads the training-time advantage over the transformer (11861 s) is gone at 2048; the 4-head arm is the efficient point. **05:10 — depth2 kernel (`results/m20_linux_depth2_s0.json`): 0.834**, repeat / non-repeat 0.21 / 1.17, 13.95M, 9553 s. Full slope, kernel code at 20M:

| heads | placement | params | bpb | repeat / non-repeat | wall-clock (shared) |
|---:|---|---:|---:|---|---:|
| 0 | — | 12.9M | 1.295 | 1.03 / 1.44 | 6000 s |
| 1 | top | 13.4M | 0.910 | 0.31 / 1.24 | 7137 s |
| 2 | after 4, top | 13.95M | 0.834 | 0.21 / 1.17 | 9553 s |
| 4 | after 2/4/6, top | 15.0M | 0.811 / 0.814 | 0.18 / 1.15 | 10620 s |
| 8 | every layer | 17.15M | 0.802 | 0.17 / 1.14 | 17210 s |
| transformer | attention every layer | 17.09M | 0.781 | 0.16 / 1.12 | 11861 s |

Gains per doubling of heads: 0.076, 0.022, 0.010. Each doubling is worth about a third of the previous one, so the rule "heads ∝ depth" has a coefficient: one head per two layers is where the marginal head stops paying for its 0.5M parameters and its step time. **07:55 — depth8 enwik8 (`results/m20_enwik8_depth8_s0.json`): 1.349**, repeat / non-repeat 0.26 / 1.49, 17.15M, 15842 s. enwik8 at 20M: 1 head 1.375 / 1.380 (two seeds), 8 heads 1.349, transformer 1.360 / 1.355 (two seeds, 17.09M). At matched parameters the headed recurrence leads on prose by 0.006–0.011, with a 0.005 seed spread per arm: a marginal lead, not a tie. Depth is worth 0.03 on prose vs 0.11 on code, consistent with prose having a third the repeat mass. depth4 enwik8 running on GPU 1.

**08:25 — depth4 enwik8 (`results/m20_enwik8_depth4_s0.json`): 1.351**, repeat / non-repeat 0.27 / 1.49, 15.0M, 12145 s. Prose at 20M is settled: 1 head 1.375 / 1.380, 4 heads 1.351, 8 heads 1.349, transformer 1.360 / 1.355. Four heads lead the transformer on prose with 12 % fewer parameters; the 4 → 8 step is worth 0.002 on prose (0.009 on code): saturation at one head per two layers holds on both corpora.

**Run 38 summary table (20M, d=512, 8 layers, seq 2048, 260M tokens):**

| arm | params | kernel code | CPython held out | enwik8 |
|---|---:|---:|---:|---:|
| minGRU | 12.9M | 1.295 | 2.419 | 1.543 |
| + 1 head | 13.4M | 0.910 | 1.683 | 1.375 / 1.380 |
| + 2 heads | 13.95M | 0.834 | 1.543 | — |
| **+ 4 heads** | **15.0M** | 0.811 / 0.814 | 1.502 | **1.351** |
| + 8 heads | 17.15M | 0.802 | 1.480 | 1.349 |
| transformer | 17.09M | **0.781** | **1.460** | 1.360 / 1.355 |

**17:45 — kernel transformer seed 1 (`results/m20_linux_s1.json`): 0.789** (seed 0: 0.781; mean 0.785, spread 0.008). Residuals are firm: 4 heads 0.027, 8 heads 0.017. Wall-clock on an *unshared* GPU 1: 6657 s, vs 11861 s for seed 0 on a shared GPU; the 32768-slot 4-head arm, also unshared, took 11152 s. So at 2048 tokens this implementation of the head arm trains *slower* than the transformer (about 1.5–1.7x), not faster; the "~60 % / ~90 % of the transformer's time" statements from Runs 37–38 compared shared-GPU wall-clocks and are withdrawn. What stands on cost: per-token cost of the head arm is flat in window length while attention's is not, which is why it is the one still standing at 32k–64k.

**Claim:** with one sparse-key memory head per two layers, a Mamba-class recurrence is within 0.03 of a tuned transformer on in-domain code and 0.04 out of domain at 12 % fewer parameters, and ahead of it on prose; at matched parameters (8 heads) the gaps are 0.02 / 0.02 / −0.01. The gain from depth saturates at about one head per two layers. **Not a claim:** more than one seed on most cells (two on 4-head code and on 1-head / transformer prose); anything beyond 17M parameters.

**08:05 — where the residual lives** (Run 31 sweep on the 20M kernel checkpoints, `results/m20_linux_{s0,gate_s0,depth4_s0,depth8_s0}_masksweep.json`, 16 batches, bytes binned by longest earlier match):

| longest match | bytes | transformer | 1 head | 4 heads | 8 heads | 8 heads − transformer | share of residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 34 % | 1.706 | 1.830 | 1.737 | 1.733 | +0.026 | 43 % |
| 2-gram | 9 % | 0.745 | 0.889 | 0.804 | 0.792 | +0.047 | 19 % |
| 3-gram | 6 % | 0.422 | 0.526 | 0.465 | 0.457 | +0.035 | 10 % |
| 4-gram | 9 % | 0.324 | 0.414 | 0.357 | 0.340 | +0.015 | 7 % |
| 6-gram | 6 % | 0.264 | 0.377 | 0.283 | 0.269 | +0.005 | 2 % |
| 8-gram+ | 35 % | 0.160 | 0.309 | 0.182 | 0.172 | +0.012 | 20 % |

Overall residual +0.021. Three readings. (1) **Long exact retrieval is solved**: 6-gram and 8-gram bins are within 0.005 / 0.012 of attention. (2) **43 % of what is left is on bytes with no earlier match**, where memory cannot help: this is the recurrence being a slightly weaker general model than attention (1.733 vs 1.706). Depth heads helped here too (1.830 → 1.737), by freeing the recurrence, but the floor is the backbone. (3) **29 % is on 2–3-gram matches**, where many earlier contexts share the key and the memory returns a mass-normalised superposition of their continuations; attention can weigh candidates by content. So the residual is not multi-hop composition; the two-hop test is deprioritised. The next mechanistic lever is disambiguation on short matches (e.g. a sharper read: temperature on the top-k match weights, or a second key of a different n-gram order so 2-gram and 4-gram evidence are read separately), and the other is simply a better backbone on unmatched bytes. Readout: if depth4 lands near 0.78 the gap was capacity and the headline changes; if it stays near 0.91 the gap is structural (the write-up says so). Queued behind the enwik8 seed-1 pair on both GPUs (`scratchpad/cap_queue.sh`, retries on OOM; the other project holds 10.4 GB per GPU).

---

**Cross-lab note (2026-09-14, no code crossed).** A sibling lab of mine was testing low-precision training of a same-size dense transformer and shared its results privately; the numbers are theirs to publish, so only the conclusion is recorded here. Two things came out of it. Their transformer is seed-stable at short (256-character) windows, which says our 0.3 bpb swings are a property of the retrieval mode — whether attention learns to look back over thousands of bytes within the budget — rather than of the transformer recipe. And their precision deficit is flat across match length rather than concentrated on repeatable bytes, so a recall head would not close it: a precision-starved dense model does not lose its recall first, it loses everywhere. For us that is a useful negative — the head's gain is not "precision the backbone lacks", it is a mechanism the backbone lacks. The merge idea is closed.

## Run 39 — do slots need to grow, and with what? (`scripts/run_slot_rules.sh`, `results/slotwidth_*.json`, `results/slotwin_*.json`) — launched 2026-09-14 19:05, GPU 1

Run 38's slots16k is a bigger memory, not a growing one. A growth rule needs a slope: slots varied along an axis at several points. Two candidate rules, head arm only (the backbone does not depend on slot count; its numbers are already in Run 29 / Runs 35–36):

- **slots ∝ width** — Run 29 ladder setting (T=512, 4 layers, 3000 steps, code + cpp, 2 seeds): slots = 4·d and 16·d at d = 64…384, against the existing fixed 1024. Load at fixed 1024 is ~2 writes/slot (512 writes x 4 active); 4·d at d=64 is 8 writes/slot, 16·d at d=384 is ~0.3. Readout: Run 29's gain was flat with width at fixed slots; if the 16·d curve tilts upward with width, slots should scale with the model; if all three curves sit on top of each other, they should not.
- **slots ∝ window** — tuned d=256 CPython setting (8000 steps, tokens/step 16384 held equal): windows 1024 / 2048 / 4096 x slots 1024 / 4096 / 16384 (two cells exist: 2048x4096 = 1.084, 4096x4096 = 1.042). Readout: the head's bpb as a function of writes-per-slot; if the cells collapse onto one curve in writes/slot, the rule is slots ∝ window with a fixed load target, and the target is read off the curve.

**22:45 — slots ∝ width: complete, 24 runs, two seeds each. Verdict: no.** Head-arm bpb vs the Run 29 fixed-1024 cell at the same width (T=512, so writes/slot = 2048 / slots):

| corpus | d | slots | writes/slot | slot arm (s0/s1) | fixed 1024 (s0/s1) | delta |
|---|---:|---:|---:|---|---|---:|
| code | 64 | 256 | 8.0 | 1.798 / 1.779 | 1.733 / 1.733 | +0.055 |
| code | 128 | 512 | 4.0 | 1.490 / 1.486 | 1.481 / 1.487 | +0.004 |
| code | 128 | 2048 | 1.0 | 1.480 / 1.480 | 1.481 / 1.487 | -0.003 |
| code | 256 | 4096 | 0.5 | 1.343 / 1.329 | 1.332 / 1.328 | +0.006 |
| code | 384 | 1536 | 1.3 | 1.302 / 1.277 | 1.285 / 1.269 | +0.013 |
| code | 384 | 6144 | 0.3 | 1.295 / 1.286 | 1.285 / 1.269 | +0.014 |
| cpp | 64 | 256 | 8.0 | 1.944 / 1.930 | 1.841 / 1.854 | +0.090 |
| cpp | 128 | 512 | 4.0 | 1.550 / 1.545 | 1.534 / 1.539 | +0.011 |
| cpp | 128 | 2048 | 1.0 | 1.522 / 1.524 | 1.534 / 1.539 | -0.014 |
| cpp | 256 | 4096 | 0.5 | 1.318 / 1.319 | 1.325 / 1.308 | +0.002 |
| cpp | 384 | 1536 | 1.3 | 1.234 / 1.247 | 1.251 / 1.242 | -0.006 |
| cpp | 384 | 6144 | 0.3 | 1.243 / 1.245 | 1.251 / 1.242 | -0.002 |

Only the overloaded cell (8 writes/slot) costs anything (0.055 code, 0.090 cpp). Everything from 4 writes/slot down to 0.3 is within the ±0.015 seed spread, at every width. Slots do not need to follow the model; they need to stay under a load ceiling somewhere between 4 and 8 writes per slot, i.e. slots ≥ window x active units / 4.

**Window cells complete (tuned d=256 CPython, head arm, backbone 1.450 at 2048 / 1.408 at 4096):**

| window | slots | writes/slot | bpb | vs ≤1 write/slot at same window |
|---:|---:|---:|---:|---:|
| 1024 | 1024 | 4 | 1.093 | (no lower-load cell) |
| 2048 | 1024 | 8 | 1.107 | +0.023 vs 4096 slots (2 w/s) 1.084 |
| 4096 | 1024 | 16 | 1.081 | +0.054 |
| 4096 | 4096 | 4 | 1.042 | +0.015 |
| 4096 | 16384 | 1 | 1.027 | — |

Same curve as the width ladder: 4 writes/slot costs 0.00–0.015, 8 costs 0.02–0.09, 16 costs 0.05; and even at 16 the head keeps 0.33 of its 0.37 gain. Rule: slots ≈ window x active units / 2 (4096 for a 2048 window, which is what the 20M runs use; 8192 would be right at 4096). Slots are a frozen buffer, so this costs no parameters. Verdict for Run 39: slots follow the window with a load target of about 2 writes per slot; they do not follow the model.

**19:50 — window test trimmed to 4 cells** (`scripts/run_slot_window.sh`): 1024 slots at windows 1024/2048/4096 (4/8/16 writes per slot) and 16384 slots at 4096 (0.5). The four cells at 0.5–1 writes/slot are foregone after the width results. Cost: 24 small runs (~1 h) + 4 medium runs (~2.5 h), sharing GPU 1 with the enwik8 seed-1 transformer arm and, later, the slots16k 20M arm.

---

## Run 40 — context extrapolation: evaluate the 20M checkpoints far past their 2048 training window (`scripts/cross_eval.py --lens`, `results/extrap_m20_{linux,enwik8}.json`) — 2026-09-15 02:40

Forward-only, GPU 1, 16 batches per length, in-domain validation (kernel bytes past 200.1M; enwik8 past 90.1M). All arms score the same windows at a given length, so comparisons within a row are exact; across rows the window sample changes (the plain recurrence's ±0.06 drift is that sampling, not the model).

**Kernel code (trained at 2048):**

| eval window | minGRU | + 1 head | + 4 heads in depth | transformer (RoPE) |
|---:|---:|---:|---:|---:|
| 2048 | 1.291 | 0.893 | 0.791 | **0.760** |
| 4096 | 1.301 | 0.871 | **0.752** | 1.048 |
| 8192 | 1.341 | 0.918 | **0.762** | 2.061 |
| 16384 | 1.215 | 0.908 | **0.743** | 3.142 |
| 32768 | 1.229 | 1.024 | **0.826** | 3.926 |

**enwik8 (trained at 2048):**

| eval window | minGRU | + 1 head | transformer (RoPE) |
|---:|---:|---:|---:|
| 2048 | 1.525 | 1.368 | **1.356** |
| 4096 | 1.567 | **1.380** | 2.076 |
| 8192 | 1.559 | **1.396** | 3.810 |
| 16384 | 1.477 | **1.322** | 4.728 |
| 32768 | 1.562 | **1.426** | 4.949 |

- **The head arms keep working at 16x their training length.** Four heads: 0.79 / 0.75 / 0.76 / 0.74 / 0.83 from 2k to 32k. The head's gain over its backbone is 0.40–0.43 (one head) and 0.50–0.58 (four) through 8192, then narrows (0.31 / 0.21 for one head at 16k / 32k; 0.47 / 0.40 for four) exactly where the slot load passes the Run 39 ceiling: 16k x 4 active / 4096 slots = 16 writes per slot, 32 at 32k. Longer windows need more slots (a frozen buffer, no parameters), nothing else.
- **The transformer fails at 2x its training length**, 0.76 → 1.05 → 2.06 → 3.14 → 3.93 (last-eighth bpb 0.65 → 5.0). This is the standard RoPE extrapolation failure and is not a surprise; the fair statement is that a transformer needs a length-extension method (NTK/YaRN-style scaling, or training at the target length at quadratic cost) to use a longer window, while the recurrence + head uses it as is. At 4096 the four-head arm already leads the transformer by 0.30; at 16k by 2.4.
- **Depth heads extrapolate better than one head**: the four-head gain over the backbone widens relative to the one-head gain as the window grows (0.10 at 2k, 0.17 at 16k), i.e. the extra heads are not redundant with the first one under load.
- The per-position split says the same thing: the head arms' last-eighth bpb stays below their first-eighth at every length up to 16k (more context helps); the transformer's last eighth is where it breaks.

**06:40 — the fair transformer control: NTK-aware RoPE scaling at eval time** (`TransformerBackbone.set_rope_scale`, `cross_eval.py --ntk-train-len 2048`; base' = 10000 x s^(Dh/(Dh-2)) for s = T / 2048; `results/extrap_m20_{linux,enwik8}_ntk.json`). Same checkpoints, same windows:

| eval window | + 4 heads | transformer plain | **transformer NTK** | enwik8: + 1 head | transformer NTK |
|---:|---:|---:|---:|---:|---:|
| 2048 | 0.791 | 0.760 | **0.760** | 1.368 | **1.356** |
| 4096 | 0.752 | 1.048 | **0.726** | 1.380 | **1.369** |
| 8192 | **0.762** | 2.061 | 0.807 | **1.396** | 1.491 |
| 16384 | **0.743** | 3.142 | 1.029 | **1.322** | 1.743 |
| 32768 | **0.826** | 3.926 | 1.597 | **1.426** | 2.397 |

The standard length-extension trick rescues the transformer at 2x (it *wins* at 4096, 0.726 vs 0.752) and slows the decay after that, but does not stop it: 0.81 at 8k, 1.03 at 16k, 1.60 at 32k. The head arm is flat.

**08:50 — can slots be added at eval time? No.** `KeyCode.expand_slots` (append fresh random rows, mean trained theta; `cross_eval.py --slots`) on the 4-head kernel checkpoint at a 32k window: 4096 slots 0.826, 16384 slots **0.863** (worse). Slot count is not only capacity: top-4 over a larger pool of random detectors is a *more specific* address, so similar-but-different contexts share fewer units than the trained value/output projections expect. Run 39 trained each slot count from scratch, which is why it saw no penalty. Testing longer windows at low load therefore needs a checkpoint trained with the big table: `m20_linux_depth4x64k_s0` (65536 slots) launched on GPU 0 at 12:34 after making the index form's memory independent of slot count (sliced no-grad top-k + sparse rescoring in `sparse_code`; index-equality intra-chunk scores; peak 10.7 GB at batch 8), and `m20_linux_depth4x32k_s0` (32768 slots) on GPU 1 at 12:40. Readout once trained: 4-head bpb at 32k and 64k windows vs slots {4096 (Run 40), 32768, 65536}, i.e. writes/slot {32, 4, 2} at 32k and {8, 4} at 64k. ~6–9 h each on shared GPUs. Also: the index form snapshots the state once per chunk, so 65536 slots x 128 chunks at 32k is 17 GB; `cross_eval.py --chunk 4096` at eval keeps it at ~2 GB per head.

**15:50 — trained with 32768 slots (`m20_linux_depth4x32k_s0`, 0.809 at 2048, i.e. unchanged in-domain as Run 39 predicted), evaluated long** (`results/extrap_m20_linux_depth4x32k.json`, `results/extrap_m20_linux_depth4_T64k.json`, chunk 4096 at eval):

| eval window | 4-head, 4096 slots | 4-head, 32768 slots | writes/slot (4096 → 32768) |
|---:|---:|---:|---|
| 2048 | 0.791 | 0.791 | 2 → 0.25 |
| 8192 | 0.762 | 0.744 | 8 → 1 |
| 32768 | 0.826 | **0.772** | 32 → 4 |
| 65536 | 0.956 | **0.880** | 64 → 8 |

More slots recover most of the 32k drift (0.826 → 0.772) and a third of the 64k drift (0.956 → 0.880), but the big table still degrades from 32k to 64k, and its last-eighth loss (1.03) sits well above its first-eighth (0.71): the memory gets worse as the window fills even at 8 writes/slot. Hypothesis: a second ceiling that is not slot collisions — the *same* short key (a bigram context) is legitimately written thousands of times in 64k bytes, so with decay 1.0 its read is a mass-normalised blend of every continuation ever seen. Slots cannot fix that; recency (decay < 1) or a longer key could. The 65536-slot arm (4 writes/slot at 64k) separates the two: if it lands near the 32k number, slots were the limit; if it lands near 0.88, it is same-key blending.

Same-window baselines at 64k (`results/extrap_m20_linux_T64k_baselines.json`): minGRU 1.340 (its own last eighth 1.47 vs first 1.31, so these windows are harder late), 1 head 1.145, NTK transformer 2.415. Gain of the 4-head arm over its backbone by window, 4096 → 32768 slots: 2048 0.50 / 0.50; 8192 0.58 / 0.60; 32768 0.40 / 0.46; 65536 0.38 / 0.46. With enough slots the gain is 0.46 at both 32k and 64k, against 0.50–0.60 at short windows: most of the absolute rise at 64k is the windows, not the memory. The remaining ~0.1 of gain lost past 8k is what the 65536-slot arm and, if needed, recency will address.

**22:10 — trained with 65536 slots (`m20_linux_depth4x64k_s0`, 0.811 at 2048, unchanged), evaluated long** (`results/extrap_m20_linux_depth4x64k.json`). The full slot x window grid, 4-head kernel arm, same windows down each column (backbone minGRU 1.291 / 1.341 / 1.229 / 1.340):

| eval window | 4096 slots | 32768 slots | 65536 slots | writes/slot at 65536 | gain over backbone (65536) |
|---:|---:|---:|---:|---:|---:|
| 2048 | 0.791 | 0.791 | 0.791 | 0.13 | 0.50 |
| 8192 | 0.762 | 0.744 | 0.741 | 0.5 | 0.60 |
| 32768 | 0.826 | 0.772 | **0.756** | 2 | 0.47 |
| 65536 | 0.956 | 0.880 | **0.858** | 4 | 0.48 |

Readings. (1) **Slots explain the drift.** At 64k, 64 → 8 → 4 writes/slot gives 0.956 → 0.880 → 0.858; at 32k, 32 → 4 → 2 gives 0.826 → 0.772 → 0.756. With slots scaled to the window the gain over the backbone is 0.47–0.50 at 32k and 64k, the same as at 2k (0.50); the absolute rise from 32k to 64k (0.10) matches the backbone's own rise on those windows (0.11), i.e. harder windows, not a failing memory. (2) **A residual ~0.1 of gain is lost between 8k and 32k that slots do not recover** (0.60 at 8k → 0.47 at 32k even at 2 writes/slot, while at 8k load barely matters: 0.58 at 8 writes/slot). That is the same-key blending hypothesis: with decay 1.0 a short key written thousands of times reads as a blend of every continuation; recency (decay < 1 in the chunked form) is the lever, untested. (3) The transformer with NTK scaling is at 1.60 (32k) and 2.42 (64k) on these windows.

**Claim (final for this run):** with the slot table scaled to the window (≈ window x active units / 2, a frozen buffer with no parameters and no retraining beyond choosing it), a model trained at 2048 keeps its full gain over the backbone at 32x its training length: 0.756 at 32k and 0.858 at 64k against 0.791 at 2k, where the same-windows backbone rises from 1.29 to 1.34. There is no hard context limit; the soft one is slot load, and past that a ~0.1 blending loss that slots do not fix. **Not a claim:** beyond 64k; more than one seed; that a slot table can be grown *after* training (it cannot, 08:50 above).

**Claim (revised):** models trained at 2048 with the sparse-key memory generalise to 32k-byte inputs with their gain intact up to the slot-load ceiling, with no retraining and no positional trick. A same-recipe RoPE transformer does not generalise past 2x untreated; with NTK-aware scaling it holds to 2x and degrades from 4x, and the head arm leads from 8192 on (0.05 at 8k, 0.29 at 16k, 0.77 at 32k on code; 0.10 / 0.42 / 0.97 on prose). **Not a claim:** what a transformer *trained* at 32k would do (16x attention cost per token); YaRN's per-band interpolation may do somewhat better than plain NTK; one seed per arm.

**Held-out CPython with depth heads (06:20, `results/cross_m20_linux_depth_to_cpython.json`, Run 37 settings):** 1 head 1.683 → 2 heads 1.543 → 4 heads 1.502 → 8 heads 1.480; transformer 1.460. The out-of-domain gap (0.22 in Run 37) closes the same way as in-domain: 0.04 at four heads, 0.02 at eight.

---

## Run 41 — designs, not controls: five ways to get better results (`scratchpad/screen_jobs.txt`, `results/scr_*.json`) — launched 2026-09-15 21:20, GPU 0

The Run 38 bin sweep says the remaining loss to a transformer is (a) blended reads on short, many-times-seen contexts and (b) the recurrence as a general model on unmatched bytes; the Run 40 long-window drift is the same blending. Nothing so far attacked either. Screening at the Run 35 tuned CPython setting (d=256, 4 layers, seq 2048, 8000 steps; references: 1-head gated 1.084, transformer 1.010, minGRU 1.450), all arms on a 2-head base (`--mem-every 2`), one seed, ~40 min each:

| arm | design | mechanism | prediction |
|---|---|---|---|
| `scr_base2` | 2 heads | baseline for this size | ~1.05 |
| `scr_keys24` | key widths 2 and 4 across heads (`--key-convs 2,4`) | short and long evidence through separate gates; free | −0.01 |
| `scr_ff1024` | d_ff 512 → 1024 | spend parameters on the unmatched-byte residual instead of heads | −0.02 |
| `scr_count` | **count memory** (`--count-head`): one-hot next bytes written under the context key, read back as a next-byte histogram, mixed into the output distribution with a mass-driven gate | turns the blend into the right object (an empirical continuation distribution); pointer-cache with sparse addressing; KC→MBON drives the output directly | −0.03 on the 2–3-gram bins |
| `scr_decay` | **learned recency** (`--learn-decay --decay 0.999`): per-head decay in the exact chunked read | recent writes dominate, as attention gets from position; also the long-window fix | −0.02 |
| `scr_count_decay` | both | | −0.04 |
| `scr_local512` | **one sliding-window attention layer** mid-stack (`--local-attn 512`, block-wise, RoPE, O(T·512)) | content-weighted selection over the last 512 bytes; memory heads for everything further | −0.04, and the most likely to reach 1.01 |
| `scr_all` | local + count + decay | | best arm; if < 1.01 the hybrid beats the transformer at 2048 too |

**Results (2026-09-16 01:00, first eight arms; transformer at this setting 1.010 / 2.24M):**

| arm | params | bpb | copyable / non | note |
|---|---:|---:|---|---|
| `scr_base2` 2 heads | 1.99M | 1.043 | 0.333 / 1.450 | |
| `scr_keys24` | 1.99M | 1.037 | 0.312 / 1.454 | −0.006, free, on the repeat bytes |
| `scr_ff1024` | 3.04M | 0.992 | 0.319 / 1.378 | −0.051 but 36 % more params than the transformer; shows the unmatched-byte residual is backbone capacity (non 1.450 → 1.378 = the transformer's 1.382) |
| `scr_count` | 1.99M | 1.039 | 0.330 / 1.446 | −0.004: **prediction (−0.03) wrong**; the value heads already carry the continuation histogram implicitly. 3x slower. Out. |
| `scr_decay` learned recency | 1.99M | 1.030 | 0.336 / 1.429 | −0.013, on the non-8-gram bytes as intended; learned half-lives 662 / 435 bytes (init 693) |
| `scr_count_decay` | 1.99M | 1.025 | 0.334 / 1.423 | additive |
| **`scr_local512`** one sliding-window attention layer | **2.25M** | **0.966** | 0.312 / 1.342 | **−0.077; beats the transformer by 0.044 at matched parameters (2.25M vs 2.24M)**, on both repeat (0.31 vs 0.36) and non-repeat (1.34 vs 1.38) bytes |

Extrapolation of the winner (CPython val past 40.1M, same windows per column; `results/extrap_scr_*.json`, `extrap_tuned_tf_ntk.json`): 2 heads 0.960 / 0.984 / 1.075 at 2k / 8k / 32k; **2 heads + local 0.891 / 0.886 / 0.900**; NTK transformer 0.928 / 1.239 / 2.642. The hybrid is flat to 16x its training window and ahead of the transformer at every length including 2048. Launched at 20M (`run_capacity.sh ... depth4local`: 4 heads + local 512 + keys 2,4; kernel then enwik8) at 01:05 on the first GPU with room. **Remaining arms (04:55):**

| arm | params | bpb | copyable / non | note |
|---|---:|---:|---|---|
| `scr_all` local + count + recency | 2.26M | 0.972 | 0.318 / 1.348 | no better than local alone (0.966); count and recency add nothing once attention is present; ~2x slower |
| `scr_ff640` (param-matched FFN) | 2.25M | 1.024 | 0.329 / 1.423 | same parameters as the attention layer buy 0.019 here vs 0.077 there |
| `scr_ff640_keys24` | 2.25M | 1.019 | 0.309 / 1.427 | keys add their 0.005 |
| `scr_archive_recent` (top 1.0, inner learned 0.999) | 1.99M | 1.028 | 0.332 / 1.428 | = both decaying (1.030), keeps one head with unlimited reach |
| `scr_recent_archive` (top learned, inner 1.0) | 1.99M | 1.032 | 0.334 / 1.433 | |

**Screening verdict.** One sliding-window attention layer is worth 0.077 at matched parameters and is the only design that beats the transformer at 2048 (0.966 vs 1.010); it also keeps the head arm's flat extrapolation (0.89–0.90 to 32k). Everything else is ≤ 0.013: recency 0.013 (use archive + recent if at all), mixed keys 0.005, count memory 0.004 (out), FFN width 0.019 at matched params. Attention within a short window plus sparse-key memory for everything beyond it is the design. Wall-clock (GPU shared with the 20M run for the later arms): local512 2302 s vs base 719 s unshared; not comparable, measure later.

Plumbing (all tested, 141 passing): `CountHead` (`flybrain/model/hybrid.py`), `sparse_memory_read` factored out of `Block` with scalar/learned decay (chunked == parallel with decay 0.9 verified incl. grads), `LocalAttention` (equals a banded full-attention reference; sees exactly its window), `--key-convs`. Winners go to 20M (kernel + enwik8, 4 heads) overnight.

---

## Run 42 — precision: what the memory needs, per head (`cross_eval.py --state-bits --bf16`, `results/prec256_*.json`) — 2026-09-15 22:10

Everything in this lab is fp32. Eval-only on the 4-head kernel checkpoint: the memory state carried between 256-byte chunks is fake-quantised (symmetric per-row, `fake_quant`) to b bits per head, or the whole model is cast to bf16. Head order [top, after-L2, after-L4, after-L6]. Kernel val, same windows as Run 40.

| setting | 2048 | 32768 |
|---|---:|---:|
| fp32 | 0.791 | 0.826 |
| state 16 bits, all heads | 0.791 | 0.826 |
| state 8 bits, all heads | 0.793 | 0.881 |
| state 6 bits, all heads | 0.832 | 1.308 |
| 8 bits on the top head only | 0.791 | 0.827 |
| 8 bits on the L6 head only | 0.791 | 0.830 |
| 8 bits on the L4 head only | 0.791 | 0.840 |
| 8 bits on the L2 head only | 0.792 | 0.845 |
| bf16 weights (whole model) | 0.791 | 0.826 |
| bf16 weights + 8-bit state | 0.793 | 0.877 |
| Author's layout: L2 fp32, L4 16, L6 8, top 8 | 0.791 | 0.832 |

- **16-bit memory state is lossless** at both lengths; bf16 weights are lossless. Half the state memory for free.
- **8 bits costs almost nothing at 2048 (+0.002) and 0.055 at 32k.** The damage grows with the window because a per-row absmax scale gets coarser as rows accumulate mass; a scale tied to the row's own mass (or quantising the normalised read instead of the raw sums) would likely fix most of it. Untested.
- **Precision by depth goes early-fine, late-coarse** (the author's intuition, not mine): at 8 bits the top head costs 0.001, L6 0.004, L4 0.014, L2 0.019. The deep layers amplify an early head's error; the top head's output has nothing after it to amplify it. the author's mixed layout (fp32 / 16 / 8 / 8) is within 0.006 of fp32 at 32k.
- 6 bits breaks it (2048: +0.04; 32k: +0.5).

**Claim:** the memory state runs at 16 bits with no loss and at 8 bits with +0.002 at the training window; the sensitivity to state precision decreases with depth. Weights at bf16 are free. **Not a claim:** integer weights (not tested here); a trained-at-low-precision model; the count head's precision (not in this checkpoint).

---

## Run 43 — the design at 20M: 4 heads + one sliding-window attention layer (`scripts/run_capacity.sh <dev> depth4local <seed> 16000 linux|enwik8`, `results/m20_linux_depth4local_s0.json`) — 2026-09-16 06:08

Run 41's winner (heads after layers 2/4/6 + top, one 512-window attention layer after layer 4, key widths cycled 2,4) at the 20M kernel recipe (d 512, 8 layers, seq 2048, 16k steps; same corpus, windows and steps as Runs 37–38). Trained on GPU 0 shared with the screening tail until ~04:55, so the wall-clock is not comparable.

| arm | params | kernel bpb | copyable / non | held-out CPython | train s |
|---|---:|---:|---|---:|---:|
| transformer (2 seeds) | 17.09M | 0.781 / 0.789 | 0.160 / 1.117 | 1.460 | 6657 (unshared) |
| 4 heads (2 seeds) | 15.01M | 0.811 / 0.814 | 0.182 / 1.151 | 1.502 | ~11000 (unshared) |
| 8 heads | 17.15M | 0.802 | 0.172 / 1.143 | 1.480 | 17210 (shared) |
| **4 heads + local 512 + keys 2,4** (2 seeds) | 16.07M | **0.764 / 0.762** | **0.154 / 1.094**, 0.155 / 1.090 | **1.459 / 1.439** | 18944 (shared), **6866 (unshared)** |

- **First code win at 20M, replicated**: 0.764 / 0.762 vs the transformer's 0.781 / 0.789, at 6 % fewer parameters. Seed spread 0.002 (seed 1 ran 06:10–08:05 on GPU 1, `results/m20_linux_depth4local_s1.json`); the margin is 0.017–0.027 on every seed pair.
- **Wall-clock is a tie.** Seed 1 on an unshared GPU: 6866 s against the transformer's 6657 s (unshared seed 1) for the same steps at 2048. The earlier '4-head arm ~11000 s unshared' figure in Run 38 was extrapolated from shared runs and is withdrawn; with the sliced top-k of Run 40 the head arm costs about the same as attention at 2048 and, unlike attention, does not grow with the window.
- The gain over 4 heads (−0.047) is smaller than at d=256 (−0.077) and lands on *both* halves: copyable 0.182 → 0.154 (now below the transformer's 0.160) and non-copyable 1.151 → 1.094 (below the transformer's 1.117). (First write-up said the gain must be the short-match bins; Run 44's bin sweep shows it is the *unmatched* bytes, 1.737 → 1.656, with the 2–4-gram bins still 0.005–0.022 behind the transformer.)
- **Held-out CPython 1.459 / 1.439 vs transformer 1.460** (4 heads 1.502, 8 heads 1.480). The out-of-domain gap is closed on one seed and reversed on the other.

Extrapolation, trained at 2048 (`cross_eval.py --lens 2048..32768 --chunk 4096 --n-batches 4`, kernel val past 200.1M; `results/extrap_m20_linux_depth4local.json`; the other columns from Run 40 on the same windows):

| eval window | 4 heads | **4 heads + local** | NTK transformer |
|---:|---:|---:|---:|
| 2048 | 0.791 | **0.706 / 0.703** | 0.76 |
| 4096 | 0.752 | **0.749 / 0.749** | 0.73 |
| 8192 | 0.762 | **0.700 / 0.702** | 0.81 |
| 16384 | 0.743 | 0.773 / 0.771 | 1.03 |
| 32768 | 0.826 | **0.811 / 0.806** | 1.60 |

The attention layer does not break length generalisation (its RoPE only ever sees offsets ≤ 512): 0.70–0.81 from 2k to 32k, the same shape as the 4-head arm, with the same 4096-slot drift past 16k that Run 40 showed a window-sized slot table removes. Four batches per length, so ±0.03 between neighbouring rows is noise; the 2048 column is 16 windows and reads 0.085 below the 4-head arm where the full validation set says 0.047.

**enwik8, same arm, 20M (`results/m20_enwik8_depth4local_s0.json`, 11568 s, GPU 0 unshared after 08:05):**

| arm | params | enwik8 bpb | copyable / non |
|---|---:|---:|---|
| transformer (2 seeds) | 17.09M | 1.360 / 1.355 | 0.319 / 1.491 |
| 4 heads | 15.01M | 1.351 | 0.272 / 1.487 |
| 8 heads | 17.15M | 1.349 | 0.263 / 1.486 |
| **4 heads + local 512 + keys 2,4** (2 seeds) | 16.07M | **1.307 / 1.308** | **0.253 / 1.440**, 0.250 / 1.442 |

Seed 1 (`m20_enwik8_depth4local_s1.json`, 6458 s unshared vs the transformer's 11870 / 17539 s, both shared) lands at 1.308. −0.044 against the best head arm and −0.047…−0.053 against the transformer seeds: the prose margin is larger than the code margin, and again it is the non-copyable bytes that move (1.487 → 1.440). Extrapolation on enwik8 (`extrap_m20_enwik8_depth4local.json`, 16 batches, offset 90.1M, same windows as Run 40): **1.302 / 1.303 / 1.305 / 1.231 / 1.306** at 2k / 4k / 8k / 16k / 32k. Flat. The one-head arm on the same windows was 1.37–1.43, the NTK transformer 1.356 / 1.369 / 1.491 / 1.743 / 2.397.

**Claim (two seeds each side on code and prose):** at 20M on the kernel, a minGRU with four sparse-key memory heads through its depth plus one 512-byte sliding-window attention layer beats a tuned RoPE transformer in domain (0.764 / 0.762 vs 0.781 / 0.789), ties or beats it out of domain (1.459 / 1.439 vs 1.460), has 6 % fewer parameters, trains in the same wall-clock at 2048 (6866 s vs 6657 s, unshared), has per-token cost flat in window length, and evaluates unchanged to 16× its training window (two seeds within 0.005 of each other at every length). On enwik8 it beats the transformer by 0.05 (1.307 / 1.308 vs 1.355 / 1.360) and is flat to 32k (1.30–1.31) where the NTK transformer reaches 2.40. **Not yet:** anything above 17M parameters; a transformer trained at the long window.

---

## Run 44 — the local layer's window, and where its gain lands (`scratchpad/gpu1_window.sh`, `results/scr_local{128,256,1024}.json`, `results/m20_linux_depth4local_s0_masksweep.json`) — 2026-09-16 08:07–09:00, GPU 1

Same setting as Run 41 (d=256 tuned CPython, 2 heads, 8000 steps; base 1.043, transformer 1.010), varying only the window of the one attention layer after layer 2:

| window | bpb | copyable / non | train s (unshared) |
|---:|---:|---|---:|
| 128 | 0.984 | 0.323 / 1.364 | 848 |
| 256 | 0.974 | 0.319 / 1.351 | 870 |
| **512** (Run 41) | **0.966** | 0.318 / 1.348 | (shared) |
| 1024 | 0.971 | 0.316 / 1.347 | 1069 |

Saturates by 512; 128 keeps three quarters of the gain. 512 stays the setting. (Seed noise at this setting is ~0.005; 256 vs 1024 is not a difference.)

Match-length bins of the 20M kernel arms (`mask_sweep.py`, same validation windows; the 4-head and 8-head rows from Run 38):

| longest earlier match | share of bytes | transformer | 4 heads | 8 heads | **4 heads + local** |
|---|---:|---:|---:|---:|---:|
| none | 0.344 | 1.706 | 1.737 | 1.733 | **1.656** |
| 2-gram | 0.085 | **0.745** | 0.804 | 0.792 | 0.750 |
| 3-gram | 0.063 | **0.422** | 0.465 | 0.457 | 0.444 |
| 4–5 | 0.091 | **0.324** | 0.357 | 0.340 | 0.332 |
| 6–7 | 0.065 | 0.264 | 0.283 | 0.269 | **0.261** |
| ≥ 8 | 0.351 | 0.160 | 0.182 | 0.172 | **0.154** |

- **I had this wrong in Run 43's first write-up.** The attention layer's win is not on the short-match bins; it is on the 34 % of bytes with *no* earlier match in the window, where the hybrid is now 0.05 ahead of the transformer and 0.08 ahead of the 4-head arm (weighted: −0.017 of the −0.020 total margin comes from this bin). Attention over 512 bytes is a better local-syntax model than the recurrence, which is why the count/recency/key designs (all retrieval) gave nothing and this did.
- **The short-match bins are the transformer's remaining lead**: +0.005 / +0.022 / +0.007 on 2 / 3 / 4-gram bins, ~+0.003 weighted. That is the one place the design still loses. Long retrieval (≥ 6-gram) is now ahead of attention.

**Claim:** a 512-byte attention window is the right size (flat from 256 to 1024, three quarters of the gain at 128); at 20M its gain is on unmatched bytes, and the residual against the transformer is 2–4-gram matches, worth ~0.003 in total. **Corrects** the "takes the short-match bins" sentence in Run 43 and RESULTS.

---

## Run 45 — the expensive long-context control: a transformer *trained* at 8192 (`scratchpad/gpu0_tf8k.sh`, `results/m20_linux_tf8k_s0.json`, `extrap_m20_linux_tf8k{,_ntk}.json`) — 2026-09-16 11:13–16:01, GPU 0 unshared

Same 20M transformer and recipe as Run 37 but trained with 8192-byte windows (batch 2, so the same 16k tokens per step and 16k steps). Its own validation, on 8192-byte windows, is 0.726 (copyable fraction 0.50 at that window vs 0.35 at 2048, so not comparable to the 2048 numbers). Training took **17241 s** against 6657 s for the same transformer at 2048 and 6866 / 6458 s for the hybrid. Scored on the shared extrapolation windows (Run 40 protocol, 4 batches per length):

| eval window | transformer trained @2048 (NTK) | **transformer trained @8192** | + NTK from 8192 | **hybrid trained @2048** (2 seeds) |
|---:|---:|---:|---:|---:|
| 2048 | 0.76 | 0.800 | | **0.706 / 0.703** |
| 4096 | 0.73 | 0.812 | | **0.749 / 0.749** |
| 8192 | 0.81 | 0.777 | | **0.700 / 0.702** |
| 16384 | 1.03 | 1.219 | 0.854 | **0.773 / 0.771** |
| 32768 | 1.60 | 2.370 | 0.913 | **0.811 / 0.806** |

- **The hybrid trained at a quarter of the window beats the transformer trained at 8192 at every length, including 8192 itself** (0.700 vs 0.777), at 40 % of the training time. With NTK scaling from 8192 the transformer holds to 0.85 / 0.91 at 16k / 32k, still 0.08–0.10 behind.
- Training at 8192 cost the transformer its short-window number (0.800 at 2048 vs 0.76 when trained at 2048): a quarter as many distinct windows per step and a first-eighth penalty (1.24) it never trained away. Its gain at 8192 over the 2048-trained model with NTK is 0.03.
- So the long-context claim no longer rests on the transformer being evaluated off its training length: trained *for* the length, at 2.6× the cost, it still loses.

**Claim:** at 20M on the kernel, a transformer trained at 8192 scores 0.777 on 8192-byte windows and 0.85 / 0.91 (NTK) at 16k / 32k; the recurrence + heads + local-attention hybrid trained at 2048 scores 0.70 / 0.77 / 0.81 on the same windows at 40 % of the training time. **Not a claim:** a transformer trained at 32k (4× the cost again); more than one seed of the 8192 control.

---

## Run 46 — placement and number of local layers (`--local-at`, `scratchpad/gpu{1,0}_place*.sh`, `results/scr_localat*.json`, `scr_local512_ff640.json`) — 2026-09-16 08:57–16:53

Run 41 setting (d=256, 4 minGRU layers, heads after layer 2 and top; transformer 1.010 at 2.24M). `--local-at` (new, tested: single-layer checkpoints load unchanged) places the 512-window attention layer after chosen layers, one or several.

| local layer(s) after | params | bpb | copyable / non |
|---|---:|---:|---|
| 1 | 2.25M | 0.984 | 0.321 / 1.365 |
| 2 (Run 41) | 2.25M | 0.966 | 0.312 / 1.342 |
| 3 | 2.25M | 0.964 | 0.306 / 1.342 |
| 1 and 3 | 2.52M | **0.943** | 0.300 / 1.313 |
| 2 and 4 | 2.52M | 0.952 | 0.314 / 1.319 |
| 2, FFN 640 (param-matched control) | 2.51M | 0.954 | 0.313 / 1.323 |

- **Placement:** after the first layer is worse by 0.02; after 2 or 3 is the same. The attention layer wants at least one recurrent layer of context below it and does not care beyond that.
- **A second layer is not worth its parameters.** Two layers at 1+3 beat one layer by 0.023, but the same 0.26M spent on FFN width buys 0.012 (0.954), so the second attention layer's own value is 0.011 at 1+3 and −0.002 at 2+4, at the edge of the ~0.005 seed noise. One layer stays the design; the 20M Run 43 configuration is not changed.
- The arms that ran while GPU 1 was shared with the other project's 20 GB process OOMed twice (two local layers need ~3.5 GB at this size); they were moved to GPU 0.

**Claim:** one 512-window attention layer, placed anywhere from the middle of the stack on, is the whole effect; a second buys ≤ 0.01 over the same parameters as FFN width. **Not a claim:** more than one seed per cell; behaviour at 20M (not rerun: the expected gain is inside noise).

---

## Run 47 — trained low precision at 20M (`--amp` bf16 autocast with the memory head pinned to fp32, `--state-bits 16`; `results/m20_linux_{prec,state16,amp}_s0.json`) — 2026-09-16 22:25 → 2026-09-17 15:53

Run 43's kernel arm (4 heads + local 512 + keys 2,4) trained under bf16 autocast with every head's carried state fake-quantised to 16 bits between chunks. Same data, steps and seed as Run 43 seed 0.

| arm | bpb | copyable / non | train s (unshared) |
|---|---:|---:|---:|
| fp32 (Run 43, 2 seeds) | 0.764 / 0.762 | 0.154 / 1.094 | 6866 |
| bf16 autocast + 16-bit state | 0.784 | **0.197** / 1.102 | 6935 |
| 16-bit state alone | 0.784 | 0.197 / 1.101 | 8716 (shared) |
| **bf16 autocast alone** | **0.761** | 0.160 / 1.086 | **4919** |
| 16-bit state, straight-through fix | **0.761** | 0.156 / 1.089 | 7658 |
| 8-bit state, straight-through fix | 0.839 | 0.214 / 1.177 | 7561 |

- **The split is clean: the state quantiser caused all of it.** 16-bit state alone reproduces 0.784 / 0.197 exactly; bf16 autocast alone is 0.761 (fp32 seeds 0.764 / 0.762) in **4919 s against 6866 s, 28 % faster**.
- **Why, and it was a bug:** `fake_quant` used `.round()`, whose gradient is zero in PyTorch. The forward value was right, so Run 42's eval-only sweep saw nothing, but in training the memory's write path was cut at every chunk boundary — the heads learned to write with no gradient from anything older than the current 256-byte chunk, which is exactly a retrieval loss (copyable 0.154 → 0.197, non-copyable unchanged). Fixed with a straight-through estimator (`S + (q - S).detach()`, quantised forward / identity backward, tested); reruns queued (`m20_linux_state16ste_s0`, `m20_linux_state8ste_s0`).
- Autocast costs memory rather than saving it here (cached bf16 weight copies: the 50M hybrid needs 18.3 GB at batch 4 under autocast, 14.4 GB in fp32), and the minGRU scan (fp64) and the head (pinned fp32) stay out of it, so the 28 % is from the projections, FFNs and the one attention layer.

- **With the estimator fixed, 16-bit state is free**: 0.761 (repeat bytes 0.156 vs fp32's 0.154), i.e. the whole 0.02 was the missing gradient.
- **8-bit is not free even with the fix**: 0.839, +0.075, again concentrated on repeat bytes (0.214). Eval-only 8-bit cost 0.002 at this window (Run 42), so the damage is to *learning* with a coarse state, not to reading one: the model trains against a state whose small entries are rounded away, and the writes it learns are the ones that survive that rounding. The state-bit budget for training is 16.

**Claim:** at 20M both halves of low precision are free — bf16 autocast 0.761 in 4919 s (28 % faster than fp32's 6866 s) and a 16-bit memory state 0.761 — once `fake_quant` passes gradients straight through. The 0.02 in the first run was a bug in this lab's code, not a property of quantised memory. 8-bit trained costs 0.075 (eval-only it cost 0.002), so 16 bits is the training budget. **Not yet:** the two combined; 50M under this recipe (Run 49 was fp32).

---

## Run 48 — sharper addresses for 2–4-gram matches (`scratchpad/gpu1_shortmatch.sh`, `results/scr_local_{keys33,keys23,topk2,topk8}.json`) — 2026-09-16 22:22 → 2026-09-17 12:00, GPU 1 / GPU 0

Run 44's bin sweep left the transformer ahead only on 2–4-gram matches (~0.003 weighted). Two ways to sharpen the address at the Run 41 setting (d=256, 2 heads + local 512; 0.966, copyable 0.312): longer key orders on the heads, and fewer / more active slots per key.

| arm | bpb | copyable / non |
|---|---:|---:|
| local512 (keys 2,2, top-4) | 0.966 | 0.312 / 1.342 |
| keys 3,3 | 0.961 | 0.299 / 1.342 |
| keys 2,3 | 0.965 | 0.304 / 1.344 |
| top-2 | 0.979 | 0.344 / 1.344 |
| top-8 | 0.965 | 0.306 / 1.343 |

Key order moves repeat bytes by ≤ 0.013 and the total by ≤ 0.005, inside seed noise; top-2 is worse (0.013, all on repeat bytes: fewer active units per key is a blurrier address, not a sharper one); top-8 equals top-4. (top-8 ran on GPU 0 beside the 50M run after the other project's server OOMed it twice on GPU 1; 3684 s is shared time.)

**Claim:** the address is not the limit on 2–4-gram matches: key order 2–3 and 4–8 active units are equivalent, 2 active units is worse. The transformer's ~0.003 lead on those bins is not recoverable by sharpening the memory's key. Closed.

---

## Run 49 — the scale step: 50M (`scripts/run_50m.sh`, `results/m50_linux_{hybrid,tf}_s0.json`, `cross_m50_to_cpython.json`, `extrap_m50_{hybrid,tf_ntk}.json`, `m50_linux_*_masksweep.json`) — 2026-09-17 06:09 → 21:40, GPU 0

d 768, 12 layers, d_ff 1536, batch 4 × 32k steps (the same 260M training bytes as the 20M runs), lr 4e-4, warmup 2000, cosine, wd 0.1, fp32, kernel corpus. Hybrid: 6 heads (one per two layers) + one 512-window attention layer after layer 6 + key widths 2,4. Both arms shared GPU 0 with another project's run for part of the time, so the wall-clocks (hybrid 31237 s, transformer 24228 s) are not comparable.

| | params | kernel | copyable / non | held-out CPython |
|---|---:|---:|---|---:|
| transformer, 20M | 17.09M | 0.781 / 0.789 | 0.160 / 1.117 | 1.460 |
| hybrid, 20M | 16.07M | 0.764 / 0.762 | 0.154 / 1.094 | 1.459 / 1.439 |
| transformer, 50M | 57.12M | 0.735 | 0.147 / 1.066 | 1.405 |
| **hybrid, 50M** | 52.45M | **0.696** | **0.138 / 1.009** | **1.386** |

- **The margin grows with scale**: 0.017–0.025 at 20M → **0.039** at 50M, at 8 % fewer parameters. Out of domain it goes from a tie (1.459 / 1.439 vs 1.460) to a 0.019 lead.
- **Extrapolation, trained at 2048** (4 batches per length, kernel val past 200.1M): hybrid 0.644 / 0.690 / 0.636 / 0.723 / 0.746 at 2k / 4k / 8k / 16k / 32k; NTK transformer 0.673 / 0.723 / 0.723 / 0.941 / 1.404. **The hybrid now leads at every length including the training window** (at 20M the NTK transformer still won 2k and 4k).
- **Bin sweep, and it explains Run 48.** At 50M the hybrid leads every match-length bin except 6–7-grams (+0.006):

| longest earlier match | share | transformer | hybrid | weighted |
|---|---:|---:|---:|---:|
| none | 0.340 | 1.633 | **1.531** | −0.035 |
| 2-gram | 0.084 | 0.699 | **0.678** | −0.002 |
| 3-gram | 0.062 | 0.402 | 0.402 | 0.000 |
| 4–5 | 0.090 | 0.305 | **0.302** | −0.000 |
| 6–7 | 0.064 | **0.245** | 0.252 | +0.000 |
| ≥ 8 | 0.360 | 0.147 | **0.138** | −0.003 |

The 20M residual on 2–4-gram matches is gone at 50M without any change to the memory — so Run 48 found nothing because the address was never the limit; capacity was. Nine tenths of the 50M margin (−0.035 of −0.039) is on bytes with no earlier match, where the local attention layer and the freed recurrence do the work.

**Claim:** at 50M on the kernel the hybrid beats a tuned transformer of the same recipe by 0.039 with 8 % fewer parameters, beats it out of domain by 0.019, leads at every evaluated context length from 2k to 32k, and leads every match-length bin but one. The advantage widens from 20M to 50M. **Not a claim:** more than one seed at 50M; wall-clock at 50M (both runs shared the GPU); enwik8 at 50M; anything above 57M parameters or 260M training bytes.

---

## Run 50 — what could related files share? Exact-match census of the held-out kernel files (`scripts/census.py`, `results/census_linux_{dev,test}.json`) — 2026-10-03, CPU

Every long-context number in Runs 40–49 crossed into unrelated files, because the corpus is a file shuffle. This run asks what *related* files could share. Targets are the held-out files past byte 202.1M (split by file order: dev = the first 500, test = the other 499) with at least four same-directory files entirely inside the training bytes: 353 on dev, 339 on test (median 14 related files each). The scored bytes are the first 4,096 body bytes of each target (the whole body if shorter): 1,299,363 on test. Each scored byte is binned by exact n-gram match: **A** continues an n-gram seen earlier in the target itself (header included); **B** is not A but the n-gram and its continuation occur in the related prefix (the target's same-directory training files, path order, headers kept, cut at P bytes); **B'** is the same test against an equal-size random prefix from other directories (seeded; never the target's directory, never a file with the target's content); **C** is neither. Shares are byte-weighted over targets; the margin carries a 95 % interval from resampling the targets (files). Test, n = 8:

| P | A | B | B' | C | B - B' (95 % interval) |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.357 | 0.098 | 0.029 | 0.545 | +0.070 (+0.061, +0.079) |
| 16,384 | 0.357 | 0.124 | 0.036 | 0.518 | +0.088 (+0.079, +0.098) |
| 32,768 | 0.357 | 0.152 | 0.048 | 0.491 | +0.104 (+0.093, +0.115) |

The same at P = 32,768 for the other match lengths:

| n | A | B | B' | B - B' (95 % interval) |
|---:|---:|---:|---:|---:|
| 8 | 0.357 | 0.152 | 0.048 | +0.104 (+0.093, +0.115) |
| 12 | 0.255 | 0.130 | 0.029 | +0.101 (+0.090, +0.113) |
| 16 | 0.183 | 0.115 | 0.021 | +0.095 (+0.083, +0.108) |

Targets whose directory holds at least two other files in their own split (83 on test, 313,317 bytes): the same bytes against related files the weights trained on and against related files they never saw (n = 8):

| P | B, trained files | B, never-trained files | B' (random, same size as the never-trained prefix) |
|---:|---:|---:|---:|
| 8,192 | 0.092 | 0.092 | 0.027 |
| 16,384 | 0.109 | 0.110 | 0.034 |
| 32,768 | 0.132 | 0.127 | 0.043 |

**Claim:** on the 339 test targets at n = 8, a 32,768-byte related prefix holds the exact n-gram and next byte for 15.2% of the scored bytes that the target itself cannot supply, against 4.8% for an equal-size random prefix from other directories (35.7% are self-copyable, 49.1% match nowhere); B at P = 8,192, 16,384, 32,768 is 9.8%, 12.4%, 15.2%, B' 2.9%, 3.6%, 4.8%. The related margin B - B' at P = 32,768 is +0.104 (+0.093, +0.115) at n = 8, +0.101 (+0.090, +0.113) at n = 12 and +0.095 (+0.083, +0.108) at n = 16 (95 % intervals over targets; dev, n = 8: +0.116 (+0.104, +0.130)). For never-trained related files (83 targets) B is 12.7% against 13.2% for trained files on the same targets. **Not a claim:** that a model can turn these matches into lower loss (Run 51 measures that); which related files to read (path order is arbitrary and deterministic); anything outside kernel C and header files; that the 95 % intervals cover variation between directories (they resample targets, which are files; resampling whole directories instead of files widens the gate-number interval of Run 51 by about 1.4 times on dev and 1.1 times on test (final review: dev [0.039, 0.055] became about [0.036, 0.058]), and this entry's own intervals were not remeasured that way); the dev numbers, which are shown only for consistency.

---

## Run 51 — does reading related files help with no new model code? (`scripts/probe_commons.py --mode context`, `results/commons_ctx_{x64k,local,m50,tf}_{dev,test}.json`) — 2026-10-03 to 2026-10-04, GPU 0 (the 65,536-slot model on dev, the three reference models) and GPU 1 (the 65,536-slot model on test)

Plain `forward`, batch 1, one sequence per condition per test target (339 targets), scoring only the target's bytes: **alone** (header + target), **related** (its same-directory training files with headers in path order, cut at P bytes, then header + target), **random** (an equal-size seeded draw from other directories) and **never trained** (related files from the target's own split; 83 targets: the "gain never trained" column covers those and no others, the other columns all 339). Heads run at chunk 4096 for the long sequences. Every condition scores the same target bytes, so only the differences are read: gains are alone minus the condition (positive = fewer bits/byte) with 95 % intervals from resampling the targets (files). First 4,096 target bytes (the transformer: its first 2,048, prefix 2,048, NTK-scaled):

| model | P | alone (bits/byte) | gain related | gain random | related minus random | gain never trained |
|---|---:|---:|---:|---:|---:|---:|
| 4 heads x 65,536 slots (`m20_linux_depth4x64k_s0`) | 8,192 | 0.782 | -0.084 (-0.089, -0.079) | -0.102 (-0.105, -0.099) | +0.018 (+0.013, +0.024) | -0.088 (-0.094, -0.081) |
| 4 heads x 65,536 slots (`m20_linux_depth4x64k_s0`) | 16,384 | 0.782 | -0.130 (-0.137, -0.122) | -0.161 (-0.165, -0.157) | +0.031 (+0.024, +0.039) | -0.132 (-0.143, -0.122) |
| 4 heads x 65,536 slots (`m20_linux_depth4x64k_s0`) | 32,768 | 0.782 | -0.190 (-0.200, -0.180) | -0.241 (-0.247, -0.234) | +0.050 (+0.041, +0.062) | -0.173 (-0.189, -0.158) |
| 4 heads x 4,096 + local 512 (`m20_linux_depth4local_s0`) | 2,048 | 0.732 | -0.012 (-0.013, -0.010) | -0.019 (-0.020, -0.018) | +0.007 (+0.005, +0.009) | -0.014 (-0.017, -0.011) |
| 4 heads x 4,096 + local 512 (`m20_linux_depth4local_s0`) | 4,096 | 0.732 | -0.023 (-0.025, -0.021) | -0.032 (-0.033, -0.030) | +0.009 (+0.007, +0.011) | -0.027 (-0.032, -0.023) |
| 50M hybrid (`m50_linux_hybrid_s0`) | 2,048 | 0.671 | -0.012 (-0.013, -0.010) | -0.017 (-0.018, -0.016) | +0.005 (+0.004, +0.007) | -0.013 (-0.016, -0.011) |
| 50M hybrid (`m50_linux_hybrid_s0`) | 4,096 | 0.671 | -0.023 (-0.025, -0.021) | -0.029 (-0.031, -0.028) | +0.007 (+0.005, +0.009) | -0.026 (-0.029, -0.022) |
| transformer 17.09M (`m20_linux_s0_transformer`) | 2,048 | 0.762 | -0.032 (-0.034, -0.030) | -0.043 (-0.045, -0.041) | +0.011 (+0.009, +0.013) | -0.035 (-0.038, -0.031) |

The first 2,048 scored bytes of every model, so the rows compare:

| model | P | alone (bits/byte) | gain related | gain random | related minus random |
|---|---:|---:|---:|---:|---:|
| 4 heads x 65,536 slots (`m20_linux_depth4x64k_s0`) | 8,192 | 0.786 | -0.092 (-0.098, -0.085) | -0.119 (-0.122, -0.115) | +0.027 (+0.020, +0.035) |
| 4 heads x 65,536 slots (`m20_linux_depth4x64k_s0`) | 16,384 | 0.786 | -0.136 (-0.145, -0.127) | -0.179 (-0.184, -0.173) | +0.043 (+0.034, +0.052) |
| 4 heads x 65,536 slots (`m20_linux_depth4x64k_s0`) | 32,768 | 0.786 | -0.191 (-0.202, -0.179) | -0.256 (-0.264, -0.248) | +0.065 (+0.055, +0.077) |
| 4 heads x 4,096 + local 512 (`m20_linux_depth4local_s0`) | 2,048 | 0.738 | -0.009 (-0.012, -0.006) | -0.021 (-0.022, -0.020) | +0.012 (+0.009, +0.015) |
| 4 heads x 4,096 + local 512 (`m20_linux_depth4local_s0`) | 4,096 | 0.738 | -0.020 (-0.023, -0.017) | -0.032 (-0.034, -0.030) | +0.012 (+0.010, +0.015) |
| 50M hybrid (`m50_linux_hybrid_s0`) | 2,048 | 0.676 | -0.010 (-0.013, -0.008) | -0.018 (-0.019, -0.017) | +0.008 (+0.005, +0.011) |
| 50M hybrid (`m50_linux_hybrid_s0`) | 4,096 | 0.676 | -0.020 (-0.022, -0.017) | -0.029 (-0.031, -0.027) | +0.009 (+0.007, +0.012) |
| transformer 17.09M (`m20_linux_s0_transformer`) | 2,048 | 0.762 | -0.032 (-0.034, -0.030) | -0.043 (-0.045, -0.041) | +0.011 (+0.009, +0.013) |

Where the gain lands, 4 heads x 65,536 slots (`m20_linux_depth4x64k_s0`), P = 32,768 (census bins at n = 8; the share-weighted gains of the three bins add up to the total gain; the B' row bins by the random prefix and gains from reading it):

| bin | share | alone (bits/byte) | share-weighted gain |
|---|---:|---:|---:|
| A | 0.357 | 0.205 | -0.095 (-0.100, -0.091) |
| B | 0.152 | 0.514 | +0.017 (+0.013, +0.021) |
| C | 0.491 | 1.285 | -0.112 (-0.117, -0.107) |
| B' | 0.048 | 0.271 | -0.000 (-0.001, -0.000) |

Headroom H(X) = share(X) x (loss(X) - loss(A)) of the alone condition:

| P | share B | share B' | H(B) | H(B') | H(B) - H(B') |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 0.098 | 0.029 | +0.018 (+0.014, +0.022) | +0.000 (-0.000, +0.001) | +0.018 (+0.013, +0.022) |
| 16,384 | 0.124 | 0.036 | +0.032 (+0.026, +0.039) | +0.002 (+0.001, +0.002) | +0.030 (+0.024, +0.037) |
| 32,768 | 0.152 | 0.048 | +0.047 (+0.040, +0.056) | +0.003 (+0.002, +0.004) | +0.044 (+0.037, +0.052) |

**Claim:** with 4 heads x 65,536 slots (`m20_linux_depth4x64k_s0`) and no new model code, reading 32,768 bytes of same-directory training files makes bits/byte on the first 4,096 target bytes worse than reading the target alone (0.782): a gain of -0.190 (-0.200, -0.180) bits/byte (positive = fewer bits, so a negative gain is more bits; 95 % interval over targets), in which bin B gains +0.017 (+0.013, +0.021) and bins A and C lose; the same number of random kernel bytes gives -0.241 (-0.247, -0.234). Related minus random, +0.050 (+0.041, +0.062), is the content effect: both kinds of prefix make the score worse than reading the file alone (in every row of both tables), related files less so, and the difference is not a net gain from context. The gate number H(B) - H(B') at n = 8 and P = 32,768 is **+0.044 (+0.037, +0.052)** bits/byte on test (dev +0.047 (+0.039, +0.055)): at or above the 0.02 threshold, which the Gate 1 table reads as Stage 2 (train a brain with a shared memory) — the decision is Kyle's. Related gains at the other models' trained load: local-attention model -0.023 (-0.025, -0.021) (P = 4,096), 50M hybrid -0.023 (-0.025, -0.021) (P = 4,096); on the first 2,048 bytes the transformer reference (2,048-byte prefix, NTK-scaled) -0.032 (-0.034, -0.030), against -0.092 (-0.098, -0.085) for the 65,536-slot model at 8,192 bytes. **Not a claim:** that the headroom is realised (bin B recovers +0.017 of its +0.047, and the total is negative for every model here); that the gain comes from the memory heads rather than the attention layer (the rows differ in both); that a brain trained to share memory would gain this much (Stage 2); anything about a different file-selection rule or other directories' size; that the 95 % intervals cover variation between directories (they resample targets, which are files; resampling whole directories instead of files widens the gate-number interval of Run 51 by about 1.4 times on dev and 1.1 times on test (final review: dev [0.039, 0.055] became about [0.036, 0.058]), and this entry's own intervals were not remeasured that way); more than one seed per checkpoint.

---

## Run 52 — can a group's pooled memory be handed to an agent that did not read the files? (`scripts/probe_commons.py --mode handoff`, `results/commons_handoff_x64k_{dev,test}.json`) — 2026-10-04, GPU 0 (dev) and GPU 1 (test)

`m20_linux_depth4x64k_s0` (no local attention) through the stream API on 353 dev and 339 test targets. One fresh reader agent per related file (its first 8,192 bytes, header included; 32,768 reader bytes at most per target, 5.5 files on average on test); `merge` their tables; `preload` the sum into a fresh agent at alpha; score the target's first 4,096 bytes. A sequential reader (one agent, the same files in order, `new_document()` between files) and the plain-forward context row on exactly the same bytes complete the comparison, so every row scores the same bytes. Gains are alone minus the row (positive = fewer bits/byte); the transfer ratio is (alone - handoff) / (alone - context); intervals are 95 % from resampling the targets (files):

| split | condition | bits/byte | gain vs alone | transfer ratio |
|---|---|---:|---:|---:|
| dev | alone | 0.771 | | |
| test | alone | 0.782 | | |
| dev | context (same bytes as a plain prefix) | 0.940 | -0.169 (-0.180, -0.158) | n/a |
| test | context (same bytes as a plain prefix) | 0.960 | -0.178 (-0.188, -0.168) | n/a |
| dev | handoff, alpha 0.03 | 0.762 | +0.009 (+0.007, +0.011) | n/a |
| dev | handoff, alpha 0.1 | 0.771 | -0.001 (-0.004, +0.003) | n/a |
| dev | handoff, alpha 0.3 | 0.817 | -0.047 (-0.053, -0.041) | n/a |
| dev | handoff, alpha 1.0 | 0.930 | -0.159 (-0.169, -0.149) | n/a |
| test | handoff, alpha 0.03 (chosen on dev) | 0.775 | +0.007 (+0.006, +0.009) | n/a |
| dev | sequential reader | 0.939 | -0.169 (-0.179, -0.158) | n/a |
| test | sequential reader | 0.959 | -0.177 (-0.188, -0.167) | n/a |

Relative difference between the merged reader tables and the sequential agent's, per head (`all_heads()` order: the post head first, then the inner heads bottom-up), test:

| head | S | Ksum |
|---:|---:|---:|
| 0 | 4.8e-01 | 2.4e-01 |
| 1 | 8.9e-08 | 9.0e-08 |
| 2 | 4.3e-01 | 2.6e-01 |
| 3 | 4.9e-01 | 3.1e-01 |

**Claim:** an agent that starts from the pooled tables of reader agents, scaled by alpha = 0.03 (chosen on dev), reads the test targets at 0.007 (0.006, 0.009) fewer bits/byte than the same agent starting empty (dev: 0.009 (0.007, 0.011)), with weights that were never trained to start from a loaded memory. The gain is small and falls away above alpha 0.03 (dev: -0.001 at 0.1, -0.047 at 0.3, -0.159 at 1.0). The context gain on test is -0.178 (-0.188, -0.168) bits/byte: reading the same files as a plain prefix made the score worse than reading the target alone, which is below the 0.01 floor, so no transfer ratio is reported. Merged reader tables equal the sequential agent's for head 1 (relative difference 8.9e-08) and differ by 4.9e-01 at most in S across the other heads (head 0, the post head, among them), whose keys depend on earlier reads. **Not a claim:** that handoff does not work. No checkpoint was trained to start from a loaded memory, and loaded matches dilute the agent's own through the normaliser, so this probe can confirm and cannot refute; alpha was chosen on dev (0.03) and test was scored once at it. Nor do the 95 % intervals cover variation between directories (they resample targets, which are files; resampling whole directories instead of files widens the gate-number interval of Run 51 by about 1.4 times on dev and 1.1 times on test (final review: dev [0.039, 0.055] became about [0.036, 0.058]), and this entry's own intervals were not remeasured that way).

---

## Run 53 — the spine: hybrid vs RoPE transformer vs zero-head local-attention cell at 16M, one card, one recipe (`scripts/run_spine.sh`, `scripts/spine_report.py`, `results/spine16_*.json`, `results/spine16_score_*.json`) — 2026-10-03 16:52 → 2026-10-04 19:27, GPU 1 at 150 W

16M class (d 512, 8 layers, d_ff 1024; kernel corpus; 16,000 steps of 8 × 2,048 bytes; lr 6e-4, warmup 1000, cosine, wd 0.1): the Run 37 / Run 43 recipe, all three arms under bf16 autocast (`--amp`; the transformer had never run under it), with the yardstick of the group-learning spec. One arm after another on GPU 1 (RTX 3090, power cap 150 W, model server stopped, nothing else on the card except where the `contended` column says otherwise), from a worktree pinned at `d8906a3`. The three seed-0 arms (hybrid, then transformer, then zero-head) ran as `flybrain-spine-s0.service` from 2026-10-03 16:52 to 2026-10-04 01:46. The table also has two `_eqsec` rows, the equal-seconds reruns that the report's rules called for: the transformer at 47,682 steps and the zero-head cell at 34,361, step counts set so that `train_seconds` land at the hybrid's. They ran as `flybrain-spine-followups.service` on 2026-10-04 from 09:36 to 19:27 (an earlier start of that unit at 09:15 was stopped after 40 seconds with nothing written). The table has no `_s1` rows: every arm is one seed. Start and end times are from `journalctl --user`. The seed-0 arms ran with both cards visible and device `cuda:1`, which left an idle 256 MiB context on GPU 0 for the length of each run (no work on it); the reruns ran with only GPU 1 visible, so their JSONs record device `cuda:0`, which is the same physical card (the meter resolves it by UUID; `gpu_uuid` reads GPU-51164b62-fd30-253a-5738-55844d46af9c in every JSON). `train_seconds` excludes evals and saves; peak VRAM is `torch.cuda.max_memory_allocated`; joules, `contended` and the temperature come from one nvidia-smi sample a second. Scored by `cross_eval.py --fixed` in fp32 at length 2048: dev = the first 1,024 windows from byte 202,117,357, test = the first 1,024 windows from byte 211,576,468, legacy bytes = all 976 windows from byte 200,100,000. Test windows were scored twice for the three seed-0 arms: the first scoring run wrote all four score files under one wrong name (a quoting bug in the launch command), so the identical commands were run again from a script file; the printed numbers of the two runs are the same and nothing was changed between them. The two reruns' test windows were scored once. Cold = mean over the window, warm = mean over its second half.

| run | params | dev cold / warm | test cold / warm | legacy bytes cold | train_seconds | peak VRAM GiB | joules | contended |
|---|---:|---|---|---:|---:|---:|---:|---|
| `spine16_hybrid_s0` | 16.07M | 0.785 / 0.711 | 0.786 / 0.712 | 0.741 | 17,586 | 7.4 | 2,667,984 | no |
| `spine16_tf_s0` | 17.09M | 0.807 / 0.726 | 0.810 / 0.730 | 0.761 | 5,901 | 2.6 | 889,055 | no |
| `spine16_zeroheads_s0` | 13.93M | 0.830 / 0.778 | 0.832 / 0.781 | 0.789 | 8,189 | 4.7 | 1,235,520 | no |
| `spine16_tf_eqsec_s0` | 17.09M | 0.723 / 0.645 | 0.720 / 0.642 | 0.681 | 17,622 | 2.6 | 2,653,752 | no |
| `spine16_zeroheads_eqsec_s0` | 13.93M | 0.778 / 0.728 | 0.779 / 0.730 | 0.742 | 17,616 | 4.7 | 2,657,118 | yes |

- hybrid - tf: cold bpb dev -0.022, test -0.024, legacy -0.020; train_seconds 17,586 vs 5,901 (+198.0%)
- hybrid - zeroheads: cold bpb dev -0.045, test -0.046, legacy -0.048; train_seconds 17,586 vs 8,189 (+114.8%)
- tf - zeroheads: cold bpb dev -0.023, test -0.022, legacy -0.028; train_seconds 5,901 vs 8,189 (-27.9%)
- spine16_tf_eqsec_s0 - spine16_hybrid_s0 (matched on train_seconds): cold bpb dev -0.0618, test -0.0659, legacy -0.0592; train_seconds 17,622 vs 17,586 (ratio 100.2%)
- spine16_tf_eqsec_s0 - spine16_tf_s0 (own arm, seed 0): cold bpb dev -0.0839, test -0.0901, legacy -0.0794; train_seconds 17,622 vs 5,901 (ratio 298.6%)
- spine16_zeroheads_eqsec_s0 - spine16_hybrid_s0 (matched on train_seconds): cold bpb dev -0.0068, test -0.0068, legacy +0.0018; train_seconds 17,616 vs 17,586 (ratio 100.2%)
- spine16_zeroheads_eqsec_s0 - spine16_zeroheads_s0 (own arm, seed 0): cold bpb dev -0.0515, test -0.0528, legacy -0.0465; train_seconds 17,616 vs 8,189 (ratio 215.1%)
- runner val_bpb (128 random windows, the number Runs 37-49 report): hybrid 0.7613, tf 0.7828, zeroheads 0.8079
- Sign conventions: in the `_eqsec` lines the difference is rerun minus reference, so a negative number means the rerun reads fewer bits; in the other pair lines it is the first arm minus the second, so `hybrid - tf` negative means the hybrid reads fewer bits.
- Consistency: the in-run bf16 score of the 128 dev windows and its fp32 re-score differ by +0.0001 for every run except `spine16_zeroheads_eqsec_s0` (+0.0000) (the report's `consistency` lines).
- Joules cover the whole arm, evals and saves included, while `train_seconds` excludes them (the report's own note).
- Joules are logged, not a headline: at a fixed 150 W cap they are close to 150 W × seconds (mean power 149.7 to 150.5 W here), so they restate the seconds column.
- Pace drifts within a run, and the reruns ran in the daytime while the machine was in use, so the per-1,000-step intervals were read before quoting seconds. Hybrid: 1,075 s for the first 1,000 steps, then 1,091 to 1,107 s for each of the other 15 (the second took 1,099 s), against 0.988 s/step in the 60-step smoke run before the launch; the smoke runs of the other two arms read 0.343 and 0.485 s/step against full-run paces of 0.369 and 0.512, so short timings understate the long-run pace by 5 to 10 %. Transformer rerun: 351 s for the first 1,000 steps, then 362 to 375 s for each of the next 46 full intervals (0.370 s/step over the run, against 0.369 for the 16,000-step arm). Zero-head rerun: 500 s for the first, then 508 to 517 s for each of the next 33 full intervals (0.513 s/step over the run, against 0.512). The zero-head rerun's `contended` flag is set: another process was listed on the card at some sample, and the report prints a WARNING for it (`spine16_zeroheads_eqsec_s0: contended=True telemetry_lost=False; its train_seconds is not clean`); its steady intervals, the same range as the seed-0 zero-head arm's, are the evidence that its seconds are usable.

**Claim:** At 16M and 16,000 steps the hybrid reads 0.022 fewer bits/byte than the transformer on dev and 0.024 fewer on test (hybrid - tf: -0.022 / -0.024; legacy bytes -0.020) but trains for 17,586 s against 5,901 s (+198.0 %); at equal seconds the order reverses: the transformer rerun (`spine16_tf_eqsec_s0`, cold 0.723 dev / 0.720 test at 17,622 s) is ahead of the hybrid by 0.0618 on dev and 0.0659 on test (its line reads -0.0618 / -0.0659, rerun minus reference, so the negative sign here means the transformer reads fewer bits; the hybrid's per-step lead did not survive equal seconds) and ahead of its own 16,000-step run by 0.0839 / 0.0901. The zero-head cell reads 0.045 / 0.046 bits/byte more than the hybrid at 16,000 steps (hybrid - zeroheads: -0.045 / -0.046, at 8,189 s against 17,586 s, +114.8 %) and 0.023 / 0.022 more than the transformer (tf - zeroheads: -0.023 / -0.022); at equal seconds with the hybrid (17,616 s against 17,586 s) it reads 0.007 fewer than the hybrid (0.0068 on dev and on test; zeroheads_eqsec - hybrid: -0.0068 / -0.0068; legacy bytes +0.0018, the other way), which is inside the 0.01 band with one seed, so that pair is not ranked; the seed-1 pair that would rank it was not run, because both of its runs would use step speeds that a later profiling round made obsolete (last sentence below). Dev and test agree in sign on every gap. **Not a claim:** joules as a result; more than one seed (the table has no `_s1` rows); anything beyond 16M, 16,000 steps (the `_eqsec` rows ran longer) and this kernel corpus; that the order at equal seconds holds beyond these two eager reruns of one seed each; a ranking of the hybrid and the zero-head cell at equal seconds; a comparison of the fixed-window numbers with the random-window `val_bpb` of Runs 37 to 49 (different windows, different numbers; the `runner val_bpb` line above is the comparable one). These seconds are for the code as it ran (eager PyTorch; the hybrid's heads in fp32 and its recurrence in float64): a profiling round on 2026-10-04 (notes/SPEED.md on branch feat/speed-round) found that torch.compile with the heads and the recurrence under bf16 takes the hybrid from 1.10 to about 0.45 s/step and the transformer from 0.37 to 0.29, so the seconds ratio of this entry is not a property of the architecture; the comparison at those speeds is that round's, not this entry's.

## Run 54 — one table (Qwen's n-gram idea inside the memory's address), and the training step 289 → 89 ms (`docs/superpowers/specs/2026-11-05-merged-table-design.md`, `scripts/run_merged.sh`, `results/mt_linux_*_4000_s*.json`, `cross_mt_4000_*.json`, `*_masksweep.json`; `flybrain/model/scan_kernel.py`, `select_kernel.py`, `tests/test_speed_exact.py`) — 2026-11-05 07:04 → 10:08, GPU 1 at 300 W (follow-ups on GPU 0)

Qwen3.8-Flash-Next carries a 20M-row hashed n-gram embedding (51B parameters, fixed after training) beside its fast-weight layers. This lab's memory head already has the fast weights and the sparse address; the question was whether a *learned* table under the same address helps (one table: phrasebook plus notepad), with Qwen's separate embedding as the equal-parameter control. Early read: the Run 41 winner (d 512, 8 minGRU layers, 4 heads of 4096 slots, one 512-byte local-attention layer), kernel corpus, 4000 steps of 8 × 2048 bytes, lr 6e-4, warmup 1000, cosine, wd 0.1, bf16 autocast, seed 0 (seed 1 where marked). `addr` shifts each head's slot index by a hash of the last 2 or 3 input bytes (heads alternate), reads and writes alike; `prior` starts each head's table from a learned (V0, m0) per slot; `ngram` adds a separate hashed bigram/trigram embedding (8192 rows per order) after layer 2.

| arm | params | val bpb | CPython 2048 | bins none / 2 / 3 / 4 / 6 / 8+ |
|---|---:|---|---|---|
| A base s0 / s1 | 16.07M | 0.925 / 0.925 | 1.448 / 1.455 | 1.977 / 0.911 / 0.564 / 0.401 / 0.317 / 0.210 |
| B ngram embedding | 24.72M | 0.925 | 1.450 | 1.979 / 0.913 / 0.552 / 0.396 / 0.314 / 0.209 |
| C learned prior | 24.47M | 0.924 | 1.456 | 1.978 / 0.908 / 0.559 / 0.396 / 0.313 / 0.209 |
| **E n-gram address only** | **16.07M** | **0.905** | **1.409** | 1.966 / 0.889 / 0.508 / 0.374 / 0.298 / 0.190 |
| D prior + address s0 / s1 | 24.47M | 0.906 / 0.906 | 1.407 / 1.399 | 1.971 / 0.893 / 0.494 / 0.364 / 0.291 / 0.192 |

- The gain is the address term, not the learned table: E equals D with no extra parameters; C and B equal A. Both seeds of A and D agree to the third decimal, against a 0.019 gap.
- Where it lands: every bin, most in the 3- and 4-byte-match bins (0.564 → 0.508, 0.401 → 0.374), the bins where Run 44 found the transformer's last lead; bytes with no earlier match move 0.011.
- Held out: CPython 1.448 → 1.409 (seed 1: 1.455 → 1.399), so it is recall, not memorised kernel.
- The prior's slots did learn (80–98 % of slots moved, values up to norm 3), but their weights m0 stayed near the 0.69 init: a learned value counts as one write and the content address drifts under it during training. Not pursued further at this size.

**Speed (same math).** Profiling the 20M step (0.289 s) found 20 % matmul and the rest elementwise passes, copies and 2217 kernel launches, with the CPU issuing work at ~130 ms a step. Four exact changes, each pinned by a test against the old path (`tests/test_speed_exact.py`, with a logits fixture from the old code): the minGRU recurrence as one fused Triton kernel per direction, two-pass chunked (the fp64 closed form with its cast copies and 63-step Python carry loop is gone; 10.2 → 0.7 ms a layer, within 2e-6 of fp64 at 2048); the head's one-pass read under `torch.compile` (1840 → 400 launches a head); slot selection as one kernel (4 signed picks a slot, threshold, top-4, instead of a 512-wide matmul and a general topk); the table update in one `index_add` and the key rescored from its 4 nonzeros. Two further changes reorder bf16 arithmetic and are opt-in: flex attention for the sliding window (`FLYBRAIN_FLEX=1`) and whole-model compile for the training step (`--compile-model`).

| path | ms/step | A at 4000 steps | train s |
|---|---:|---|---:|
| as of Run 53 | 289 | 0.925 | 1156 |
| exact kernels + compiled head read | 117 | 0.925 (bins within 0.004) | 500 |
| + two-pass scan, flex attention | 103 | 0.928 | 436 |
| + whole-model compile | 89 | 0.924 | 390 |

- The exact paths reproduce A to the third decimal; the two bf16-reordering options land ±0.003 around it, the seed spread, so they are for experiments whose arms all share them, not for comparing against earlier runs.
- Gotchas recorded for the next kernel: Triton launches on the current device, so wrap launches in `torch.cuda.device(t.device)` for cuda:1; `torch.cuda.synchronize()` without a device syncs cuda:0 only, which made a first benchmark read 0.1 ms for a 0.7 ms kernel; the first parallel-scan rewrite moved more bytes than the fp64 form and gained nothing, the win came from fusing the walk into one kernel.

**Claim:** at 20M and 4000 steps an n-gram term in the memory's slot address (a hash of the last 2 or 3 bytes, no parameters) reads 0.020 fewer bits per byte on the kernel (0.925 → 0.905, two seeds each) and 0.04–0.05 fewer on held-out CPython, with the gain concentrated in the 3–4-byte-match bins; a learned table under the same address, and Qwen's separate n-gram embedding, add nothing at this size and budget. The training step is 3.2× faster (2.7× on the bit-exact paths), which bears on Run 53's equal-seconds result and is the next thing to rerun. **Not yet:** 16k steps; enwik8; 50M; the equal-seconds pair of Run 53 on the fast path.

---

## Path to V3 (ladder plumbing)

V0 **can** beat unigram once flatten and ordered hops are combined (Run 13, fly **3.586**). Mean-head and parallel-flatten still cannot. The ladder below is plumbing, not language quality.

### V0 status

Mean-head d=4 (`results/wiki_mean_d4.json`): fly **4.600** / shuffle **4.600** / GRU **4.080**. Flatten parallel (`results/wiki_cpu_n128.json`): fly 4.522 / shuffle 4.471 / GRU **3.502**. Flatten + ordered (`results/wiki_flatten_d4_ordered.json`): fly **3.586** / shuffle **3.621** / GRU **3.572** vs unigram **4.564** / bigram **3.729**.

### V1 prune/grow (`results/v1_prune_grow.json`)

N=64 mean-head. Drop 16 weakest edges, grow 16 random unused pairs. Sparse budget, no `[N,N]` tensor.

| Arm | E before | after prune | after grow | sign locked | dense N×N | loss (1 batch) |
|-----|---------:|------------:|-----------:|:-----------:|:---------:|---------------:|
| fly | 239 | 223 | 239 | yes | no | 5.674 |
| shuffle | 239 | 223 | 239 | yes | no | 5.523 |

**Claim:** topology can change on the shipped sparse graph; signs on surviving/new edges stay `sign·softplus`.  
**Not a claim:** plasticity improves bits/byte.

### V2 slot pool (`results/v2_slots.json`)

N=32 allocated **capacity=40**. Live 32 → activate reserved slot **33** → deactivate **32**. Capacity stays 40. Inactive slot L1=0; after activate L1=0.96; after deactivate that slot’s state is frozen.

**Claim:** growth is a live-mask over a preallocated pool, not a resized neuron tensor.  
**Not a claim:** extra slots help language.

### V3 sandbox loop (`results/v3_launch_1.json`, `v3_launch_2.json`)

Entry: `uv run python scripts/run_v3.py --run-name v3_launch_{1,2}`.  
Kind: **in-process relative allowlist** (`echo`/`cat`/`ls`/`pwd` with paths confined to the sandbox root). No host bash. Docker was present; this class is what tests and `run_v3.py` call.

Playbook (because V0 decode is replacement-char garbage, not shell):

```text
echo v3-ok > notes.txt
cat notes.txt
ls
```

Both launches:

| Launch | `notes.txt` | `cat` observation | `ls` observation | \(H_t\) mean |
|--------|-------------|-------------|-------------|------:|
| 1 | `v3-ok\n` | `…cat notes.txt\nv3-ok\n` | `…ls\nnotes.txt\n` | 0.245 |
| 2 | `v3-ok\n` | same pattern | same pattern | 0.284 |

The file is created by `Sandbox.step`, not by the test writing the path. Brain `DECODED` is `'�'` — the fly net did **not** type the command. \(H_t\) is returned state and changes across steps; files live only under the sandbox root. Tests refuse `cat /etc/passwd`, `..`, and `curl`.

**Claim:** the perception–action loop exists; filesystem ≠ \(H_t\); default sandbox cannot reach host secrets or the network.  
**Not a claim:** biological design uses a computer.

Tests after this ladder: **43 passed** (`uv run pytest -q` twice).

---

## Next

Run 54: the n-gram address term (hash of the last 2–3 bytes mixed into each head's slot index, zero parameters) is worth 0.020 bpb on the kernel and 0.04–0.05 held out at 20M/4k steps, two seeds; the learned prior and Qwen's separate n-gram embedding add nothing. Training step 289 → 89 ms (2.7× bit-exact, 3.2× with the bf16-reordering options). Next: rerun Run 53's equal-seconds pair on the fast path; then the address term at 16k steps, enwik8 and 50M.

Run 28: the head generalizes to minGRU (Mamba-class) and gated linear attention with the same margins; learned-key head does not. Next: scale (d=256, 50 MB, seq 2048) on GLA/minGRU; transformer backbone as the negative control; real terminal transcripts.

Run 27 confirmed Run 26 (seeds, wider/tuned/longer GRU, leakage control, held-out prose and code). Next: scale (d=256, 50 MB, seq 2048) and a transformer backbone with the same head.

Run 26: GRU + sparse head beats plain GRU, wider GRU and a learned-key head on code (1.46 vs 1.60 / 1.55 / 1.52) and prose (2.01 vs 2.08 / 2.04 / 2.04). First bpb win over the best conventional baseline. Next: seeds on wiki; scale (d=256, 50 MB); longer windows (2048) where the memory should matter more; a transformer backbone with the same head.

Run 25: mixed (learned + sparse head) is the best fast-weight arm on prose and code and wins repeat bytes, but a GRU still wins overall at ~500k scalars; sparse retrieval works to 1024 tokens, fails at 2048 for reasons not yet separated (signal fraction vs cross-talk); no cross-domain in-context adaptation. Open: 2048 with 16k units / signed keys; multi-span at 512; a GRU + sparse-head hybrid (the GRU keeps winning, so put the memory on the GRU, not on attention).

Run 24: normalizer + bigram key gives sparse 94 % copy@512 (all controls at chance) but on text sparse is the weakest attention arm (3.16 vs learned 2.89 vs GRU 2.59). Next honest test is a mixed model (learned + sparse heads) on text-with-repeats / code / terminal transcripts at a size where attention beats a GRU.

Run 23: sparse top‑k is a learning-speed/robustness win at 64 and the only arm that learns copy at 512, but a one-token key plateaus at 57 % there (ambiguity, not capacity) and every attention-style arm loses to a GRU on text by 0.65 bpb. Under test: 2–3-token key conv at 512. After that: does *any* of this lower text bpb at a size where attention beats a GRU (4+ blocks, d≥128, 10 MB)?

Run 22: sparse top‑k keys beat gated-linear-attention keys and orthogonal Gaussian keys on in-context copy at every width and lr tried; gate can be constant; k=8 of 1024 is best. Mechanism open. Next: top‑k of a Gaussian projection and signed sparse keys (is it top‑k or claws?), delta-rule write, longer sequences on GPU, adaptation on shifted text.

Run 21 opened a line: sparse code + Hebbian fast weights copies (91 % at 1500 steps) where dense codes and a GRU sit at chance. Next honest checks: (a) modulator ablation (constant gate), (b) key sparsity sweep (top‑k 8…512) to show the overlap story directly, (c) adaptation-speed eval on a shifted corpus, (d) longer spans / longer sequences, (e) a tiny attention arm as the ceiling reference.

Run 20 closed: with 11 seeds the both-matched block beats a dense FFN at 600 steps by 0.04 bpb (t≈2, 9/11) and ties a GRU; at 1500 steps (5 seeds) dense +0.04 and GRU +0.07 ahead. An early-learning edge, not a quality win. Next honest question is whether the late-training loss is an lr/schedule artefact (cosine decay, or lower lr after step 600) or a capacity ceiling of top‑k. Not a fly result.

Run 19 closed: the coincidence block is a real sequence layer but does not beat a dense FFN or GRU at matched scalars on WikiText (3.36 vs 3.28 vs 3.16). It is cheaper per step, not better. Do not scale it to claim a win.

Run 18 closed: coincidence mixer works above chance and does **not** beat a matched bag MLP (the MLP sees the whole set). Fly-hash is for huge sparse keys, not 12–48 bit bags. Do not load MaleCNS.
