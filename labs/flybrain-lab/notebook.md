# flybrain-lab notebook

Sticky across sessions. Raw dumps live in [`results/`](results/). This file is the **interpreted record**: protocol, numbers, what we will and will not claim.

Handoff: Fly Brain Text Generation (private ChatGPT conversation).  
Sibling delayed-identity lab: [`../biointel-lab`](../biointel-lab/README.md).

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

Tests after this change: **109 passed** (`uv run pytest -q`).

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

Run 25: mixed (learned + sparse head) is the best fast-weight arm on prose and code and wins repeat bytes, but a GRU still wins overall at ~500k scalars; sparse retrieval works to 1024 tokens, fails at 2048 for reasons not yet separated (signal fraction vs cross-talk); no cross-domain in-context adaptation. Open: 2048 with 16k units / signed keys; multi-span at 512; a GRU + sparse-head hybrid (the GRU keeps winning, so put the memory on the GRU, not on attention).

Run 24: normalizer + bigram key gives sparse 94 % copy@512 (all controls at chance) but on text sparse is the weakest attention arm (3.16 vs learned 2.89 vs GRU 2.59). Next honest test is a mixed model (learned + sparse heads) on text-with-repeats / code / terminal transcripts at a size where attention beats a GRU.

Run 23: sparse top‑k is a learning-speed/robustness win at 64 and the only arm that learns copy at 512, but a one-token key plateaus at 57 % there (ambiguity, not capacity) and every attention-style arm loses to a GRU on text by 0.65 bpb. Under test: 2–3-token key conv at 512. After that: does *any* of this lower text bpb at a size where attention beats a GRU (4+ blocks, d≥128, 10 MB)?

Run 22: sparse top‑k keys beat gated-linear-attention keys and orthogonal Gaussian keys on in-context copy at every width and lr tried; gate can be constant; k=8 of 1024 is best. Mechanism open. Next: top‑k of a Gaussian projection and signed sparse keys (is it top‑k or claws?), delta-rule write, longer sequences on GPU, adaptation on shifted text.

Run 21 opened a line: sparse code + Hebbian fast weights copies (91 % at 1500 steps) where dense codes and a GRU sit at chance. Next honest checks: (a) modulator ablation (constant gate), (b) key sparsity sweep (top‑k 8…512) to show the overlap story directly, (c) adaptation-speed eval on a shifted corpus, (d) longer spans / longer sequences, (e) a tiny attention arm as the ceiling reference.

Run 20 closed: with 11 seeds the both-matched block beats a dense FFN at 600 steps by 0.04 bpb (t≈2, 9/11) and ties a GRU; at 1500 steps (5 seeds) dense +0.04 and GRU +0.07 ahead. An early-learning edge, not a quality win. Next honest question is whether the late-training loss is an lr/schedule artefact (cosine decay, or lower lr after step 600) or a capacity ceiling of top‑k. Not a fly result.

Run 19 closed: the coincidence block is a real sequence layer but does not beat a dense FFN or GRU at matched scalars on WikiText (3.36 vs 3.28 vs 3.16). It is cheaper per step, not better. Do not scale it to claim a win.

Run 18 closed: coincidence mixer works above chance and does **not** beat a matched bag MLP (the MLP sees the whole set). Fly-hash is for huge sparse keys, not 12–48 bit bags. Do not load MaleCNS.
