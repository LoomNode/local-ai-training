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

**22:43 — depth8 launched on GPU 0** (`--mem-every 1`, a head after every layer, 17.15M vs the transformer's 17.09M: parameter-matched to 0.3 %). Then depth8 on enwik8 (does the prose tie become a win?). depth4 kernel seed 1 queued on GPU 1 behind the Run 39b window cells. Readout: if depth4 lands near 0.78 the gap was capacity and the headline changes; if it stays near 0.91 the gap is structural (the write-up says so). Queued behind the enwik8 seed-1 pair on both GPUs (`scratchpad/cap_queue.sh`, retries on OOM; the other project holds 10.4 GB per GPU).

---

**Cross-lab note (2026-09-14, from the sibling lab / sibling lab session, no code crossed):** their transformer is seed-stable: FP32 [cross-lab numbers redacted]

| bin | bytes | [redacted] | share of gap |
|---|---:|---:|---:|---:|
| no match | [redacted] |
| 1 | [redacted] |
| 2–3 | [redacted] |
| 4–7 | [redacted] |
| 8+ | [redacted] |

The sibling lab's arm's deficit is flat across match length (0.05–0.09 nats everywhere, proportional to byte mass), and QAT loses nothing on copyable bytes. So a precision-starved dense model does not lose its recall first; it loses everywhere. A recall head would absorb at most the copyable share of their gap (3 % at 256-char windows; under half even at code-window copyable fractions if the per-bin gap stays flat). Their reading, which I share: the merge is not a gap-closing story. Caveat: 256-char windows, text8, 27-symbol vocabulary. For us it is a useful negative: the head's gain is not "precision the backbone lacks", it is a mechanism the backbone lacks.

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
