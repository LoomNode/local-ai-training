# flybrain-lab program

**Hypothesis:** a fly-derived *organizational prior* (sparse modular graph, cell-type-specific recurrent units, E/I signs, sensory/readout bottlenecks) learns next-byte prediction more efficiently than the same units on shuffled wiring, and can be compared honestly to a unique-weight GRU at matched trainable-scalar budget.

Origin: ChatGPT handoff Fly Brain Text Generation (private ChatGPT conversation). Sibling lab: [biointel-lab](../biointel-lab/README.md) (shared-rule delayed identity; connectome as a *later* prior). This lab is the sequence-learning experiment that brief eventually asked for.

## What this is not

- Not a molecule-for-molecule fly, not LIF/spiking, not BANC/FlyWire-scale V0.
- Not “make a fly write English” as the first success criterion.
- Not a separate world-model head, terminal agent, or neurogenesis (V1–V3).

## V0 (this slice)

Tiny end-to-end next-byte model:

```text
UTF-8 bytes
    → 256 → n_sensory embedding (not all neurons)
    → 4-d GRUCell per cell type, sign-locked synapses (scalar / vector2 / vector / projected)
    → designated readout population
    → 256 logits
```

Topology is **frozen**. Trainable: synapse strengths (sign-constrained), type programs, emit vectors, encoder, decoder. Recurrent state \(H_t\) is **not** \(\theta\) and resets at document boundaries.

Graph for V0 is **synthetic fly-like motifs** (sensory → Kenyon-like expansion → recurrent core → readout, plus a DA-like population, mixed E/I, log-normal contact counts). `load_npz` exists for a later real connectome; it is not required to train.

Three arms, matched as closely as the architecture allows:

| Arm | What it is |
|-----|------------|
| `fly` | synthetic fly-like wiring |
| `shuffle` | degree-matched destination shuffle; same signs, counts, cell types, I/O sets |
| `gru` | unique-weight byte GRU sized to the fly trainable-scalar count |

## Success for V0

Plumbing: loss drops on a tiny repeating corpus; tests pin sparsity, bottlenecks, sign constraint, type-program N-independence, and honest `numel()` accounting.

Science: fly vs shuffle vs GRU **held-out bits-per-byte** on a WikiText-2 byte slice (`scripts/run_eval.py`). A fly/shuffle split is the result that would change what we do next. Flatten readout is the V0 spec; `readout=mean` is a control so the linear head cannot dominate the scalar budget.

**Recorded (see `notebook.md`):** mean-head knobs sit at unigram (~4.60 vs 4.564). Flatten + ordered hops beats unigram (Run 13 fly **3.586**). Type-aware shuffle (Run 14) is **3.577** vs fly **3.600** — motifs do not win. **Do not scale N.**

Frozen-LM + MaleCNS reservoir is a **different** thesis: [`notes/FLM.md`](notes/FLM.md).

## Later (science still gated)

V1 prune/grow, V2 slot pool, and V3 sandbox loop are **implemented as plumbing** (`scripts/run_v{1,2,3}.py`). V3 does not type commands. Do not scale N. WikiText: region shuffle ≥ fly. Unplanted 3-claw binding: shuffle ≥ fly (Run 16). Unique-PN 4-claw coincidence (Run 17): fly > shuffle on two seeds. Coincidence mixer without a connectome (Run 18) is above chance and loses to a matched bag MLP. The same block as a recurrent byte LM layer (Run 19) beats unigram/bigram and trails a matched dense FFN and GRU (3.36 / 3.28 / 3.16) with 34% fewer FLOPs per step. FLOP-matched at d=48 (Run 20) it beats the dense FFN on every seed at 600 steps and ties the GRU; both dense arms pass it by 1500 steps. Adding KC→MBON-style Hebbian fast weights (Run 21) makes the sparse code copy a 16-byte span in-context at 91 % where an all-active code and a matched GRU stay at chance — the first wide, fair win for a biological design principle here, on in-context memory rather than bpb. Run 22: the same memory with gated-linear-attention keys (learned key+query) or orthogonal Gaussian keys stays at chance at every width and lr tried; sparse k=8 of 1024 reaches 84 %; the write gate can be a constant. Run 23 (parallel GPU form): at the right lr every key learns copy@64 and sparse is 3–4× faster; at 512 only sparse top‑k learns (57 %, one-token-key plateau); on WikiText@512 a GRU beats every attention-style arm by 0.65 bpb. Run 24: with a bigram key and the linear-attention normalizer, sparse top‑4 copies at 512 at 92–94 % while learned keys, softmax, dense random keys and a GRU stay at chance; on text the same sparse key is the weakest attention arm (3.16 vs learned 2.89 vs GRU 2.59). Sparse keys are an exact-retrieval memory, not a better byte LM. Run 25 (4 blocks, d=128, 9 MB): a mixed learned+sparse head is the best fast-weight arm on prose and code and wins repeat bytes by 0.3–0.7 bpb, but a GRU still wins every corpus overall; sparse retrieval works to 2048 tokens with a 4096-unit memory. Run 26: the sparse memory head on a GRU beats the plain GRU, a wider GRU and a learned-key head on code (1.46 vs 1.60) and prose (2.01 vs 2.08) — the first bits-per-byte win over the best conventional baseline in this lab; Run 27 confirmed it on three seeds per corpus, against a wider/tuned/longer-trained GRU, with a leakage control, and on held-out prose and held-out code (1.52 vs 1.76). Run 28: same gain on a minGRU (Mamba-class) and on gated linear attention; a learned-key head does not reproduce it. Run 29: across a 28× width ladder the minGRU gain is stable (code 0.15→0.12, prose 0.04) while GLA, which has its own learned-key memory, closes the code gap at d≥256. Run 30: a transformer that has formed attention retrieval is made worse by the head; the head is for models without exact recall. Runs 31–33: the gain is entirely on bytes with an earlier match (down to 2-grams); a match-strength gate adds 0.05. Run 34 (d=256, 2048 tokens, CPython / enwik8 / held-out Linux kernel): the gain grows with context (code 0.38, enwik8 0.21) and the gated head beats a same-recipe transformer at equal training with 2/3 of the parameters (1.11 vs 1.22 code; 1.63 vs 1.84 enwik8), tying it out of domain. Runs 35–37: with a tuned recipe (warmup, weight decay, RoPE) the transformer gains 0.2 and wins code by 0.04–0.13 while tying prose; the head's gain over its backbone is unchanged by recipe, context (4096) or size (13M), and the recurrence + head trains in ~60 % of the transformer's time. Not FlyWire / MaleCNS.
- Real FlyWire / BANC subgraph once the tiny pipeline is boringly solid.
- [nftechie/flm](https://github.com/nftechie/flm) is not V0: frozen 1.2B LM, frozen 166k-node graph, adapter-only. Their no-graph control wins.
