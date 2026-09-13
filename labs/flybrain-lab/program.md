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

V1 prune/grow, V2 slot pool, and V3 sandbox loop are **implemented as plumbing** (`scripts/run_v{1,2,3}.py`). V3 does not type commands. Do not scale N. WikiText: region shuffle ≥ fly. Unplanted 3-claw binding: shuffle ≥ fly (Run 16). Unique-PN 4-claw coincidence (Run 17): fly > shuffle on two seeds. Coincidence mixer without a connectome (Run 18) is above chance and loses to a matched bag MLP. The same block as a recurrent byte LM layer (Run 19) beats unigram/bigram and trails a matched dense FFN and GRU (3.36 / 3.28 / 3.16) with 34% fewer FLOPs per step. FLOP-matched at d=48 (Run 20) it beats the dense FFN on every seed at 600 steps and ties the GRU; both dense arms pass it by 1500 steps. Adding KC→MBON-style Hebbian fast weights (Run 21) makes the sparse code copy a 16-byte span in-context at 91 % where an all-active code and a matched GRU stay at chance — the first wide, fair win for a biological design principle here, on in-context memory rather than bpb. Run 22: the same memory with gated-linear-attention keys (learned key+query) or orthogonal Gaussian keys stays at chance at every width and lr tried; sparse k=8 of 1024 reaches 84 %; the write gate can be a constant. Not FlyWire / MaleCNS.
- Real FlyWire / BANC subgraph once the tiny pipeline is boringly solid.
- [nftechie/flm](https://github.com/nftechie/flm) is not V0: frozen 1.2B LM, frozen 166k-node graph, adapter-only. Their no-graph control wins.
