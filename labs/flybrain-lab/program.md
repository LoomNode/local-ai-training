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

V1 prune/grow, V2 slot pool, and V3 sandbox loop are **implemented as plumbing** (`scripts/run_v{1,2,3}.py`). V3 does not type commands. Do not scale N. Flatten+ordered WikiText beats unigram; region shuffle ≥ fly on language. Trained small-head MB binding (Run 16) is fly > region-shuffle without planted labels. Not FlyWire / MaleCNS.
- Real FlyWire / BANC subgraph once the tiny pipeline is boringly solid.
- [nftechie/flm](https://github.com/nftechie/flm) is not V0: frozen 1.2B LM, frozen 166k-node graph, adapter-only. Their no-graph control wins.
