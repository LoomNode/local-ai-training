# Parameter inefficiency and folding weights with memory

**Origin:** lab discussions ~2026-07-17 (weight + memory layer) and ~2026-07-20–22 (modular specialists / sticky experts), refined in envelope-lab planning 2026-08.

**Related:** [tokens-as-parameters.md](tokens-as-parameters.md), [../CAMPAIGN.md](../CAMPAIGN.md)

---

## Claim

Standard deep Transformers buy **computational depth** largely by stacking **new unique parameters** at every stage:

```text
Layer1 weights → Layer2 weights → … → LayerN weights
```

Each parameter tensor is typically used **once per token per forward stage**, then the model moves on to another equally large set of weights. Capability becomes tightly coupled to **total resident parameter count**, not to **how much useful work each parameter performs**.

That is **parameter inefficiency**:

1. Parameters are **spatially duplicated** to create depth.  
2. Neighboring layers may learn **overlapping operations**.  
3. Every parameter occupies memory even though it participates in only one fixed stage.  
4. “More thinking” ≈ “more unique weight matrices,” which is hostile to **single-GPU / local** deployment.

MoE improves **active FLOPs** (few experts fire) but can **worsen** this warehouse problem: even more parameters sit idle until rarely routed, and routing thrash costs bandwidth (see Colibri / expert paging — related but different).

---

## Not the same as

| Idea | Difference |
|------|------------|
| **Quantization** | Same graph, fewer bits — does not fix one-use depth stages |
| **Colibri / expert paging** | Same full stack, move shards across NVMe/RAM/VRAM |
| **Just “add residual memory” to every unique layer** | Still N unique blocks; memory is garnish unless paired with **reuse** |

---

## Folding thesis

Prefer something closer to:

```text
shared core weights
    ↓
applied repeatedly while internal state changes
    ↓
refined representation / next token
```

So **effective depth ≫ unique weight depth**.

### Why memory / state is required

Blindly reusing one identical block can collapse to repetitive or limited refinement. A **changing state** (writable memory, iteration index, small adapters, routers) lets the same weights mean different things on successive passes:

> “Where am I in the computation, what have I already established, what should I refine next?”

Thus the architecture is jointly:

- **Fixed (or slowly learned) compute weights** — the reusable core  
- **Persistent / writable memory** — the evolving state  
- Optional: **tiny per-step or per-token modifiers** — so reuse is not sterile  

---

## Two levels of the program

### 1. Fold an existing model (harder)

- Detect redundant neighboring layers  
- Merge / share weights  
- Keep small residual adapters for differences  
- Needs strong eval; risk of silent quality loss  

### 2. Train for folding from the start (right for this lab)

- One or few **shared** blocks applied **K times**  
- Memory slots or recurrent state across applications  
- Same frozen train budget and hardware envelope as a stacked baseline  

Level 2 is the experimental path for envelope-lab on solo GPU1.

---

## Connection to modular specialists (~July 20–22)

A scaled picture of the same efficiency instinct:

- Shared foundation (reusable core)  
- Modular / semantic specialists (addressable capacity)  
- **Reuse** the same specialists across consecutive tokens when possible  
- Penalize unnecessary expert load/switch (sticky routing)  
- Large **total** capacity (e.g. conceptual 200B specialized) with modest **active** set (e.g. 20–40B) for a task  

That is “capacity without N stacked unique full-depth copies of everything.”

---

## Experimental implications for envelope-lab

**Primary metric:** lower `val_bpb` under fixed time budget.  
**Hard constraints:** peak VRAM, min decode tok/s, solo GPU1 for Campaign 1.  
**Density / folding metrics to log:**

- unique parameters vs effective depth (applications of shared blocks)  
- params in embed vs core  
- active params roughly touched per token (when applicable)  

**Minimal A/B that tests the thesis:**

| Candidate | Structure |
|-----------|-----------|
| Baseline | Stacked L unique blocks |
| Folded | U unique blocks × R recurrences, U×R ≥ L, unique params ≪ stacked |
| Folded + memory | Same with writable state across recurrences |

If folded (+ memory) wins bpb at lower unique params under the envelope, the parameter-inefficiency claim has local evidence.

---

## Status in code (as of 2026-08)

| Piece | Status |
|-------|--------|
| `USE_MEMORY_BLOCK` / slots | Partial: memory **per stacked layer**, not full shared recurrence |
| Campaign 1 | Density / memory-vs-depth under solo GPU1 |
| Shared recurrent core | **TODO** — see [TODO.md](TODO.md) |

---

## One-line thesis

> Current LLMs may underuse each parameter by purchasing depth as stacks of one-use weights; a smaller reusable core plus evolving memory could deliver comparable effective computation with far better hardware efficiency.
