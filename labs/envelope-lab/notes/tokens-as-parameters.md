# Tokens as parameters — unify tokenizer and weights

**Origin:** envelope-lab architecture discussion 2026-08 (follows parameter-folding thesis).  
**Related:** [parameter-inefficiency-and-folding.md](parameter-inefficiency-and-folding.md)

---

## Motivation

If depth-via-unique-layers wastes parameters, another split also looks arbitrary:

```text
frozen tokenizer  →  discrete ids
embedding table   →  first big parameter store keyed by token
layer stack       →  most parameters, not keyed by token identity
lm_head (± tied)  →  back to ids
```

**Part** of the model is already “token id ↔ parameter row.” Most of the rest is a deep stack that only sees activations.  

**Tokens as parameters** asks: should **addressable identity** (token / code / concept) own more of the capacity, with only a **small reusable computer** doing iterative compute?

That unifies:

- **Tokenizer-side** (what discrete symbols exist)  
- **Weight-side knowledge** (what is stored for those symbols)  
- **Compute** (thin machine + state), not 80 photocopied machines  

---

## Readings of the idea

| Reading | Meaning |
|---------|---------|
| **Lexicon-heavy model** | Most capacity in token-indexed tables (embed, per-token adapters, product-key / codebook memory). Small shared core computes. |
| **Joint tokenize + model** | No frozen BPE; bytes/patches → learned discrete or continuous tokens end-to-end. |
| **Soft tokens** | Trainable virtual tokens that *are* parameters (prefix / prompt tuning). |
| **Token-conditioned weights** | Hypernet or router: token/concept selects or generates weights for this step. |
| **Codebook = tokenizer = memory** | VQ / residual codes are both “tokens of thought” and dictionary parameters the model reads/writes. |

The interesting local-hardware story is roughly:

> **Large addressable store (by token/code) + small reused computer + writable state.**

---

## Link to folding / parameter inefficiency

Stacked Transformer:

```text
params ≈ f(depth × width²) + vocab×d
each stage weight used ~once per position
```

Token-centric / unified:

```text
params ≈ large store keyed by discrete identity
       + small recurrent / shared core
       + state (what I’ve concluded so far)
```

| Role | Where it lives |
|------|----------------|
| Knowledge | Lookup / retrieve by token or code |
| Thinking | Iterate the same computer while state changes |
| Depth | Iterations × state, not N unique layer tensors |

MoE is only a half-step: more parameters in a warehouse, still often one-shot when fired—not “this token’s long-term parameter packet.”

---

## Why this helps single-GPU / local AI

| Lever | Hardware effect |
|-------|-----------------|
| Smaller unique core | Fits 24 GB; better cache reuse |
| Large token/code store | Can be quantized hard, paged by **address** (entry paging, not only layer paging) |
| Active set | Dominated by tokens in the current sequence (+ neighbors) |
| Depth via recurrence | More refinement without more resident matrices |

Colibri pages **layer/expert shards**. Token-as-params suggests paging **dictionary entries**—same “don’t keep the whole warehouse hot,” different grain.

---

## Not the same as Mamba

| Token-as-params / fold | Mamba / SSM |
|------------------------|-------------|
| Efficiency of **parameter use & addressable knowledge** | Efficiency of **long sequence compute** |
| Orthogonal | Can combine: SSM as the small computer over a big token memory |

---

## Risks

- Expressivity: lookup alone is not multi-step reasoning; need a strong recurrent core + state.  
- Training dynamics: deep weight sharing is harder than residual stacks.  
- Joint tokenization can be unstable.  
- Huge vocab tables are still huge—you reorganize scale so **reuse and paging** make sense, not magic free capacity.  
- Must beat **same-budget** baseline under envelope metrics.

---

## Research goal phrasing (for KAT / campaigns)

> **Token-parameterized compute:** Under fixed train time and one 3090 envelope, beat a stacked Transformer baseline by shifting capacity into **token-addressable parameters** (embed / codebook / per-token adapters) and a **small shared recurrent computer with writable state**, so unique weight count for “depth” shrinks while quality holds or improves under VRAM and decode floors.

---

## Experimental ladder (suggested)

1. **Campaign 1 (current):** density + per-layer memory slots (weak cousin).  
2. **Shared recurrent core + state** (true folding) — same param budget A/B.  
3. **Grow embed / codebook, shrink stack** — measure embed_frac vs core_frac.  
4. **Optional:** learned codes / stronger token-side memory; SSM core variant.  
5. **Later:** page token tables; modular specialists with sticky routing.

### Metrics to add when implementing

- `params_embed_M`, `params_core_M`  
- `unique_blocks`, `recurrence`, `effective_depth`  
- `val_bpb`, `peak_vram_mb`, `decode_tps` (existing)  
- density: quality per unique param, per GB  

---

## One-line thesis

> Treat discrete identity as the index into most model capacity, and treat deep stacks of unique matrices as the inefficient way to buy thought—unify the token store with weights, and iterate a small computer over evolving state.
