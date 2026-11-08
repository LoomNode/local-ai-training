# nftechie/flm vs this lab

Source: [github.com/nftechie/flm](https://github.com/nftechie/flm) (commit `d60610f`). Paper: [Flies Are All You Need](https://artificialscientific.com/papers/flies-are-all-you-need) (2026-09-11 preprint; not peer-reviewed). Not a run in this lab.

## What FLM is

Frozen **LiquidAI LFM2.5-1.2B-Instruct** (1.17B params) plus the **full retained MaleCNS v1.0** graph: 166,700 nodes, 25,582,938 directed edges. Token embeddings drive the **fixed** graph. Only a **278,528-parameter** bias-free adapter is trained. It adds a bounded residual to the frozen LM’s logits.

Recurrence (every node, every token):

```text
x = tanh(W @ (0.6 * x + 0.4 * B @ embed))
```

`W` is incoming-normalized nonnegative contact counts (frozen). `B` assigns each cell one random ±1 input channel (128 channels from the 2048-d embedding). `P` bins all cells into 128 output features. Language competence is the pretrained backbone.

They state this explicitly: it does **not** mean a biological fly understands language.

## What they measured

On 32 held-out conversations (1,236 target tokens), three seeds:

| Condition | NLL nats/token |
|-----------|---------------:|
| Frozen backbone | 1.3820 |
| Fly readout | 1.3598 |
| **Direct-input readout** (no graph) | **1.3593** |
| Relabel nodes, no refit | 1.3813 |
| No edges | 1.3820 (exact) |

Fly minus direct-input = **+0.000488** nats/token (interval above zero). The graph residual is real (disconnect zeroes it; it flips the argmax at ~1.5% of positions). The **matched adapter without the connectome is slightly better**. Relabeling is an interface control, not a topology win — they say so.

Raw reservoir state contracts by ≤0.6 per token; after ~20 steps the init is gone. That is a design bound, not fly memory.

## Contrast with flybrain-lab V0

| | This lab (V0) | nftechie/flm |
|--|---------------|--------------|
| Question | Can a fly-*like prior* **learn** next-byte better than shuffle? | Can a **frozen** connectome **condition** a frozen LM? |
| Language source | Trained from scratch (bytes) | Pretrained 1.2B LM |
| Graph | Synthetic N=128 motifs, **trainable** synapses | Real 166,700-node MaleCNS, **frozen** |
| I/O | Sensory/readout bottlenecks | **Every** cell gets a random channel of the token; **every** cell is pooled |
| Current token | Default: 2 hops late. Ordered: through the graph same token | Mixed into **all** nodes **before** `W` (`0.4 * input`) |
| Trained | encoder, type-GRUs, synapses, decoder | adapter only (0.024% of backbone) |
| Control | degree-matched shuffle; matched GRU | parameter-matched **direct-input** (no `W`) |
| Result | fly ≈ shuffle ≈ unigram 4.56 bpb | fly residual yes; **direct-input wins** |

They already did the thing V0 forbade as a cheat: broadcast the current token onto the whole graph, then let a small head read a pool of everyone. That is how a connectome demo chats. It is also why it does not test “bottlenecked fly organization learns language.”

Our Run 10 (ordered hops, still unigram) and their direct-input win are consistent: **wiring participation ≠ wiring helping**.

## What not to do with this pointer

- Do not treat FLM chat as a reason to load MaleCNS / FlyWire into V0.
- Do not train their 1.2B stack in this lab unless asked (multi-GB downloads, frozen LM, different success criterion).
- Do not read their title as “attention is replaced by a fly.” The paper rejects that.

## If we ever want a reservoir slice here

Keep N=128, **no** 1.2B backbone: freeze our synthetic `W`, inject current byte into **all** live neurons (their `B`), pool all neurons (their `P`), train only a small readout vs a no-graph direct-input control. That copies their *question* at our scale. It abandons V0’s bottlenecks on purpose. Gate it behind an explicit ask; it is a new program, not Run 11.
