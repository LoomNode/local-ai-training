# FlyBrainLM V0 (tiny)

- **Date:** 2026-09-12
- **Status:** implemented (tiny pipeline)
- **Source brief:** a private ChatGPT conversation

## Goal

Prove `bytes → sensory → recurrent sparse graph → readout → loss → backward → generate` on a **1k-or-less** neuron stand-in for a fly-derived brain, with a shuffled-wiring control and a matched-budget GRU.

## Non-goals (V0)

- Loading FlyWire FAFB / BANC (loader stub only)
- Spikes, STDP, e-prop, dopamine as a learning rule
- Edge/neuron growth
- Terminal sandbox / world-model head
- Readable English as a success metric

## Locked decisions (from the brief)

| Choice | V0 |
|--------|----|
| Neuron | 4-d state, learned usage, `GRUCell(1, 4)` per cell type |
| Synapse | `sign * softplus(p)`; kinds `scalar` / `vector2` / `vector` / `projected` |
| Topology | frozen synthetic motifs |
| Input | byte embedding onto a sensory subset |
| Output | linear head on a readout subset |
| Train | Adam + BPTT, next-byte CE |
| Controls | degree-matched shuffle; GRU within ~35% scalar budget |
| Metric | bits-per-byte, tokens, time, memory, greedy sample |

## Layout

```
flybrain/data/connectome.py   synthetic graph + npz loader
flybrain/data/text.py         UTF-8 windows
flybrain/model/brain.py       FlyLanguageModel
flybrain/baselines/gru.py     matched ByteGRU
flybrain/accounting.py        numel() report
flybrain/train.py             shared train_steps / generate
scripts/run_tiny.py           three-arm smoke
```
