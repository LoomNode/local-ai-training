# biointel-lab shared-rule PoC

- **Date:** 2026-09-12
- **Status:** implemented (slice 0)
- **Source brief:** a private ChatGPT conversation

## Goal

A tiny PyTorch framework where neuron model, synapse model, and graph topology swap independently, with honest resource accounting. First task is delayed identity, not the fly connectome.

## Non-goals (slice 0)

- Loading FlyWire / BANC
- Learned plasticity / e-prop
- Neuromodulation
- GPU envelope (CPU is enough; graphs are N≤64)

## Components

- `SparseGraph` + Erdős–Rényi + degree-matched destination shuffle
- `SharedRuleNet`: shared message Linear + point tanh / GRUCell / two-compartment GRUCell
- `UniqueGRU` baseline
- `ResourceReport`: scalars, bits, state, neurons, synapses, FLOPs/step
- Ladder writer in `scripts/run_ladder.py`

## Success for this slice

- Tests pin the accounting cheat, N-independence of shared θ, degree-matched control, and a recurrent train smoke that beats chance.
- A ladder JSON exists with the same keys on every arm.

## Slice 1 (two-timescale)

- `synapse="stateful"`: `s_ij` and a readout adapter are `requires_grad=False` parameters (state). Trainable scalar count matches `synapse="shared"`.
- `inner_adapt` freezes θ, Adam-updates adapters on a label derangement, then freezes adapters again.
- Tests: state vs params; program tensors unchanged; adapt acc ≥ frozen + 0.15.
- Not yet: learned local G_φ without BPTT.

## Slice 2 (local delta rule)

- `local_adapt` runs under `no_grad` and must not call `Tensor.backward`.
- Label derangement: adapters learn B; `reset_adapters` restores A.
- Unique GRU fine-tune on the same derangement overwrites A (no adapter snapshot).
- First-vs-last cue does not make the GRU forget A; do not use it as a forgetting task.
