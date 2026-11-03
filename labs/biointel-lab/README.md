# biointel-lab

> **DORMANT — nothing open (2026-11-03).** Paused; see the top-level README.

Shared-rule neuron/synapse experiments: **capability per trainable bit**, not a fly-scale simulator.

Origin: ChatGPT research brief Estimate Intelligence Parameters (private ChatGPT conversation). Sibling thesis: envelope-lab [parameter inefficiency and folding](../envelope-lab/notes/parameter-inefficiency-and-folding.md). Next-byte fly-prior experiment: [`../flybrain-lab`](../flybrain-lab/README.md).

## Setup

```bash
cd /games/ailab/lab/biointel-lab
uv sync --extra dev
uv run pytest -q
```

## How to run

```bash
# unit tests (CPU)
uv run pytest -q

# tiny ladder (plumbing; not a quality claim)
uv run python scripts/run_ladder.py --tiny

# slightly larger CPU comparison
uv run python scripts/run_ladder.py

# two-timescale label shift (adapters only vs GRU fine-tune)
uv run python scripts/run_shift.py

# local G_φ (no BPTT) vs GRU fine-tune; reset adapters restores identity
uv run python scripts/run_local.py
```

JSON lands in `results/{run_name}.json`.

## Ladder arms

| Arm | What it is |
|-----|------------|
| `unique_gru` | Conventional unique-weight GRU (not a graph) |
| `shared_point` | Shared F_θ, no self-recurrence |
| `shared_recurrent` | Shared GRUCell on an Erdős–Rényi graph |
| `shared_dendritic` | Two dendritic aggregates + shared GRUCell |
| `shared_recurrent_shuffled` | Same program on a degree-matched rewire |

Shared-program parameter count does **not** grow with neuron count. Persistent state does.

First `ladder_cpu` (seed 0, 80 steps, N=32, delay=4): unique GRU (**1124** scalars) hit 1.0 eval. Shared point stayed at chance. Unshuffled shared-recurrent did **not** beat chance; the degree-matched shuffle of the same program did. Wiring already matters, and this slice is not a parameter-efficiency win over the GRU.

First `shift_cpu`: after a label derangement, frozen accuracy is 0. Inner-loop synapse+readout adapters restore the shared-rule net to 1.0 **without changing θ**. A unique GRU also recovers, by updating θ. Inner-loop still backprops through time into adapters — not a local Hebbian rule yet.

First `local_cpu` (no `backward`): local delta-rule adapters take B from 0 → 1.0 with θ frozen. Identity with adapters on is 0; **reset adapters restores identity to 1.0**. GRU fine-tune also gets B=1.0 and **destroys identity** (0.0) because it overwrites θ. Caveat: this is still a linear remap of frozen features, with a hand-written delta rule, not a meta-learned G_φ.

## Honest counting

`biointel.accounting.count_resources` sums `numel()` of trainable tensors. A single `Parameter` of shape `(8, 32)` is 256 scalars, not one “complex parameter.”
