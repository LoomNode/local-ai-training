# flybrain-lab

Next-byte language model with a **fly-derived architectural prior**. V0 is a tiny sparse RNN, not a 139k-neuron simulator.

**Findings so far:** [`notes/RESULTS.md`](notes/RESULTS.md). Short version: fly *anatomy* gave nothing over fair controls; fly *computation* (sparse random expansion + fast synapses) as a memory head on a GRU gives 5–15 % lower per-byte perplexity for +6 % parameters, confirmed on seeds, wider/tuned/longer baselines, a leakage control, and held-out prose and code.

Handoff: Fly Brain Text Generation (private ChatGPT conversation). Shared-rule delayed-identity work stays in [`../biointel-lab`](../biointel-lab/README.md).

## Setup

```bash
cd /games/ailab/lab/flybrain-lab
uv sync --extra dev
uv run pytest -q
```

## How to run

```bash
uv run pytest -q
uv run python scripts/run_tiny.py
# held-out WikiText-2 byte slice (local file); flatten head + mean-pool head
uv run python scripts/run_eval.py --device cpu
uv run python scripts/run_v1.py
uv run python scripts/run_v2.py
uv run python scripts/run_v3.py --run-name v3_episode
```

JSON lands in `results/{run_name}.json`. **Interpreted record:** [`notebook.md`](notebook.md) (resume: [`notes/HANDOFF.md`](notes/HANDOFF.md); directions: [`notes/DIRECTIONS.md`](notes/DIRECTIONS.md); FLM comparison: [`notes/FLM.md`](notes/FLM.md)). `run_tiny.py` is plumbing. `run_eval.py` is held-out bits/byte. Neither is a FlyWire result.

## V0 rules the coding agent is not allowed to “simplify away”

- 4-d continuous state per neuron; dimensions have no preset meaning.
- Cell type selects the update program (`GRUCell` per type), shared across neurons of that type.
- Default synapse is scalar; `message=` can be `scalar`, `vector2`, `vector` (4-d × one \(p\)), or `projected` (4-d \(w\) per edge). Still no dense \(N\times N\).
- \(w_{ij} = \mathrm{sign}_{ij}\cdot\mathrm{softplus}(p_{ij})\) (broadcast over channels for `projected`). Signs do not flip.
- Bytes enter a sensory population; logits come from a readout population. No \(256\times N\) broadcast, no \(N\to 256\) from every neuron.
- Topology frozen. \(H_t\) is checkpointed separately from \(\theta\) (returned state, not a `Parameter`).
- No dense \(N\times N\) matrix.

## Honest counting

`flybrain.accounting.count_resources` sums `numel()` of trainable tensors. Persistent neural state is reported separately and is not a trainable scalar.
