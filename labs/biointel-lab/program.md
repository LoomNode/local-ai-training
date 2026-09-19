# biointel-lab program

**Hypothesis:** a small shared learned program, instantiated across many stateful neurons and synapses, can produce more useful computation per trainable scalar than unique-weight point-neuron nets.

This is the same folding instinct as envelope-lab's [parameter inefficiency notes](/games/ailab/lab/envelope-lab/notes/parameter-inefficiency-and-folding.md), approached from the bio-inspired side: richer units, recurrent state, optional dendritic compartments, connectome-as-prior later.

## What this is not

- Not a molecule-for-molecule fly brain.
- Not “minimize the number printed as `parameters`.”
- Not starting from FlyWire FAFB (139,255 neurons) or BANC v888 (158,262 cells). Those are *later* structural priors.

## Accounting rule

If a “complex parameter” is 256 floats, it counts as 256 trainable scalars.

Every experiment reports: trainable scalars, trainable bits, persistent dynamic state, neuron count, synapse count, FLOPs/step, accuracy, and (when relevant) adaptation after a graph shuffle.

## Ablation ladder

`point neuron → recurrent internal state → dendritic compartments → unique vs shared synapses → degree-matched shuffled wiring.`

The question at each step is not “does it run?” It is: **does this mechanism buy capability per trainable bit (or per FLOP) versus the simpler arm?**

## First task

Delayed identity: a one-hot cue on step 0, silence, then report the class. Chance is 1/K. Recurrence should beat a point neuron here; a unique-weight GRU is the conventional baseline.

## Slice 0 result (do not overclaim)

On delayed identity, a unique-weight GRU is currently the thing to beat. Shared θ does not automatically win; the first CPU ladder already shows a wiring split (ER recurrent failed, degree-matched shuffle of the same θ succeeded).

## Slice 1 — two-timescale adapters

Stateful synapses (`s_ij`) plus a readout adapter are **state**, not trainable scalars. Outer loop trains the shared program with backprop. Inner loop updates only adapters after a label permutation.

First `shift_cpu` (N=16, delay=3, 80 outer / 60 inner): both the plastic net and the GRU hit 1.0 on identity and 0.0 frozen after a derangement. Inner-loop adapters restored the plastic net to **1.0 with θ unchanged**. The GRU also recovered to 1.0, but by **fine-tuning θ**.

Caveat: inner-loop still uses BPTT to move adapters. That is two-timescale learning, not yet a learned local G_φ (Hebbian/e-prop). Remapping delayed-identity labels is close to permuting a linear readout on features that already encode the cue.

## Slice 2 — local G_φ, no BPTT

`local_adapt` updates readout/synapse adapters with an explicit delta rule under `torch.no_grad()`. `Tensor.backward` is not called. Resetting adapters returns the developmental program.

First vs last cue was a dead end for forgetting: a unique GRU trained on last-cue still held first-cue (~0.79). Label derangement is the shift that actually overwrites a GRU readout.

See `results/local_cpu.json` after `scripts/run_local.py`.

## Later (not this slice)

- Outer-loop *learned* G_φ coefficients (meta-plasticity), not a hand delta rule.
- Neuromodulation / task context so adapters do not have to be manually reset.
- Small biological subnetworks vs degree-matched random graphs.
- BANC v888 as a sensorimotor prior, never as the first model.
