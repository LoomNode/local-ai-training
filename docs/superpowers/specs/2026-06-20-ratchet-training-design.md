# Quinary and Septenary Ratchet Training Design

## Question

Can a tiny Transformer learn when its dense matrices have only five or seven persistent
weight states and an integer pressure counter, with no floating-point master copy?

The first phase tests algorithmic viability. It does not test low-bit throughput because
ordinary PyTorch must materialize floating-point effective weights for its dense kernels.

## Precision Boundary

Every attention, MLP, and untied language-head matrix uses a configurable discrete ratchet
layer. Persistent layer state is an `int8` code matrix, an `int8` pressure matrix, and one
fixed FP32 scale per output row. Forward creates a temporary differentiable
`scale * code` tensor. Backward creates its temporary gradient. The post-backward update
converts that gradient into pressure and code moves, records statistics, and clears both.

Token embeddings and RMSNorm parameters remain floating point, matching the practical
boundary of low-bit Transformer research. Fixed sinusoidal positions and bias-free linear
layers minimize unrelated trainable state. AdamW only sees the small floating-point support
tensors. A runtime audit rejects floating-point ratchet weight parameters.

## Update Rule

For each output row, normalize the effective-weight gradient by its RMS plus epsilon.
Bucket normalized magnitudes below 0.5 to zero, magnitudes in `[0.5, 1.5)` to one, and
magnitudes at or above 1.5 to two. Negate the gradient sign so pressure follows gradient
descent. At the default pressure threshold eight, move the code one step and retain residual
pressure. Perform update arithmetic in a wider integer type before clamping to `int8`.

Quinary uses codes `[-2, 2]`; septenary uses `[-3, 3]`. Outward moves at a boundary consume
their threshold and are recorded as blocked moves so pressure cannot wind up indefinitely.
V1 scales are fixed. A seed produces one logical floating-point initialization per matrix;
each arm quantizes it with `row_max_abs / max_code`.

## Comparison

Use a pinned script-free Hugging Face Tiny Shakespeare mirror, character tokenization, and
a deterministic final-ten-percent validation split. For each seed, both arms receive the
same batch index schedule and evaluation batches. The research configuration uses seeds
1337, 1338, and 1339. Viability requires finite improving validation loss below its initial
value and the random-character reference, without rapid near-total saturation.

CSV logs include loss, validation loss, throughput, code and pressure histograms, zero and
saturation rates, positive/negative/blocked moves, persistent state bytes, and device memory.
Checkpoints use safetensors plus validated JSON metadata.

## Safety

Pin the dataset revision, disable remote dataset code, validate downloaded schema, keep
artifacts under ignored project directories, validate every configuration and checkpoint,
fail on non-finite loss or scale, and assert ratchet state invariants after updates.

