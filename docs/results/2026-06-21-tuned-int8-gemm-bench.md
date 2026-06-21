# Tuned int8 GEMM Re-Benchmark: Results

## Setup
torchao Int8DynamicActivationInt8WeightConfig vs bf16 nn.Linear, single RTX 3090, nonary-100M
layer shapes at a 16384-token batch. torchao install: uv add, version 0.17.0.

## Numbers
| shape (N x K, T) | bf16 ms | torchao int8 ms | torchao vs bf16 |
| --- | ---: | ---: | ---: |
| 768 x 768, 16384 | 0.308 | 1.289 | 0.24x |
| 2304 x 768, 16384 | 0.955 | 2.333 | 0.41x |
| 3072 x 768, 16384 | 1.165 | 2.879 | 0.40x |

## Verdict
NO-GO -- a tuned int8 kernel failed to beat bf16 and is actually significantly slower (0.24x - 0.41x the speed). This closes the speed question definitively (not a torch._int_mm artifact). int8 genuinely does not beat bf16 on this hardware at these shapes.
