# Tiny Shakespeare Full Comparison

## Protocol

The full `configs/ratchet_tiny.toml` comparison trained quinary and septenary models for
2,000 steps with matched seeds 1337, 1338, and 1339. An orphaned Ollama process initially
held GPU memory; after it was stopped, all six runs completed without source changes.

## Results

| Arm | Seed | Initial validation | Final validation | Final train | Final saturation | Final zeros | Moves at logged steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Quinary | 1337 | 4.3608 | 1.8218 | 1.7017 | 57.83% | 11.15% | 49,849 |
| Quinary | 1338 | 4.3492 | 1.8285 | 1.6987 | 57.47% | 11.36% | 49,393 |
| Quinary | 1339 | 4.3556 | 1.8296 | 1.6664 | 57.47% | 11.38% | 48,265 |
| Septenary | 1337 | 4.3601 | 1.8017 | 1.6732 | 47.13% | 8.36% | 53,828 |
| Septenary | 1338 | 4.3333 | 1.7967 | 1.6580 | 47.00% | 8.52% | 53,388 |
| Septenary | 1339 | 4.3116 | 1.7839 | 1.6303 | 47.89% | 8.21% | 55,887 |

Mean final validation loss was 1.8266 for quinary and 1.7941 for septenary. Septenary was
better on every seed by this metric and averaged about 10.25 percentage points less final
saturation. This is positive evidence that both master-weight-free ratchets learn on the
small corpus, with a modest and consistent advantage for the seven-state version.

The move column is the sum of per-step moves at the 20 evaluation/logging steps. The current
CSV does not persist cumulative moves across all 2,000 updates, so these values must not be
reported as total training moves.

Artifacts are under ignored `runs/tiny-shakespeare/`, including all CSVs, checkpoints, and
`comparison.png`.

