# Training runs on this backend are not bit-repeatable: measured run-to-run spread — 2026-09-15

**Runs:** `runs/repeatability-2026-09-15/` (400-step pairs and 5k repeats), plus
`runs/momentum-grid-2026-09-13/plain` and `runs/init-scale-screen-2026-09-15/k1`
**Status:** Measured. Corrects the statement, repeated in the momentum, lever-screen, and ladder
notes, that the 25M text8 model is bit-repeatable with the default attention backend.

## Finding

Two fresh runs with identical source, config, seed, card, and dataset diverge within the first 200
steps and end with different checkpoints, in every arm tried (`configs/scaleup_text8_25m_5k.toml`
with 400 steps, `runs/repeatability-2026-09-15/det400.toml`, GPU 0, source `634f0d6`):

| Pair, 400 steps | Run A | Run B | Checkpoints |
| --- | ---: | ---: | --- |
| plain ratchet | 2.0074 | 2.0787 | differ |
| FP32 control | 1.8185 | 1.8188 | differ |
| ratchet, `deterministic_attention` | 2.0307 | 1.9871 | differ |
| FP32, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 1.8190 | 1.8179 | differ |

Validation loss at step 400. The FP32 control drifts by a hair; the ratchet's integer thresholding
amplifies the hair into hundredths early in training. Neither the deterministic-attention flag nor
pinning the cuBLAS workspace removes the divergence, so the remaining sources are the FP32 token
embedding's backward (atomic scatter-adds), the flash attention backward (atomic dQ accumulation,
already documented in `model.py`), and BF16 autocast matmuls. The repository sets
`torch.use_deterministic_algorithms` nowhere.

Where the earlier claim came from: the 1B determinism probe (`docs/results/2026-09-13-provenance-reruns.md`)
found head size 64 with 256-token windows repeatable, but that probe used int8 matmuls, a compiled
update, and a ratcheted embedding. The text8 configs use BF16 autocast, an FP32 `nn.Embedding`, and
the fp32 eager matmul path, none of which the probe covered. The CPU smoke config is repeatable
across commits, so this is a CUDA-path property, not a code regression.

## Run-to-run spread at the 5k screen

Four runs of the plain ratchet at `configs/scaleup_text8_25m_5k.toml`, seed 1337, 15 codes, on
three different days and two cards: best validation 1.1671, 1.1646, 1.1678, 1.1661; mean 1.1664,
sample SD 0.0014, range 0.0032.

## Consequences

- **30k three-seed results stand.** Their seed spread (0.002 to 0.009) already contains this noise,
  and every conclusion there rests on effects of 0.05 or more.
- **5k single-seed screens need a bar.** A cell must differ from the plain mean by about 0.005 (three
  SDs) to mean anything. Re-read against that bar: the momentum grid's winner (0.015 below plain) was
  real; the update-rule screen's best cell (0.002 below plain) was noise, which is what the note
  concluded; the init-scale screen's cells are all far outside the bar in the wrong direction.
- **Sharing the cards did not bias anything.** Nondeterminism of this kind is unbiased noise; it only
  means "sharing changes wall-clock only" should have read "sharing changes wall-clock and adds no
  bias".
- **Identity checks cannot be run as bit-equality on this backend.** The init-scale driver's
  identity check fails by design; drivers should compare against a measured spread instead. An
  opt-in deterministic mode (`torch.use_deterministic_algorithms(True)` with the cuBLAS workspace
  setting) would restore paired comparisons at some speed cost; not implemented.
- The ladder note's caveat about the 7M rung's repeatability applies to every rung.
