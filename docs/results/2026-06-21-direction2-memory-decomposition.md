# Direction 2 Phase A: 2048-Width Memory Decomposition

**Date:** 2026-06-21
**Commit measured:** `6a5e20fcd5084b86b3d998f53d2af6dbfd78f61b`
**GPU:** NVIDIA GeForce RTX 3090, UUID
`GPU-51164b62-fd30-253a-5738-55844d46af9c`, 24,576 MiB
**Result:** the reported ~6.9 GiB was not a training-step peak. It was the cumulative high-water
mark of `collect_ratchet_metrics`. After resetting the allocator peak following warmup, the completed
int8 training step peaks **22.88 MiB below BF16**, not above it. The proposed int8-specific Phase B
optimization therefore has no measured target and should not begin without review.

## Method

`scripts/memory_decomposition.py` launches each mode three times in fresh processes on idle GPU 1.
Each child builds the matched 2048-width sweep configuration (batch 2, context 32, 16 layers, 16
heads, checkpointing, quinary codes, seed 1337), warms one complete forward/backward/update/Adam
step, empties the cache, resets peak statistics, and records a second completed step. JSON is written
only after the measured step, update, and optimizer step complete.

The probe also records evaluation and `collect_ratchet_metrics` in isolated peak-stat windows. A
separate diagnostic step wraps the functions imported by `ratchet.py`; its local peak resets cannot
contaminate the primary whole-step measurement. `saved_tensors_hooks` deduplicates storage by CUDA
storage pointer and attributes checkpoint inputs to their transformer block.

Command:

```bash
CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=/games/ailab/.uv-cache \
  uv run python scripts/memory_decomposition.py --repeats 3
```

Raw completion-guarded JSON is under ignored `runs/memory-decomposition/`.

## The Historical ~6.9 GiB Peak

All three runs produced identical allocator byte counts, so spread is `0.00 MiB` throughout.

| Region (median allocated peak) | BF16 MiB | int8 MiB |
| --- | ---: | ---: |
| Persistent baseline after warmup | 788.07 | 771.82 |
| Training forward | 911.07 | 894.82 |
| Completed training backward | **954.71** | **931.83** |
| Evaluation (`no_grad`, checkpointing inactive) | 903.57 | 887.32 |
| `collect_ratchet_metrics` | **6,933.08** | **6,916.83** |
| Historical sweep (`max_memory_allocated`, never reset) | 6,934 | 6,937 |

The 1-20 MiB difference from the old sweep is expected allocator/run context variation. The decisive
result is categorical: the isolated training, evaluation, and warmup peaks are all below 1.1 GiB;
only ratchet histogram collection reproduces ~6.9 GiB. `train.py` reports CUDA's cumulative peak
after evaluation and immediately calls `collect_ratchet_metrics`, so later CSV rows retain that
observability high-water mark.

The complete historical BF16 peak is therefore:

| Component at observability peak | MiB |
| --- | ---: |
| Persistent allocated baseline | 788.07 |
| Temporary/unattributed `collect_ratchet_metrics` working set | 6,145.02 |
| Total observed peak | **6,933.08** |

The 6,145.02 MiB is left explicitly unattributed within the metrics function. Its code, pressure,
saturation concatenations and property-level unpack temporaries overlap; assigning their logical
tensor sizes additively would overcount. It is not assigned to activations or int8 buffers.

## Persistent State

Both modes have identical named model/optimizer tensors. The BF16 process retains an additional
16.25 MiB of unenumerated CUDA allocations after warmup; int8 retains 0.02 MiB. This runtime residue
is included in the allocator baseline, not silently added to a named tensor category.

| State | dtype | tensors | bytes | MiB |
| --- | --- | ---: | ---: | ---: |
| Ratchet packed code + pressure | `torch.uint8` | 65 | 805,439,488 | 768.13 |
| Ratchet row scales | `torch.float32` | 65 | 1,179,908 | 1.13 |
| Embedding + RMSNorm support | `torch.float32` | 35 | 1,064,960 | 1.02 |
| Adam support state | `torch.float32` | 102 | 1,605,768 | 1.53 |
| Named tensor total | | 267 | 809,290,124 | 771.80 |

No floating-point code-mirror parameter is present.

## Steady-State Training Attribution

These rows reconcile each isolated backward peak. “Residual” is the observed peak minus persistent
allocation, unique saved storage, and final support gradients. Those three categories do not
necessarily coexist at their maxima, so the residual is conservative and deliberately not named.

| Component | BF16 MiB | int8 MiB |
| --- | ---: | ---: |
| Persistent allocated baseline | 788.07 | 771.82 |
| Unique saved-tensor storage | 9.15 | 9.15 |
| Final support gradients (`torch.float32`) | 0.77 | 0.77 |
| Residual/recompute/temporary overlap | 156.72 | 150.10 |
| Completed backward peak | **954.71** | **931.83** |
| Post-backward allocated | 788.83 | 772.58 |
| Post-update + optimizer allocated | 788.07 | 771.82 |

Checkpointing saves one 524,288-byte (`0.50 MiB`) input storage for each of blocks 00-15. Ten
outside-block storages total 1,208,072 bytes (`1.15 MiB`). Total unique saved storage is 9,596,680
bytes (`9.15 MiB`) in both modes.

| Blocks | Per-block MiB | Block total MiB | Outside MiB | Total MiB |
| --- | ---: | ---: | ---: | ---: |
| 00-15 | 0.50 | 8.00 | 1.15 | 9.15 |

## Direct int8 Allocation Evidence

The separate diagnostic step observed existing int8 operations directly:

| Operation | calls | summed output MiB | largest output MiB | largest local peak delta MiB |
| --- | ---: | ---: | ---: | ---: |
| `quantize_rows` | 1,347 | 65.31 | 0.50 | 4.00 |
| `quantize_columns` | 65 | 15.01 | 0.53 | 4.03 |
| tiled `scaled_int8_mm` | 1,347 | 1,636.51 | 4.00 | 4.02 |

Summed output bytes are allocation traffic, not simultaneous live storage, and must not be added to
the peak. The directly observed full `quantize_columns(flat_inputs)` result is at most 0.53 MiB per
call; its local peak delta is 4.03 MiB. Meanwhile the matched whole-step difference is:

`int8 - BF16 = 931.83 - 954.71 = -22.88 MiB`.

Thus both direct evidence and the matched comparison reject a positive int8-specific peak at this
configuration. They do not identify how much of the int8 peak is uniquely quantization workspace;
that uncertainty is retained rather than converted into an optimization claim.

## Limitations And Review Gate

- `memory_allocated` measures live PyTorch CUDA allocations, not reserved cache or driver memory.
- Operation-local peaks come from a later diagnostic step because resetting them during the primary
  step would destroy the whole-step high-water mark.
- Saved-storage attribution describes retained autograd storage, not recompute temporaries.
- Identical runs demonstrate deterministic allocator behavior here, not cross-driver portability.
- The old 3072 int8 OOM needs remeasurement with observability excluded before it can support a
  kernel-memory claim; the current evidence does not.

**Stop here for attribution review.** Phase B's proposed int8-buffer optimization is not justified
by this measurement. The next decision should first correct/re-run the scaling methodology so
training peaks and observability peaks are reported separately; no kernel or memory behavior has
been changed in Phase A.
