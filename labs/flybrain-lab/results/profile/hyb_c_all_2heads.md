USDT:2026-10-04 09:31:28 3746616:3746616 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:31:30 3746616:3746616 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_c_all_2heads.json
# Step profile

## Arm `mingru_sparse_post` (14,998,788 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused, --compile, --head-amp, --scan, --sparse-claws; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.4150 | 15 steps, one sync at the end, data draw included; final loss 5.163 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0004 |
| split: forward | 0.1747 | forward + loss; median 0.1750, min 0.1708, max 0.1771 |
| split: backward | 0.2410 | zero_grad + backward; median 0.2419, min 0.2311, max 0.2444 |
| split: optimizer | 0.0026 | clip_grad_norm_ 0.0014 + optimizer step and schedule 0.0012 (median 0.0012, max 0.0014) |
| split: sum | 0.4186 | sync at every boundary: 1.01 x the plain step |
| energy per step (J) | 62.3 | mean power 150 W (limit 150 W), max 59 C, contended False |

Warmup steps (s): 42.046, 0.442, 0.395, 0.415, 0.426 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 382.1 ms, the parts add up to 382.1 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 379.5 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 174.7 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 140.85 | 36.9 |
| recurrence scan (all layers) | 137.23 | 35.9 |
| layers, rest (LN, projections, FFN) | 78.00 | 20.4 |
| local attention | 22.76 | 6.0 |
| model glue (residual adds, loops over layers) | 1.33 | 0.3 |
| decoder | 0.80 | 0.2 |
| final LayerNorm | 0.52 | 0.1 |
| loss | 0.34 | 0.1 |
| embedding | 0.27 | 0.1 |
| unattributed (outside the model call and the loss) | 0.04 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.03 | 0.3 | 0.7 |
| key conv | 2.93 | 0.8 | 2.1 |
| value projection | 2.42 | 0.6 | 1.7 |
| fp32 pin: casts, glue | 0.03 | 0.0 | 0.0 |
| address: no-grad matmul + top-k | 49.32 | 12.9 | 35.0 |
| address: recompute selected | 1.68 | 0.4 | 1.2 |
| write/read (rest of sparse_memory_read) | 76.51 | 20.0 | 54.3 |
| normaliser + gate | 2.17 | 0.6 | 1.5 |
| output projection | 2.44 | 0.6 | 1.7 |
| other (padding, residual) | 2.32 | 0.6 | 1.6 |
| **all heads** | 140.85 | 36.9 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.50 | 1.41 | 1.21 | 0.01 | 24.65 | 0.83 | 38.25 | 1.08 | 1.22 | 1.15 | 70.31 | 18.4 |
| head after layer 4 | 0.53 | 1.51 | 1.21 | 0.02 | 24.68 | 0.84 | 38.26 | 1.10 | 1.22 | 1.17 | 70.54 | 18.5 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 17.16 | 4.5 | 9.77 | 2.6 |
| 2 | 17.14 | 4.5 | 9.76 | 2.6 |
| 3 | 17.14 | 4.5 | 9.75 | 2.6 |
| 4 | 17.14 | 4.5 | 9.75 | 2.6 |
| 5 | 17.14 | 4.5 | 9.76 | 2.6 |
| 6 | 17.13 | 4.5 | 9.75 | 2.6 |
| 7 | 17.25 | 4.5 | 9.73 | 2.5 |
| 8 | 17.14 | 4.5 | 9.73 | 2.5 |

### Top operators by self device time, forward + backward, per step (all operators: 414.1 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 90 | 136.34 | 32.9 |
| `aten::addmm` | 27 | 51.15 | 12.4 |
| `aten::topk` | 8 | 22.78 | 5.5 |
| `aten::bmm` | 56 | 22.41 | 5.4 |
| `aten::_efficient_attention_backward` | 1 | 15.79 | 3.8 |
| `triton_poi_fused_add_mul_select_backward_slice_97` | 8 | 7.58 | 1.8 |
| `triton_poi_fused_add_mul_select_backward_slice_94` | 8 | 7.16 | 1.7 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_select_backward_sigmoid_sigmoid_backward_slice_view_where_100` | 8 | 6.64 | 1.6 |
| `aten::_efficient_attention_forward` | 1 | 6.01 | 1.5 |
| `triton_poi_fused_add_mul_select_backward_slice_92` | 8 | 5.02 | 1.2 |
| `triton_poi_fused_add_mul_select_backward_slice_95` | 8 | 4.86 | 1.2 |
| `triton_poi_fused_add_mul_select_backward_slice_98` | 8 | 4.77 | 1.2 |
| `triton_poi_fused_add_mul_select_backward_slice_91` | 8 | 4.50 | 1.1 |
| `triton_poi_fused_add_mul_select_backward_slice_96` | 8 | 4.02 | 1.0 |
| `triton_poi_fused_add_mul_select_backward_slice_99` | 8 | 4.00 | 1.0 |
| `triton_poi_fused_add_mul_select_backward_slice_93` | 8 | 3.80 | 0.9 |
| `triton_per_fused__to_copy_add_cat_mul_native_layer_norm_native_layer_norm_backward_slice_unsqueeze_view_18` | 6 | 3.52 | 0.8 |
| `aten::convolution_backward` | 2 | 3.23 | 0.8 |
| `triton_poi_fused_new_zeros_41` | 15 | 3.15 | 0.8 |
| `triton_poi_fused_gelu_gelu_backward_view_86` | 8 | 3.03 | 0.7 |
| `triton_poi_fused_add_cat_mul_slice_10` | 8 | 2.84 | 0.7 |
| `triton_poi_fused_add_cat_mul_slice_11` | 8 | 2.83 | 0.7 |
| `triton_poi_fused_add_cat_mul_slice_9` | 8 | 2.82 | 0.7 |
| `triton_poi_fused_add_cat_mul_slice_6` | 8 | 2.81 | 0.7 |
| `triton_poi_fused_add_cat_mul_slice_7` | 8 | 2.79 | 0.7 |

Kernel time in the profiled forward + backward: 414.1 ms; the split run's forward + backward took 415.7 ms, so the device was busy 100 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 38 | 51.27 | 12.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 21 | 31.43 | 7.6 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 29.01 | 7.0 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 22.14 | 5.3 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 17.45 | 4.2 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 17.23 | 4.2 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 15.79 | 3.8 |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_nt` | 8 | 15.58 | 3.8 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_tn` | 12 | 12.13 | 2.9 |
| `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::Tenso` | 32 | 9.16 | 2.2 |
| `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned` | 8 | 7.60 | 1.8 |
| `triton_poi_fused_add_mul_select_backward_slice_97` | 8 | 7.58 | 1.8 |
| `triton_poi_fused_add_mul_select_backward_slice_94` | 8 | 7.16 | 1.7 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_select_backward_sigmoid_sig` | 8 | 6.64 | 1.6 |
| `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass:` | 1 | 6.01 | 1.5 |

### Memory

Peak allocated 8.22 GiB, peak reserved 8.44 GiB over the timed training steps; the forward leaves 7.56 GiB of activations.

