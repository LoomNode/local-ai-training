USDT:2026-10-04 09:32:49 3747813:3747813 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:32:51 3747813:3747813 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_c_all_chunk512.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused, --compile, --head-amp, --scan, --sparse-claws; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.4771 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.2216 | forward + loss; median 0.2206, min 0.2179, max 0.2296 |
| split: backward | 0.2566 | zero_grad + backward; median 0.2568, min 0.2512, max 0.2618 |
| split: optimizer | 0.0028 | clip_grad_norm_ 0.0015 + optimizer step and schedule 0.0013 (median 0.0013, max 0.0013) |
| split: sum | 0.4813 | sync at every boundary: 1.01 x the plain step |
| energy per step (J) | 71.5 | mean power 150 W (limit 150 W), max 60 C, contended False |

Warmup steps (s): 46.608, 0.426, 0.499, 0.449, 0.502 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 552.8 ms, the parts add up to 552.7 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 546.9 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 221.6 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 307.10 | 55.6 |
| recurrence scan (all layers) | 139.20 | 25.2 |
| layers, rest (LN, projections, FFN) | 79.26 | 14.3 |
| local attention | 23.13 | 4.2 |
| model glue (residual adds, loops over layers) | 2.07 | 0.4 |
| decoder | 0.81 | 0.1 |
| final LayerNorm | 0.52 | 0.1 |
| loss | 0.34 | 0.1 |
| embedding | 0.27 | 0.0 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 2.11 | 0.4 | 0.7 |
| key conv | 5.93 | 1.1 | 1.9 |
| value projection | 4.92 | 0.9 | 1.6 |
| fp32 pin: casts, glue | 0.07 | 0.0 | 0.0 |
| address: no-grad matmul + top-k | 99.42 | 18.0 | 32.4 |
| address: recompute selected | 3.34 | 0.6 | 1.1 |
| write/read (rest of sparse_memory_read) | 178.35 | 32.3 | 58.1 |
| normaliser + gate | 3.25 | 0.6 | 1.1 |
| output projection | 4.97 | 0.9 | 1.6 |
| other (padding, residual) | 4.73 | 0.9 | 1.5 |
| **all heads** | 307.10 | 55.6 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.51 | 1.43 | 1.23 | 0.02 | 24.80 | 0.82 | 44.74 | 0.81 | 1.24 | 1.17 | 76.76 | 13.9 |
| head after layer 2 | 0.53 | 1.53 | 1.23 | 0.02 | 24.97 | 0.85 | 44.88 | 0.83 | 1.24 | 1.18 | 77.27 | 14.0 |
| head after layer 4 | 0.54 | 1.44 | 1.24 | 0.02 | 24.79 | 0.83 | 43.94 | 0.82 | 1.25 | 1.20 | 76.06 | 13.8 |
| head after layer 6 | 0.53 | 1.53 | 1.23 | 0.02 | 24.87 | 0.83 | 44.79 | 0.80 | 1.24 | 1.17 | 77.01 | 13.9 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 17.38 | 3.1 | 9.92 | 1.8 |
| 2 | 17.41 | 3.1 | 9.92 | 1.8 |
| 3 | 17.40 | 3.1 | 9.90 | 1.8 |
| 4 | 17.40 | 3.1 | 9.91 | 1.8 |
| 5 | 17.41 | 3.1 | 9.92 | 1.8 |
| 6 | 17.40 | 3.1 | 9.91 | 1.8 |
| 7 | 17.39 | 3.1 | 9.89 | 1.8 |
| 8 | 17.40 | 3.1 | 9.89 | 1.8 |

### Top operators by self device time, forward + backward, per step (all operators: 478.8 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 102 | 137.05 | 28.6 |
| `aten::addmm` | 27 | 47.36 | 9.9 |
| `aten::topk` | 16 | 46.21 | 9.7 |
| `aten::bmm` | 64 | 46.20 | 9.6 |
| `aten::_efficient_attention_backward` | 1 | 15.17 | 3.2 |
| `triton_poi_fused_add_mul_select_backward_slice_68` | 8 | 6.77 | 1.4 |
| `aten::convolution_backward` | 4 | 6.38 | 1.3 |
| `triton_poi_fused_add_mul_select_backward_slice_65` | 8 | 6.22 | 1.3 |
| `triton_red_fused__to_copy_add_arange_clamp_div_eq_expand_lift_fresh_mul_neg_pow_slice_sub_sum_unsqueeze_29` | 3 | 6.19 | 1.3 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_select_backward_sigmoid_sigmoid_backward_slice_view_where_71` | 8 | 5.88 | 1.2 |
| `aten::_efficient_attention_forward` | 1 | 5.76 | 1.2 |
| `aten::_conv_depthwise2d` | 4 | 5.10 | 1.1 |
| `triton_poi_fused_relu_sub_23` | 16 | 4.61 | 1.0 |
| `triton_poi_fused_add_mul_select_backward_slice_66` | 8 | 4.30 | 0.9 |
| `triton_poi_fused_add_mul_select_backward_slice_63` | 8 | 4.29 | 0.9 |
| `triton_poi_fused_add_mul_select_backward_slice_69` | 8 | 4.26 | 0.9 |
| `triton_poi_fused_add_mul_select_backward_slice_62` | 8 | 3.86 | 0.8 |
| `triton_poi_fused_add_mul_select_backward_slice_70` | 8 | 3.57 | 0.7 |
| `triton_poi_fused_add_mul_select_backward_slice_67` | 8 | 3.57 | 0.7 |
| `triton_red_fused_eq_expand_mul_slice_sum_unsqueeze_27` | 4 | 3.25 | 0.7 |
| `triton_poi_fused_add_mul_select_backward_slice_64` | 8 | 3.24 | 0.7 |
| `triton_poi_fused__to_copy__unsafe_view_add_arange_lift_fresh_mul_pow_rsub_slice_unsqueeze_view_46` | 13 | 2.92 | 0.6 |
| `triton_poi_fused__to_copy_add_arange_div_expand_index_add_lift_fresh_mul_pow_slice_softplus_unsqueeze_view_31` | 4 | 2.92 | 0.6 |
| `triton_poi_fused_add_cat_mul_slice_14` | 8 | 2.64 | 0.6 |
| `triton_poi_fused_add_cat_mul_slice_10` | 8 | 2.63 | 0.5 |

Kernel time in the profiled forward + backward: 478.8 ms; the split run's forward + backward took 478.3 ms, so the device was busy 100 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 50.00 | 10.4 |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_nt` | 32 | 35.95 | 7.5 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 35.27 | 7.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 27.01 | 5.6 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 20.35 | 4.3 |
| `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::Tenso` | 64 | 18.43 | 3.9 |
| `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned` | 16 | 15.60 | 3.3 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_tn` | 16 | 15.55 | 3.2 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 15.17 | 3.2 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 14.90 | 3.1 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 14.82 | 3.1 |
| `triton_poi_fused_add_mul_select_backward_slice_68` | 8 | 6.77 | 1.4 |
| `triton_poi_fused_add_mul_select_backward_slice_65` | 8 | 6.22 | 1.3 |
| `triton_red_fused__to_copy_add_arange_clamp_div_eq_expand_lift_fresh_mul_neg_pow_slice_sub_sum_unsqueeze_29` | 3 | 6.19 | 1.3 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_select_backward_sigmoid_sig` | 8 | 5.88 | 1.2 |

### Memory

Peak allocated 8.49 GiB, peak reserved 8.72 GiB over the timed training steps; the forward leaves 8.11 GiB of activations.

