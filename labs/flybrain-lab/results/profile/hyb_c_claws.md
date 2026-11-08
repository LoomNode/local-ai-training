USDT:2026-10-04 09:25:39 3739108:3739108 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:25:40 3739108:3739108 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_c_claws.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --compile, --sparse-claws; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.4973 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0002 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.2038 | forward + loss; median 0.2035, min 0.1973, max 0.2079 |
| split: backward | 0.2937 | zero_grad + backward; median 0.2932, min 0.2808, max 0.3034 |
| split: optimizer | 0.0037 | clip_grad_norm_ 0.0016 + optimizer step and schedule 0.0020 (median 0.0020, max 0.0021) |
| split: sum | 0.5014 | sync at every boundary: 1.01 x the plain step |
| energy per step (J) | 74.4 | mean power 150 W (limit 150 W), max 55 C, contended False |

Warmup steps (s): 78.508, 0.484, 0.518, 0.488, 0.513 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 407.1 ms, the parts add up to 407.0 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 407.7 ms, so the instrumented one is 1.00 x it (the syncs remove the overlap of launches with work); the forward in the split run took 203.8 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 231.67 | 56.9 |
| recurrence scan (all layers) | 106.09 | 26.1 |
| layers, rest (LN, projections, FFN) | 55.72 | 13.7 |
| local attention | 10.15 | 2.5 |
| model glue (residual adds, loops over layers) | 1.78 | 0.4 |
| decoder | 0.80 | 0.2 |
| loss | 0.34 | 0.1 |
| embedding | 0.25 | 0.1 |
| final LayerNorm | 0.24 | 0.1 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 0.94 | 0.2 | 0.4 |
| key conv | 2.74 | 0.7 | 1.2 |
| value projection | 2.23 | 0.5 | 1.0 |
| fp32 pin: casts, glue | 1.11 | 0.3 | 0.5 |
| address: no-grad matmul + top-k | 79.16 | 19.4 | 34.2 |
| address: recompute selected | 2.21 | 0.5 | 1.0 |
| write/read (rest of sparse_memory_read) | 132.18 | 32.5 | 57.1 |
| normaliser + gate | 3.81 | 0.9 | 1.6 |
| output projection | 4.90 | 1.2 | 2.1 |
| other (padding, residual) | 2.39 | 0.6 | 1.0 |
| **all heads** | 231.67 | 56.9 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.23 | 0.66 | 0.55 | 0.28 | 17.89 | 0.49 | 32.81 | 0.94 | 1.22 | 0.60 | 55.67 | 13.7 |
| head after layer 2 | 0.22 | 0.68 | 0.53 | 0.26 | 17.90 | 0.47 | 32.08 | 0.94 | 1.23 | 0.57 | 54.89 | 13.5 |
| head after layer 4 | 0.24 | 0.68 | 0.57 | 0.29 | 25.24 | 0.75 | 35.37 | 1.01 | 1.21 | 0.61 | 65.96 | 16.2 |
| head after layer 6 | 0.24 | 0.73 | 0.58 | 0.29 | 18.12 | 0.49 | 31.93 | 0.92 | 1.24 | 0.61 | 55.15 | 13.5 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 15.75 | 3.9 | 7.67 | 1.9 |
| 2 | 8.92 | 2.2 | 4.87 | 1.2 |
| 3 | 17.34 | 4.3 | 8.79 | 2.2 |
| 4 | 9.82 | 2.4 | 5.62 | 1.4 |
| 5 | 17.03 | 4.2 | 8.79 | 2.2 |
| 6 | 9.98 | 2.5 | 5.78 | 1.4 |
| 7 | 17.46 | 4.3 | 8.58 | 2.1 |
| 8 | 9.79 | 2.4 | 5.63 | 1.4 |

### Top operators by self device time, forward + backward, per step (all operators: 504.5 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 102 | 70.03 | 13.9 |
| `triton_per_fused__to_copy__unsafe_view_clamp_min_cumsum_div_exp_log_mul_rsub_sigmoid_view_3` | 8 | 64.83 | 12.8 |
| `triton_poi_fused__to_copy__unsafe_view_add_div_exp_flip_mul_neg_select_select_backward_sigmoid_squeeze_stack_unsqueeze_view_zeros_103` | 8 | 55.22 | 10.9 |
| `triton_per_fused_cumsum_exp_flip_mul_102` | 16 | 40.00 | 7.9 |
| `aten::bmm` | 112 | 36.88 | 7.3 |
| `triton_poi_fused_add_exp_mul_select_select_backward_squeeze_100` | 8 | 31.44 | 6.2 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_exp_flip_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoid_backward_view_where_104` | 8 | 28.81 | 5.7 |
| `aten::addmm` | 27 | 26.34 | 5.2 |
| `triton_poi_fused__to_copy_add_exp_mul_select_select_backward_squeeze_view_101` | 8 | 19.43 | 3.9 |
| `triton_per_fused__to_copy_exp_mul_sum_view_98` | 8 | 17.89 | 3.5 |
| `aten::topk` | 16 | 16.68 | 3.3 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_75` | 4 | 10.05 | 2.0 |
| `aten::_efficient_attention_backward` | 1 | 6.47 | 1.3 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_103` | 1 | 5.11 | 1.0 |
| `triton_poi_fused__to_copy_convolution_backward_transpose_80` | 4 | 3.42 | 0.7 |
| `aten::_efficient_attention_forward` | 1 | 2.90 | 0.6 |
| `aten::convolution_backward` | 4 | 2.90 | 0.6 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_8` | 2 | 2.81 | 0.6 |
| `triton_poi_fused_relu_sub_17` | 16 | 2.58 | 0.5 |
| `triton_poi_fused_new_zeros_25` | 29 | 2.55 | 0.5 |
| `triton_poi_fused_mul_56` | 4 | 2.02 | 0.4 |
| `triton_poi_fused_mul_50` | 4 | 2.02 | 0.4 |
| `aten::_conv_depthwise2d` | 4 | 1.92 | 0.4 |
| `triton_poi_fused__to_copy_add_arange_div_expand_index_add_lift_fresh_mul_pow_slice_softplus_unsqueeze_view_19` | 4 | 1.90 | 0.4 |
| `triton_poi_fused__to_copy_add_cat_div_mul_softplus_101` | 3 | 1.77 | 0.4 |

Kernel time in the profiled forward + backward: 504.5 ms; the split run's forward + backward took 497.5 ms, so the device was busy 101 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `triton_per_fused__to_copy__unsafe_view_clamp_min_cumsum_div_exp_log_mul_rsub_sigmoid_view_3` | 8 | 64.83 | 12.8 |
| `triton_poi_fused__to_copy__unsafe_view_add_div_exp_flip_mul_neg_select_select_backward_sigmoid_squeeze_stack_u` | 8 | 55.22 | 10.9 |
| `triton_per_fused_cumsum_exp_flip_mul_102` | 16 | 40.00 | 7.9 |
| `triton_poi_fused_add_exp_mul_select_select_backward_squeeze_100` | 8 | 31.44 | 6.2 |
| `ampere_sgemm_128x128_tn` | 16 | 29.64 | 5.9 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_exp_flip_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoi` | 8 | 28.81 | 5.7 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 26.30 | 5.2 |
| `triton_poi_fused__to_copy_add_exp_mul_select_select_backward_squeeze_view_101` | 8 | 19.43 | 3.9 |
| `triton_per_fused__to_copy_exp_mul_sum_view_98` | 8 | 17.89 | 3.5 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 16.64 | 3.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 14.18 | 2.8 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 12.16 | 2.4 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_75` | 4 | 10.05 | 2.0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 8.89 | 1.8 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 8.46 | 1.7 |

### Memory

Peak allocated 4.82 GiB, peak reserved 5.03 GiB over the timed training steps; the forward leaves 4.13 GiB of activations.

