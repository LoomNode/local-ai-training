USDT:2026-10-04 09:27:26 3741227:3741227 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:27:27 3741227:3741227 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_c_detach.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --compile, --detach-state; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.4786 | 15 steps, one sync at the end, data draw included; final loss 5.153 |
| split: data | 0.0002 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.2041 | forward + loss; median 0.2043, min 0.1983, max 0.2096 |
| split: backward | 0.2748 | zero_grad + backward; median 0.2739, min 0.2716, max 0.2791 |
| split: optimizer | 0.0036 | clip_grad_norm_ 0.0016 + optimizer step and schedule 0.0020 (median 0.0020, max 0.0021) |
| split: sum | 0.4827 | sync at every boundary: 1.01 x the plain step |
| energy per step (J) | 71.6 | mean power 150 W (limit 150 W), max 56 C, contended False |

Warmup steps (s): 73.229, 0.473, 0.488, 0.465, 0.494 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 437.8 ms, the parts add up to 437.7 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 432.6 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 204.1 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 254.54 | 58.1 |
| recurrence scan (all layers) | 109.23 | 24.9 |
| layers, rest (LN, projections, FFN) | 59.36 | 13.6 |
| local attention | 11.12 | 2.5 |
| model glue (residual adds, loops over layers) | 1.80 | 0.4 |
| decoder | 0.80 | 0.2 |
| loss | 0.34 | 0.1 |
| final LayerNorm | 0.28 | 0.1 |
| embedding | 0.27 | 0.1 |
| unattributed (outside the model call and the loss) | 0.08 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.03 | 0.2 | 0.4 |
| key conv | 2.96 | 0.7 | 1.2 |
| value projection | 2.41 | 0.5 | 0.9 |
| fp32 pin: casts, glue | 1.19 | 0.3 | 0.5 |
| address: no-grad matmul + top-k | 80.25 | 18.3 | 31.5 |
| address: recompute selected | 13.89 | 3.2 | 5.5 |
| write/read (rest of sparse_memory_read) | 141.28 | 32.3 | 55.5 |
| normaliser + gate | 4.11 | 0.9 | 1.6 |
| output projection | 4.86 | 1.1 | 1.9 |
| other (padding, residual) | 2.57 | 0.6 | 1.0 |
| **all heads** | 254.54 | 58.1 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.26 | 0.75 | 0.63 | 0.31 | 19.90 | 3.46 | 35.42 | 1.02 | 1.22 | 0.67 | 63.65 | 14.5 |
| head after layer 2 | 0.25 | 0.74 | 0.58 | 0.29 | 19.05 | 2.89 | 33.71 | 1.02 | 1.23 | 0.63 | 60.40 | 13.8 |
| head after layer 4 | 0.25 | 0.68 | 0.57 | 0.29 | 22.44 | 4.92 | 36.89 | 1.06 | 1.20 | 0.61 | 68.90 | 15.7 |
| head after layer 6 | 0.27 | 0.78 | 0.62 | 0.30 | 18.86 | 2.63 | 35.26 | 1.01 | 1.22 | 0.66 | 61.59 | 14.1 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 16.84 | 3.8 | 8.51 | 1.9 |
| 2 | 9.15 | 2.1 | 5.32 | 1.2 |
| 3 | 17.84 | 4.1 | 9.73 | 2.2 |
| 4 | 9.70 | 2.2 | 6.24 | 1.4 |
| 5 | 17.89 | 4.1 | 9.17 | 2.1 |
| 6 | 9.54 | 2.2 | 5.67 | 1.3 |
| 7 | 17.79 | 4.1 | 8.67 | 2.0 |
| 8 | 10.48 | 2.4 | 6.03 | 1.4 |

### Top operators by self device time, forward + backward, per step (all operators: 488.1 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `triton_per_fused__to_copy__unsafe_view_clamp_min_cumsum_div_exp_log_mul_rsub_sigmoid_view_3` | 8 | 65.51 | 13.4 |
| `aten::mm` | 102 | 64.46 | 13.2 |
| `triton_poi_fused__to_copy__unsafe_view_add_div_exp_flip_mul_neg_select_select_backward_sigmoid_squeeze_stack_unsqueeze_view_zeros_69` | 8 | 55.84 | 11.4 |
| `triton_per_fused_cumsum_exp_flip_mul_68` | 16 | 41.29 | 8.5 |
| `aten::bmm` | 112 | 37.96 | 7.8 |
| `triton_poi_fused_add_exp_mul_select_select_backward_squeeze_66` | 8 | 32.28 | 6.6 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_exp_flip_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoid_backward_view_where_70` | 8 | 31.48 | 6.4 |
| `aten::addmm` | 27 | 26.72 | 5.5 |
| `triton_per_fused__to_copy_exp_mul_sum_view_64` | 8 | 18.25 | 3.7 |
| `aten::topk` | 16 | 17.04 | 3.5 |
| `triton_poi_fused__to_copy_add_exp_mul_select_select_backward_squeeze_view_67` | 8 | 12.58 | 2.6 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_73` | 4 | 9.73 | 2.0 |
| `aten::_efficient_attention_backward` | 1 | 5.81 | 1.2 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_101` | 1 | 5.03 | 1.0 |
| `aten::_efficient_attention_forward` | 1 | 3.17 | 0.6 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_8` | 2 | 2.84 | 0.6 |
| `aten::convolution_backward` | 4 | 2.79 | 0.6 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_102` | 1 | 2.67 | 0.5 |
| `triton_poi_fused__to_copy__unsafe_view_add_arange_lift_fresh_mul_new_zeros_pow_rsub_slice_unsqueeze_view_28` | 28 | 2.61 | 0.5 |
| `triton_poi_fused_relu_sub_17` | 16 | 2.58 | 0.5 |
| `aten::_conv_depthwise2d` | 4 | 2.22 | 0.5 |
| `triton_poi_fused_mul_53` | 4 | 2.01 | 0.4 |
| `triton_poi_fused__to_copy_add_cat_div_mul_softplus_99` | 3 | 1.80 | 0.4 |
| `triton_poi_fused_add_exp_mul_select_stack_zeros_6` | 472 | 1.70 | 0.3 |
| `triton_per_fused__to_copy_add_native_layer_norm_backward_view_63` | 9 | 1.60 | 0.3 |

Kernel time in the profiled forward + backward: 488.1 ms; the split run's forward + backward took 478.9 ms, so the device was busy 102 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `triton_per_fused__to_copy__unsafe_view_clamp_min_cumsum_div_exp_log_mul_rsub_sigmoid_view_3` | 8 | 65.51 | 13.4 |
| `triton_poi_fused__to_copy__unsafe_view_add_div_exp_flip_mul_neg_select_select_backward_sigmoid_squeeze_stack_u` | 8 | 55.84 | 11.4 |
| `triton_per_fused_cumsum_exp_flip_mul_68` | 16 | 41.29 | 8.5 |
| `triton_poi_fused_add_exp_mul_select_select_backward_squeeze_66` | 8 | 32.28 | 6.6 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_exp_flip_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoi` | 8 | 31.48 | 6.4 |
| `ampere_sgemm_128x128_tn` | 16 | 30.74 | 6.3 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 23.26 | 4.8 |
| `triton_per_fused__to_copy_exp_mul_sum_view_64` | 8 | 18.25 | 3.7 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 15.58 | 3.2 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 15.05 | 3.1 |
| `triton_poi_fused__to_copy_add_exp_mul_select_select_backward_squeeze_view_67` | 8 | 12.58 | 2.6 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 11.67 | 2.4 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_73` | 4 | 9.73 | 2.0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 7.97 | 1.6 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_tn` | 16 | 7.44 | 1.5 |

### Memory

Peak allocated 4.70 GiB, peak reserved 4.79 GiB over the timed training steps; the forward leaves 4.06 GiB of activations.

