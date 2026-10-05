USDT:2026-10-04 09:18:55 3729294:3729294 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:18:57 3729294:3729294 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_c.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --compile; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.5222 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0002 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.2115 | forward + loss; median 0.2107, min 0.2074, max 0.2177 |
| split: backward | 0.3093 | zero_grad + backward; median 0.3096, min 0.3057, max 0.3142 |
| split: optimizer | 0.0036 | clip_grad_norm_ 0.0016 + optimizer step and schedule 0.0020 (median 0.0020, max 0.0021) |
| split: sum | 0.5246 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 78.3 | mean power 150 W (limit 150 W), max 58 C, contended False |

Warmup steps (s): 82.373, 0.523, 0.532, 0.522, 0.518 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 446.5 ms, the parts add up to 446.5 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 442.3 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 211.5 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 260.28 | 58.3 |
| recurrence scan (all layers) | 111.70 | 25.0 |
| layers, rest (LN, projections, FFN) | 59.54 | 13.3 |
| local attention | 11.42 | 2.6 |
| model glue (residual adds, loops over layers) | 1.81 | 0.4 |
| decoder | 0.82 | 0.2 |
| loss | 0.35 | 0.1 |
| final LayerNorm | 0.27 | 0.1 |
| embedding | 0.26 | 0.1 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.02 | 0.2 | 0.4 |
| key conv | 2.96 | 0.7 | 1.1 |
| value projection | 2.41 | 0.5 | 0.9 |
| fp32 pin: casts, glue | 1.19 | 0.3 | 0.5 |
| address: no-grad matmul + top-k | 83.26 | 18.6 | 32.0 |
| address: recompute selected | 14.23 | 3.2 | 5.5 |
| write/read (rest of sparse_memory_read) | 143.54 | 32.1 | 55.1 |
| normaliser + gate | 4.17 | 0.9 | 1.6 |
| output projection | 4.95 | 1.1 | 1.9 |
| other (padding, residual) | 2.55 | 0.6 | 1.0 |
| **all heads** | 260.28 | 58.3 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.26 | 0.76 | 0.64 | 0.31 | 20.13 | 2.71 | 34.83 | 1.06 | 1.25 | 0.67 | 62.64 | 14.0 |
| head after layer 2 | 0.24 | 0.72 | 0.57 | 0.28 | 21.27 | 3.53 | 35.10 | 1.02 | 1.24 | 0.60 | 64.58 | 14.5 |
| head after layer 4 | 0.25 | 0.69 | 0.58 | 0.29 | 22.85 | 4.99 | 37.71 | 1.06 | 1.23 | 0.61 | 70.25 | 15.7 |
| head after layer 6 | 0.27 | 0.79 | 0.62 | 0.30 | 19.02 | 2.99 | 35.90 | 1.02 | 1.24 | 0.66 | 62.81 | 14.1 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 16.90 | 3.8 | 7.83 | 1.8 |
| 2 | 9.32 | 2.1 | 5.04 | 1.1 |
| 3 | 17.98 | 4.0 | 9.56 | 2.1 |
| 4 | 10.66 | 2.4 | 6.44 | 1.4 |
| 5 | 18.19 | 4.1 | 9.23 | 2.1 |
| 6 | 9.74 | 2.2 | 5.82 | 1.3 |
| 7 | 17.85 | 4.0 | 9.28 | 2.1 |
| 8 | 11.06 | 2.5 | 6.34 | 1.4 |

### Top operators by self device time, forward + backward, per step (all operators: 522.4 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 102 | 77.82 | 14.9 |
| `triton_per_fused__to_copy__unsafe_view_clamp_min_cumsum_div_exp_log_mul_rsub_sigmoid_view_3` | 8 | 71.37 | 13.7 |
| `triton_poi_fused__to_copy__unsafe_view_add_div_exp_flip_mul_neg_select_select_backward_sigmoid_squeeze_stack_unsqueeze_view_zeros_101` | 8 | 59.27 | 11.3 |
| `triton_per_fused_cumsum_exp_flip_mul_100` | 16 | 42.10 | 8.1 |
| `aten::bmm` | 112 | 37.52 | 7.2 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_exp_flip_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoid_backward_view_where_102` | 8 | 32.18 | 6.2 |
| `triton_poi_fused_add_exp_mul_select_select_backward_squeeze_98` | 8 | 31.83 | 6.1 |
| `aten::addmm` | 27 | 25.70 | 4.9 |
| `triton_per_fused__to_copy_exp_mul_sum_view_96` | 8 | 18.81 | 3.6 |
| `aten::topk` | 16 | 17.29 | 3.3 |
| `triton_poi_fused__to_copy_add_exp_mul_select_select_backward_squeeze_view_99` | 8 | 13.24 | 2.5 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_73` | 4 | 9.79 | 1.9 |
| `aten::_efficient_attention_backward` | 1 | 6.61 | 1.3 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_101` | 1 | 4.44 | 0.9 |
| `aten::convolution_backward` | 4 | 3.02 | 0.6 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_102` | 1 | 3.01 | 0.6 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_8` | 2 | 2.95 | 0.6 |
| `triton_poi_fused_new_zeros_23` | 29 | 2.81 | 0.5 |
| `aten::_efficient_attention_forward` | 1 | 2.70 | 0.5 |
| `triton_poi_fused_relu_sub_17` | 16 | 2.58 | 0.5 |
| `triton_poi_fused_mul_54` | 4 | 2.10 | 0.4 |
| `triton_poi_fused__to_copy_add_cat_div_mul_softplus_99` | 3 | 2.02 | 0.4 |
| `triton_poi_fused_mul_50` | 4 | 1.96 | 0.4 |
| `aten::_conv_depthwise2d` | 4 | 1.91 | 0.4 |
| `triton_per_fused__to_copy_add_native_layer_norm_backward_view_95` | 9 | 1.78 | 0.3 |

Kernel time in the profiled forward + backward: 522.4 ms; the split run's forward + backward took 520.7 ms, so the device was busy 100 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `triton_per_fused__to_copy__unsafe_view_clamp_min_cumsum_div_exp_log_mul_rsub_sigmoid_view_3` | 8 | 71.37 | 13.7 |
| `triton_poi_fused__to_copy__unsafe_view_add_div_exp_flip_mul_neg_select_select_backward_sigmoid_squeeze_stack_u` | 8 | 59.27 | 11.3 |
| `triton_per_fused_cumsum_exp_flip_mul_100` | 16 | 42.10 | 8.1 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_exp_flip_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoi` | 8 | 32.18 | 6.2 |
| `triton_poi_fused_add_exp_mul_select_select_backward_squeeze_98` | 8 | 31.83 | 6.1 |
| `ampere_sgemm_128x128_tn` | 16 | 30.25 | 5.8 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 29.22 | 5.6 |
| `triton_per_fused__to_copy_exp_mul_sum_view_96` | 8 | 18.81 | 3.6 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 18.02 | 3.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 14.94 | 2.9 |
| `triton_poi_fused__to_copy_add_exp_mul_select_select_backward_squeeze_view_99` | 8 | 13.24 | 2.5 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 10.76 | 2.1 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 9.98 | 1.9 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_73` | 4 | 9.79 | 1.9 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 9.27 | 1.8 |

### Memory

Peak allocated 4.82 GiB, peak reserved 5.03 GiB over the timed training steps; the forward leaves 4.12 GiB of activations.

