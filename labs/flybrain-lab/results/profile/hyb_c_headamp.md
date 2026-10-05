USDT:2026-10-04 09:23:47 3736965:3736965 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:23:48 3736965:3736965 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_c_headamp.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --compile, --head-amp; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.4522 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0002 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.1662 | forward + loss; median 0.1653, min 0.1619, max 0.1727 |
| split: backward | 0.2900 | zero_grad + backward; median 0.2882, min 0.2821, max 0.2988 |
| split: optimizer | 0.0036 | clip_grad_norm_ 0.0016 + optimizer step and schedule 0.0020 (median 0.0020, max 0.0021) |
| split: sum | 0.4601 | sync at every boundary: 1.02 x the plain step |
| energy per step (J) | 67.6 | mean power 149 W (limit 150 W), max 56 C, contended False |

Warmup steps (s): 75.933, 0.425, 0.475, 0.427, 0.460 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 388.3 ms, the parts add up to 388.2 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 381.1 ms, so the instrumented one is 1.02 x it (the syncs remove the overlap of launches with work); the forward in the split run took 166.2 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 206.86 | 53.3 |
| recurrence scan (all layers) | 110.68 | 28.5 |
| layers, rest (LN, projections, FFN) | 56.24 | 14.5 |
| local attention | 10.82 | 2.8 |
| model glue (residual adds, loops over layers) | 1.88 | 0.5 |
| decoder | 0.82 | 0.2 |
| loss | 0.34 | 0.1 |
| final LayerNorm | 0.28 | 0.1 |
| embedding | 0.28 | 0.1 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.03 | 0.3 | 0.5 |
| key conv | 2.96 | 0.8 | 1.4 |
| value projection | 2.42 | 0.6 | 1.2 |
| fp32 pin: casts, glue | 0.06 | 0.0 | 0.0 |
| address: no-grad matmul + top-k | 52.31 | 13.5 | 25.3 |
| address: recompute selected | 8.84 | 2.3 | 4.3 |
| write/read (rest of sparse_memory_read) | 127.89 | 32.9 | 61.8 |
| normaliser + gate | 3.80 | 1.0 | 1.8 |
| output projection | 4.96 | 1.3 | 2.4 |
| other (padding, residual) | 2.59 | 0.7 | 1.3 |
| **all heads** | 206.86 | 53.3 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.26 | 0.76 | 0.64 | 0.01 | 13.13 | 2.02 | 31.58 | 0.94 | 1.24 | 0.68 | 51.25 | 13.2 |
| head after layer 2 | 0.24 | 0.72 | 0.57 | 0.01 | 12.07 | 1.88 | 30.82 | 0.91 | 1.24 | 0.62 | 49.10 | 12.6 |
| head after layer 4 | 0.27 | 0.72 | 0.61 | 0.02 | 15.30 | 3.16 | 37.22 | 1.08 | 1.24 | 0.65 | 60.26 | 15.5 |
| head after layer 6 | 0.26 | 0.76 | 0.60 | 0.02 | 11.81 | 1.77 | 28.28 | 0.86 | 1.24 | 0.65 | 46.25 | 11.9 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 16.93 | 4.4 | 7.73 | 2.0 |
| 2 | 9.15 | 2.4 | 5.21 | 1.3 |
| 3 | 18.13 | 4.7 | 8.91 | 2.3 |
| 4 | 10.61 | 2.7 | 5.56 | 1.4 |
| 5 | 17.95 | 4.6 | 8.70 | 2.2 |
| 6 | 9.66 | 2.5 | 5.49 | 1.4 |
| 7 | 17.83 | 4.6 | 8.79 | 2.3 |
| 8 | 10.41 | 2.7 | 5.85 | 1.5 |

### Top operators by self device time, forward + backward, per step (all operators: 459.2 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 102 | 67.72 | 14.7 |
| `triton_poi_fused__to_copy__unsafe_view_add_div_exp_flip_mul_neg_select_select_backward_sigmoid_squeeze_stack_unsqueeze_view_zeros_93` | 8 | 59.58 | 13.0 |
| `triton_per_fused__to_copy__unsafe_view_clamp_min_cumsum_div_exp_log_mul_rsub_sigmoid_view_3` | 8 | 51.68 | 11.3 |
| `triton_per_fused_cumsum_exp_flip_mul_92` | 16 | 40.72 | 8.9 |
| `triton_poi_fused_add_exp_mul_select_select_backward_squeeze_90` | 8 | 30.07 | 6.5 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_exp_flip_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoid_backward_view_where_94` | 8 | 27.80 | 6.1 |
| `aten::addmm` | 27 | 25.70 | 5.6 |
| `aten::topk` | 16 | 18.75 | 4.1 |
| `triton_per_fused__to_copy_exp_mul_sum_view_88` | 8 | 18.05 | 3.9 |
| `aten::bmm` | 112 | 16.80 | 3.7 |
| `triton_poi_fused__to_copy_add_exp_mul_select_select_backward_squeeze_view_91` | 8 | 12.17 | 2.6 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_73` | 4 | 11.55 | 2.5 |
| `aten::_efficient_attention_backward` | 1 | 7.13 | 1.6 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_102` | 1 | 4.00 | 0.9 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_8` | 2 | 2.86 | 0.6 |
| `aten::_conv_depthwise2d` | 4 | 2.53 | 0.6 |
| `triton_poi_fused_new_zeros_24` | 29 | 2.44 | 0.5 |
| `aten::convolution_backward` | 4 | 2.41 | 0.5 |
| `aten::_efficient_attention_forward` | 1 | 2.21 | 0.5 |
| `triton_poi_fused_relu_sub_18` | 16 | 2.09 | 0.5 |
| `triton_poi_fused_mul_54` | 4 | 1.86 | 0.4 |
| `triton_poi_fused_mul_43` | 4 | 1.86 | 0.4 |
| `triton_poi_fused__to_copy_add_arange_cat_cos_div_mul_neg_permute_pow_select_sin_slice_sub_unsqueeze_view_76` | 1 | 1.85 | 0.4 |
| `triton_per_fused__to_copy_add_native_layer_norm_backward_view_87` | 9 | 1.66 | 0.4 |
| `triton_poi_fused__to_copy_add_arange_div_expand_index_add_lift_fresh_mul_pow_slice_softplus_unsqueeze_view_18` | 4 | 1.63 | 0.4 |

Kernel time in the profiled forward + backward: 459.2 ms; the split run's forward + backward took 456.2 ms, so the device was busy 101 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `triton_poi_fused__to_copy__unsafe_view_add_div_exp_flip_mul_neg_select_select_backward_sigmoid_squeeze_stack_u` | 8 | 59.58 | 13.0 |
| `triton_per_fused__to_copy__unsafe_view_clamp_min_cumsum_div_exp_log_mul_rsub_sigmoid_view_3` | 8 | 51.68 | 11.3 |
| `triton_per_fused_cumsum_exp_flip_mul_92` | 16 | 40.72 | 8.9 |
| `triton_poi_fused_add_exp_mul_select_select_backward_squeeze_90` | 8 | 30.07 | 6.5 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_exp_flip_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoi` | 8 | 27.80 | 6.1 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 24.86 | 5.4 |
| `triton_per_fused__to_copy_exp_mul_sum_view_88` | 8 | 18.05 | 3.9 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 15.63 | 3.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 13.72 | 3.0 |
| `triton_poi_fused__to_copy_add_exp_mul_select_select_backward_squeeze_view_91` | 8 | 12.17 | 2.6 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 11.97 | 2.6 |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_nt` | 16 | 11.69 | 2.5 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_73` | 4 | 11.55 | 2.5 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 9.28 | 2.0 |
| `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::Tenso` | 64 | 8.12 | 1.8 |

### Memory

Peak allocated 4.80 GiB, peak reserved 5.02 GiB over the timed training steps; the forward leaves 4.11 GiB of activations.

