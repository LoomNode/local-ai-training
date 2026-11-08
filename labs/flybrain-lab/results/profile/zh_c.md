USDT:2026-10-04 09:33:51 3749678:3749678 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:33:52 3749678:3749678 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/zh_c.json
# Step profile

## Arm `mingru` (13,930,752 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --compile; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.3150 | 15 steps, one sync at the end, data draw included; final loss 5.238 |
| split: data | 0.0002 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.0865 | forward + loss; median 0.0847, min 0.0811, max 0.0935 |
| split: backward | 0.2276 | zero_grad + backward; median 0.2279, min 0.2225, max 0.2337 |
| split: optimizer | 0.0030 | clip_grad_norm_ 0.0012 + optimizer step and schedule 0.0019 (median 0.0018, max 0.0020) |
| split: sum | 0.3173 | sync at every boundary: 1.01 x the plain step |
| energy per step (J) | 47.2 | mean power 150 W (limit 150 W), max 60 C, contended False |

Warmup steps (s): 40.699, 0.327, 0.285, 0.310, 0.342 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 137.7 ms, the parts add up to 137.7 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 138.5 ms, so the instrumented one is 0.99 x it (the syncs remove the overlap of launches with work); the forward in the split run took 86.5 ms.

| part | ms | % of forward |
|:---|---:|---:|
| recurrence scan (all layers) | 79.12 | 57.5 |
| layers, rest (LN, projections, FFN) | 44.12 | 32.0 |
| local attention | 12.94 | 9.4 |
| decoder | 0.42 | 0.3 |
| model glue (residual adds, loops over layers) | 0.41 | 0.3 |
| final LayerNorm | 0.26 | 0.2 |
| embedding | 0.20 | 0.1 |
| loss | 0.19 | 0.1 |
| unattributed (outside the model call and the loss) | 0.04 | 0.0 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 9.53 | 6.9 | 5.25 | 3.8 |
| 2 | 9.06 | 6.6 | 4.89 | 3.6 |
| 3 | 8.34 | 6.1 | 4.68 | 3.4 |
| 4 | 8.42 | 6.1 | 5.29 | 3.8 |
| 5 | 12.82 | 9.3 | 7.23 | 5.3 |
| 6 | 12.46 | 9.0 | 6.44 | 4.7 |
| 7 | 9.35 | 6.8 | 5.20 | 3.8 |
| 8 | 9.13 | 6.6 | 5.13 | 3.7 |

### Top operators by self device time, forward + backward, per step (all operators: 313.0 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 78 | 52.47 | 16.8 |
| `triton_poi_fused__to_copy__unsafe_view_add_div_exp_flip_mul_neg_select_select_backward_sigmoid_squeeze_stack_unsqueeze_view_zeros_17` | 8 | 49.51 | 15.8 |
| `triton_per_fused__to_copy__unsafe_view_clamp_min_cumsum_div_exp_log_mul_rsub_sigmoid_view_3` | 8 | 39.36 | 12.6 |
| `triton_per_fused_cumsum_exp_flip_mul_16` | 16 | 36.47 | 11.7 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_exp_flip_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoid_backward_view_where_18` | 8 | 29.47 | 9.4 |
| `triton_poi_fused_add_exp_mul_select_select_backward_squeeze_14` | 8 | 25.68 | 8.2 |
| `aten::addmm` | 27 | 15.26 | 4.9 |
| `triton_per_fused__to_copy_exp_mul_sum_view_12` | 8 | 14.97 | 4.8 |
| `triton_poi_fused__to_copy_add_exp_mul_select_select_backward_squeeze_view_15` | 8 | 10.46 | 3.3 |
| `aten::_efficient_attention_backward` | 1 | 7.06 | 2.3 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_24` | 4 | 5.88 | 1.9 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_8` | 4 | 4.97 | 1.6 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_13` | 3 | 3.63 | 1.2 |
| `triton_per_fused__to_copy_add_native_layer_norm_backward_view_11` | 9 | 1.44 | 0.5 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_12` | 1 | 1.37 | 0.4 |
| `aten::_efficient_attention_forward` | 1 | 1.36 | 0.4 |
| `triton_poi_fused_gelu_gelu_backward_view_8` | 8 | 1.34 | 0.4 |
| `triton_per_fused__to_copy_add_native_layer_norm_backward_view_20` | 7 | 1.32 | 0.4 |
| `triton_poi_fused_add_exp_mul_select_squeeze_13` | 8 | 1.11 | 0.4 |
| `triton_poi_fused_add_exp_mul_select_stack_zeros_6` | 472 | 1.01 | 0.3 |
| `triton_poi_fused_add_exp_mul_select_stack_zeros_4` | 8 | 0.86 | 0.3 |
| `triton_poi_fused__to_copy_add_arange_cos_div_mul_neg_permute_pow_sin_slice_slice_backward_unsqueeze_view_23` | 1 | 0.82 | 0.3 |
| `triton_poi_fused_gelu_view_11` | 8 | 0.75 | 0.2 |
| `aten::sum` | 36 | 0.70 | 0.2 |
| `triton_poi_fused__to_copy_view_4` | 9 | 0.57 | 0.2 |

Kernel time in the profiled forward + backward: 313.0 ms; the split run's forward + backward took 314.1 ms, so the device was busy 100 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `triton_poi_fused__to_copy__unsafe_view_add_div_exp_flip_mul_neg_select_select_backward_sigmoid_squeeze_stack_u` | 8 | 49.51 | 15.8 |
| `triton_per_fused__to_copy__unsafe_view_clamp_min_cumsum_div_exp_log_mul_rsub_sigmoid_view_3` | 8 | 39.36 | 12.6 |
| `triton_per_fused_cumsum_exp_flip_mul_16` | 16 | 36.47 | 11.7 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_div_exp_flip_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoi` | 8 | 29.47 | 9.4 |
| `triton_poi_fused_add_exp_mul_select_select_backward_squeeze_14` | 8 | 25.68 | 8.2 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 34 | 20.39 | 6.5 |
| `triton_per_fused__to_copy_exp_mul_sum_view_12` | 8 | 14.97 | 4.8 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 17 | 11.20 | 3.6 |
| `triton_poi_fused__to_copy_add_exp_mul_select_select_backward_squeeze_view_15` | 8 | 10.46 | 3.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 8.91 | 2.8 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 7.52 | 2.4 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 7.51 | 2.4 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 7.06 | 2.3 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 6.36 | 2.0 |
| `triton_per_fused__to_copy_add_exp_mul_native_layer_norm_native_layer_norm_backward_stack_unsqueeze_view_24` | 4 | 5.88 | 1.9 |

### Memory

Peak allocated 3.31 GiB, peak reserved 3.49 GiB over the timed training steps; the forward leaves 3.03 GiB of activations.

