USDT:2026-10-04 09:14:32 3724810:3724810 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:14:33 3724810:3724810 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/tf_compile.json
# Step profile

## Arm `transformer` (17,085,696 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused, --compile; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.2759 | 15 steps, one sync at the end, data draw included; final loss 5.285 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.0955 | forward + loss; median 0.0957, min 0.0893, max 0.1001 |
| split: backward | 0.1766 | zero_grad + backward; median 0.1772, min 0.1674, max 0.1860 |
| split: optimizer | 0.0021 | clip_grad_norm_ 0.0011 + optimizer step and schedule 0.0010 (median 0.0010, max 0.0010) |
| split: sum | 0.2744 | sync at every boundary: 0.99 x the plain step |
| energy per step (J) | 41.2 | mean power 149 W (limit 150 W), max 60 C, contended False |

Warmup steps (s): 10.179, 0.370, 0.243, 0.241, 0.283 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 116.9 ms, the parts add up to 116.8 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 113.6 ms, so the instrumented one is 1.03 x it (the syncs remove the overlap of launches with work); the forward in the split run took 95.5 ms.

| part | ms | % of forward |
|:---|---:|---:|
| transformer attention | 76.78 | 65.7 |
| transformer FFN | 28.75 | 24.6 |
| transformer LayerNorm | 5.75 | 4.9 |
| model glue (residual adds, loops over layers) | 4.17 | 3.6 |
| decoder | 0.54 | 0.5 |
| final LayerNorm | 0.35 | 0.3 |
| loss | 0.24 | 0.2 |
| embedding | 0.23 | 0.2 |
| unattributed (outside the model call and the loss) | 0.04 | 0.0 |

Layers, per forward:

| layer | attention ms | attention % | FFN ms | FFN % | LayerNorm ms |
|:---|---:|---:|---:|---:|---:|
| 1 | 9.68 | 8.3 | 3.65 | 3.1 | 0.75 |
| 2 | 9.56 | 8.2 | 3.54 | 3.0 | 0.72 |
| 3 | 9.26 | 7.9 | 3.51 | 3.0 | 0.70 |
| 4 | 9.47 | 8.1 | 3.52 | 3.0 | 0.71 |
| 5 | 9.56 | 8.2 | 3.59 | 3.1 | 0.71 |
| 6 | 9.92 | 8.5 | 3.74 | 3.2 | 0.73 |
| 7 | 9.81 | 8.4 | 3.69 | 3.2 | 0.73 |
| 8 | 9.51 | 8.1 | 3.51 | 3.0 | 0.70 |

### Top operators by self device time, forward + backward, per step (all operators: 268.3 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 66 | 89.05 | 33.2 |
| `aten::_flash_attention_backward` | 8 | 56.61 | 21.1 |
| `aten::addmm` | 33 | 44.08 | 16.4 |
| `aten::_flash_attention_forward` | 8 | 21.84 | 8.1 |
| `triton_poi_fused__to_copy_add_arange_cos_div_mul_neg_pow_sin_slice_slice_backward_unsqueeze_13` | 16 | 19.13 | 7.1 |
| `triton_poi_fused__to_copy_add_arange_cat_cos_div_mul_neg_permute_pow_select_sin_slice_sub_unsqueeze_view_3` | 8 | 10.07 | 3.8 |
| `triton_poi_fused__to_copy_add_arange_cat_cos_div_mul_neg_permute_pow_select_sin_slice_sub_unsqueeze_view_4` | 8 | 9.20 | 3.4 |
| `triton_per_fused__to_copy_add_native_layer_norm_backward_view_11` | 14 | 2.68 | 1.0 |
| `triton_poi_fused_gelu_gelu_backward_view_8` | 8 | 2.01 | 0.7 |
| `triton_poi_fused_add_clone_permute_select_backward_14` | 8 | 1.47 | 0.5 |
| `triton_poi_fused_gelu_view_11` | 8 | 1.44 | 0.5 |
| `aten::copy_` | 13 | 1.42 | 0.5 |
| `triton_poi_fused__to_copy_view_4` | 15 | 0.95 | 0.4 |
| `aten::sum` | 34 | 0.92 | 0.3 |
| `triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_13` | 4 | 0.77 | 0.3 |
| `triton_red_fused__unsafe_view_add_clone_permute_select_backward_sum_view_15` | 8 | 0.72 | 0.3 |
| `triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_12` | 4 | 0.66 | 0.2 |
| `triton_red_fused_sum_5` | 16 | 0.55 | 0.2 |
| `triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_16` | 2 | 0.55 | 0.2 |
| `triton_red_fused_gelu_gelu_backward_sum_view_9` | 8 | 0.51 | 0.2 |
| `triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_15` | 3 | 0.45 | 0.2 |
| `triton_poi_fused_clone_transpose_5` | 8 | 0.42 | 0.2 |
| `triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_17` | 1 | 0.26 | 0.1 |
| `triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_view_14` | 1 | 0.25 | 0.1 |
| `triton_poi_fused__to_copy_7` | 32 | 0.23 | 0.1 |

Kernel time in the profiled forward + backward: 268.3 ms; the split run's forward + backward took 272.1 ms, so the device was busy 99 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<32, 128, 128, 8, 4, 4` | 8 | 55.06 | 20.5 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 16 | 27.01 | 10.1 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 25 | 25.06 | 9.3 |
| `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<32, 128, 128, 4, false, false, cutlass::bfloat16_` | 8 | 21.84 | 8.1 |
| `triton_poi_fused__to_copy_add_arange_cos_div_mul_neg_pow_sin_slice_slice_backward_unsqueeze_13` | 16 | 19.13 | 7.1 |
| `ampere_bf16_s1688gemm_bf16_128x256_ldg8_f2f_stages_32x1_nn` | 8 | 17.35 | 6.5 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 17 | 17.08 | 6.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_nt_align8>(cutlass_80_tensoro` | 8 | 15.19 | 5.7 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 11.49 | 4.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 11.32 | 4.2 |
| `triton_poi_fused__to_copy_add_arange_cat_cos_div_mul_neg_permute_pow_select_sin_slice_sub_unsqueeze_view_3` | 8 | 10.07 | 3.8 |
| `triton_poi_fused__to_copy_add_arange_cat_cos_div_mul_neg_permute_pow_select_sin_slice_sub_unsqueeze_view_4` | 8 | 9.20 | 3.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 8 | 7.74 | 2.9 |
| `triton_per_fused__to_copy_add_native_layer_norm_backward_view_11` | 14 | 2.68 | 1.0 |
| `triton_poi_fused_gelu_gelu_backward_view_8` | 8 | 2.01 | 0.7 |

### Memory

Peak allocated 2.57 GiB, peak reserved 2.61 GiB over the timed training steps; the forward leaves 2.21 GiB of activations.

