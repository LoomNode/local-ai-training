USDT:2026-10-04 09:35:16 3751466:3751466 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:35:16 3751466:3751466 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/zh_c_d.json
# Step profile

## Arm `mingru` (13,930,752 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused, --compile, --scan; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.2429 | 15 steps, one sync at the end, data draw included; final loss 5.238 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.0849 | forward + loss; median 0.0850, min 0.0805, max 0.0902 |
| split: backward | 0.1543 | zero_grad + backward; median 0.1553, min 0.1447, max 0.1598 |
| split: optimizer | 0.0022 | clip_grad_norm_ 0.0011 + optimizer step and schedule 0.0011 (median 0.0011, max 0.0012) |
| split: sum | 0.2417 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 36.3 | mean power 149 W (limit 150 W), max 60 C, contended False |

Warmup steps (s): 21.962, 0.302, 0.231, 0.219, 0.227 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 223.9 ms, the parts add up to 223.9 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 218.6 ms, so the instrumented one is 1.02 x it (the syncs remove the overlap of launches with work); the forward in the split run took 84.9 ms.

| part | ms | % of forward |
|:---|---:|---:|
| recurrence scan (all layers) | 126.79 | 56.6 |
| layers, rest (LN, projections, FFN) | 74.21 | 33.1 |
| local attention | 20.53 | 9.2 |
| decoder | 0.72 | 0.3 |
| model glue (residual adds, loops over layers) | 0.59 | 0.3 |
| final LayerNorm | 0.47 | 0.2 |
| loss | 0.31 | 0.1 |
| embedding | 0.26 | 0.1 |
| unattributed (outside the model call and the loss) | 0.04 | 0.0 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 15.08 | 6.7 | 8.66 | 3.9 |
| 2 | 15.76 | 7.0 | 9.18 | 4.1 |
| 3 | 16.28 | 7.3 | 9.54 | 4.3 |
| 4 | 15.75 | 7.0 | 9.11 | 4.1 |
| 5 | 15.37 | 6.9 | 9.58 | 4.3 |
| 6 | 16.68 | 7.4 | 9.72 | 4.3 |
| 7 | 16.12 | 7.2 | 9.32 | 4.2 |
| 8 | 15.75 | 7.0 | 9.09 | 4.1 |

### Top operators by self device time, forward + backward, per step (all operators: 235.5 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 78 | 107.20 | 45.5 |
| `aten::addmm` | 27 | 42.33 | 18.0 |
| `aten::_efficient_attention_backward` | 1 | 14.80 | 6.3 |
| `aten::_efficient_attention_forward` | 1 | 4.11 | 1.7 |
| `triton_poi_fused__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoid_backward_slice_slice_backward_view_where_22` | 8 | 3.70 | 1.6 |
| `triton_poi_fused_add_mul_slice_slice_backward_19` | 8 | 3.47 | 1.5 |
| `triton_poi_fused_add_mul_slice_slice_backward_16` | 8 | 3.20 | 1.4 |
| `triton_poi_fused__to_copy_add_mul_slice_slice_backward_14` | 8 | 3.00 | 1.3 |
| `triton_poi_fused__to_copy_add_mul_slice_slice_backward_13` | 8 | 2.79 | 1.2 |
| `triton_poi_fused_gelu_gelu_backward_view_8` | 8 | 2.73 | 1.2 |
| `triton_poi_fused_add_mul_slice_slice_backward_17` | 8 | 2.28 | 1.0 |
| `triton_poi_fused_add_mul_slice_slice_backward_20` | 8 | 2.25 | 1.0 |
| `triton_per_fused__to_copy_add_native_layer_norm_backward_view_11` | 8 | 2.03 | 0.9 |
| `triton_poi_fused_gelu_view_16` | 8 | 1.99 | 0.8 |
| `triton_per_fused__to_copy_add_native_layer_norm_backward_view_24` | 7 | 1.93 | 0.8 |
| `triton_poi_fused__to_copy_add_arange_cos_div_mul_neg_permute_pow_sin_slice_slice_backward_unsqueeze_view_27` | 1 | 1.79 | 0.8 |
| `triton_poi_fused_add_mul_slice_slice_backward_18` | 8 | 1.78 | 0.8 |
| `triton_poi_fused_add_mul_slice_slice_backward_21` | 8 | 1.78 | 0.8 |
| `triton_per_fused__to_copy_add_cat_mul_native_layer_norm_native_layer_norm_backward_slice_unsqueeze_view_20` | 5 | 1.75 | 0.7 |
| `triton_poi_fused_add_mul_slice_slice_backward_15` | 8 | 1.72 | 0.7 |
| `triton_poi_fused__unsafe_view_add_cat_clamp_min_mul_rsub_sigmoid_slice_view_3` | 8 | 1.67 | 0.7 |
| `triton_poi_fused__to_copy_add_arange_cat_cos_div_mul_neg_permute_pow_select_sin_slice_sub_unsqueeze_view_24` | 1 | 1.41 | 0.6 |
| `triton_poi_fused__to_copy_add_arange_cat_cos_div_mul_neg_permute_pow_select_sin_slice_sub_unsqueeze_view_23` | 1 | 1.41 | 0.6 |
| `triton_poi_fused__to_copy_add_mul_slice_slice_backward_unsqueeze_12` | 8 | 1.39 | 0.6 |
| `aten::sum` | 36 | 1.27 | 0.5 |

Kernel time in the profiled forward + backward: 235.5 ms; the split run's forward + backward took 239.2 ms, so the device was busy 98 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 34 | 41.39 | 17.6 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 24.16 | 10.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 17 | 21.48 | 9.1 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 18.16 | 7.7 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 15.36 | 6.5 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 15.34 | 6.5 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 14.80 | 6.3 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_tn` | 8 | 6.74 | 2.9 |
| `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass:` | 1 | 4.11 | 1.7 |
| `triton_poi_fused__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoid_backward_slice_slice` | 8 | 3.70 | 1.6 |
| `triton_poi_fused_add_mul_slice_slice_backward_19` | 8 | 3.47 | 1.5 |
| `triton_poi_fused_add_mul_slice_slice_backward_16` | 8 | 3.20 | 1.4 |
| `triton_poi_fused__to_copy_add_mul_slice_slice_backward_14` | 8 | 3.00 | 1.3 |
| `ampere_bf16_s1688gemm_bf16_128x256_ldg8_f2f_stages_32x1_nn` | 1 | 2.99 | 1.3 |
| `triton_poi_fused__to_copy_add_mul_slice_slice_backward_13` | 8 | 2.79 | 1.2 |

### Memory

Peak allocated 4.76 GiB, peak reserved 4.80 GiB over the timed training steps; the forward leaves 4.48 GiB of activations.

