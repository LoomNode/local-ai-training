USDT:2026-10-04 09:34:35 3750730:3750730 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:34:36 3750730:3750730 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/zh_c_d32.json
# Step profile

## Arm `mingru` (13,930,752 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused, --compile, --scan; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.3199 | 15 steps, one sync at the end, data draw included; final loss 5.238 |
| split: data | 0.0002 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.1120 | forward + loss; median 0.1117, min 0.1094, max 0.1145 |
| split: backward | 0.2069 | zero_grad + backward; median 0.2071, min 0.2012, max 0.2112 |
| split: optimizer | 0.0022 | clip_grad_norm_ 0.0012 + optimizer step and schedule 0.0011 (median 0.0011, max 0.0012) |
| split: sum | 0.3214 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 48.0 | mean power 150 W (limit 150 W), max 60 C, contended False |

Warmup steps (s): 21.548, 0.331, 0.316, 0.306, 0.332 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 244.2 ms, the parts add up to 244.2 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 241.8 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 112.0 ms.

| part | ms | % of forward |
|:---|---:|---:|
| recurrence scan (all layers) | 139.28 | 57.0 |
| layers, rest (LN, projections, FFN) | 79.21 | 32.4 |
| local attention | 23.12 | 9.5 |
| decoder | 0.82 | 0.3 |
| model glue (residual adds, loops over layers) | 0.63 | 0.3 |
| final LayerNorm | 0.53 | 0.2 |
| loss | 0.35 | 0.1 |
| embedding | 0.28 | 0.1 |
| unattributed (outside the model call and the loss) | 0.04 | 0.0 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 17.44 | 7.1 | 9.93 | 4.1 |
| 2 | 17.40 | 7.1 | 9.92 | 4.1 |
| 3 | 17.40 | 7.1 | 9.89 | 4.0 |
| 4 | 17.42 | 7.1 | 9.89 | 4.0 |
| 5 | 17.40 | 7.1 | 9.90 | 4.1 |
| 6 | 17.40 | 7.1 | 9.90 | 4.1 |
| 7 | 17.41 | 7.1 | 9.89 | 4.0 |
| 8 | 17.40 | 7.1 | 9.89 | 4.1 |

### Top operators by self device time, forward + backward, per step (all operators: 323.0 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 78 | 126.58 | 39.2 |
| `aten::addmm` | 27 | 52.92 | 16.4 |
| `aten::_efficient_attention_backward` | 1 | 18.34 | 5.7 |
| `triton_poi_fused_add_mul_select_backward_slice_19` | 8 | 7.76 | 2.4 |
| `triton_poi_fused_add_mul_select_backward_slice_16` | 8 | 7.37 | 2.3 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_select_backward_sigmoid_sigmoid_backward_slice_view_where_22` | 8 | 6.80 | 2.1 |
| `aten::_efficient_attention_forward` | 1 | 5.75 | 1.8 |
| `triton_poi_fused_add_mul_select_backward_slice_14` | 8 | 5.15 | 1.6 |
| `triton_poi_fused_add_mul_select_backward_slice_17` | 8 | 4.96 | 1.5 |
| `triton_poi_fused_add_mul_select_backward_slice_20` | 8 | 4.90 | 1.5 |
| `triton_poi_fused_add_mul_select_backward_slice_13` | 8 | 4.64 | 1.4 |
| `triton_poi_fused_add_mul_select_backward_slice_18` | 8 | 4.13 | 1.3 |
| `triton_poi_fused_add_mul_select_backward_slice_21` | 8 | 4.12 | 1.3 |
| `triton_poi_fused_add_mul_select_backward_slice_15` | 8 | 3.92 | 1.2 |
| `triton_per_fused__to_copy_add_cat_mul_native_layer_norm_native_layer_norm_backward_slice_unsqueeze_view_18` | 6 | 3.67 | 1.1 |
| `triton_poi_fused_gelu_gelu_backward_view_8` | 8 | 3.19 | 1.0 |
| `triton_poi_fused_add_cat_mul_slice_6` | 8 | 3.01 | 0.9 |
| `triton_poi_fused_add_cat_mul_slice_10` | 8 | 2.97 | 0.9 |
| `triton_poi_fused_add_cat_mul_slice_11` | 8 | 2.95 | 0.9 |
| `triton_poi_fused_add_cat_mul_slice_9` | 8 | 2.94 | 0.9 |
| `triton_poi_fused_add_cat_mul_slice_8` | 8 | 2.94 | 0.9 |
| `triton_poi_fused_add_cat_mul_slice_7` | 8 | 2.91 | 0.9 |
| `triton_poi_fused_add_cat_mul_slice_12` | 8 | 2.91 | 0.9 |
| `triton_poi_fused_add_cat_mul_slice_4` | 8 | 2.91 | 0.9 |
| `triton_poi_fused_add_cat_mul_slice_5` | 8 | 2.89 | 0.9 |

Kernel time in the profiled forward + backward: 323.0 ms; the split run's forward + backward took 318.9 ms, so the device was busy 101 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 34 | 48.52 | 15.0 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 30.17 | 9.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 17 | 25.62 | 7.9 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 22.75 | 7.0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 18.35 | 5.7 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 18.34 | 5.7 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 17.96 | 5.6 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_tn` | 8 | 8.46 | 2.6 |
| `triton_poi_fused_add_mul_select_backward_slice_19` | 8 | 7.76 | 2.4 |
| `triton_poi_fused_add_mul_select_backward_slice_16` | 8 | 7.37 | 2.3 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_select_backward_sigmoid_sig` | 8 | 6.80 | 2.1 |
| `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass:` | 1 | 5.75 | 1.8 |
| `triton_poi_fused_add_mul_select_backward_slice_14` | 8 | 5.15 | 1.6 |
| `triton_poi_fused_add_mul_select_backward_slice_17` | 8 | 4.96 | 1.5 |
| `triton_poi_fused_add_mul_select_backward_slice_20` | 8 | 4.90 | 1.5 |

### Memory

Peak allocated 7.28 GiB, peak reserved 7.31 GiB over the timed training steps; the forward leaves 7.00 GiB of activations.

