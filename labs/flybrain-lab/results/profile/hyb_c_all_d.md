USDT:2026-10-04 09:28:53 3743438:3743438 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:28:54 3743438:3743438 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_c_all_d.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused, --compile, --head-amp, --scan, --sparse-claws; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.4003 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.1881 | forward + loss; median 0.1873, min 0.1844, max 0.1930 |
| split: backward | 0.2113 | zero_grad + backward; median 0.2119, min 0.2070, max 0.2156 |
| split: optimizer | 0.0025 | clip_grad_norm_ 0.0014 + optimizer step and schedule 0.0011 (median 0.0011, max 0.0012) |
| split: sum | 0.4022 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 59.9 | mean power 150 W (limit 150 W), max 57 C, contended False |

Warmup steps (s): 55.425, 0.426, 0.365, 0.426, 0.387 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 466.1 ms, the parts add up to 466.0 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 461.0 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 188.1 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 235.36 | 50.5 |
| recurrence scan (all layers) | 131.45 | 28.2 |
| layers, rest (LN, projections, FFN) | 75.72 | 16.2 |
| local attention | 19.82 | 4.3 |
| model glue (residual adds, loops over layers) | 1.87 | 0.4 |
| decoder | 0.74 | 0.2 |
| final LayerNorm | 0.47 | 0.1 |
| loss | 0.31 | 0.1 |
| embedding | 0.26 | 0.1 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.81 | 0.4 | 0.8 |
| key conv | 5.06 | 1.1 | 2.1 |
| value projection | 4.21 | 0.9 | 1.8 |
| fp32 pin: casts, glue | 0.05 | 0.0 | 0.0 |
| address: no-grad matmul + top-k | 85.45 | 18.3 | 36.3 |
| address: recompute selected | 2.75 | 0.6 | 1.2 |
| write/read (rest of sparse_memory_read) | 123.90 | 26.6 | 52.6 |
| normaliser + gate | 3.69 | 0.8 | 1.6 |
| output projection | 4.39 | 0.9 | 1.9 |
| other (padding, residual) | 4.06 | 0.9 | 1.7 |
| **all heads** | 235.36 | 50.5 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.45 | 1.28 | 1.10 | 0.01 | 21.72 | 0.70 | 31.31 | 0.90 | 1.08 | 1.05 | 59.60 | 12.8 |
| head after layer 2 | 0.45 | 1.29 | 1.04 | 0.01 | 22.04 | 0.66 | 30.46 | 0.93 | 1.07 | 1.00 | 58.97 | 12.7 |
| head after layer 4 | 0.45 | 1.16 | 0.98 | 0.01 | 21.11 | 0.73 | 32.52 | 0.96 | 1.19 | 0.98 | 60.08 | 12.9 |
| head after layer 6 | 0.46 | 1.33 | 1.08 | 0.01 | 20.58 | 0.66 | 29.61 | 0.89 | 1.05 | 1.03 | 56.70 | 12.2 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 16.22 | 3.5 | 9.30 | 2.0 |
| 2 | 14.86 | 3.2 | 8.72 | 1.9 |
| 3 | 16.95 | 3.6 | 9.72 | 2.1 |
| 4 | 16.48 | 3.5 | 9.38 | 2.0 |
| 5 | 16.99 | 3.6 | 10.01 | 2.1 |
| 6 | 16.54 | 3.5 | 9.31 | 2.0 |
| 7 | 17.03 | 3.7 | 9.81 | 2.1 |
| 8 | 16.39 | 3.5 | 9.46 | 2.0 |

### Top operators by self device time, forward + backward, per step (all operators: 402.0 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 102 | 122.67 | 30.5 |
| `aten::addmm` | 27 | 41.74 | 10.4 |
| `aten::topk` | 16 | 39.37 | 9.8 |
| `aten::bmm` | 112 | 38.30 | 9.5 |
| `aten::_efficient_attention_backward` | 1 | 13.69 | 3.4 |
| `aten::convolution_backward` | 4 | 5.78 | 1.4 |
| `aten::_efficient_attention_forward` | 1 | 5.74 | 1.4 |
| `triton_poi_fused_new_zeros_32` | 29 | 5.51 | 1.4 |
| `aten::_conv_depthwise2d` | 4 | 4.69 | 1.2 |
| `triton_poi_fused_relu_sub_24` | 16 | 3.74 | 0.9 |
| `triton_poi_fused__to_copy_add_cat_div_mul_softplus_107` | 3 | 3.39 | 0.8 |
| `triton_poi_fused__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoid_backward_slice_slice_backward_view_where_100` | 8 | 3.37 | 0.8 |
| `triton_poi_fused__to_copy_add_arange_div_expand_index_add_lift_fresh_mul_pow_slice_softplus_unsqueeze_view_18` | 4 | 3.34 | 0.8 |
| `triton_poi_fused_add_mul_slice_slice_backward_97` | 8 | 3.21 | 0.8 |
| `triton_poi_fused_mul_43` | 4 | 3.10 | 0.8 |
| `triton_poi_fused_add_mul_slice_slice_backward_94` | 8 | 2.90 | 0.7 |
| `triton_poi_fused__to_copy_add_mul_slice_slice_backward_92` | 8 | 2.76 | 0.7 |
| `triton_poi_fused_mul_62` | 4 | 2.75 | 0.7 |
| `triton_poi_fused__to_copy_add_arange_div_expand_index_add_lift_fresh_mul_pow_slice_softplus_unsqueeze_view_39` | 4 | 2.55 | 0.6 |
| `triton_poi_fused__to_copy_add_mul_slice_slice_backward_91` | 8 | 2.55 | 0.6 |
| `triton_poi_fused_gelu_gelu_backward_view_86` | 8 | 2.50 | 0.6 |
| `triton_poi_fused_convolution_backward_transpose_74` | 4 | 2.34 | 0.6 |
| `triton_poi_fused_add_mul_slice_slice_backward_95` | 8 | 2.07 | 0.5 |
| `triton_poi_fused_add_mul_slice_slice_backward_98` | 8 | 2.06 | 0.5 |
| `triton_poi_fused_index_select_mul_sum_view_26` | 8 | 1.98 | 0.5 |

Kernel time in the profiled forward + backward: 402.0 ms; the split run's forward + backward took 399.4 ms, so the device was busy 101 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 45.26 | 11.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 29.91 | 7.4 |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_nt` | 16 | 26.22 | 6.5 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 23.46 | 5.8 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 18.27 | 4.5 |
| `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::Tenso` | 64 | 15.89 | 4.0 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 14.05 | 3.5 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 14.04 | 3.5 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 13.69 | 3.4 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_tn` | 16 | 13.25 | 3.3 |
| `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned` | 16 | 13.12 | 3.3 |
| `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass:` | 1 | 5.74 | 1.4 |
| `triton_poi_fused_new_zeros_32` | 29 | 5.51 | 1.4 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nt_align8>(cutlass_80_wmma_ten` | 32 | 5.35 | 1.3 |
| `void at::native::(anonymous namespace)::conv_depthwise2d_forward_kernel_generic<c10::BFloat16, int>(torch::hea` | 4 | 4.69 | 1.2 |

### Memory

Peak allocated 6.26 GiB, peak reserved 6.48 GiB over the timed training steps; the forward leaves 5.57 GiB of activations.

