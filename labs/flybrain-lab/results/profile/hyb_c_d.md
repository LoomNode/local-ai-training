USDT:2026-10-04 09:22:00 3735230:3735230 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:22:02 3735230:3735230 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_c_d.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --compile, --scan; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.4600 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0002 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.2494 | forward + loss; median 0.2496, min 0.2433, max 0.2539 |
| split: backward | 0.2091 | zero_grad + backward; median 0.2094, min 0.2049, max 0.2130 |
| split: optimizer | 0.0042 | clip_grad_norm_ 0.0015 + optimizer step and schedule 0.0027 (median 0.0026, max 0.0032) |
| split: sum | 0.4629 | sync at every boundary: 1.01 x the plain step |
| energy per step (J) | 68.8 | mean power 150 W (limit 150 W), max 58 C, contended False |

Warmup steps (s): 55.649, 0.462, 0.425, 0.490, 0.423 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 531.4 ms, the parts add up to 531.3 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 531.1 ms, so the instrumented one is 1.00 x it (the syncs remove the overlap of launches with work); the forward in the split run took 249.4 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 296.91 | 55.9 |
| recurrence scan (all layers) | 133.41 | 25.1 |
| layers, rest (LN, projections, FFN) | 77.01 | 14.5 |
| local attention | 20.12 | 3.8 |
| model glue (residual adds, loops over layers) | 1.97 | 0.4 |
| decoder | 0.80 | 0.2 |
| final LayerNorm | 0.45 | 0.1 |
| loss | 0.34 | 0.1 |
| embedding | 0.27 | 0.1 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.83 | 0.3 | 0.6 |
| key conv | 5.19 | 1.0 | 1.7 |
| value projection | 4.32 | 0.8 | 1.5 |
| fp32 pin: casts, glue | 1.90 | 0.4 | 0.6 |
| address: no-grad matmul + top-k | 128.69 | 24.2 | 43.3 |
| address: recompute selected | 15.22 | 2.9 | 5.1 |
| write/read (rest of sparse_memory_read) | 127.03 | 23.9 | 42.8 |
| normaliser + gate | 3.68 | 0.7 | 1.2 |
| output projection | 4.89 | 0.9 | 1.6 |
| other (padding, residual) | 4.15 | 0.8 | 1.4 |
| **all heads** | 296.91 | 55.9 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.44 | 1.24 | 1.09 | 0.47 | 31.62 | 3.76 | 31.75 | 0.92 | 1.22 | 1.03 | 73.55 | 13.8 |
| head after layer 2 | 0.48 | 1.40 | 1.13 | 0.50 | 32.53 | 3.77 | 31.89 | 0.93 | 1.24 | 1.07 | 74.94 | 14.1 |
| head after layer 4 | 0.45 | 1.18 | 0.99 | 0.44 | 31.85 | 3.89 | 31.34 | 0.92 | 1.20 | 0.99 | 73.23 | 13.8 |
| head after layer 6 | 0.47 | 1.37 | 1.11 | 0.48 | 32.68 | 3.81 | 32.06 | 0.91 | 1.24 | 1.06 | 75.18 | 14.1 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 16.37 | 3.1 | 9.28 | 1.7 |
| 2 | 15.52 | 2.9 | 9.06 | 1.7 |
| 3 | 17.30 | 3.3 | 10.14 | 1.9 |
| 4 | 16.66 | 3.1 | 9.52 | 1.8 |
| 5 | 17.29 | 3.3 | 10.13 | 1.9 |
| 6 | 16.51 | 3.1 | 9.45 | 1.8 |
| 7 | 17.29 | 3.3 | 10.11 | 1.9 |
| 8 | 16.48 | 3.1 | 9.32 | 1.8 |

### Top operators by self device time, forward + backward, per step (all operators: 460.9 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 102 | 128.59 | 27.9 |
| `aten::bmm` | 112 | 91.81 | 19.9 |
| `aten::addmm` | 27 | 43.30 | 9.4 |
| `aten::topk` | 16 | 38.17 | 8.3 |
| `aten::_efficient_attention_backward` | 1 | 13.56 | 2.9 |
| `aten::_efficient_attention_forward` | 1 | 6.09 | 1.3 |
| `aten::convolution_backward` | 4 | 5.17 | 1.1 |
| `triton_poi_fused_new_zeros_29` | 29 | 5.13 | 1.1 |
| `aten::_conv_depthwise2d` | 4 | 4.60 | 1.0 |
| `triton_poi_fused_relu_sub_23` | 16 | 3.97 | 0.9 |
| `triton_poi_fused__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoid_backward_slice_slice_backward_view_where_106` | 8 | 3.66 | 0.8 |
| `triton_poi_fused_add_mul_slice_slice_backward_103` | 8 | 3.44 | 0.7 |
| `triton_poi_fused__to_copy_add_cat_div_mul_softplus_104` | 3 | 3.20 | 0.7 |
| `triton_poi_fused_add_mul_slice_slice_backward_100` | 8 | 3.16 | 0.7 |
| `triton_poi_fused__to_copy_add_arange_div_expand_index_add_lift_fresh_mul_pow_slice_softplus_unsqueeze_view_19` | 4 | 3.15 | 0.7 |
| `triton_poi_fused__to_copy_add_mul_slice_slice_backward_98` | 8 | 2.99 | 0.6 |
| `triton_poi_fused_mul_50` | 4 | 2.77 | 0.6 |
| `triton_poi_fused__to_copy_add_mul_slice_slice_backward_97` | 8 | 2.76 | 0.6 |
| `triton_poi_fused_gelu_gelu_backward_view_92` | 8 | 2.69 | 0.6 |
| `triton_poi_fused_mul_60` | 4 | 2.68 | 0.6 |
| `triton_poi_fused__to_copy_add_arange_div_expand_index_add_lift_fresh_mul_pow_slice_softplus_unsqueeze_view_46` | 4 | 2.33 | 0.5 |
| `triton_poi_fused_add_mul_slice_slice_backward_101` | 8 | 2.25 | 0.5 |
| `triton_poi_fused_add_mul_slice_slice_backward_104` | 8 | 2.23 | 0.5 |
| `triton_poi_fused__to_copy_add_arange_cat_cos_div_mul_neg_permute_pow_select_sin_slice_sub_unsqueeze_view_83` | 1 | 2.01 | 0.4 |
| `triton_poi_fused__to_copy_add_arange_cat_cos_div_mul_neg_permute_pow_select_sin_slice_sub_unsqueeze_view_82` | 1 | 2.01 | 0.4 |

Kernel time in the profiled forward + backward: 460.9 ms; the split run's forward + backward took 458.5 ms, so the device was busy 101 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `ampere_sgemm_128x128_tn` | 16 | 78.20 | 17.0 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 47.21 | 10.2 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 31.50 | 6.8 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 24.25 | 5.3 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 19.06 | 4.1 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 15.34 | 3.3 |
| `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::Tenso` | 64 | 15.33 | 3.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 15.08 | 3.3 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 13.56 | 2.9 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_tn` | 16 | 13.28 | 2.9 |
| `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned` | 16 | 12.73 | 2.8 |
| `ampere_sgemm_128x128_nn` | 32 | 6.14 | 1.3 |
| `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass:` | 1 | 6.09 | 1.3 |
| `triton_poi_fused_new_zeros_29` | 29 | 5.13 | 1.1 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nt_align8>(cutlass_80_wmma_ten` | 32 | 4.88 | 1.1 |

### Memory

Peak allocated 6.27 GiB, peak reserved 6.49 GiB over the timed training steps; the forward leaves 5.58 GiB of activations.

