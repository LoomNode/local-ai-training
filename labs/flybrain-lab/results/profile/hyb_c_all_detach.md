USDT:2026-10-04 09:30:17 3745060:3745060 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:30:18 3745060:3745060 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_c_all_detach.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused, --compile, --head-amp, --detach-state, --scan, --sparse-claws; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.4669 | 15 steps, one sync at the end, data draw included; final loss 5.153 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.2309 | forward + loss; median 0.2293, min 0.2267, max 0.2362 |
| split: backward | 0.2363 | zero_grad + backward; median 0.2362, min 0.2307, max 0.2434 |
| split: optimizer | 0.0028 | clip_grad_norm_ 0.0015 + optimizer step and schedule 0.0012 (median 0.0013, max 0.0013) |
| split: sum | 0.4702 | sync at every boundary: 1.01 x the plain step |
| energy per step (J) | 70.0 | mean power 150 W (limit 150 W), max 58 C, contended False |

Warmup steps (s): 50.346, 0.400, 0.497, 0.431, 0.479 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 524.1 ms, the parts add up to 524.0 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 519.9 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 230.9 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 281.91 | 53.8 |
| recurrence scan (all layers) | 137.24 | 26.2 |
| layers, rest (LN, projections, FFN) | 78.06 | 14.9 |
| local attention | 22.78 | 4.3 |
| model glue (residual adds, loops over layers) | 2.08 | 0.4 |
| decoder | 0.80 | 0.2 |
| final LayerNorm | 0.53 | 0.1 |
| loss | 0.34 | 0.1 |
| embedding | 0.27 | 0.1 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 2.09 | 0.4 | 0.7 |
| key conv | 5.87 | 1.1 | 2.1 |
| value projection | 4.85 | 0.9 | 1.7 |
| fp32 pin: casts, glue | 0.06 | 0.0 | 0.0 |
| address: no-grad matmul + top-k | 98.51 | 18.8 | 34.9 |
| address: recompute selected | 3.39 | 0.6 | 1.2 |
| write/read (rest of sparse_memory_read) | 153.12 | 29.2 | 54.3 |
| normaliser + gate | 4.46 | 0.9 | 1.6 |
| output projection | 4.88 | 0.9 | 1.7 |
| other (padding, residual) | 4.68 | 0.9 | 1.7 |
| **all heads** | 281.91 | 53.8 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.50 | 1.42 | 1.21 | 0.02 | 24.61 | 0.84 | 38.25 | 1.11 | 1.22 | 1.17 | 70.33 | 13.4 |
| head after layer 2 | 0.52 | 1.51 | 1.21 | 0.02 | 24.62 | 0.84 | 38.27 | 1.10 | 1.22 | 1.16 | 70.47 | 13.4 |
| head after layer 4 | 0.54 | 1.42 | 1.21 | 0.01 | 24.64 | 0.84 | 38.29 | 1.12 | 1.22 | 1.17 | 70.48 | 13.4 |
| head after layer 6 | 0.53 | 1.52 | 1.21 | 0.01 | 24.65 | 0.87 | 38.31 | 1.13 | 1.22 | 1.18 | 70.63 | 13.5 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 17.15 | 3.3 | 9.75 | 1.9 |
| 2 | 17.13 | 3.3 | 9.73 | 1.9 |
| 3 | 17.14 | 3.3 | 9.76 | 1.9 |
| 4 | 17.25 | 3.3 | 9.76 | 1.9 |
| 5 | 17.14 | 3.3 | 9.76 | 1.9 |
| 6 | 17.14 | 3.3 | 9.77 | 1.9 |
| 7 | 17.14 | 3.3 | 9.76 | 1.9 |
| 8 | 17.14 | 3.3 | 9.77 | 1.9 |

### Top operators by self device time, forward + backward, per step (all operators: 470.0 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 102 | 139.92 | 29.8 |
| `aten::addmm` | 27 | 47.72 | 10.2 |
| `aten::topk` | 16 | 43.84 | 9.3 |
| `aten::bmm` | 112 | 43.76 | 9.3 |
| `aten::_efficient_attention_backward` | 1 | 15.36 | 3.3 |
| `triton_poi_fused_add_mul_select_backward_slice_73` | 8 | 6.73 | 1.4 |
| `aten::convolution_backward` | 4 | 6.70 | 1.4 |
| `triton_poi_fused_add_mul_select_backward_slice_70` | 8 | 6.44 | 1.4 |
| `aten::_efficient_attention_forward` | 1 | 6.04 | 1.3 |
| `triton_poi_fused__to_copy__unsafe_view_add_arange_lift_fresh_mul_new_zeros_pow_rsub_slice_unsqueeze_view_35` | 28 | 6.01 | 1.3 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_select_backward_sigmoid_sigmoid_backward_slice_view_where_76` | 8 | 5.87 | 1.2 |
| `aten::_conv_depthwise2d` | 4 | 5.21 | 1.1 |
| `triton_poi_fused_add_mul_select_backward_slice_68` | 8 | 4.45 | 0.9 |
| `triton_poi_fused_relu_sub_23` | 16 | 4.44 | 0.9 |
| `triton_poi_fused_add_mul_select_backward_slice_71` | 8 | 4.26 | 0.9 |
| `triton_poi_fused_add_mul_select_backward_slice_74` | 8 | 4.19 | 0.9 |
| `triton_poi_fused_add_mul_select_backward_slice_67` | 8 | 3.98 | 0.8 |
| `triton_poi_fused__to_copy_add_cat_div_mul_softplus_106` | 3 | 3.97 | 0.8 |
| `triton_poi_fused_convolution_backward_transpose_50` | 4 | 3.88 | 0.8 |
| `triton_poi_fused_add_mul_select_backward_slice_72` | 8 | 3.60 | 0.8 |
| `triton_poi_fused_add_mul_select_backward_slice_75` | 8 | 3.54 | 0.8 |
| `triton_poi_fused_add_mul_select_backward_slice_69` | 8 | 3.39 | 0.7 |
| `triton_poi_fused_mul_60` | 4 | 3.17 | 0.7 |
| `triton_poi_fused_add_cat_mul_slice_6` | 8 | 2.73 | 0.6 |
| `triton_poi_fused_add_cat_mul_slice_10` | 8 | 2.70 | 0.6 |

Kernel time in the profiled forward + backward: 470.0 ms; the split run's forward + backward took 467.2 ms, so the device was busy 101 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 51.22 | 10.9 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 35.66 | 7.6 |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_nt` | 16 | 29.68 | 6.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 27.15 | 5.8 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 20.57 | 4.4 |
| `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::Tenso` | 64 | 17.63 | 3.7 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 15.74 | 3.3 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_tn` | 16 | 15.44 | 3.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 15.38 | 3.3 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 15.36 | 3.3 |
| `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned` | 16 | 14.64 | 3.1 |
| `triton_poi_fused_add_mul_select_backward_slice_73` | 8 | 6.73 | 1.4 |
| `triton_poi_fused_add_mul_select_backward_slice_70` | 8 | 6.44 | 1.4 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nt_align8>(cutlass_80_wmma_ten` | 32 | 6.24 | 1.3 |
| `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass:` | 1 | 6.04 | 1.3 |

### Memory

Peak allocated 8.61 GiB, peak reserved 8.69 GiB over the timed training steps; the forward leaves 8.02 GiB of activations.

