USDT:2026-10-04 09:20:30 3733570:3733570 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:20:32 3733570:3733570 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_c_d32.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --compile, --scan; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.5352 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0002 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.2784 | forward + loss; median 0.2784, min 0.2768, max 0.2820 |
| split: backward | 0.2565 | zero_grad + backward; median 0.2565, min 0.2547, max 0.2580 |
| split: optimizer | 0.0049 | clip_grad_norm_ 0.0016 + optimizer step and schedule 0.0033 (median 0.0033, max 0.0034) |
| split: sum | 0.5401 | sync at every boundary: 1.01 x the plain step |
| energy per step (J) | 80.2 | mean power 150 W (limit 150 W), max 57 C, contended False |

Warmup steps (s): 56.445, 0.486, 0.556, 0.514, 0.543 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 594.6 ms, the parts add up to 594.5 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 583.1 ms, so the instrumented one is 1.02 x it (the syncs remove the overlap of launches with work); the forward in the split run took 278.4 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 349.94 | 58.9 |
| recurrence scan (all layers) | 138.58 | 23.3 |
| layers, rest (LN, projections, FFN) | 78.83 | 13.3 |
| local attention | 23.07 | 3.9 |
| model glue (residual adds, loops over layers) | 2.09 | 0.4 |
| decoder | 0.82 | 0.1 |
| final LayerNorm | 0.53 | 0.1 |
| loss | 0.34 | 0.1 |
| embedding | 0.28 | 0.0 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 2.12 | 0.4 | 0.6 |
| key conv | 5.94 | 1.0 | 1.7 |
| value projection | 4.91 | 0.8 | 1.4 |
| fp32 pin: casts, glue | 2.15 | 0.4 | 0.6 |
| address: no-grad matmul + top-k | 153.29 | 25.8 | 43.8 |
| address: recompute selected | 19.49 | 3.3 | 5.6 |
| write/read (rest of sparse_memory_read) | 148.04 | 24.9 | 42.3 |
| normaliser + gate | 4.32 | 0.7 | 1.2 |
| output projection | 4.97 | 0.8 | 1.4 |
| other (padding, residual) | 4.72 | 0.8 | 1.3 |
| **all heads** | 349.94 | 58.9 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.51 | 1.44 | 1.23 | 0.53 | 38.12 | 4.81 | 36.74 | 1.05 | 1.24 | 1.18 | 86.84 | 14.6 |
| head after layer 2 | 0.53 | 1.53 | 1.23 | 0.54 | 38.27 | 4.90 | 37.35 | 1.12 | 1.25 | 1.18 | 87.90 | 14.8 |
| head after layer 4 | 0.55 | 1.44 | 1.24 | 0.54 | 38.49 | 4.84 | 36.90 | 1.08 | 1.24 | 1.19 | 87.49 | 14.7 |
| head after layer 6 | 0.53 | 1.53 | 1.22 | 0.53 | 38.42 | 4.93 | 37.05 | 1.07 | 1.24 | 1.17 | 87.70 | 14.8 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 16.99 | 2.9 | 9.69 | 1.6 |
| 2 | 17.29 | 2.9 | 9.75 | 1.6 |
| 3 | 17.39 | 2.9 | 9.91 | 1.7 |
| 4 | 17.39 | 2.9 | 9.89 | 1.7 |
| 5 | 17.39 | 2.9 | 9.90 | 1.7 |
| 6 | 17.38 | 2.9 | 9.90 | 1.7 |
| 7 | 17.36 | 2.9 | 9.88 | 1.7 |
| 8 | 17.39 | 2.9 | 9.91 | 1.7 |

### Top operators by self device time, forward + backward, per step (all operators: 535.7 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 102 | 138.55 | 25.9 |
| `aten::bmm` | 112 | 101.96 | 19.0 |
| `aten::addmm` | 27 | 44.89 | 8.4 |
| `aten::topk` | 16 | 41.62 | 7.8 |
| `aten::_efficient_attention_backward` | 1 | 15.36 | 2.9 |
| `triton_poi_fused_add_mul_slice_slice_backward_103` | 8 | 6.84 | 1.3 |
| `triton_poi_fused_add_mul_slice_slice_backward_100` | 8 | 6.47 | 1.2 |
| `aten::convolution_backward` | 4 | 6.39 | 1.2 |
| `aten::_efficient_attention_forward` | 1 | 6.12 | 1.1 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_sigmoid_sigmoid_backward_slice_slice_backward_view_where_106` | 8 | 5.99 | 1.1 |
| `triton_poi_fused_new_zeros_28` | 29 | 5.56 | 1.0 |
| `aten::_conv_depthwise2d` | 4 | 4.90 | 0.9 |
| `triton_poi_fused_add_mul_slice_slice_backward_98` | 8 | 4.51 | 0.8 |
| `triton_poi_fused_add_mul_slice_slice_backward_104` | 8 | 4.39 | 0.8 |
| `triton_poi_fused_add_mul_slice_slice_backward_101` | 8 | 4.38 | 0.8 |
| `triton_poi_fused_relu_sub_22` | 16 | 4.17 | 0.8 |
| `triton_poi_fused_add_mul_slice_slice_backward_97` | 8 | 4.06 | 0.8 |
| `triton_poi_fused__to_copy_add_arange_div_expand_index_add_lift_fresh_mul_pow_slice_softplus_unsqueeze_view_19` | 4 | 3.77 | 0.7 |
| `triton_poi_fused_add_mul_slice_slice_backward_105` | 8 | 3.67 | 0.7 |
| `triton_poi_fused_add_mul_slice_slice_backward_102` | 8 | 3.64 | 0.7 |
| `triton_poi_fused__to_copy_add_cat_div_mul_softplus_103` | 3 | 3.54 | 0.7 |
| `triton_poi_fused_add_mul_slice_slice_backward_99` | 8 | 3.42 | 0.6 |
| `triton_poi_fused_mul_50` | 4 | 3.42 | 0.6 |
| `triton_poi_fused_mul_59` | 4 | 3.03 | 0.6 |
| `triton_poi_fused__to_copy_add_arange_div_expand_index_add_lift_fresh_mul_pow_slice_softplus_unsqueeze_view_46` | 4 | 2.86 | 0.5 |

Kernel time in the profiled forward + backward: 535.7 ms; the split run's forward + backward took 534.9 ms, so the device was busy 100 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `ampere_sgemm_128x128_tn` | 16 | 85.86 | 16.0 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 50.96 | 9.5 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 34.91 | 6.5 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 25.52 | 4.8 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 19.38 | 3.6 |
| `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::Tenso` | 64 | 16.74 | 3.1 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 16.18 | 3.0 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 15.56 | 2.9 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 15.36 | 2.9 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_tn` | 16 | 14.65 | 2.7 |
| `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned` | 16 | 13.86 | 2.6 |
| `ampere_sgemm_128x128_nn` | 32 | 6.95 | 1.3 |
| `triton_poi_fused_add_mul_slice_slice_backward_103` | 8 | 6.84 | 1.3 |
| `triton_poi_fused_add_mul_slice_slice_backward_100` | 8 | 6.47 | 1.2 |
| `fmha_cutlassF_bf16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, cutlass:` | 1 | 6.12 | 1.1 |

### Memory

Peak allocated 8.78 GiB, peak reserved 8.99 GiB over the timed training steps; the forward leaves 8.09 GiB of activations.

