USDT:2026-10-04 09:14:02 3722643:3722643 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:14:04 3722643:3722643 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_all_compile.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused, --compile, --head-amp, --scan, --sparse-claws; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.4999 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0002 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0003 |
| split: forward | 0.2302 | forward + loss; median 0.2341, min 0.1821, max 0.2367 |
| split: backward | 0.2693 | zero_grad + backward; median 0.2694, min 0.2642, max 0.2755 |
| split: optimizer | 0.0028 | clip_grad_norm_ 0.0016 + optimizer step and schedule 0.0012 (median 0.0013, max 0.0013) |
| split: sum | 0.5027 | sync at every boundary: 1.01 x the plain step |
| energy per step (J) | 75.0 | mean power 150 W (limit 150 W), max 59 C, contended False |

Warmup steps (s): 65.212, 0.447, 0.517, 0.479, 0.512 (the first includes torch.compile)

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 531.4 ms, the parts add up to 531.3 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 527.1 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 230.2 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 285.84 | 53.8 |
| recurrence scan (all layers) | 139.15 | 26.2 |
| layers, rest (LN, projections, FFN) | 79.21 | 14.9 |
| local attention | 23.11 | 4.3 |
| model glue (residual adds, loops over layers) | 2.07 | 0.4 |
| decoder | 0.82 | 0.2 |
| final LayerNorm | 0.52 | 0.1 |
| loss | 0.35 | 0.1 |
| embedding | 0.26 | 0.0 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 2.10 | 0.4 | 0.7 |
| key conv | 5.92 | 1.1 | 2.1 |
| value projection | 4.92 | 0.9 | 1.7 |
| fp32 pin: casts, glue | 0.06 | 0.0 | 0.0 |
| address: no-grad matmul + top-k | 100.08 | 18.8 | 35.0 |
| address: recompute selected | 3.41 | 0.6 | 1.2 |
| write/read (rest of sparse_memory_read) | 155.25 | 29.2 | 54.3 |
| normaliser + gate | 4.43 | 0.8 | 1.5 |
| output projection | 4.96 | 0.9 | 1.7 |
| other (padding, residual) | 4.70 | 0.9 | 1.6 |
| **all heads** | 285.84 | 53.8 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.51 | 1.43 | 1.23 | 0.01 | 25.00 | 0.85 | 38.80 | 1.09 | 1.24 | 1.17 | 71.32 | 13.4 |
| head after layer 2 | 0.53 | 1.53 | 1.23 | 0.02 | 25.09 | 0.86 | 38.83 | 1.11 | 1.24 | 1.17 | 71.61 | 13.5 |
| head after layer 4 | 0.54 | 1.44 | 1.23 | 0.02 | 24.99 | 0.85 | 38.81 | 1.11 | 1.24 | 1.18 | 71.41 | 13.4 |
| head after layer 6 | 0.53 | 1.53 | 1.23 | 0.02 | 25.00 | 0.86 | 38.80 | 1.11 | 1.24 | 1.18 | 71.50 | 13.5 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 17.39 | 3.3 | 9.92 | 1.9 |
| 2 | 17.40 | 3.3 | 9.90 | 1.9 |
| 3 | 17.39 | 3.3 | 9.91 | 1.9 |
| 4 | 17.40 | 3.3 | 9.90 | 1.9 |
| 5 | 17.39 | 3.3 | 9.90 | 1.9 |
| 6 | 17.39 | 3.3 | 9.90 | 1.9 |
| 7 | 17.39 | 3.3 | 9.91 | 1.9 |
| 8 | 17.40 | 3.3 | 9.88 | 1.9 |

### Top operators by self device time, forward + backward, per step (all operators: 509.0 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 102 | 142.66 | 28.0 |
| `aten::addmm` | 27 | 50.32 | 9.9 |
| `aten::topk` | 16 | 47.53 | 9.3 |
| `aten::bmm` | 112 | 46.13 | 9.1 |
| `aten::_efficient_attention_backward` | 1 | 16.57 | 3.3 |
| `triton_poi_fused_add_mul_select_backward_slice_97` | 8 | 7.01 | 1.4 |
| `triton_poi_fused_new_zeros_31` | 29 | 6.72 | 1.3 |
| `triton_poi_fused_add_mul_select_backward_slice_94` | 8 | 6.60 | 1.3 |
| `aten::convolution_backward` | 4 | 6.51 | 1.3 |
| `aten::_efficient_attention_forward` | 1 | 6.09 | 1.2 |
| `triton_poi_fused__to_copy__unsafe_view_add_clamp_min_gt_mul_neg_rsub_scalar_tensor_select_backward_sigmoid_sigmoid_backward_slice_view_where_100` | 8 | 6.03 | 1.2 |
| `aten::_conv_depthwise2d` | 4 | 5.34 | 1.0 |
| `triton_poi_fused_relu_sub_23` | 16 | 4.76 | 0.9 |
| `triton_poi_fused_add_mul_select_backward_slice_92` | 8 | 4.63 | 0.9 |
| `triton_poi_fused_add_mul_select_backward_slice_95` | 8 | 4.48 | 0.9 |
| `triton_poi_fused_add_mul_select_backward_slice_98` | 8 | 4.38 | 0.9 |
| `triton_poi_fused_add_mul_select_backward_slice_91` | 8 | 4.16 | 0.8 |
| `triton_poi_fused__to_copy_add_cat_div_mul_softplus_106` | 3 | 3.98 | 0.8 |
| `triton_poi_fused__to_copy_add_arange_div_expand_index_add_lift_fresh_mul_pow_slice_softplus_unsqueeze_view_18` | 4 | 3.86 | 0.8 |
| `triton_poi_fused_add_mul_select_backward_slice_96` | 8 | 3.74 | 0.7 |
| `triton_poi_fused_add_mul_select_backward_slice_99` | 8 | 3.66 | 0.7 |
| `triton_poi_fused_mul_43` | 4 | 3.53 | 0.7 |
| `triton_poi_fused_add_mul_select_backward_slice_93` | 8 | 3.51 | 0.7 |
| `triton_poi_fused_mul_61` | 4 | 3.47 | 0.7 |
| `triton_poi_fused__to_copy_add_arange_div_expand_index_add_lift_fresh_mul_pow_slice_softplus_unsqueeze_view_39` | 4 | 2.94 | 0.6 |

Kernel time in the profiled forward + backward: 509.0 ms; the split run's forward + backward took 499.6 ms, so the device was busy 102 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 51.60 | 10.1 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 35.94 | 7.1 |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_nt` | 16 | 31.84 | 6.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 28.68 | 5.6 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 21.64 | 4.3 |
| `void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::Tenso` | 64 | 19.04 | 3.7 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 16.57 | 3.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 16.27 | 3.2 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_tn` | 16 | 16.21 | 3.2 |
| `void at::native::mbtopk::gatherTopK<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned` | 16 | 15.93 | 3.1 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 15.89 | 3.1 |
| `triton_poi_fused_add_mul_select_backward_slice_97` | 8 | 7.01 | 1.4 |
| `triton_poi_fused_new_zeros_31` | 29 | 6.72 | 1.3 |
| `triton_poi_fused_add_mul_select_backward_slice_94` | 8 | 6.60 | 1.3 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_64x1_nt_align8>(cutlass_80_wmma_ten` | 32 | 6.10 | 1.2 |

### Memory

Peak allocated 8.78 GiB, peak reserved 8.96 GiB over the timed training steps; the forward leaves 8.08 GiB of activations.

