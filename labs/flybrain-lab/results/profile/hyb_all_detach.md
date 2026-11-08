USDT:2026-10-04 09:10:18 3718872:3718872 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:10:22 3718872:3718872 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_all_detach.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused, --head-amp, --detach-state, --scan, --sparse-claws; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 1.2062 | 15 steps, one sync at the end, data draw included; final loss 5.152 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0008 |
| split: forward | 0.5287 | forward + loss; median 0.5286, min 0.5284, max 0.5293 |
| split: backward | 0.6784 | zero_grad + backward; median 0.6784, min 0.6767, max 0.6801 |
| split: optimizer | 0.0034 | clip_grad_norm_ 0.0021 + optimizer step and schedule 0.0013 (median 0.0013, max 0.0015) |
| split: sum | 1.2108 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 191.1 | mean power 158 W (limit 150 W), max 62 C, contended False |

Warmup steps (s): 1.209, 1.205, 1.206, 1.207, 1.207

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 536.5 ms, the parts add up to 536.4 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 528.9 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 528.7 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 289.19 | 53.9 |
| recurrence scan (all layers) | 139.54 | 26.0 |
| layers, rest (LN, projections, FFN) | 80.10 | 14.9 |
| local attention | 23.16 | 4.3 |
| model glue (residual adds, loops over layers) | 2.36 | 0.4 |
| decoder | 0.82 | 0.2 |
| final LayerNorm | 0.57 | 0.1 |
| loss | 0.35 | 0.1 |
| embedding | 0.30 | 0.1 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 2.23 | 0.4 | 0.8 |
| key conv | 6.03 | 1.1 | 2.1 |
| value projection | 4.99 | 0.9 | 1.7 |
| fp32 pin: casts, glue | 0.08 | 0.0 | 0.0 |
| address: no-grad matmul + top-k | 100.30 | 18.7 | 34.7 |
| address: recompute selected | 3.70 | 0.7 | 1.3 |
| write/read (rest of sparse_memory_read) | 156.54 | 29.2 | 54.1 |
| normaliser + gate | 5.32 | 1.0 | 1.8 |
| output projection | 5.05 | 0.9 | 1.7 |
| other (padding, residual) | 4.95 | 0.9 | 1.7 |
| **all heads** | 289.19 | 53.9 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.52 | 1.45 | 1.24 | 0.02 | 25.04 | 0.90 | 39.08 | 1.35 | 1.27 | 1.23 | 72.13 | 13.4 |
| head after layer 2 | 0.58 | 1.56 | 1.24 | 0.02 | 25.08 | 0.91 | 39.15 | 1.32 | 1.25 | 1.24 | 72.36 | 13.5 |
| head after layer 4 | 0.56 | 1.45 | 1.24 | 0.02 | 25.08 | 0.92 | 39.14 | 1.33 | 1.27 | 1.23 | 72.24 | 13.5 |
| head after layer 6 | 0.58 | 1.56 | 1.26 | 0.02 | 25.10 | 0.97 | 39.16 | 1.32 | 1.26 | 1.25 | 72.47 | 13.5 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 17.45 | 3.3 | 10.01 | 1.9 |
| 2 | 17.44 | 3.3 | 10.03 | 1.9 |
| 3 | 17.44 | 3.3 | 10.01 | 1.9 |
| 4 | 17.44 | 3.3 | 9.99 | 1.9 |
| 5 | 17.43 | 3.2 | 9.99 | 1.9 |
| 6 | 17.47 | 3.3 | 10.01 | 1.9 |
| 7 | 17.43 | 3.2 | 10.03 | 1.9 |
| 8 | 17.44 | 3.3 | 10.03 | 1.9 |

### Top operators by self device time, forward + backward, per step (all operators: 1237.1 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mul` | 1442 | 231.39 | 18.7 |
| `aten::mm` | 102 | 168.13 | 13.6 |
| `aten::copy_` | 1322 | 144.82 | 11.7 |
| `aten::add_` | 735 | 111.19 | 9.0 |
| `aten::fill_` | 871 | 93.10 | 7.5 |
| `aten::cat` | 202 | 80.61 | 6.5 |
| `aten::addmm` | 27 | 53.56 | 4.3 |
| `aten::sum` | 359 | 50.28 | 4.1 |
| `aten::topk` | 16 | 48.49 | 3.9 |
| `aten::bmm` | 112 | 47.87 | 3.9 |
| `aten::add` | 420 | 46.34 | 3.7 |
| `aten::native_layer_norm_backward` | 22 | 22.06 | 1.8 |
| `aten::_efficient_attention_backward` | 1 | 18.40 | 1.5 |
| `aten::sub` | 126 | 16.67 | 1.3 |
| `aten::scatter_add_` | 68 | 12.98 | 1.0 |
| `aten::eq` | 36 | 12.69 | 1.0 |
| `aten::native_layer_norm` | 22 | 10.98 | 0.9 |
| `aten::_index_put_impl_` | 4 | 9.60 | 0.8 |
| `aten::div` | 278 | 7.15 | 0.6 |
| `aten::convolution_backward` | 4 | 6.93 | 0.6 |
| `aten::_efficient_attention_forward` | 1 | 6.13 | 0.5 |
| `aten::_conv_depthwise2d` | 4 | 5.78 | 0.5 |
| `aten::clamp_min` | 28 | 5.39 | 0.4 |
| `aten::where` | 10 | 4.74 | 0.4 |
| `aten::gather` | 69 | 4.03 | 0.3 |

Kernel time in the profiled forward + backward: 1237.1 ms; the split run's forward + backward took 1207.1 ms, so the device was busy 102 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | 1170 | 185.52 | 15.0 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, ` | 809 | 121.42 | 9.8 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 697 | 97.70 | 7.9 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>, fals` | 781 | 89.86 | 7.3 |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<4u>,` | 180 | 77.28 | 6.2 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 61.46 | 5.0 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 41.14 | 3.3 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::nativ` | 264 | 36.29 | 2.9 |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tt` | 16 | 32.55 | 2.6 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 30.57 | 2.5 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 136 | 29.96 | 2.4 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<flo` | 157 | 24.28 | 2.0 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 22.99 | 1.9 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 19.84 | 1.6 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 19.68 | 1.6 |

### Memory

Peak allocated 9.82 GiB, peak reserved 9.90 GiB over the timed training steps; the forward leaves 9.25 GiB of activations.

