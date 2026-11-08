USDT:2026-10-04 09:09:11 3717204:3717204 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:09:15 3717204:3717204 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_all.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused, --head-amp, --scan, --sparse-claws; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 1.2681 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0008 |
| split: forward | 0.5290 | forward + loss; median 0.5289, min 0.5285, max 0.5298 |
| split: backward | 0.7411 | zero_grad + backward; median 0.7409, min 0.7394, max 0.7432 |
| split: optimizer | 0.0034 | clip_grad_norm_ 0.0021 + optimizer step and schedule 0.0013 (median 0.0013, max 0.0015) |
| split: sum | 1.2738 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 201.1 | mean power 159 W (limit 150 W), max 61 C, contended False |

Warmup steps (s): 1.264, 1.267, 1.269, 1.268, 1.269

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 536.1 ms, the parts add up to 536.0 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 528.8 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 529.0 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 288.79 | 53.9 |
| recurrence scan (all layers) | 139.65 | 26.0 |
| layers, rest (LN, projections, FFN) | 80.01 | 14.9 |
| local attention | 23.17 | 4.3 |
| model glue (residual adds, loops over layers) | 2.32 | 0.4 |
| decoder | 0.82 | 0.2 |
| final LayerNorm | 0.57 | 0.1 |
| loss | 0.35 | 0.1 |
| embedding | 0.32 | 0.1 |
| unattributed (outside the model call and the loss) | 0.08 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 2.22 | 0.4 | 0.8 |
| key conv | 6.02 | 1.1 | 2.1 |
| value projection | 4.98 | 0.9 | 1.7 |
| fp32 pin: casts, glue | 0.07 | 0.0 | 0.0 |
| address: no-grad matmul + top-k | 100.24 | 18.7 | 34.7 |
| address: recompute selected | 3.70 | 0.7 | 1.3 |
| write/read (rest of sparse_memory_read) | 156.38 | 29.2 | 54.1 |
| normaliser + gate | 5.20 | 1.0 | 1.8 |
| output projection | 5.03 | 0.9 | 1.7 |
| other (padding, residual) | 4.95 | 0.9 | 1.7 |
| **all heads** | 288.79 | 53.9 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.52 | 1.45 | 1.23 | 0.02 | 25.03 | 0.91 | 39.02 | 1.32 | 1.25 | 1.22 | 71.96 | 13.4 |
| head after layer 2 | 0.58 | 1.55 | 1.24 | 0.02 | 25.05 | 0.91 | 39.12 | 1.29 | 1.25 | 1.25 | 72.27 | 13.5 |
| head after layer 4 | 0.57 | 1.48 | 1.26 | 0.02 | 25.09 | 0.91 | 39.18 | 1.31 | 1.26 | 1.24 | 72.32 | 13.5 |
| head after layer 6 | 0.55 | 1.54 | 1.24 | 0.02 | 25.07 | 0.97 | 39.06 | 1.28 | 1.27 | 1.24 | 72.25 | 13.5 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 17.46 | 3.3 | 9.99 | 1.9 |
| 2 | 17.45 | 3.3 | 10.03 | 1.9 |
| 3 | 17.47 | 3.3 | 9.97 | 1.9 |
| 4 | 17.48 | 3.3 | 10.03 | 1.9 |
| 5 | 17.42 | 3.3 | 9.97 | 1.9 |
| 6 | 17.46 | 3.3 | 10.01 | 1.9 |
| 7 | 17.46 | 3.3 | 10.01 | 1.9 |
| 8 | 17.46 | 3.3 | 9.99 | 1.9 |

### Top operators by self device time, forward + backward, per step (all operators: 1306.0 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mul` | 1630 | 258.58 | 19.8 |
| `aten::mm` | 102 | 167.94 | 12.9 |
| `aten::copy_` | 1350 | 145.42 | 11.1 |
| `aten::add_` | 811 | 121.73 | 9.3 |
| `aten::fill_` | 927 | 100.18 | 7.7 |
| `aten::cat` | 202 | 82.82 | 6.3 |
| `aten::sum` | 415 | 55.16 | 4.2 |
| `aten::addmm` | 27 | 54.42 | 4.2 |
| `aten::topk` | 16 | 48.49 | 3.7 |
| `aten::bmm` | 112 | 47.88 | 3.7 |
| `aten::add` | 476 | 47.67 | 3.7 |
| `aten::scatter_add_` | 124 | 23.45 | 1.8 |
| `aten::native_layer_norm_backward` | 22 | 22.01 | 1.7 |
| `aten::_efficient_attention_backward` | 1 | 18.36 | 1.4 |
| `aten::sub` | 126 | 16.69 | 1.3 |
| `aten::eq` | 36 | 12.69 | 1.0 |
| `aten::native_layer_norm` | 22 | 11.12 | 0.9 |
| `aten::_index_put_impl_` | 4 | 10.19 | 0.8 |
| `aten::gather` | 125 | 7.25 | 0.6 |
| `aten::div` | 278 | 7.14 | 0.5 |
| `aten::convolution_backward` | 4 | 6.93 | 0.5 |
| `aten::_efficient_attention_forward` | 1 | 6.13 | 0.5 |
| `aten::_conv_depthwise2d` | 4 | 5.78 | 0.4 |
| `aten::clamp_min` | 28 | 5.40 | 0.4 |
| `aten::where` | 10 | 4.74 | 0.4 |

Kernel time in the profiled forward + backward: 1306.0 ms; the split run's forward + backward took 1270.1 ms, so the device was busy 103 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | 1330 | 204.66 | 15.7 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, ` | 913 | 132.20 | 10.1 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 697 | 97.59 | 7.5 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>, fals` | 837 | 96.94 | 7.4 |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<4u>,` | 180 | 79.49 | 6.1 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 61.46 | 4.7 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 40.81 | 3.1 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::nativ` | 292 | 38.46 | 2.9 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 164 | 38.01 | 2.9 |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tt` | 16 | 32.55 | 2.5 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 31.08 | 2.4 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<flo` | 157 | 24.76 | 1.9 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 23.34 | 1.8 |
| `void at::native::vectorized_scatter_add_kernel<16, float, long>(float*, float const*, long const*, int, long, ` | 60 | 20.27 | 1.6 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 19.82 | 1.5 |

### Memory

Peak allocated 10.27 GiB, peak reserved 10.35 GiB over the timed training steps; the forward leaves 9.69 GiB of activations.

