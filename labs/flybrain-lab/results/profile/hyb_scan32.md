USDT:2026-10-04 09:04:52 3711912:3711912 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:04:56 3711912:3711912 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_scan32.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --scan; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 1.3610 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0004 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0009 |
| split: forward | 0.6003 | forward + loss; median 0.6001, min 0.5999, max 0.6012 |
| split: backward | 0.7603 | zero_grad + backward; median 0.7601, min 0.7582, max 0.7628 |
| split: optimizer | 0.0057 | clip_grad_norm_ 0.0023 + optimizer step and schedule 0.0034 (median 0.0034, max 0.0036) |
| split: sum | 1.3667 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 215.0 | mean power 158 W (limit 150 W), max 61 C, contended False |

Warmup steps (s): 1.413, 1.361, 1.361, 1.361, 1.363

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 608.7 ms, the parts add up to 608.6 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 600.0 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 600.3 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 361.47 | 59.4 |
| recurrence scan (all layers) | 139.55 | 22.9 |
| layers, rest (LN, projections, FFN) | 79.93 | 13.1 |
| local attention | 23.18 | 3.8 |
| model glue (residual adds, loops over layers) | 2.39 | 0.4 |
| decoder | 0.82 | 0.1 |
| final LayerNorm | 0.58 | 0.1 |
| loss | 0.35 | 0.1 |
| embedding | 0.31 | 0.1 |
| unattributed (outside the model call and the loss) | 0.09 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 2.25 | 0.4 | 0.6 |
| key conv | 6.06 | 1.0 | 1.7 |
| value projection | 4.96 | 0.8 | 1.4 |
| fp32 pin: casts, glue | 2.25 | 0.4 | 0.6 |
| address: no-grad matmul + top-k | 155.49 | 25.5 | 43.0 |
| address: recompute selected | 20.40 | 3.4 | 5.6 |
| write/read (rest of sparse_memory_read) | 154.82 | 25.4 | 42.8 |
| normaliser + gate | 5.20 | 0.9 | 1.4 |
| output projection | 5.07 | 0.8 | 1.4 |
| other (padding, residual) | 4.97 | 0.8 | 1.4 |
| **all heads** | 361.47 | 59.4 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.52 | 1.45 | 1.25 | 0.56 | 38.86 | 5.07 | 38.61 | 1.30 | 1.26 | 1.23 | 90.11 | 14.8 |
| head after layer 2 | 0.57 | 1.56 | 1.24 | 0.57 | 38.87 | 5.11 | 38.70 | 1.28 | 1.29 | 1.25 | 90.42 | 14.9 |
| head after layer 4 | 0.59 | 1.50 | 1.24 | 0.56 | 38.87 | 5.09 | 38.71 | 1.31 | 1.27 | 1.24 | 90.39 | 14.8 |
| head after layer 6 | 0.57 | 1.55 | 1.24 | 0.57 | 38.89 | 5.13 | 38.80 | 1.30 | 1.26 | 1.25 | 90.56 | 14.9 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 17.44 | 2.9 | 9.96 | 1.6 |
| 2 | 17.47 | 2.9 | 10.00 | 1.6 |
| 3 | 17.43 | 2.9 | 9.98 | 1.6 |
| 4 | 17.44 | 2.9 | 10.01 | 1.6 |
| 5 | 17.44 | 2.9 | 9.99 | 1.6 |
| 6 | 17.46 | 2.9 | 10.02 | 1.6 |
| 7 | 17.43 | 2.9 | 9.97 | 1.6 |
| 8 | 17.44 | 2.9 | 10.00 | 1.6 |

### Top operators by self device time, forward + backward, per step (all operators: 1391.4 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mul` | 1630 | 268.72 | 19.3 |
| `aten::mm` | 102 | 167.75 | 12.1 |
| `aten::copy_` | 1222 | 144.93 | 10.4 |
| `aten::bmm` | 112 | 130.29 | 9.4 |
| `aten::add_` | 811 | 124.73 | 9.0 |
| `aten::fill_` | 923 | 101.99 | 7.3 |
| `aten::cat` | 202 | 79.81 | 5.7 |
| `aten::sum` | 419 | 58.64 | 4.2 |
| `aten::addmm` | 27 | 52.86 | 3.8 |
| `aten::topk` | 16 | 48.49 | 3.5 |
| `aten::add` | 476 | 46.73 | 3.4 |
| `aten::scatter_add_` | 120 | 22.36 | 1.6 |
| `aten::native_layer_norm_backward` | 22 | 22.07 | 1.6 |
| `aten::_efficient_attention_backward` | 1 | 18.38 | 1.3 |
| `aten::eq` | 36 | 12.69 | 0.9 |
| `aten::native_layer_norm` | 22 | 10.85 | 0.8 |
| `aten::_index_put_impl_` | 4 | 10.26 | 0.7 |
| `aten::sub` | 126 | 9.20 | 0.7 |
| `aten::div` | 278 | 7.14 | 0.5 |
| `aten::gather` | 121 | 7.06 | 0.5 |
| `aten::convolution_backward` | 4 | 6.93 | 0.5 |
| `aten::_efficient_attention_forward` | 1 | 6.14 | 0.4 |
| `aten::_conv_depthwise2d` | 4 | 5.78 | 0.4 |
| `aten::clamp_min` | 28 | 5.38 | 0.4 |
| `aten::where` | 10 | 4.74 | 0.3 |

Kernel time in the profiled forward + backward: 1391.4 ms; the split run's forward + backward took 1360.6 ms, so the device was busy 102 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | 1394 | 232.15 | 16.7 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, ` | 1001 | 140.21 | 10.1 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>, fals` | 869 | 101.17 | 7.3 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 729 | 98.88 | 7.1 |
| `ampere_sgemm_128x128_tt` | 16 | 96.18 | 6.9 |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<4u>,` | 180 | 76.48 | 5.5 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 61.44 | 4.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 40.81 | 2.9 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::nativ` | 292 | 39.96 | 2.9 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<flo` | 173 | 31.54 | 2.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 30.02 | 2.2 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 22.84 | 1.6 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 104 | 20.78 | 1.5 |
| `void at::native::vectorized_scatter_add_kernel<16, float, long>(float*, float const*, long const*, int, long, ` | 60 | 20.23 | 1.5 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 19.84 | 1.4 |

### Memory

Peak allocated 10.84 GiB, peak reserved 10.94 GiB over the timed training steps; the forward leaves 10.21 GiB of activations.

