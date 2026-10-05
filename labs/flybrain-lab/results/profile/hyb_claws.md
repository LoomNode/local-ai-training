USDT:2026-10-04 09:03:37 3710907:3710907 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:03:40 3710907:3710907 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_claws.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --sparse-claws; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 1.0336 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0011 |
| split: forward | 0.4473 | forward + loss; median 0.4500, min 0.4349, max 0.4561 |
| split: backward | 0.5836 | zero_grad + backward; median 0.5822, min 0.5761, max 0.5924 |
| split: optimizer | 0.0055 | clip_grad_norm_ 0.0023 + optimizer step and schedule 0.0032 (median 0.0033, max 0.0036) |
| split: sum | 1.0367 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 155.3 | mean power 150 W (limit 150 W), max 61 C, contended False |

Warmup steps (s): 1.141, 0.979, 1.030, 1.026, 1.033

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 455.6 ms, the parts add up to 455.5 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 448.2 ms, so the instrumented one is 1.02 x it (the syncs remove the overlap of launches with work); the forward in the split run took 447.3 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 267.46 | 58.7 |
| recurrence scan (all layers) | 112.71 | 24.7 |
| layers, rest (LN, projections, FFN) | 59.18 | 13.0 |
| local attention | 12.21 | 2.7 |
| model glue (residual adds, loops over layers) | 2.12 | 0.5 |
| decoder | 0.82 | 0.2 |
| loss | 0.35 | 0.1 |
| final LayerNorm | 0.34 | 0.1 |
| embedding | 0.31 | 0.1 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.17 | 0.3 | 0.4 |
| key conv | 3.18 | 0.7 | 1.2 |
| value projection | 2.62 | 0.6 | 1.0 |
| fp32 pin: casts, glue | 1.32 | 0.3 | 0.5 |
| address: no-grad matmul + top-k | 94.20 | 20.7 | 35.2 |
| address: recompute selected | 3.03 | 0.7 | 1.1 |
| write/read (rest of sparse_memory_read) | 148.89 | 32.7 | 55.7 |
| normaliser + gate | 5.10 | 1.1 | 1.9 |
| output projection | 5.07 | 1.1 | 1.9 |
| other (padding, residual) | 2.87 | 0.6 | 1.1 |
| **all heads** | 267.46 | 58.7 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.28 | 0.80 | 0.69 | 0.34 | 21.16 | 0.67 | 36.25 | 1.24 | 1.26 | 0.74 | 63.42 | 13.9 |
| head after layer 2 | 0.27 | 0.74 | 0.59 | 0.30 | 22.79 | 0.76 | 37.45 | 1.27 | 1.28 | 0.66 | 66.11 | 14.5 |
| head after layer 4 | 0.31 | 0.78 | 0.67 | 0.34 | 29.34 | 0.90 | 38.40 | 1.34 | 1.26 | 0.72 | 74.05 | 16.3 |
| head after layer 6 | 0.32 | 0.86 | 0.68 | 0.34 | 20.92 | 0.71 | 36.78 | 1.26 | 1.27 | 0.75 | 63.88 | 14.0 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 17.26 | 3.8 | 7.87 | 1.7 |
| 2 | 8.97 | 2.0 | 5.07 | 1.1 |
| 3 | 17.93 | 3.9 | 9.27 | 2.0 |
| 4 | 10.95 | 2.4 | 6.39 | 1.4 |
| 5 | 18.10 | 4.0 | 9.13 | 2.0 |
| 6 | 10.69 | 2.3 | 6.19 | 1.4 |
| 7 | 17.87 | 3.9 | 9.28 | 2.0 |
| 8 | 10.93 | 2.4 | 5.98 | 1.3 |

### Top operators by self device time, forward + backward, per step (all operators: 1018.3 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mul` | 2654 | 162.90 | 16.0 |
| `aten::mm` | 102 | 146.71 | 14.4 |
| `aten::bmm` | 112 | 86.64 | 8.5 |
| `aten::div` | 318 | 69.54 | 6.8 |
| `aten::copy_` | 1646 | 62.47 | 6.1 |
| `aten::add_` | 1387 | 59.95 | 5.9 |
| `aten::sum` | 423 | 55.75 | 5.5 |
| `aten::log` | 8 | 49.35 | 4.8 |
| `aten::fill_` | 1335 | 38.92 | 3.8 |
| `aten::add` | 1396 | 38.85 | 3.8 |
| `aten::addmm` | 27 | 36.29 | 3.6 |
| `aten::topk` | 16 | 30.23 | 3.0 |
| `aten::exp` | 8 | 24.31 | 2.4 |
| `aten::scatter_add_` | 124 | 22.04 | 2.2 |
| `aten::native_layer_norm_backward` | 22 | 19.99 | 2.0 |
| `aten::_efficient_attention_backward` | 1 | 16.43 | 1.6 |
| `aten::eq` | 36 | 12.11 | 1.2 |
| `aten::_index_put_impl_` | 4 | 10.33 | 1.0 |
| `aten::cumsum` | 32 | 8.29 | 0.8 |
| `aten::flip` | 32 | 7.73 | 0.8 |
| `aten::native_layer_norm` | 22 | 7.44 | 0.7 |
| `aten::gather` | 125 | 6.98 | 0.7 |
| `aten::convolution_backward` | 4 | 6.93 | 0.7 |
| `aten::sub` | 126 | 5.91 | 0.6 |
| `aten::neg` | 92 | 3.80 | 0.4 |

Kernel time in the profiled forward + backward: 1018.3 ms; the split run's forward + backward took 1030.9 ms, so the device was busy 99 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | 846 | 92.17 | 9.1 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 40 | 62.66 | 6.2 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 54.34 | 5.3 |
| `ampere_sgemm_128x128_tt` | 16 | 53.85 | 5.3 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::log_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 49.35 | 4.8 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, ` | 553 | 41.98 | 4.1 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::nativ` | 292 | 37.07 | 3.6 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul>,` | 1032 | 36.57 | 3.6 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 35.38 | 3.5 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 24.31 | 2.4 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 544 | 22.54 | 2.2 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 22.16 | 2.2 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>, fals` | 249 | 21.22 | 2.1 |
| `void at::native::vectorized_scatter_add_kernel<16, float, long>(float*, float const*, long const*, int, long, ` | 60 | 19.69 | 1.9 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 104 | 19.29 | 1.9 |

### Memory

Peak allocated 6.98 GiB, peak reserved 7.01 GiB over the timed training steps; the forward leaves 6.38 GiB of activations.

