USDT:2026-10-04 08:59:41 3705278:3705278 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 08:59:44 3705278:3705278 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_base.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: none; batch 8 x 2048; warmup 5, measured 20 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 1.0142 | 20 steps, one sync at the end, data draw included; final loss 4.872 |
| split: data | 0.0004 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0014 |
| split: forward | 0.4438 | forward + loss; median 0.4438, min 0.4288, max 0.4571 |
| split: backward | 0.5835 | zero_grad + backward; median 0.5836, min 0.5772, max 0.5904 |
| split: optimizer | 0.0054 | clip_grad_norm_ 0.0022 + optimizer step and schedule 0.0032 (median 0.0033, max 0.0036) |
| split: sum | 1.0331 | sync at every boundary: 1.02 x the plain step |
| energy per step (J) | 152.2 | mean power 150 W (limit 150 W), max 59 C, contended False |

Warmup steps (s): 1.122, 0.934, 1.001, 1.002, 0.996

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 463.5 ms, the parts add up to 463.4 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 457.7 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 443.8 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 278.62 | 60.1 |
| recurrence scan (all layers) | 110.71 | 23.9 |
| layers, rest (LN, projections, FFN) | 58.77 | 12.7 |
| local attention | 11.37 | 2.5 |
| model glue (residual adds, loops over layers) | 2.12 | 0.5 |
| decoder | 0.83 | 0.2 |
| loss | 0.35 | 0.1 |
| final LayerNorm | 0.32 | 0.1 |
| embedding | 0.31 | 0.1 |
| unattributed (outside the model call and the loss) | 0.08 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.17 | 0.3 | 0.4 |
| key conv | 3.11 | 0.7 | 1.1 |
| value projection | 2.54 | 0.5 | 0.9 |
| fp32 pin: casts, glue | 1.31 | 0.3 | 0.5 |
| address: no-grad matmul + top-k | 90.21 | 19.5 | 32.4 |
| address: recompute selected | 16.33 | 3.5 | 5.9 |
| write/read (rest of sparse_memory_read) | 150.89 | 32.6 | 54.2 |
| normaliser + gate | 5.19 | 1.1 | 1.9 |
| output projection | 5.05 | 1.1 | 1.8 |
| other (padding, residual) | 2.81 | 0.6 | 1.0 |
| **all heads** | 278.62 | 60.1 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.27 | 0.79 | 0.66 | 0.34 | 20.67 | 3.56 | 37.49 | 1.25 | 1.26 | 0.72 | 67.01 | 14.5 |
| head after layer 2 | 0.29 | 0.77 | 0.61 | 0.32 | 20.79 | 3.72 | 37.06 | 1.32 | 1.26 | 0.70 | 66.86 | 14.4 |
| head after layer 4 | 0.29 | 0.74 | 0.62 | 0.31 | 27.92 | 5.10 | 38.63 | 1.37 | 1.26 | 0.67 | 76.90 | 16.6 |
| head after layer 6 | 0.32 | 0.82 | 0.65 | 0.33 | 20.83 | 3.95 | 37.70 | 1.26 | 1.27 | 0.72 | 67.86 | 14.6 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 16.48 | 3.6 | 8.06 | 1.7 |
| 2 | 9.43 | 2.0 | 5.44 | 1.2 |
| 3 | 18.21 | 3.9 | 9.36 | 2.0 |
| 4 | 9.70 | 2.1 | 5.91 | 1.3 |
| 5 | 17.98 | 3.9 | 8.84 | 1.9 |
| 6 | 10.46 | 2.3 | 5.89 | 1.3 |
| 7 | 18.23 | 3.9 | 9.17 | 2.0 |
| 8 | 10.23 | 2.2 | 6.09 | 1.3 |

### Top operators by self device time, forward + backward, per step (all operators: 1025.2 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mul` | 2654 | 177.18 | 17.3 |
| `aten::mm` | 102 | 143.00 | 13.9 |
| `aten::bmm` | 112 | 85.61 | 8.4 |
| `aten::div` | 318 | 70.28 | 6.9 |
| `aten::copy_` | 1646 | 61.16 | 6.0 |
| `aten::add_` | 1387 | 58.54 | 5.7 |
| `aten::sum` | 427 | 58.19 | 5.7 |
| `aten::log` | 8 | 47.01 | 4.6 |
| `aten::add` | 1396 | 38.76 | 3.8 |
| `aten::fill_` | 1331 | 37.66 | 3.7 |
| `aten::addmm` | 27 | 35.57 | 3.5 |
| `aten::topk` | 16 | 30.05 | 2.9 |
| `aten::exp` | 8 | 23.98 | 2.3 |
| `aten::scatter_add_` | 120 | 21.58 | 2.1 |
| `aten::native_layer_norm_backward` | 22 | 19.34 | 1.9 |
| `aten::_efficient_attention_backward` | 1 | 15.89 | 1.5 |
| `aten::_index_put_impl_` | 4 | 14.33 | 1.4 |
| `aten::eq` | 36 | 12.04 | 1.2 |
| `aten::cumsum` | 32 | 8.34 | 0.8 |
| `aten::flip` | 32 | 7.77 | 0.8 |
| `aten::native_layer_norm` | 22 | 7.41 | 0.7 |
| `aten::convolution_backward` | 4 | 6.93 | 0.7 |
| `aten::gather` | 121 | 6.70 | 0.7 |
| `aten::sub` | 126 | 5.98 | 0.6 |
| `aten::neg` | 92 | 3.82 | 0.4 |

Kernel time in the profiled forward + backward: 1025.2 ms; the split run's forward + backward took 1027.4 ms, so the device was busy 100 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | 850 | 107.66 | 10.5 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 40 | 63.51 | 6.2 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 53.35 | 5.2 |
| `ampere_sgemm_128x128_tt` | 16 | 53.29 | 5.2 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::log_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 47.01 | 4.6 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, ` | 553 | 41.42 | 4.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::nativ` | 292 | 37.86 | 3.7 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul>,` | 1032 | 35.67 | 3.5 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 34.66 | 3.4 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 23.98 | 2.3 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 544 | 22.26 | 2.2 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 21.73 | 2.1 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>, fals` | 245 | 20.53 | 2.0 |
| `void at::native::vectorized_scatter_add_kernel<16, float, long>(float*, float const*, long const*, int, long, ` | 60 | 19.53 | 1.9 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 104 | 18.80 | 1.8 |

### Memory

Peak allocated 7.38 GiB, peak reserved 7.46 GiB over the timed training steps; the forward leaves 6.74 GiB of activations.

