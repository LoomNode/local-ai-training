USDT:2026-10-04 09:01:38 3708911:3708911 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:01:41 3708911:3708911 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_tf32.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 1.0163 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0008 |
| split: forward | 0.4394 | forward + loss; median 0.4408, min 0.4255, max 0.4505 |
| split: backward | 0.5750 | zero_grad + backward; median 0.5751, min 0.5681, max 0.5793 |
| split: optimizer | 0.0055 | clip_grad_norm_ 0.0022 + optimizer step and schedule 0.0033 (median 0.0033, max 0.0036) |
| split: sum | 1.0203 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 152.6 | mean power 150 W (limit 150 W), max 60 C, contended False |

Warmup steps (s): 1.168, 0.962, 1.017, 1.017, 1.007

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 436.3 ms, the parts add up to 436.2 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 436.1 ms, so the instrumented one is 1.00 x it (the syncs remove the overlap of launches with work); the forward in the split run took 439.4 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 245.74 | 56.3 |
| recurrence scan (all layers) | 113.74 | 26.1 |
| layers, rest (LN, projections, FFN) | 61.19 | 14.0 |
| local attention | 11.72 | 2.7 |
| model glue (residual adds, loops over layers) | 2.02 | 0.5 |
| decoder | 0.82 | 0.2 |
| loss | 0.35 | 0.1 |
| final LayerNorm | 0.31 | 0.1 |
| embedding | 0.29 | 0.1 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.19 | 0.3 | 0.5 |
| key conv | 3.24 | 0.7 | 1.3 |
| value projection | 2.66 | 0.6 | 1.1 |
| fp32 pin: casts, glue | 1.33 | 0.3 | 0.5 |
| address: no-grad matmul + top-k | 72.41 | 16.6 | 29.5 |
| address: recompute selected | 13.38 | 3.1 | 5.4 |
| write/read (rest of sparse_memory_read) | 138.90 | 31.8 | 56.5 |
| normaliser + gate | 4.71 | 1.1 | 1.9 |
| output projection | 5.03 | 1.2 | 2.0 |
| other (padding, residual) | 2.89 | 0.7 | 1.2 |
| **all heads** | 245.74 | 56.3 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.28 | 0.81 | 0.68 | 0.33 | 16.06 | 2.69 | 34.86 | 1.20 | 1.26 | 0.74 | 58.93 | 13.5 |
| head after layer 2 | 0.31 | 0.84 | 0.66 | 0.34 | 16.73 | 2.89 | 33.84 | 1.16 | 1.26 | 0.72 | 58.73 | 13.5 |
| head after layer 4 | 0.29 | 0.74 | 0.64 | 0.32 | 23.09 | 4.87 | 35.52 | 1.19 | 1.24 | 0.69 | 68.58 | 15.7 |
| head after layer 6 | 0.31 | 0.85 | 0.68 | 0.34 | 16.53 | 2.94 | 34.68 | 1.16 | 1.28 | 0.74 | 59.51 | 13.6 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 16.95 | 3.9 | 8.97 | 2.1 |
| 2 | 10.13 | 2.3 | 5.75 | 1.3 |
| 3 | 18.18 | 4.2 | 8.98 | 2.1 |
| 4 | 10.74 | 2.5 | 6.09 | 1.4 |
| 5 | 18.13 | 4.2 | 9.51 | 2.2 |
| 6 | 10.87 | 2.5 | 6.23 | 1.4 |
| 7 | 18.10 | 4.1 | 9.36 | 2.1 |
| 8 | 10.65 | 2.4 | 6.31 | 1.4 |

### Top operators by self device time, forward + backward, per step (all operators: 1007.2 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mul` | 2654 | 178.56 | 17.7 |
| `aten::mm` | 102 | 149.36 | 14.8 |
| `aten::div` | 318 | 72.41 | 7.2 |
| `aten::copy_` | 1646 | 62.48 | 6.2 |
| `aten::add_` | 1387 | 60.38 | 6.0 |
| `aten::sum` | 427 | 58.64 | 5.8 |
| `aten::bmm` | 112 | 54.04 | 5.4 |
| `aten::log` | 8 | 47.85 | 4.8 |
| `aten::add` | 1396 | 39.06 | 3.9 |
| `aten::fill_` | 1331 | 38.45 | 3.8 |
| `aten::addmm` | 27 | 36.87 | 3.7 |
| `aten::topk` | 16 | 28.16 | 2.8 |
| `aten::exp` | 8 | 24.22 | 2.4 |
| `aten::scatter_add_` | 120 | 21.67 | 2.2 |
| `aten::native_layer_norm_backward` | 22 | 20.16 | 2.0 |
| `aten::_efficient_attention_backward` | 1 | 17.66 | 1.8 |
| `aten::eq` | 36 | 11.72 | 1.2 |
| `aten::_index_put_impl_` | 4 | 10.30 | 1.0 |
| `aten::cumsum` | 32 | 8.52 | 0.8 |
| `aten::flip` | 32 | 8.01 | 0.8 |
| `aten::native_layer_norm` | 22 | 7.69 | 0.8 |
| `aten::convolution_backward` | 4 | 6.93 | 0.7 |
| `aten::gather` | 121 | 6.81 | 0.7 |
| `aten::sub` | 126 | 5.76 | 0.6 |
| `aten::neg` | 92 | 3.97 | 0.4 |

Kernel time in the profiled forward + backward: 1007.2 ms; the split run's forward + backward took 1014.5 ms, so the device was busy 99 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | 850 | 107.26 | 10.6 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 40 | 65.47 | 6.5 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 55.58 | 5.5 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::log_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 47.85 | 4.8 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, ` | 553 | 42.01 | 4.2 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::nativ` | 292 | 37.81 | 3.8 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul>,` | 1032 | 36.89 | 3.7 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 36.49 | 3.6 |
| `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_tt_align4>(cutlass_80_tensorop_s1688gemm_256x6` | 16 | 36.32 | 3.6 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 24.22 | 2.4 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 544 | 23.06 | 2.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 22.52 | 2.2 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>, fals` | 245 | 20.71 | 2.1 |
| `void at::native::vectorized_scatter_add_kernel<16, float, long>(float*, float const*, long const*, int, long, ` | 60 | 19.62 | 1.9 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 104 | 19.42 | 1.9 |

### Memory

Peak allocated 7.38 GiB, peak reserved 7.46 GiB over the timed training steps; the forward leaves 6.74 GiB of activations.

