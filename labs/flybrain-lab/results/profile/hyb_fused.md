USDT:2026-10-04 09:08:01 3715829:3715829 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:08:04 3715829:3715829 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_fused.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --fused; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 1.0561 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0007 |
| split: forward | 0.4663 | forward + loss; median 0.4677, min 0.4561, max 0.4742 |
| split: backward | 0.5898 | zero_grad + backward; median 0.5897, min 0.5835, max 0.5946 |
| split: optimizer | 0.0033 | clip_grad_norm_ 0.0021 + optimizer step and schedule 0.0012 (median 0.0012, max 0.0013) |
| split: sum | 1.0596 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 158.6 | mean power 150 W (limit 150 W), max 61 C, contended False |

Warmup steps (s): 1.189, 1.012, 1.055, 1.057, 1.060

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 475.1 ms, the parts add up to 475.0 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 466.2 ms, so the instrumented one is 1.02 x it (the syncs remove the overlap of launches with work); the forward in the split run took 466.3 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 288.91 | 60.8 |
| recurrence scan (all layers) | 112.36 | 23.7 |
| layers, rest (LN, projections, FFN) | 57.92 | 12.2 |
| local attention | 11.95 | 2.5 |
| model glue (residual adds, loops over layers) | 2.09 | 0.4 |
| decoder | 0.84 | 0.2 |
| loss | 0.35 | 0.1 |
| final LayerNorm | 0.30 | 0.1 |
| embedding | 0.28 | 0.1 |
| unattributed (outside the model call and the loss) | 0.07 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.16 | 0.2 | 0.4 |
| key conv | 3.18 | 0.7 | 1.1 |
| value projection | 2.63 | 0.6 | 0.9 |
| fp32 pin: casts, glue | 1.33 | 0.3 | 0.5 |
| address: no-grad matmul + top-k | 97.90 | 20.6 | 33.9 |
| address: recompute selected | 17.13 | 3.6 | 5.9 |
| write/read (rest of sparse_memory_read) | 152.24 | 32.0 | 52.7 |
| normaliser + gate | 5.46 | 1.1 | 1.9 |
| output projection | 5.06 | 1.1 | 1.8 |
| other (padding, residual) | 2.84 | 0.6 | 1.0 |
| **all heads** | 288.91 | 60.8 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.27 | 0.77 | 0.65 | 0.33 | 23.56 | 4.43 | 38.48 | 1.39 | 1.27 | 0.73 | 71.88 | 15.1 |
| head after layer 2 | 0.27 | 0.77 | 0.63 | 0.32 | 23.48 | 3.74 | 37.10 | 1.33 | 1.27 | 0.68 | 69.60 | 14.7 |
| head after layer 4 | 0.30 | 0.78 | 0.68 | 0.33 | 28.31 | 5.11 | 38.57 | 1.38 | 1.26 | 0.70 | 77.42 | 16.3 |
| head after layer 6 | 0.31 | 0.86 | 0.68 | 0.34 | 22.55 | 3.85 | 38.09 | 1.36 | 1.26 | 0.73 | 70.02 | 14.7 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 16.85 | 3.5 | 7.76 | 1.6 |
| 2 | 9.57 | 2.0 | 5.27 | 1.1 |
| 3 | 17.96 | 3.8 | 9.17 | 1.9 |
| 4 | 10.48 | 2.2 | 5.97 | 1.3 |
| 5 | 17.96 | 3.8 | 9.05 | 1.9 |
| 6 | 10.79 | 2.3 | 6.15 | 1.3 |
| 7 | 18.02 | 3.8 | 8.61 | 1.8 |
| 8 | 10.73 | 2.3 | 5.94 | 1.3 |

### Top operators by self device time, forward + backward, per step (all operators: 1041.9 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mul` | 2654 | 181.82 | 17.5 |
| `aten::mm` | 102 | 149.03 | 14.3 |
| `aten::bmm` | 112 | 85.03 | 8.2 |
| `aten::div` | 318 | 71.86 | 6.9 |
| `aten::copy_` | 1646 | 62.27 | 6.0 |
| `aten::add_` | 1387 | 60.35 | 5.8 |
| `aten::sum` | 427 | 59.99 | 5.8 |
| `aten::log` | 8 | 47.81 | 4.6 |
| `aten::add` | 1396 | 39.03 | 3.7 |
| `aten::fill_` | 1331 | 38.76 | 3.7 |
| `aten::addmm` | 27 | 35.05 | 3.4 |
| `aten::topk` | 16 | 29.99 | 2.9 |
| `aten::exp` | 8 | 24.02 | 2.3 |
| `aten::scatter_add_` | 120 | 22.05 | 2.1 |
| `aten::native_layer_norm_backward` | 22 | 20.23 | 1.9 |
| `aten::_efficient_attention_backward` | 1 | 16.83 | 1.6 |
| `aten::eq` | 36 | 12.46 | 1.2 |
| `aten::_index_put_impl_` | 4 | 10.25 | 1.0 |
| `aten::cumsum` | 32 | 8.44 | 0.8 |
| `aten::flip` | 32 | 7.96 | 0.8 |
| `aten::native_layer_norm` | 22 | 7.34 | 0.7 |
| `aten::convolution_backward` | 4 | 6.93 | 0.7 |
| `aten::gather` | 121 | 6.91 | 0.7 |
| `aten::sub` | 126 | 5.84 | 0.6 |
| `aten::neg` | 92 | 3.97 | 0.4 |

Kernel time in the profiled forward + backward: 1041.9 ms; the split run's forward + backward took 1056.0 ms, so the device was busy 99 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | 850 | 110.58 | 10.6 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 40 | 64.89 | 6.2 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 55.63 | 5.3 |
| `ampere_sgemm_128x128_tt` | 16 | 51.80 | 5.0 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::log_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 47.81 | 4.6 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, ` | 553 | 42.44 | 4.1 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::nativ` | 292 | 39.02 | 3.7 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul>,` | 1032 | 36.75 | 3.5 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 36.04 | 3.5 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 24.02 | 2.3 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 544 | 22.97 | 2.2 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 21.54 | 2.1 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>, fals` | 245 | 21.04 | 2.0 |
| `void at::native::vectorized_scatter_add_kernel<16, float, long>(float*, float const*, long const*, int, long, ` | 60 | 19.96 | 1.9 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 104 | 19.26 | 1.8 |

### Memory

Peak allocated 7.38 GiB, peak reserved 7.46 GiB over the timed training steps; the forward leaves 6.74 GiB of activations.

