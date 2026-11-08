USDT:2026-10-04 09:02:36 3709966:3709966 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:02:40 3709966:3709966 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_headamp.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --head-amp; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.9909 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0004 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0017 |
| split: forward | 0.4143 | forward + loss; median 0.4148, min 0.4035, max 0.4253 |
| split: backward | 0.5701 | zero_grad + backward; median 0.5682, min 0.5650, max 0.5773 |
| split: optimizer | 0.0052 | clip_grad_norm_ 0.0021 + optimizer step and schedule 0.0031 (median 0.0032, max 0.0034) |
| split: sum | 0.9900 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 148.9 | mean power 150 W (limit 150 W), max 61 C, contended False |

Warmup steps (s): 1.111, 0.929, 0.991, 1.000, 0.999

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 415.6 ms, the parts add up to 415.5 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 414.9 ms, so the instrumented one is 1.00 x it (the syncs remove the overlap of launches with work); the forward in the split run took 414.3 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 227.64 | 54.8 |
| recurrence scan (all layers) | 113.16 | 27.2 |
| layers, rest (LN, projections, FFN) | 58.47 | 14.1 |
| local attention | 12.34 | 3.0 |
| model glue (residual adds, loops over layers) | 2.09 | 0.5 |
| decoder | 0.84 | 0.2 |
| loss | 0.35 | 0.1 |
| final LayerNorm | 0.32 | 0.1 |
| embedding | 0.30 | 0.1 |
| unattributed (outside the model call and the loss) | 0.09 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.24 | 0.3 | 0.5 |
| key conv | 3.34 | 0.8 | 1.5 |
| value projection | 2.85 | 0.7 | 1.3 |
| fp32 pin: casts, glue | 0.07 | 0.0 | 0.0 |
| address: no-grad matmul + top-k | 61.01 | 14.7 | 26.8 |
| address: recompute selected | 10.49 | 2.5 | 4.6 |
| write/read (rest of sparse_memory_read) | 136.05 | 32.7 | 59.8 |
| normaliser + gate | 4.60 | 1.1 | 2.0 |
| output projection | 5.02 | 1.2 | 2.2 |
| other (padding, residual) | 2.98 | 0.7 | 1.3 |
| **all heads** | 227.64 | 54.8 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.29 | 0.83 | 0.71 | 0.02 | 14.08 | 2.31 | 32.52 | 1.11 | 1.25 | 0.76 | 53.89 | 13.0 |
| head after layer 2 | 0.29 | 0.81 | 0.64 | 0.02 | 13.53 | 2.19 | 32.32 | 1.12 | 1.25 | 0.71 | 52.90 | 12.7 |
| head after layer 4 | 0.32 | 0.82 | 0.79 | 0.02 | 20.02 | 3.89 | 38.39 | 1.24 | 1.26 | 0.75 | 67.49 | 16.2 |
| head after layer 6 | 0.33 | 0.88 | 0.71 | 0.02 | 13.38 | 2.10 | 32.81 | 1.12 | 1.25 | 0.76 | 53.36 | 12.8 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 16.42 | 4.0 | 8.08 | 1.9 |
| 2 | 9.61 | 2.3 | 5.55 | 1.3 |
| 3 | 17.90 | 4.3 | 8.59 | 2.1 |
| 4 | 10.75 | 2.6 | 5.92 | 1.4 |
| 5 | 17.74 | 4.3 | 8.87 | 2.1 |
| 6 | 11.45 | 2.8 | 6.46 | 1.6 |
| 7 | 17.37 | 4.2 | 8.55 | 2.1 |
| 8 | 11.92 | 2.9 | 6.46 | 1.6 |

### Top operators by self device time, forward + backward, per step (all operators: 978.2 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mul` | 2654 | 178.03 | 18.2 |
| `aten::mm` | 102 | 148.93 | 15.2 |
| `aten::div` | 318 | 72.54 | 7.4 |
| `aten::copy_` | 1774 | 63.53 | 6.5 |
| `aten::add_` | 1387 | 57.07 | 5.8 |
| `aten::sum` | 427 | 56.47 | 5.8 |
| `aten::log` | 8 | 48.52 | 5.0 |
| `aten::addmm` | 27 | 41.02 | 4.2 |
| `aten::add` | 1396 | 38.82 | 4.0 |
| `aten::fill_` | 1331 | 35.80 | 3.7 |
| `aten::bmm` | 112 | 29.96 | 3.1 |
| `aten::topk` | 16 | 25.96 | 2.7 |
| `aten::exp` | 8 | 24.16 | 2.5 |
| `aten::scatter_add_` | 120 | 20.90 | 2.1 |
| `aten::native_layer_norm_backward` | 22 | 19.77 | 2.0 |
| `aten::_efficient_attention_backward` | 1 | 17.15 | 1.8 |
| `aten::eq` | 36 | 10.99 | 1.1 |
| `aten::_index_put_impl_` | 4 | 10.10 | 1.0 |
| `aten::sub` | 126 | 8.79 | 0.9 |
| `aten::cumsum` | 32 | 8.51 | 0.9 |
| `aten::native_layer_norm` | 22 | 8.27 | 0.8 |
| `aten::flip` | 32 | 7.99 | 0.8 |
| `aten::convolution_backward` | 4 | 6.93 | 0.7 |
| `aten::gather` | 121 | 6.53 | 0.7 |
| `aten::neg` | 92 | 4.00 | 0.4 |

Kernel time in the profiled forward + backward: 978.2 ms; the split run's forward + backward took 984.4 ms, so the device was busy 99 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | 786 | 84.26 | 8.6 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 40 | 65.73 | 6.7 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 55.30 | 5.7 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::log_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 48.52 | 5.0 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 168 | 42.73 | 4.4 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul>,` | 1032 | 36.96 | 3.8 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 36.00 | 3.7 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::nativ` | 292 | 35.93 | 3.7 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, ` | 465 | 33.03 | 3.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 24.43 | 2.5 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 24.16 | 2.5 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 544 | 22.97 | 2.3 |
| `void at::native::vectorized_scatter_add_kernel<16, float, long>(float*, float const*, long const*, int, long, ` | 60 | 18.90 | 1.9 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 18.33 | 1.9 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 17.95 | 1.8 |

### Memory

Peak allocated 7.24 GiB, peak reserved 7.30 GiB over the timed training steps; the forward leaves 6.65 GiB of activations.

