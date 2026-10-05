USDT:2026-10-04 09:07:00 3714115:3714115 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:07:03 3714115:3714115 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_detach.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --detach-state; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.9853 | 15 steps, one sync at the end, data draw included; final loss 5.153 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0006 |
| split: forward | 0.4645 | forward + loss; median 0.4629, min 0.4573, max 0.4786 |
| split: backward | 0.5166 | zero_grad + backward; median 0.5162, min 0.5055, max 0.5217 |
| split: optimizer | 0.0050 | clip_grad_norm_ 0.0020 + optimizer step and schedule 0.0030 (median 0.0030, max 0.0033) |
| split: sum | 0.9864 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 147.8 | mean power 150 W (limit 150 W), max 61 C, contended False |

Warmup steps (s): 1.137, 0.934, 0.979, 0.990, 0.983

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 475.6 ms, the parts add up to 475.5 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 469.8 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 464.5 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 288.42 | 60.6 |
| recurrence scan (all layers) | 111.78 | 23.5 |
| layers, rest (LN, projections, FFN) | 59.61 | 12.5 |
| local attention | 11.83 | 2.5 |
| model glue (residual adds, loops over layers) | 2.07 | 0.4 |
| decoder | 0.82 | 0.2 |
| loss | 0.35 | 0.1 |
| final LayerNorm | 0.32 | 0.1 |
| embedding | 0.30 | 0.1 |
| unattributed (outside the model call and the loss) | 0.08 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.18 | 0.2 | 0.4 |
| key conv | 3.20 | 0.7 | 1.1 |
| value projection | 2.63 | 0.6 | 0.9 |
| fp32 pin: casts, glue | 1.36 | 0.3 | 0.5 |
| address: no-grad matmul + top-k | 98.16 | 20.6 | 34.0 |
| address: recompute selected | 17.07 | 3.6 | 5.9 |
| write/read (rest of sparse_memory_read) | 151.82 | 31.9 | 52.6 |
| normaliser + gate | 5.12 | 1.1 | 1.8 |
| output projection | 5.01 | 1.1 | 1.7 |
| other (padding, residual) | 2.87 | 0.6 | 1.0 |
| **all heads** | 288.42 | 60.6 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.28 | 0.80 | 0.68 | 0.34 | 22.54 | 3.74 | 37.96 | 1.29 | 1.26 | 0.73 | 69.61 | 14.6 |
| head after layer 2 | 0.27 | 0.76 | 0.61 | 0.35 | 24.01 | 4.16 | 37.66 | 1.24 | 1.26 | 0.70 | 71.02 | 14.9 |
| head after layer 4 | 0.30 | 0.77 | 0.66 | 0.32 | 29.84 | 5.05 | 38.09 | 1.30 | 1.24 | 0.69 | 78.27 | 16.5 |
| head after layer 6 | 0.33 | 0.86 | 0.69 | 0.34 | 21.77 | 4.12 | 38.11 | 1.30 | 1.25 | 0.75 | 69.51 | 14.6 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 16.94 | 3.6 | 8.62 | 1.8 |
| 2 | 9.11 | 1.9 | 5.11 | 1.1 |
| 3 | 17.86 | 3.8 | 9.03 | 1.9 |
| 4 | 10.61 | 2.2 | 6.02 | 1.3 |
| 5 | 18.08 | 3.8 | 9.03 | 1.9 |
| 6 | 10.65 | 2.2 | 6.29 | 1.3 |
| 7 | 18.00 | 3.8 | 9.26 | 1.9 |
| 8 | 10.54 | 2.2 | 6.25 | 1.3 |

### Top operators by self device time, forward + backward, per step (all operators: 972.2 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mul` | 2466 | 156.09 | 16.1 |
| `aten::mm` | 102 | 141.58 | 14.6 |
| `aten::bmm` | 112 | 92.39 | 9.5 |
| `aten::div` | 318 | 69.83 | 7.2 |
| `aten::copy_` | 1646 | 60.77 | 6.3 |
| `aten::sum` | 371 | 54.22 | 5.6 |
| `aten::log` | 8 | 48.63 | 5.0 |
| `aten::add_` | 1311 | 47.25 | 4.9 |
| `aten::add` | 1340 | 37.20 | 3.8 |
| `aten::addmm` | 27 | 35.66 | 3.7 |
| `aten::topk` | 16 | 32.69 | 3.4 |
| `aten::fill_` | 1275 | 30.46 | 3.1 |
| `aten::exp` | 8 | 24.64 | 2.5 |
| `aten::native_layer_norm_backward` | 22 | 19.25 | 2.0 |
| `aten::_efficient_attention_backward` | 1 | 15.82 | 1.6 |
| `aten::eq` | 36 | 12.34 | 1.3 |
| `aten::scatter_add_` | 64 | 11.74 | 1.2 |
| `aten::_index_put_impl_` | 4 | 8.79 | 0.9 |
| `aten::cumsum` | 32 | 8.24 | 0.8 |
| `aten::flip` | 32 | 7.64 | 0.8 |
| `aten::native_layer_norm` | 22 | 7.43 | 0.8 |
| `aten::convolution_backward` | 4 | 6.49 | 0.7 |
| `aten::sub` | 126 | 6.49 | 0.7 |
| `aten::clamp_min` | 28 | 3.84 | 0.4 |
| `aten::neg` | 92 | 3.79 | 0.4 |

Kernel time in the profiled forward + backward: 972.2 ms; the split run's forward + backward took 981.1 ms, so the device was busy 99 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | 662 | 87.01 | 9.0 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 40 | 62.96 | 6.5 |
| `ampere_sgemm_128x128_tt` | 16 | 59.62 | 6.1 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 52.73 | 5.4 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::log_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 48.63 | 5.0 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::nativ` | 264 | 36.63 | 3.8 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul>,` | 1032 | 34.53 | 3.6 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 34.34 | 3.5 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, ` | 421 | 30.11 | 3.1 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 24.64 | 2.5 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 21.71 | 2.2 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 544 | 21.70 | 2.2 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 104 | 18.90 | 1.9 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 17.06 | 1.8 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 17.05 | 1.8 |

### Memory

Peak allocated 6.93 GiB, peak reserved 7.00 GiB over the timed training steps; the forward leaves 6.30 GiB of activations.

