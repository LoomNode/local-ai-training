USDT:2026-10-04 09:06:02 3713025:3713025 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:06:06 3713025:3713025 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_scanamp.json
# Step profile

## Arm `mingru_sparse_post` (16,066,824 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --scan; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 1.2720 | 15 steps, one sync at the end, data draw included; final loss 5.135 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0007 |
| split: forward | 0.5995 | forward + loss; median 0.6013, min 0.5738, max 0.6027 |
| split: backward | 0.6683 | zero_grad + backward; median 0.6682, min 0.6641, max 0.6729 |
| split: optimizer | 0.0054 | clip_grad_norm_ 0.0021 + optimizer step and schedule 0.0033 (median 0.0033, max 0.0037) |
| split: sum | 1.2735 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 193.9 | mean power 152 W (limit 150 W), max 61 C, contended False |

Warmup steps (s): 1.284, 1.220, 1.270, 1.271, 1.272

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 565.8 ms, the parts add up to 565.7 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 572.4 ms, so the instrumented one is 0.99 x it (the syncs remove the overlap of launches with work); the forward in the split run took 599.5 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 324.65 | 57.4 |
| recurrence scan (all layers) | 136.05 | 24.0 |
| layers, rest (LN, projections, FFN) | 79.57 | 14.1 |
| local attention | 21.16 | 3.7 |
| model glue (residual adds, loops over layers) | 2.26 | 0.4 |
| decoder | 0.82 | 0.1 |
| final LayerNorm | 0.54 | 0.1 |
| loss | 0.35 | 0.1 |
| embedding | 0.29 | 0.1 |
| unattributed (outside the model call and the loss) | 0.08 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 2.05 | 0.4 | 0.6 |
| key conv | 5.60 | 1.0 | 1.7 |
| value projection | 4.63 | 0.8 | 1.4 |
| fp32 pin: casts, glue | 2.07 | 0.4 | 0.6 |
| address: no-grad matmul + top-k | 140.52 | 24.8 | 43.3 |
| address: recompute selected | 17.26 | 3.1 | 5.3 |
| write/read (rest of sparse_memory_read) | 138.54 | 24.5 | 42.7 |
| normaliser + gate | 4.42 | 0.8 | 1.4 |
| output projection | 5.01 | 0.9 | 1.5 |
| other (padding, residual) | 4.55 | 0.8 | 1.4 |
| **all heads** | 324.65 | 57.4 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.50 | 1.39 | 1.20 | 0.53 | 34.99 | 4.25 | 35.03 | 1.10 | 1.25 | 1.16 | 81.38 | 14.4 |
| head after layer 2 | 0.51 | 1.44 | 1.16 | 0.53 | 36.12 | 4.41 | 35.46 | 1.14 | 1.25 | 1.13 | 83.14 | 14.7 |
| head after layer 4 | 0.51 | 1.31 | 1.09 | 0.49 | 35.48 | 4.38 | 34.02 | 1.09 | 1.25 | 1.10 | 80.73 | 14.3 |
| head after layer 6 | 0.53 | 1.46 | 1.18 | 0.53 | 33.93 | 4.22 | 34.03 | 1.09 | 1.26 | 1.16 | 79.39 | 14.0 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 16.96 | 3.0 | 9.65 | 1.7 |
| 2 | 15.88 | 2.8 | 9.50 | 1.7 |
| 3 | 17.35 | 3.1 | 10.17 | 1.8 |
| 4 | 17.03 | 3.0 | 9.91 | 1.8 |
| 5 | 17.34 | 3.1 | 10.20 | 1.8 |
| 6 | 17.15 | 3.0 | 9.93 | 1.8 |
| 7 | 17.33 | 3.1 | 10.19 | 1.8 |
| 8 | 16.99 | 3.0 | 10.02 | 1.8 |

### Top operators by self device time, forward + backward, per step (all operators: 1264.2 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mul` | 1630 | 261.48 | 20.7 |
| `aten::mm` | 102 | 165.16 | 13.1 |
| `aten::copy_` | 1206 | 129.08 | 10.2 |
| `aten::bmm` | 112 | 126.24 | 10.0 |
| `aten::cat` | 202 | 81.00 | 6.4 |
| `aten::add_` | 811 | 73.53 | 5.8 |
| `aten::fill_` | 923 | 62.25 | 4.9 |
| `aten::sum` | 419 | 57.47 | 4.5 |
| `aten::addmm` | 27 | 52.65 | 4.2 |
| `aten::topk` | 16 | 46.54 | 3.7 |
| `aten::add` | 476 | 44.85 | 3.5 |
| `aten::scatter_add_` | 120 | 21.97 | 1.7 |
| `aten::native_layer_norm_backward` | 22 | 21.74 | 1.7 |
| `aten::_efficient_attention_backward` | 1 | 18.36 | 1.5 |
| `aten::eq` | 36 | 12.20 | 1.0 |
| `aten::native_layer_norm` | 22 | 10.82 | 0.9 |
| `aten::_index_put_impl_` | 4 | 10.24 | 0.8 |
| `aten::sub` | 126 | 8.85 | 0.7 |
| `aten::div` | 278 | 7.04 | 0.6 |
| `aten::convolution_backward` | 4 | 6.93 | 0.5 |
| `aten::gather` | 121 | 6.92 | 0.5 |
| `aten::_efficient_attention_forward` | 1 | 6.36 | 0.5 |
| `aten::_conv_depthwise2d` | 4 | 5.58 | 0.4 |
| `aten::clamp_min` | 28 | 5.18 | 0.4 |
| `aten::where` | 10 | 4.66 | 0.4 |

Kernel time in the profiled forward + backward: 1264.2 ms; the split run's forward + backward took 1267.7 ms, so the device was busy 100 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::` | 544 | 113.64 | 9.0 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | 850 | 111.91 | 8.9 |
| `ampere_sgemm_128x128_tt` | 16 | 92.63 | 7.3 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 632 | 85.24 | 6.7 |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<2u>,` | 176 | 77.67 | 6.1 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 42 | 60.72 | 4.8 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*` | 462 | 45.41 | 3.6 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, ` | 553 | 43.41 | 3.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 25 | 40.75 | 3.2 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1u` | 640 | 40.42 | 3.2 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::nativ` | 292 | 38.82 | 3.1 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 30.04 | 2.4 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 22.61 | 1.8 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>, fals` | 245 | 21.45 | 1.7 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 104 | 20.61 | 1.6 |

### Memory

Peak allocated 8.09 GiB, peak reserved 8.15 GiB over the timed training steps; the forward leaves 7.46 GiB of activations.

