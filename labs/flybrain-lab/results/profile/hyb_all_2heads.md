USDT:2026-10-04 09:11:15 3719991:3719991 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:11:18 3719991:3719991 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/hyb_all_2heads.json
# Step profile

## Arm `mingru_sparse_post` (14,998,788 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused, --head-amp, --scan, --sparse-claws; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 1.0199 | 15 steps, one sync at the end, data draw included; final loss 5.163 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0005 |
| split: forward | 0.3859 | forward + loss; median 0.3860, min 0.3857, max 0.3862 |
| split: backward | 0.6343 | zero_grad + backward; median 0.6344, min 0.6328, max 0.6356 |
| split: optimizer | 0.0033 | clip_grad_norm_ 0.0020 + optimizer step and schedule 0.0013 (median 0.0013, max 0.0015) |
| split: sum | 1.0238 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 163.9 | mean power 161 W (limit 150 W), max 62 C, contended False |

Warmup steps (s): 1.059, 1.020, 1.019, 1.020, 1.022

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 394.3 ms, the parts add up to 394.2 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 385.7 ms, so the instrumented one is 1.02 x it (the syncs remove the overlap of launches with work); the forward in the split run took 385.9 ms.

| part | ms | % of forward |
|:---|---:|---:|
| memory heads (all) | 146.30 | 37.1 |
| recurrence scan (all layers) | 139.80 | 35.5 |
| layers, rest (LN, projections, FFN) | 80.94 | 20.5 |
| local attention | 23.18 | 5.9 |
| model glue (residual adds, loops over layers) | 1.89 | 0.5 |
| decoder | 0.85 | 0.2 |
| final LayerNorm | 0.63 | 0.2 |
| loss | 0.36 | 0.1 |
| embedding | 0.32 | 0.1 |
| unattributed (outside the model call and the loss) | 0.05 | 0.0 |

Memory heads, all of them, per forward:

| part | ms | % of forward | % of the heads |
|:---|---:|---:|---:|
| LayerNorm | 1.12 | 0.3 | 0.8 |
| key conv | 3.08 | 0.8 | 2.1 |
| value projection | 2.53 | 0.6 | 1.7 |
| fp32 pin: casts, glue | 0.04 | 0.0 | 0.0 |
| address: no-grad matmul + top-k | 50.24 | 12.7 | 34.3 |
| address: recompute selected | 1.92 | 0.5 | 1.3 |
| write/read (rest of sparse_memory_read) | 78.74 | 20.0 | 53.8 |
| normaliser + gate | 3.41 | 0.9 | 2.3 |
| output projection | 2.59 | 0.7 | 1.8 |
| other (padding, residual) | 2.62 | 0.7 | 1.8 |
| **all heads** | 146.30 | 37.1 | 100.0 |

Each head, ms per forward:

| head | LN | key conv | value | pin | select | recompute | write/read | norm+gate | out | other | total | % of forward |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| top head (after the final LN) | 0.53 | 1.47 | 1.26 | 0.02 | 25.10 | 0.95 | 39.35 | 1.71 | 1.30 | 1.28 | 72.97 | 18.5 |
| head after layer 4 | 0.60 | 1.62 | 1.27 | 0.02 | 25.13 | 0.96 | 39.38 | 1.71 | 1.29 | 1.34 | 73.33 | 18.6 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 17.46 | 4.4 | 10.04 | 2.5 |
| 2 | 17.50 | 4.4 | 10.11 | 2.6 |
| 3 | 17.45 | 4.4 | 10.12 | 2.6 |
| 4 | 17.49 | 4.4 | 10.14 | 2.6 |
| 5 | 17.45 | 4.4 | 10.04 | 2.5 |
| 6 | 17.48 | 4.4 | 10.23 | 2.6 |
| 7 | 17.50 | 4.4 | 10.11 | 2.6 |
| 8 | 17.47 | 4.4 | 10.14 | 2.6 |

### Top operators by self device time, forward + backward, per step (all operators: 1048.8 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mul` | 1108 | 191.24 | 18.2 |
| `aten::mm` | 90 | 152.41 | 14.5 |
| `aten::copy_` | 1110 | 131.12 | 12.5 |
| `aten::add_` | 645 | 111.46 | 10.6 |
| `aten::fill_` | 789 | 90.22 | 8.6 |
| `aten::cat` | 194 | 81.29 | 7.8 |
| `aten::addmm` | 27 | 53.85 | 5.1 |
| `aten::add` | 298 | 36.80 | 3.5 |
| `aten::sum` | 221 | 30.06 | 2.9 |
| `aten::topk` | 8 | 24.26 | 2.3 |
| `aten::bmm` | 56 | 23.95 | 2.3 |
| `aten::native_layer_norm_backward` | 20 | 20.06 | 1.9 |
| `aten::_efficient_attention_backward` | 1 | 18.36 | 1.8 |
| `aten::scatter_add_` | 62 | 11.69 | 1.1 |
| `aten::native_layer_norm` | 20 | 10.01 | 1.0 |
| `aten::sub` | 68 | 8.75 | 0.8 |
| `aten::eq` | 18 | 6.35 | 0.6 |
| `aten::_efficient_attention_forward` | 1 | 6.13 | 0.6 |
| `aten::_index_put_impl_` | 2 | 5.66 | 0.5 |
| `aten::where` | 10 | 4.74 | 0.5 |
| `aten::gather` | 63 | 3.69 | 0.4 |
| `aten::div` | 140 | 3.57 | 0.3 |
| `aten::convolution_backward` | 2 | 3.46 | 0.3 |
| `aten::gelu_backward` | 8 | 3.11 | 0.3 |
| `aten::clamp_min` | 18 | 3.03 | 0.3 |

Kernel time in the profiled forward + backward: 1048.8 ms; the split run's forward + backward took 1020.2 ms, so the device was busy 103 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | 942 | 161.91 | 15.4 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, ` | 699 | 118.07 | 11.3 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 659 | 95.17 | 9.1 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>, fals` | 733 | 88.30 | 8.4 |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<4u>,` | 178 | 78.32 | 7.5 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 38 | 57.05 | 5.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 21 | 34.28 | 3.3 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 30.75 | 2.9 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 23.10 | 2.2 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<flo` | 123 | 21.80 | 2.1 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 86 | 20.14 | 1.9 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 19.84 | 1.9 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 19.68 | 1.9 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::nativ` | 146 | 19.23 | 1.8 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 18.36 | 1.8 |

### Memory

Peak allocated 9.26 GiB, peak reserved 9.32 GiB over the timed training steps; the forward leaves 8.72 GiB of activations.

