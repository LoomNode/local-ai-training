USDT:2026-10-04 09:12:20 3722002:3722002 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:12:22 3722002:3722002 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/zh_scan32_fused.json
# Step profile

## Arm `mingru` (13,930,752 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused, --scan; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.7702 | 15 steps, one sync at the end, data draw included; final loss 5.238 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0006 |
| split: forward | 0.2429 | forward + loss; median 0.2429, min 0.2427, max 0.2436 |
| split: backward | 0.5273 | zero_grad + backward; median 0.5271, min 0.5266, max 0.5281 |
| split: optimizer | 0.0027 | clip_grad_norm_ 0.0015 + optimizer step and schedule 0.0012 (median 0.0012, max 0.0013) |
| split: sum | 0.7732 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 126.8 | mean power 165 W (limit 150 W), max 62 C, contended False |

Warmup steps (s): 0.805, 0.771, 0.771, 0.770, 0.770

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 246.0 ms, the parts add up to 246.0 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 242.9 ms, so the instrumented one is 1.01 x it (the syncs remove the overlap of launches with work); the forward in the split run took 242.9 ms.

| part | ms | % of forward |
|:---|---:|---:|
| recurrence scan (all layers) | 139.71 | 56.8 |
| layers, rest (LN, projections, FFN) | 80.09 | 32.6 |
| local attention | 23.17 | 9.4 |
| model glue (residual adds, loops over layers) | 0.90 | 0.4 |
| decoder | 0.84 | 0.3 |
| final LayerNorm | 0.57 | 0.2 |
| loss | 0.36 | 0.1 |
| embedding | 0.31 | 0.1 |
| unattributed (outside the model call and the loss) | 0.05 | 0.0 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 17.49 | 7.1 | 10.01 | 4.1 |
| 2 | 17.44 | 7.1 | 10.01 | 4.1 |
| 3 | 17.46 | 7.1 | 10.00 | 4.1 |
| 4 | 17.44 | 7.1 | 10.04 | 4.1 |
| 5 | 17.46 | 7.1 | 10.00 | 4.1 |
| 6 | 17.54 | 7.1 | 10.04 | 4.1 |
| 7 | 17.45 | 7.1 | 9.99 | 4.1 |
| 8 | 17.43 | 7.1 | 10.00 | 4.1 |

### Top operators by self device time, forward + backward, per step (all operators: 796.5 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 78 | 137.10 | 17.2 |
| `aten::mul` | 586 | 125.08 | 15.7 |
| `aten::copy_` | 870 | 116.88 | 14.7 |
| `aten::add_` | 479 | 101.24 | 12.7 |
| `aten::cat` | 186 | 82.18 | 10.3 |
| `aten::fill_` | 651 | 80.18 | 10.1 |
| `aten::addmm` | 27 | 54.56 | 6.9 |
| `aten::add` | 120 | 26.60 | 3.3 |
| `aten::_efficient_attention_backward` | 1 | 18.37 | 2.3 |
| `aten::native_layer_norm_backward` | 18 | 18.04 | 2.3 |
| `aten::native_layer_norm` | 18 | 9.12 | 1.1 |
| `aten::_efficient_attention_forward` | 1 | 6.14 | 0.8 |
| `aten::sum` | 27 | 4.89 | 0.6 |
| `aten::where` | 10 | 4.74 | 0.6 |
| `aten::gelu_backward` | 8 | 3.11 | 0.4 |
| `aten::gelu` | 8 | 2.31 | 0.3 |
| `aten::sigmoid` | 8 | 0.91 | 0.1 |
| `aten::sigmoid_backward` | 8 | 0.88 | 0.1 |
| `aten::neg` | 12 | 0.86 | 0.1 |
| `aten::sub` | 10 | 0.86 | 0.1 |
| `aten::clamp_min` | 8 | 0.67 | 0.1 |
| `aten::gt` | 8 | 0.50 | 0.1 |
| `aten::embedding_dense_backward` | 1 | 0.45 | 0.1 |
| `aten::nll_loss_backward` | 1 | 0.17 | 0.0 |
| `aten::gather` | 1 | 0.14 | 0.0 |

Kernel time in the profiled forward + backward: 796.5 ms; the split run's forward + backward took 770.2 ms, so the device was busy 103 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float` | 554 | 120.35 | 15.1 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>, ` | 485 | 104.05 | 13.1 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 621 | 92.65 | 11.6 |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<4u>,` | 176 | 79.59 | 10.0 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul>, fals` | 629 | 79.57 | 10.0 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 34 | 52.66 | 6.6 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 31.17 | 3.9 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 17 | 27.75 | 3.5 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 23.40 | 2.9 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 19.84 | 2.5 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 19.68 | 2.5 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<flo` | 89 | 19.39 | 2.4 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 18.37 | 2.3 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 109 | 12.84 | 1.6 |
| `void at::native::(anonymous namespace)::GammaBetaBackwardCUDAKernelTemplate<float, float, 32u, 32u, 256u, fals` | 18 | 11.92 | 1.5 |

### Memory

Peak allocated 8.05 GiB, peak reserved 8.35 GiB over the timed training steps; the forward leaves 7.75 GiB of activations.

