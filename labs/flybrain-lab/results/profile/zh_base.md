USDT:2026-10-04 09:00:44 3708122:3708122 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:00:45 3708122:3708122 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/zh_base.json
# Step profile

## Arm `mingru` (13,930,752 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: none; batch 8 x 2048; warmup 5, measured 20 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.4888 | 20 steps, one sync at the end, data draw included; final loss 5.028 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0006 |
| split: forward | 0.1500 | forward + loss; median 0.1493, min 0.1472, max 0.1555 |
| split: backward | 0.3359 | zero_grad + backward; median 0.3363, min 0.3293, max 0.3406 |
| split: optimizer | 0.0046 | clip_grad_norm_ 0.0017 + optimizer step and schedule 0.0029 (median 0.0031, max 0.0034) |
| split: sum | 0.4908 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 73.3 | mean power 150 W (limit 150 W), max 60 C, contended False |

Warmup steps (s): 0.613, 0.442, 0.501, 0.474, 0.499

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 140.8 ms, the parts add up to 140.8 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 141.6 ms, so the instrumented one is 0.99 x it (the syncs remove the overlap of launches with work); the forward in the split run took 150.0 ms.

| part | ms | % of forward |
|:---|---:|---:|
| recurrence scan (all layers) | 79.37 | 56.4 |
| layers, rest (LN, projections, FFN) | 45.89 | 32.6 |
| local attention | 13.69 | 9.7 |
| model glue (residual adds, loops over layers) | 0.66 | 0.5 |
| decoder | 0.43 | 0.3 |
| final LayerNorm | 0.29 | 0.2 |
| embedding | 0.22 | 0.2 |
| loss | 0.19 | 0.1 |
| unattributed (outside the model call and the loss) | 0.04 | 0.0 |

Layers, per forward:

| layer | scan ms | scan % | rest of the layer ms | rest % |
|:---|---:|---:|---:|---:|
| 1 | 8.99 | 6.4 | 4.93 | 3.5 |
| 2 | 8.82 | 6.3 | 5.08 | 3.6 |
| 3 | 8.95 | 6.4 | 5.31 | 3.8 |
| 4 | 9.11 | 6.5 | 6.05 | 4.3 |
| 5 | 11.52 | 8.2 | 7.07 | 5.0 |
| 6 | 13.04 | 9.3 | 6.82 | 4.8 |
| 7 | 9.66 | 6.9 | 5.30 | 3.8 |
| 8 | 9.29 | 6.6 | 5.33 | 3.8 |

### Top operators by self device time, forward + backward, per step (all operators: 472.2 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 78 | 109.20 | 23.1 |
| `aten::div` | 42 | 59.87 | 12.7 |
| `aten::mul` | 1610 | 38.45 | 8.1 |
| `aten::log` | 8 | 37.59 | 8.0 |
| `aten::add_` | 1055 | 35.45 | 7.5 |
| `aten::copy_` | 1294 | 34.84 | 7.4 |
| `aten::addmm` | 27 | 28.30 | 6.0 |
| `aten::exp` | 8 | 19.35 | 4.1 |
| `aten::fill_` | 1059 | 16.96 | 3.6 |
| `aten::add` | 1040 | 16.76 | 3.5 |
| `aten::_efficient_attention_backward` | 1 | 16.32 | 3.5 |
| `aten::native_layer_norm_backward` | 18 | 14.92 | 3.2 |
| `aten::cumsum` | 32 | 7.73 | 1.6 |
| `aten::flip` | 32 | 7.55 | 1.6 |
| `aten::sum` | 35 | 6.76 | 1.4 |
| `aten::native_layer_norm` | 18 | 4.78 | 1.0 |
| `aten::where` | 10 | 2.98 | 0.6 |
| `aten::_efficient_attention_forward` | 1 | 2.88 | 0.6 |
| `aten::neg` | 20 | 2.66 | 0.6 |
| `aten::gelu_backward` | 8 | 2.62 | 0.6 |
| `aten::cat` | 18 | 1.59 | 0.3 |
| `aten::gelu` | 8 | 1.19 | 0.3 |
| `aten::sigmoid_backward` | 8 | 0.68 | 0.1 |
| `aten::sub` | 10 | 0.55 | 0.1 |
| `aten::sigmoid` | 8 | 0.52 | 0.1 |

Kernel time in the profiled forward + backward: 472.2 ms; the split run's forward + backward took 486.0 ms, so the device was busy 97 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 40 | 59.86 | 12.7 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 34 | 42.31 | 9.0 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::log_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 37.59 | 8.0 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul>,` | 1032 | 34.95 | 7.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 17 | 21.77 | 4.6 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native` | 544 | 21.69 | 4.6 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambd` | 8 | 19.35 | 4.1 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 16.91 | 3.6 |
| `fmha_cutlassB_bf16_aligned_64x64_k64_seqaligned_sm80(PyTorchMemEffAttention::AttentionBackwardKernel<cutlass::` | 1 | 16.32 | 3.5 |
| `void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<double>, std::array<char*, 1ul>, fal` | 1034 | 16.28 | 3.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 18 | 16.13 | 3.4 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 15.96 | 3.4 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<doubl` | 1024 | 12.67 | 2.7 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 9 | 12.17 | 2.6 |
| `void at::native::(anonymous namespace)::GammaBetaBackwardCUDAKernelTemplate<float, float, 32u, 32u, 256u, fals` | 18 | 9.80 | 2.1 |

### Memory

Peak allocated 4.62 GiB, peak reserved 4.82 GiB over the timed training steps; the forward leaves 4.28 GiB of activations.

