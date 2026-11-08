USDT:2026-10-04 09:00:14 3707333:3707333 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:00:15 3707333:3707333 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/tf_base.json
# Step profile

## Arm `transformer` (17,085,696 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: none; batch 8 x 2048; warmup 5, measured 20 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.3475 | 20 steps, one sync at the end, data draw included; final loss 5.060 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0002, min 0.0002, max 0.0004 |
| split: forward | 0.1202 | forward + loss; median 0.1204, min 0.1160, max 0.1232 |
| split: backward | 0.2228 | zero_grad + backward; median 0.2228, min 0.2186, max 0.2278 |
| split: optimizer | 0.0040 | clip_grad_norm_ 0.0013 + optimizer step and schedule 0.0028 (median 0.0027, max 0.0034) |
| split: sum | 0.3472 | sync at every boundary: 1.00 x the plain step |
| energy per step (J) | 52.0 | mean power 150 W (limit 150 W), max 60 C, contended False |

Warmup steps (s): 0.533, 0.280, 0.360, 0.357, 0.322

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 115.0 ms, the parts add up to 115.0 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 115.3 ms, so the instrumented one is 1.00 x it (the syncs remove the overlap of launches with work); the forward in the split run took 120.2 ms.

| part | ms | % of forward |
|:---|---:|---:|
| transformer attention | 74.67 | 64.9 |
| transformer FFN | 27.96 | 24.3 |
| transformer LayerNorm | 6.10 | 5.3 |
| model glue (residual adds, loops over layers) | 4.81 | 4.2 |
| decoder | 0.55 | 0.5 |
| final LayerNorm | 0.37 | 0.3 |
| embedding | 0.25 | 0.2 |
| loss | 0.24 | 0.2 |
| unattributed (outside the model call and the loss) | 0.05 | 0.0 |

Layers, per forward:

| layer | attention ms | attention % | FFN ms | FFN % | LayerNorm ms |
|:---|---:|---:|---:|---:|---:|
| 1 | 9.48 | 8.2 | 3.55 | 3.1 | 0.79 |
| 2 | 9.14 | 8.0 | 3.43 | 3.0 | 0.78 |
| 3 | 9.01 | 7.8 | 3.37 | 2.9 | 0.73 |
| 4 | 9.10 | 7.9 | 3.42 | 3.0 | 0.74 |
| 5 | 9.40 | 8.2 | 3.50 | 3.0 | 0.75 |
| 6 | 9.43 | 8.2 | 3.60 | 3.1 | 0.77 |
| 7 | 9.72 | 8.5 | 3.58 | 3.1 | 0.78 |
| 8 | 9.39 | 8.2 | 3.50 | 3.0 | 0.76 |

### Top operators by self device time, forward + backward, per step (all operators: 337.1 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 66 | 102.42 | 30.4 |
| `aten::_flash_attention_backward` | 8 | 64.39 | 19.1 |
| `aten::addmm` | 33 | 50.23 | 14.9 |
| `aten::_flash_attention_forward` | 8 | 25.32 | 7.5 |
| `aten::copy_` | 338 | 23.62 | 7.0 |
| `aten::mul` | 144 | 18.20 | 5.4 |
| `aten::cat` | 26 | 12.33 | 3.7 |
| `aten::native_layer_norm_backward` | 17 | 11.30 | 3.4 |
| `aten::native_layer_norm` | 17 | 5.55 | 1.6 |
| `aten::add` | 32 | 4.90 | 1.5 |
| `aten::add_` | 64 | 4.80 | 1.4 |
| `aten::sum` | 33 | 4.55 | 1.3 |
| `aten::gelu_backward` | 8 | 2.11 | 0.6 |
| `aten::fill_` | 51 | 1.66 | 0.5 |
| `aten::gelu` | 8 | 1.51 | 0.4 |
| `aten::neg` | 32 | 1.42 | 0.4 |
| `aten::sub` | 16 | 1.27 | 0.4 |
| `aten::embedding_dense_backward` | 1 | 0.29 | 0.1 |
| `aten::pow` | 32 | 0.22 | 0.1 |
| `aten::sin` | 16 | 0.18 | 0.1 |
| `aten::cos` | 16 | 0.18 | 0.1 |
| `aten::arange` | 66 | 0.14 | 0.0 |
| `aten::nll_loss_backward` | 1 | 0.11 | 0.0 |
| `aten::gather` | 1 | 0.09 | 0.0 |
| `aten::div` | 16 | 0.08 | 0.0 |

Kernel time in the profiled forward + backward: 337.1 ms; the split run's forward + backward took 342.9 ms, so the device was busy 98 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<32, 128, 128, 8, 4, 4` | 8 | 62.71 | 18.6 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 16 | 30.89 | 9.2 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 25 | 28.74 | 8.5 |
| `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<32, 128, 128, 4, false, false, cutlass::bfloat16_` | 8 | 25.32 | 7.5 |
| `ampere_bf16_s1688gemm_bf16_128x256_ldg8_f2f_stages_32x1_nn` | 8 | 20.38 | 6.0 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 17 | 19.35 | 5.7 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_nt_align8>(cutlass_80_tensoro` | 8 | 17.48 | 5.2 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 13.12 | 3.9 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 12.99 | 3.9 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 64 | 11.75 | 3.5 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 56 | 10.20 | 3.0 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 8 | 8.69 | 2.6 |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<2u>,` | 8 | 7.47 | 2.2 |
| `void at::native::(anonymous namespace)::GammaBetaBackwardCUDAKernelTemplate<float, float, 32u, 32u, 256u, fals` | 17 | 7.41 | 2.2 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 100 | 6.87 | 2.0 |

### Memory

Peak allocated 2.56 GiB, peak reserved 2.67 GiB over the timed training steps; the forward leaves 2.23 GiB of activations.

