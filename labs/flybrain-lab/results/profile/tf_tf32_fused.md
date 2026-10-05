USDT:2026-10-04 09:11:43 3721312:3721312 SyncActivityProfilerHandler.cpp:39] profiler_start
USDT:2026-10-04 09:11:44 3721312:3721312 SyncActivityProfilerHandler.cpp:46] profiler_stop
wrote results/profile/tf_tf32_fused.json
# Step profile

## Arm `transformer` (17,085,696 trainable parameters)

Device `cuda:0` (NVIDIA GeForce RTX 3090), torch 2.14.0+cu130; autocast bf16; speed switches: --tf32, --fused; batch 8 x 2048; warmup 5, measured 15 steps.

### Seconds per step

| measure | seconds | note |
|:---|---:|:---|
| **plain step, no instrumentation** | 0.3575 | 15 steps, one sync at the end, data draw included; final loss 5.285 |
| split: data | 0.0003 | batch draw on CPU + copy to the device; median 0.0003, min 0.0002, max 0.0006 |
| split: forward | 0.1180 | forward + loss; median 0.1212, min 0.0575, max 0.1281 |
| split: backward | 0.2316 | zero_grad + backward; median 0.2330, min 0.2178, max 0.2434 |
| split: optimizer | 0.0024 | clip_grad_norm_ 0.0013 + optimizer step and schedule 0.0011 (median 0.0010, max 0.0014) |
| split: sum | 0.3523 | sync at every boundary: 0.99 x the plain step |
| energy per step (J) | 53.5 | mean power 150 W (limit 150 W), max 62 C, contended False |

Warmup steps (s): 0.459, 0.285, 0.415, 0.328, 0.351

### Forward breakdown (instrumented: synced around every part, eager)

Instrumented forward + loss 118.2 ms, the parts add up to 118.2 ms. The same forward + loss without instrumentation (eager, synced at its two ends) took 118.9 ms, so the instrumented one is 0.99 x it (the syncs remove the overlap of launches with work); the forward in the split run took 118.0 ms.

| part | ms | % of forward |
|:---|---:|---:|
| transformer attention | 77.18 | 65.3 |
| transformer FFN | 28.84 | 24.4 |
| transformer LayerNorm | 6.12 | 5.2 |
| model glue (residual adds, loops over layers) | 4.59 | 3.9 |
| decoder | 0.56 | 0.5 |
| final LayerNorm | 0.36 | 0.3 |
| embedding | 0.27 | 0.2 |
| loss | 0.24 | 0.2 |
| unattributed (outside the model call and the loss) | 0.05 | 0.0 |

Layers, per forward:

| layer | attention ms | attention % | FFN ms | FFN % | LayerNorm ms |
|:---|---:|---:|---:|---:|---:|
| 1 | 9.81 | 8.3 | 3.64 | 3.1 | 0.81 |
| 2 | 9.53 | 8.1 | 3.49 | 3.0 | 0.76 |
| 3 | 9.34 | 7.9 | 3.55 | 3.0 | 0.74 |
| 4 | 9.72 | 8.2 | 3.65 | 3.1 | 0.76 |
| 5 | 9.69 | 8.2 | 3.64 | 3.1 | 0.76 |
| 6 | 9.91 | 8.4 | 3.70 | 3.1 | 0.77 |
| 7 | 9.67 | 8.2 | 3.59 | 3.0 | 0.78 |
| 8 | 9.51 | 8.0 | 3.56 | 3.0 | 0.74 |

### Top operators by self device time, forward + backward, per step (all operators: 348.2 ms)

| operator | calls | self ms | % of total |
|:---|---:|---:|---:|
| `aten::mm` | 66 | 104.72 | 30.1 |
| `aten::_flash_attention_backward` | 8 | 67.77 | 19.5 |
| `aten::addmm` | 33 | 51.70 | 14.8 |
| `aten::_flash_attention_forward` | 8 | 26.25 | 7.5 |
| `aten::copy_` | 338 | 24.41 | 7.0 |
| `aten::mul` | 144 | 18.91 | 5.4 |
| `aten::cat` | 26 | 12.77 | 3.7 |
| `aten::native_layer_norm_backward` | 17 | 11.61 | 3.3 |
| `aten::native_layer_norm` | 17 | 5.71 | 1.6 |
| `aten::add` | 32 | 5.02 | 1.4 |
| `aten::add_` | 64 | 4.93 | 1.4 |
| `aten::sum` | 33 | 4.66 | 1.3 |
| `aten::gelu_backward` | 8 | 2.14 | 0.6 |
| `aten::fill_` | 51 | 1.71 | 0.5 |
| `aten::gelu` | 8 | 1.54 | 0.4 |
| `aten::neg` | 32 | 1.46 | 0.4 |
| `aten::sub` | 16 | 1.30 | 0.4 |
| `aten::embedding_dense_backward` | 1 | 0.34 | 0.1 |
| `aten::pow` | 32 | 0.23 | 0.1 |
| `aten::sin` | 16 | 0.19 | 0.1 |
| `aten::cos` | 16 | 0.18 | 0.1 |
| `aten::arange` | 66 | 0.15 | 0.0 |
| `aten::nll_loss_backward` | 1 | 0.11 | 0.0 |
| `aten::gather` | 1 | 0.09 | 0.0 |
| `aten::div` | 16 | 0.09 | 0.0 |

Kernel time in the profiled forward + backward: 348.2 ms; the split run's forward + backward took 349.6 ms, so the device was busy 100 % of it (a low share means launch-bound: the CPU, not the kernels, sets the pace).


Top kernels per step:

| kernel | calls | ms | % of kernel time |
|:---|---:|---:|---:|
| `void pytorch_flash::flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<32, 128, 128, 8, 4, 4` | 8 | 66.02 | 19.0 |
| `ampere_bf16_s16816gemm_bf16_256x128_ldg8_relu_f2f_stages_32x3_tn` | 16 | 31.74 | 9.1 |
| `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_nn` | 25 | 29.33 | 8.4 |
| `void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<32, 128, 128, 4, false, false, cutlass::bfloat16_` | 8 | 26.25 | 7.5 |
| `ampere_bf16_s1688gemm_bf16_128x256_ldg8_f2f_stages_32x1_nn` | 8 | 20.79 | 6.0 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8>(cutlass_80_tensoro` | 17 | 19.97 | 5.7 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_nt_align8>(cutlass_80_tensoro` | 8 | 17.99 | 5.2 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_nt_align8>(cutlass_80_tensoro` | 8 | 13.37 | 3.8 |
| `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_nt` | 8 | 13.27 | 3.8 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl<at::native::BinaryFunctor<float, float` | 64 | 12.26 | 3.5 |
| `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_` | 56 | 10.54 | 3.0 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x5_nt_align8>(cutlass_80_tensoro` | 8 | 8.93 | 2.6 |
| `void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<2u>,` | 8 | 7.72 | 2.2 |
| `void at::native::(anonymous namespace)::GammaBetaBackwardCUDAKernelTemplate<float, float, 32u, 32u, 256u, fals` | 17 | 7.62 | 2.2 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{la` | 100 | 7.15 | 2.1 |

### Memory

Peak allocated 2.56 GiB, peak reserved 2.67 GiB over the timed training steps; the forward leaves 2.23 GiB of activations.

