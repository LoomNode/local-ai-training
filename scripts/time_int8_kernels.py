import torch
import time
from local_ai_training.int8_fused import FusedRMSNormQuantizeFn, FusedGELUQuantizeFn, FusedTransposeQuantizeFn
from local_ai_training.int8_backward import rmsnorm_backward, gelu_backward
from local_ai_training.ratchet import scaled_int8_mm, quantize_columns, quantize_rows

def time_fn(name, fn, *args, iters=100):
    # warmup
    for _ in range(10):
        fn(*args)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    
    avg_ms = start.elapsed_time(end) / iters
    print(f"{name:35s}: {avg_ms:.3f} ms")

def main():
    torch.manual_seed(42)
    batch = 64
    seq = 256
    K = 512
    M = batch * seq
    
    print("--- Forward Kernels ---")
    x = torch.randn(batch, seq, K, device='cuda', dtype=torch.bfloat16)
    w_norm = torch.randn(K, device='cuda', dtype=torch.float32)
    
    time_fn("RMSNorm Quantize Fwd", FusedRMSNormQuantizeFn.apply, x, w_norm, 1e-5)
    
    x_ffn = torch.randn(batch, seq, 3 * K, device='cuda', dtype=torch.bfloat16)
    time_fn("GELU Quantize Fwd", FusedGELUQuantizeFn.apply, x_ffn)
    
    attn = torch.randn(batch, 8, seq, 64, device='cuda', dtype=torch.bfloat16)
    time_fn("Transpose Quantize Fwd", FusedTransposeQuantizeFn.apply, attn)
    
    print("\n--- Backward Kernels ---")
    grad_y = torch.randn(batch, seq, K, device='cuda', dtype=torch.bfloat16)
    time_fn("RMSNorm Bwd", rmsnorm_backward, grad_y, x, w_norm, 1e-5)
    
    rsqrt_out = torch.randn(batch * seq, device='cuda', dtype=torch.float32)
    def pytorch_grad_w(grad_y, x, rsqrt_out):
        # We simulate the exact operations PyTorch would do
        return torch.sum(grad_y * x * rsqrt_out.view(-1, 1), dim=0)
    
    time_fn("PyTorch grad_w", pytorch_grad_w, grad_y.flatten(0, -2).float(), x.flatten(0, -2).float(), rsqrt_out)
    
    print("\n--- RatchetMatmul Kernels ---")
    int8_inputs = torch.randint(-127, 127, (M, K), device='cuda', dtype=torch.int8)
    scale = torch.ones(M, device='cuda', dtype=torch.float32)
    code = torch.randint(-1, 2, (K, K), device='cuda', dtype=torch.int8)
    scale_w = torch.ones(K, device='cuda', dtype=torch.float32)
    
    time_fn("scaled_int8_mm", scaled_int8_mm, int8_inputs, code, scale, scale_w)
    
    bf16_inputs = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    time_fn("quantize_columns", quantize_columns, bf16_inputs)
    time_fn("quantize_rows", quantize_rows, bf16_inputs)
    time_fn("quantize_rows(t)", quantize_rows, bf16_inputs.t())
    
    def pytorch_quantize_columns(values):
        amax = values.abs().amax(dim=0)
        scale = torch.clamp(amax / 127.0, min=1e-5)
        out = torch.clamp(torch.round(values / scale), -127.0, 127.0).to(torch.int8)
        return out, scale
    
    time_fn("PyTorch quantize_columns", pytorch_quantize_columns, bf16_inputs)

if __name__ == "__main__":
    main()
