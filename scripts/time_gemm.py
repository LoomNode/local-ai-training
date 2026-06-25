
import torch

from local_ai_training.ratchet import scaled_int8_mm


def time_fn(name, fn, *args, iters=1000):
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
    print(f"{name:30s}: {avg_ms:.3f} ms")

def main():
    M, K, N = 16384, 512, 512
    a_int8 = torch.randint(-127, 127, (M, K), device='cuda', dtype=torch.int8)
    b_int8 = torch.randint(-1, 2, (N, K), device='cuda', dtype=torch.int8)
    scale_a = torch.ones(M, device='cuda', dtype=torch.float32)
    scale_b = torch.ones(N, device='cuda', dtype=torch.float32)
    
    a_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b_bf16 = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
    
    time_fn("scaled_int8_mm", scaled_int8_mm, a_int8, b_int8, scale_a, scale_b)
    time_fn("PyTorch bf16 matmul", torch.matmul, a_bf16, b_bf16)

if __name__ == "__main__":
    main()
