
import torch


def main():
    M, K, N = 16384, 512, 512
    a_int8 = torch.randint(-127, 127, (M, K), device='cuda', dtype=torch.int8)
    b_int8 = torch.randint(-1, 2, (N, K), device='cuda', dtype=torch.int8)
    
    try:
        # PyTorch native _int_mm requires b to be transposed (K, N)
        torch._int_mm(a_int8, b_int8.t().contiguous())
        print("torch._int_mm is available!")

        # Benchmark
        for _ in range(10):
            torch._int_mm(a_int8, b_int8.t().contiguous())
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(1000):
            torch._int_mm(a_int8, b_int8.t().contiguous())
        end.record()
        torch.cuda.synchronize()
        print(f"torch._int_mm: {start.elapsed_time(end) / 1000:.3f} ms")
        
    except Exception as e:
        print(f"torch._int_mm failed: {e}")

if __name__ == "__main__":
    main()
