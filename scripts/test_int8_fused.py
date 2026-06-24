import torch
from local_ai_training.int8_fused import fused_rmsnorm_quantize, fused_gelu_quantize
from local_ai_training.int8_matmul import quantize_rows

def test_rmsnorm():
    M, K = 128, 512
    x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    w = torch.ones(K, device='cuda', dtype=torch.float32)
    
    # Reference
    x_norm = torch.nn.functional.rms_norm(x.float(), (K,), w, 1e-5)
    ref_q, ref_s = quantize_rows(x_norm)
    
    # Fused
    fused_q, fused_s = fused_rmsnorm_quantize(x, w, 1e-5)
    
    assert torch.allclose(fused_s, ref_s), "Scale mismatch"
    assert torch.equal(fused_q, ref_q), "Quant mismatch"
    print("RMSNorm OK")

def test_gelu():
    M, K = 128, 512
    x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    
    # Reference
    x_gelu = torch.nn.functional.gelu(x.float(), approximate='tanh')
    ref_q, ref_s = quantize_rows(x_gelu)
    
    # Fused
    fused_q, fused_s = fused_gelu_quantize(x)
    
    assert torch.allclose(fused_s, ref_s), "Scale mismatch"
    assert torch.equal(fused_q, ref_q), "Quant mismatch"
    print("GELU OK")

from local_ai_training.int8_fused import fused_rmsnorm_quantize, fused_gelu_quantize, fused_transpose_quantize

def test_transpose():
    batch, n_head, seq, head_dim = 4, 8, 128, 64
    channels = n_head * head_dim
    x = torch.randn(batch, n_head, seq, head_dim, device='cuda', dtype=torch.bfloat16)
    
    # Reference
    joined = x.transpose(1, 2).contiguous().view(batch * seq, channels)
    ref_q, ref_s = quantize_rows(joined)
    
    # Fused
    fused_q, fused_s = fused_transpose_quantize(x)
    
    assert torch.allclose(fused_s, ref_s), "Scale mismatch"
    assert torch.equal(fused_q, ref_q), "Quant mismatch"
    print("Transpose OK")

if __name__ == "__main__":
    test_rmsnorm()
    test_gelu()
    test_transpose()
