import torch
from local_ai_training.int8_backward import rmsnorm_backward, gelu_backward

def test_rmsnorm_bwd():
    batch, seq, K = 4, 128, 512
    x = torch.randn(batch, seq, K, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(K, device='cuda', dtype=torch.float32, requires_grad=True)
    grad_y = torch.randn(batch, seq, K, device='cuda', dtype=torch.bfloat16)
    
    # Reference
    y_ref = torch.nn.functional.rms_norm(x.float(), (K,), w, 1e-5)
    y_ref.backward(grad_y.float())
    ref_dx = x.grad.clone()
    ref_dw = w.grad.clone()
    
    x.grad = None
    w.grad = None
    
    # Custom
    custom_dx, custom_dw = rmsnorm_backward(grad_y, x, w, 1e-5)
    
    # Compare in float32 for stable tolerance check
    assert torch.allclose(custom_dx.float(), ref_dx.float(), atol=1e-2, rtol=1e-2), f"dx mismatch: max diff {(custom_dx.float() - ref_dx.float()).abs().max()}"
    assert torch.allclose(custom_dw.float(), ref_dw.float(), atol=1e-2, rtol=1e-2), f"dw mismatch: max diff {(custom_dw.float() - ref_dw.float()).abs().max()}"
    print("RMSNorm Bwd OK")

def test_gelu_bwd():
    batch, seq, K = 4, 128, 512
    x = torch.randn(batch, seq, K, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    grad_y = torch.randn(batch, seq, K, device='cuda', dtype=torch.bfloat16)
    
    # Reference
    y_ref = torch.nn.functional.gelu(x.float(), approximate='tanh')
    y_ref.backward(grad_y.float())
    ref_dx = x.grad.clone()
    
    x.grad = None
    
    # Custom
    custom_dx = gelu_backward(grad_y, x)
    
    assert torch.allclose(custom_dx.float(), ref_dx.float(), atol=1e-2, rtol=1e-2), f"dx mismatch: max diff {(custom_dx.float() - ref_dx.float()).abs().max()}"
    print("GELU Bwd OK")

if __name__ == "__main__":
    test_rmsnorm_bwd()
    test_gelu_bwd()
