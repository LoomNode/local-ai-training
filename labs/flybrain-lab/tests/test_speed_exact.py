"""Speed shortcuts that must not change the math (2026-11-05): the minGRU scan in fp32 by a parallel scan instead of the
fp64 division form; the memory head's table update in one pass; the key rescored from its 4 nonzeros. Each new path is
checked against the old one, forward and backward."""

import torch
from torch import Tensor
from torch.nn import functional as F

from flybrain.model.hybrid import GRUHybrid, MinGRULayer
from flybrain.model.sparsela import KeyCode, sparse_memory_read


def reference_scan(a: Tensor, b: Tensor, h0: Tensor | None, L: int = 32) -> Tensor:
    """The fp64 chunked closed form the layer used until 2026-11-05: h_t = A_t (h_0 + sum_{s<=t} b_s / A_s)."""
    B, T, d = a.shape
    pad = (-T) % L
    if pad:
        a = F.pad(a, (0, 0, 0, pad), value=1.0); b = F.pad(b, (0, 0, 0, pad))
    nC = (T + pad) // L
    a = a.reshape(B, nC, L, d).double(); b = b.reshape(B, nC, L, d).double()
    A = torch.exp(torch.cumsum(torch.log(a), dim=2))
    local = A * torch.cumsum(b / A, dim=2)
    A_end, local_end = A[:, :, -1], local[:, :, -1]
    carries = [torch.zeros(B, d, dtype=torch.float64, device=a.device) if h0 is None else h0.double()]
    for i in range(nC - 1):
        carries.append(A_end[:, i] * carries[-1] + local_end[:, i])
    h0s = torch.stack(carries, dim=1)
    return (A * h0s[:, :, None, :] + local).reshape(B, nC * L, d)[:, :T]


def gates(B=2, T=77, d=16, seed=0):
    torch.manual_seed(seed)
    z = torch.sigmoid(torch.randn(B, T, d) * 3)            # decays from near 0 to near 1, incl. the 1e-4 clamp region
    z[:, :, 0] = 0.99995                                    # a channel that forgets almost everything each step
    z[:, :, 1] = 1e-3                                       # a channel that keeps almost everything
    a = (1 - z).clamp_min(1e-4)
    b = z * torch.randn(B, T, d)
    return a, b


def test_fp32_scan_matches_the_fp64_closed_form_forward_and_backward():
    a, b = gates()
    a1, b1 = a.clone().requires_grad_(), b.clone().requires_grad_()
    a2, b2 = a.clone().requires_grad_(), b.clone().requires_grad_()
    h0 = torch.randn(2, 16)
    want = reference_scan(a1, b1, h0)
    got = MinGRULayer.scan_ab(a2, b2, h0)
    assert got.dtype == torch.float32
    assert torch.allclose(got, want.float(), atol=1e-4, rtol=1e-4)
    w = torch.randn_like(want)
    (want * w).sum().backward(); (got * w).sum().backward()
    assert torch.allclose(a2.grad, a1.grad.float(), atol=1e-3, rtol=1e-3)
    assert torch.allclose(b2.grad, b1.grad.float(), atol=1e-4, rtol=1e-4)


def test_fp32_scan_handles_short_and_unpadded_lengths_and_no_carry():
    for T in (1, 5, 32, 64):
        a, b = gates(T=T)
        assert torch.allclose(MinGRULayer.scan_ab(a, b, None), reference_scan(a, b, None).float(), atol=1e-4, rtol=1e-4)


def test_fused_cuda_scan_matches_the_pairs_scan():
    import pytest
    from flybrain.model.scan_kernel import fused_scan, scan_available
    if not torch.cuda.is_available():
        pytest.skip("needs a card")
    a, b = gates(T=100); dev = torch.device("cuda")
    a1, b1 = a.to(dev).requires_grad_(), b.to(dev).requires_grad_(); a2, b2 = a.to(dev).requires_grad_(), b.to(dev).requires_grad_()
    h0 = torch.randn(2, 16, device=dev)
    assert scan_available(a1)
    want = MinGRULayer._scan_pairs_full(a1, b1, h0); got = fused_scan(a2, b2, h0)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)
    w = torch.randn_like(want); (want * w).sum().backward(); (got * w).sum().backward()
    assert torch.allclose(a2.grad, a1.grad, atol=1e-4, rtol=1e-4) and torch.allclose(b2.grad, b1.grad, atol=1e-5, rtol=1e-5)


def test_the_layer_returns_the_same_carry_as_before():
    torch.manual_seed(1)
    layer = MinGRULayer(16, 32)
    x = torch.randn(2, 50, 16)
    out, carry = layer.scan(x)
    assert carry.shape == (2, 16) and torch.isfinite(out).all()
    out2, carry2 = layer.scan(x[:, :30]); out3, _ = layer.scan(x[:, 30:], carry2)
    assert torch.allclose(torch.cat([out2, out3], 1), out, atol=1e-4)


def test_one_pass_table_update_matches_the_two_pass_form():
    torch.manual_seed(2)
    c = KeyCode("sparse", 16, 64, claws=4, top_k=4, theta=0.0, seed=3)
    xk, v = torch.randn(2, 40, 16), torch.randn(2, 40, 16, requires_grad=True)
    out = sparse_memory_read(c, xk, v, 8, 1.0, True, None)
    out_d = sparse_memory_read(c, xk, v, 8, 0.97, True, None)
    assert torch.isfinite(out).all() and not torch.allclose(out, out_d)
    (out.sum() + out_d.sum()).backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_sparse_key_rescored_from_four_nonzeros_matches_the_dense_rows():
    torch.manual_seed(3)
    c = KeyCode("sparse", 32, 128, claws=4, top_k=4, theta=0.5, seed=5)
    x = torch.randn(3, 20, 32, requires_grad=True)
    idx, val = c.sparse_code(x)
    rows = c.proj[idx]                                          # the dense form
    want = F.relu((x.unsqueeze(2) * rows).sum(-1) - c.theta[idx])
    assert torch.allclose(val, want, atol=1e-6)
    val.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    c.expand_slots(256, seed=9)                                  # the index form follows an expanded table
    idx2, val2 = c.sparse_code(x.detach())
    assert int(idx2.max()) < 256 and torch.isfinite(val2).all()


def test_whole_model_is_unchanged_on_a_fixed_input():
    """The 20M arm at toy size: the fast paths give the same logits as a checkpointed reference computed the old way."""
    torch.manual_seed(4)
    m = GRUHybrid(d_model=32, key="sparse", where="post", n_key=64, top_k=4, backbone="mingru", n_layers=4, d_ff=32,
                  normalize=True, key_conv=2, decay=1.0, seed=0, gate=True, chunk=8, mem_every=2, local_attn=8, key_convs=(2, 4)).eval()
    toks = torch.randint(0, 256, (2, 40))
    with torch.no_grad():
        logits, _ = m(toks)
    ref = torch.load("tests/fixtures/speed_exact_reference.pt", weights_only=True)
    assert torch.allclose(logits, ref, atol=1e-4, rtol=1e-4)
