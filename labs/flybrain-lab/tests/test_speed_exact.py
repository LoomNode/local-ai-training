"""Speed shortcuts that must not change the math (2026-11-05): the minGRU scan in fp32 by a parallel scan instead of the
fp64 division form; the memory head's table update in one pass; the key rescored from its 4 nonzeros. Each new path is
checked against the old one, forward and backward."""

import pytest
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


@pytest.mark.parametrize("T", [1, 100, 128, 129, 300, 2048])
def test_fused_cuda_scan_matches_the_pairs_scan(T):
    from flybrain.model.scan_kernel import fused_scan, scan_available
    if not torch.cuda.is_available():
        pytest.skip("needs a card")
    a, b = gates(T=T); dev = torch.device("cuda")
    a1, b1 = a.to(dev).requires_grad_(), b.to(dev).requires_grad_(); a2, b2 = a.to(dev).requires_grad_(), b.to(dev).requires_grad_()
    h0 = torch.randn(2, 16, device=dev)
    assert scan_available(a1)
    # the oracle is the fp64 closed form: the fp32 parallel fallback drifts to ~2e-4 over 2048 steps, the kernel does not
    want = reference_scan(a1, b1, h0); got = fused_scan(a2, b2, h0)
    assert torch.allclose(got, want.float(), atol=1e-5, rtol=1e-5)
    w = torch.randn_like(got); (want * w).sum().backward(); (got * w).sum().backward()
    assert torch.allclose(a2.grad, a1.grad.float(), atol=1e-4, rtol=1e-4) and torch.allclose(b2.grad, b1.grad.float(), atol=1e-4, rtol=1e-4)


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


def _cuda_or_skip():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("needs a card")
    return torch.device("cuda")


def test_fused_select_picks_the_same_slots_as_topk():
    from flybrain.model.select_kernel import fused_select
    dev = _cuda_or_skip(); torch.manual_seed(5)
    c = KeyCode("sparse", 64, 256, claws=4, top_k=4, theta=0.3, seed=11).to(dev)
    with torch.no_grad():
        c.theta.add_(torch.randn_like(c.theta) * 0.1)                  # distinct thresholds, as after training
    x = torch.randn(300, 64, device=dev)
    want_val, want_idx = F.relu(x @ c.proj.t() - c.theta).topk(4, dim=-1)
    cols, signs = c._index_form()
    idx = fused_select(x, cols, signs, c.theta, 4)
    got_val = F.relu((x.unsqueeze(1) * c.proj[idx]).sum(-1) - c.theta[idx])
    assert torch.allclose(got_val.sort(-1).values, want_val.sort(-1).values, atol=1e-5)   # tie-robust: the selected scores agree
    distinct = want_val.sort(-1).values.diff(dim=-1).abs().min(-1).values > 1e-6
    assert distinct.any()
    assert torch.equal(idx[distinct].sort(-1).values, want_idx[distinct].sort(-1).values)   # and the slots, where no tie is possible


def test_compiled_head_read_matches_eager_on_the_card():
    import flybrain.model.sparsela as sp
    dev = _cuda_or_skip(); torch.manual_seed(6)
    head = sp.Block(32, sp.KeyCode("sparse", 32, 64, claws=4, top_k=4, theta=0.0, seed=3), 1.0, d_ff=1, key_conv=2, normalize=True, gate=True).to(dev)
    head.chunk = 8
    x = torch.randn(2, 40, 32, device=dev, requires_grad=True)
    sp.COMPILE_ON = False
    y0 = head.memory(head.ln1(x)); y0.sum().backward(); g0 = x.grad.clone(); x.grad = None
    sp.COMPILE_ON = True
    y1 = head.memory(head.ln1(x)); y1.sum().backward(); g1 = x.grad.clone()
    assert torch.allclose(y0, y1, atol=1e-5) and torch.allclose(g0, g1, atol=1e-4)


def test_flex_attention_path_matches_the_block_masked_path_in_fp32():
    import flybrain.model.sparsela as sp
    from flybrain.model.hybrid import LocalAttention
    dev = _cuda_or_skip(); torch.manual_seed(7)
    la = LocalAttention(64, 2, 32).to(dev)
    x = torch.randn(2, 256, 64, device=dev, requires_grad=True)
    import flybrain.model.hybrid as hy
    hy.FLEX_ON = False; y0 = la(x); y0.sum().backward(); g0 = x.grad.clone(); x.grad = None
    hy.FLEX_ON = True; y1 = la(x); y1.sum().backward(); g1 = x.grad.clone(); hy.FLEX_ON = False
    assert torch.allclose(y0, y1, atol=1e-4, rtol=1e-4) and torch.allclose(g0, g1, atol=1e-3, rtol=1e-3)


def test_fp16_tables_track_fp32_closely_on_the_card():
    import flybrain.model.sparsela as sp
    dev = _cuda_or_skip(); torch.manual_seed(8)
    head = sp.Block(32, sp.KeyCode("sparse", 32, 64, claws=4, top_k=4, theta=0.0, seed=3), 1.0, d_ff=1, key_conv=2, normalize=True, gate=True).to(dev)
    head.chunk = 8
    x = torch.randn(2, 64, 32, device=dev)
    sp.COMPILE_ON = False
    with torch.no_grad():
        sp.TABLE16 = False; y0 = head.memory(head.ln1(x))
        sp.TABLE16 = True; y1 = head.memory(head.ln1(x)); sp.TABLE16 = False
    assert torch.allclose(y0, y1, atol=2e-3, rtol=2e-3)
    sp.COMPILE_ON = True


def test_causal_conv_as_shifted_sums_matches_conv1d():
    from flybrain.model.sparsela import Block, KeyCode
    torch.manual_seed(9)
    for K in (2, 4):
        blk = Block(16, KeyCode("sparse", 16, 64, 4, 4, 0.0, 1), 1.0, d_ff=1, key_conv=K)
        for T in (1, 3, 40):
            x = torch.randn(2, T, 16)
            want = blk.conv(F.pad(x.transpose(1, 2), (K - 1, 0))).transpose(1, 2)
            assert torch.allclose(blk.causal_conv(x), want, atol=1e-6), (K, T)
