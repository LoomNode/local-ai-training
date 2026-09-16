import torch
from torch.nn import functional as F

from flybrain.model.sparsela import KEY_KINDS, SparseKeyLM


def test_all_key_kinds_forward_and_are_causal():
    for kind in KEY_KINDS:
        m = SparseKeyLM(key=kind, d_model=16, n_key=64, top_k=8, n_layers=1, d_ff=32)
        x = torch.randint(0, 256, (2, 12))
        logits, _ = m(x)
        assert logits.shape == (2, 12, 256)
        x2 = x.clone()
        x2[:, 6:] = (x2[:, 6:] + 1) % 256
        logits2, _ = m(x2)
        assert torch.allclose(logits[:, :6], logits2[:, :6], atol=1e-5), kind


def test_sparse_key_is_topk_sparse():
    m = SparseKeyLM(key="sparse", d_model=16, n_key=64, top_k=8, n_layers=1)
    x = torch.randn(2, 5, 16)
    q, k = m.blocks[0].code(x)
    assert torch.all((k != 0).sum(-1) <= 8)


def test_train_step_reduces_loss():
    torch.manual_seed(0)
    m = SparseKeyLM(key="sparse", d_model=16, n_key=64, top_k=8, n_layers=1, d_ff=32)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    x = torch.randint(0, 8, (8, 16))
    y = x.roll(-1, 1)
    first = None
    for _ in range(30):
        logits, _ = m(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        first = first or float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < first


def test_key_conv_is_causal():
    m = SparseKeyLM(key="sparse", d_model=16, n_key=64, top_k=8, n_layers=1, d_ff=32, key_conv=3)
    x = torch.randint(0, 256, (2, 12))
    logits, _ = m(x)
    x2 = x.clone(); x2[:, 6:] = (x2[:, 6:] + 1) % 256
    logits2, _ = m(x2)
    assert torch.allclose(logits[:, :6], logits2[:, :6], atol=1e-5)


def test_mixed_and_tied_kinds_forward_causal():
    for kind in ("mixed", "tied"):
        m = SparseKeyLM(key=kind, d_model=16, n_key=64, top_k=4, n_layers=1, d_ff=32, learned_key=16, normalize=True, key_conv=2)
        x = torch.randint(0, 256, (2, 12))
        logits, _ = m(x)
        assert logits.shape == (2, 12, 256)
        x2 = x.clone(); x2[:, 6:] = (x2[:, 6:] + 1) % 256
        logits2, _ = m(x2)
        assert torch.allclose(logits[:, :6], logits2[:, :6], atol=1e-5), kind
    m = SparseKeyLM(key="mixed", d_model=16, n_key=64, top_k=4, n_layers=1, learned_key=16)
    assert len(m.blocks[0].codes) == 2 and m.blocks[0].codes[1].kind == "sparse"


def test_copyable_mask_and_terminal_corpus():
    from flybrain.data.corpus import copyable_mask, terminal_transcript
    from flybrain.data.task import multi_copy_batch

    x, y, m = multi_copy_batch(3, seq_len=128, span=8, n_spans=4, seed=0)
    assert torch.all(m.sum(1) == 4 * 7)
    cm = copyable_mask(x, y, n=4)
    # every recalled byte after the first 4 of a span is copyable by a 4-gram key
    assert (cm & m).sum() >= 4 * 4 * 3
    a = terminal_transcript(4000, seed=1); b = terminal_transcript(4000, seed=1)
    assert a == b and len(a) == 4000 and b"user@box" in a and b"cat " in a


def test_gru_hybrid_forward_causal_both_placements():
    from flybrain.model.hybrid import GRUHybrid

    for where in ("pre", "post"):
        for key in ("sparse", "learned"):
            m = GRUHybrid(d_model=16, hidden=24, key=key, where=where, n_key=64, top_k=4, learned_key=16)
            x = torch.randint(0, 256, (2, 12))
            logits, h = m(x)
            assert logits.shape == (2, 12, 256) and h.shape == (1, 2, 24)
            x2 = x.clone(); x2[:, 6:] = (x2[:, 6:] + 1) % 256
            logits2, _ = m(x2)
            assert torch.allclose(logits[:, :6], logits2[:, :6], atol=1e-5), (where, key)


def test_shuffled_keys_break_retrieval_but_stay_causal():
    from flybrain.model.hybrid import GRUHybrid
    torch.manual_seed(0)
    a = GRUHybrid(d_model=16, hidden=24, n_key=64, top_k=4, where="pre")
    b = GRUHybrid(d_model=16, hidden=24, n_key=64, top_k=4, where="pre", shuffle_keys=True)
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 256, (4, 12))
    la, _ = a(x); lb, _ = b(x)
    assert not torch.allclose(la, lb)
    x2 = x.clone(); x2[:, 6:] = (x2[:, 6:] + 1) % 256
    lb2, _ = b(x2)
    assert torch.allclose(lb[:, :6], lb2[:, :6], atol=1e-5)


def test_mingru_and_gla_backbones_causal_with_and_without_head():
    from flybrain.model.hybrid import GRUHybrid
    for bb in ("mingru", "gla"):
        for where in ("none", "post"):
            m = GRUHybrid(d_model=16, hidden=16, key="sparse", where=where, backbone=bb, n_layers=2, d_ff=32, n_key=64, top_k=4, learned_key=16)
            x = torch.randint(0, 256, (2, 12))
            logits, _ = m(x)
            assert logits.shape == (2, 12, 256)
            x2 = x.clone(); x2[:, 6:] = (x2[:, 6:] + 1) % 256
            logits2, _ = m(x2)
            assert torch.allclose(logits[:, :6], logits2[:, :6], atol=1e-4), (bb, where)


def test_gla_decay_gate_changes_weights():
    from flybrain.model.sparsela import SparseKeyLM
    m = SparseKeyLM(key="gla", d_model=16, n_key=16, n_layers=1, d_ff=32, learned_key=16, normalize=True)
    c = m.blocks[0].code
    x = torch.randn(2, 5, 16)
    assert c.log_decay(x).shape == (2, 5)
    assert torch.all(c.log_decay(x) <= 0)


def test_mingru_chunked_scan_matches_sequential_loop():
    from flybrain.model.hybrid import MinGRULayer
    torch.manual_seed(0)
    layer = MinGRULayer(16, 32)
    x = torch.randn(2, 70, 16)
    with torch.no_grad():
        u = layer.ln1(x); z = torch.sigmoid(layer.z(u)); cand = layer.h(u)
        a = (1 - z).clamp_min(1e-4); b = z * cand
        hs = []; hidden = torch.zeros_like(u[:, 0])
        for tt in range(70):
            hidden = a[:, tt] * hidden + b[:, tt]; hs.append(hidden)
        ref = x + torch.stack(hs, 1); ref = ref + layer.ff(layer.ln2(ref))
        out = layer(x)
    assert torch.allclose(out, ref, atol=1e-4)


def test_transformer_backbone_causal_with_and_without_head():
    from flybrain.model.hybrid import GRUHybrid
    for where in ("none", "post", "pre"):
        torch.manual_seed(0)
        m = GRUHybrid(d_model=16, hidden=16, key="sparse", where=where, backbone="transformer", n_layers=2, d_ff=32, n_key=64, top_k=4, learned_key=16)
        m.eval()
        x = torch.randint(0, 256, (2, 12))
        logits, _ = m(x)
        assert logits.shape == (2, 12, 256)
        x2 = x.clone(); x2[:, 6:] = (x2[:, 6:] + 1) % 256
        logits2, _ = m(x2)
        assert torch.allclose(logits[:, :6], logits2[:, :6], atol=1e-4), where
        # and the future does change the later positions (attention is actually wired in)
        assert not torch.allclose(logits[:, 6:], logits2[:, 6:], atol=1e-3)


def test_transformer_uses_position_beyond_bigram():
    """Full attention must be able to read a token 8 steps back: outputs at t depend on x[t-8]."""
    from flybrain.model.hybrid import GRUHybrid
    torch.manual_seed(0)
    m = GRUHybrid(d_model=16, hidden=16, key="sparse", where="none", backbone="transformer", n_layers=1, d_ff=32)
    m.eval()
    x = torch.randint(0, 256, (1, 12))
    x2 = x.clone(); x2[:, 2] = (x2[:, 2] + 1) % 256
    a, _ = m(x); b, _ = m(x2)
    assert not torch.allclose(a[:, 10], b[:, 10], atol=1e-5)


def test_flops_per_step_uses_d_ff():
    from flybrain.model.hybrid import GRUHybrid
    a = GRUHybrid(d_model=16, hidden=16, where="none", backbone="mingru", n_layers=1, d_ff=32)
    b = GRUHybrid(d_model=16, hidden=16, where="none", backbone="mingru", n_layers=1, d_ff=64)
    assert b.flops_per_step() > a.flops_per_step()


def test_match_gate_zeroes_head_when_nothing_matched_and_keeps_causality():
    from flybrain.model.hybrid import GRUHybrid
    torch.manual_seed(0)
    g = GRUHybrid(d_model=16, hidden=16, key="sparse", where="post", backbone="mingru", n_layers=1, d_ff=32, n_key=64, top_k=4, gate=True)
    g.eval()
    x = torch.randint(0, 256, (2, 12))
    z = g.mem.ln1(g.core(g.embed(x)))
    head = g.mem.memory(z)
    assert torch.allclose(head[:, :2], torch.zeros_like(head[:, :2]), atol=1e-6)  # slots 0 and 1 have nothing to read
    x2 = x.clone(); x2[:, 6:] = (x2[:, 6:] + 1) % 256
    a, _ = g(x); b, _ = g(x2)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-4)
    # a repeated 4-gram gets a nonzero read at its continuation
    r = torch.tensor([[1, 2, 3, 4, 9, 9, 1, 2, 3, 4, 9, 9]])
    zr = g.mem.ln1(g.core(g.embed(r)))
    assert float(g.mem.memory(zr)[0, 9].abs().sum()) > 0
    # tau is trainable
    assert g.mem.gate_tau.requires_grad


def test_transformer_max_len_follows_argument():
    from flybrain.model.hybrid import GRUHybrid
    m = GRUHybrid(d_model=16, hidden=16, where="none", backbone="transformer", n_layers=1, d_ff=32, max_len=64)
    logits, _ = m(torch.randint(0, 256, (1, 64)))
    assert logits.shape == (1, 64, 256)


def test_rope_scores_depend_only_on_relative_position():
    from flybrain.model.hybrid import rope
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, 16); k = torch.randn(1, 1, 1, 16)
    def score(i, j):
        Q = torch.zeros(1, 1, max(i, j) + 1, 16); K = Q.clone(); Q[..., i, :] = q; K[..., j, :] = k
        return float((rope(Q)[..., i, :] * rope(K)[..., j, :]).sum())
    assert abs(score(5, 2) - score(9, 6)) < 1e-4 and abs(score(5, 2) - score(12, 2)) > 1e-2
    assert torch.allclose(rope(q).norm(), q.norm())


def test_rope_transformer_causal_and_reads_far_back():
    from flybrain.model.hybrid import GRUHybrid
    torch.manual_seed(0)
    m = GRUHybrid(d_model=32, hidden=32, where="none", backbone="transformer", n_layers=1, d_ff=64, rope=True); m.eval()
    assert m.core.pos is None
    x = torch.randint(0, 256, (2, 12)); a, _ = m(x)
    x2 = x.clone(); x2[:, 6:] = (x2[:, 6:] + 1) % 256; b, _ = m(x2)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-4) and not torch.allclose(a[:, 6:], b[:, 6:], atol=1e-3)
    x3 = x.clone(); x3[:, 1] = (x3[:, 1] + 1) % 256; c, _ = m(x3)
    assert not torch.allclose(a[:, 11], c[:, 11], atol=1e-5)


def test_lr_schedule_shape():
    from scripts.run_sparsela import lr_lambda
    f = lr_lambda(1000, 100, True)
    assert abs(f(0) - 0.01) < 1e-9 and abs(f(99) - 1.0) < 1e-9 and abs(f(100) - 1.0) < 1e-6 and abs(f(999) - 0.1) < 1e-3
    g = lr_lambda(1000, 0, False)
    assert g(0) == 1.0 and g(999) == 1.0


def test_chunked_head_matches_parallel_form():
    """Chunked (linear-cost) memory equals the T x T parallel form for the hybrid's settings (decay 1, sparse keys)."""
    from flybrain.model.sparsela import Block, KeyCode
    for gate in (False, True):
        for T, L in ((37, 8), (64, 16), (16, 64)):
            torch.manual_seed(0)
            b = Block(24, KeyCode("sparse", 24, 96, 4, 4, 1.0, 0), 1.0, d_ff=1, key_conv=2, normalize=True, gate=gate)
            if gate:
                with torch.no_grad():
                    b.gate_tau.fill_(0.3)
            x = torch.randn(3, T, 24)
            ref = b.memory(x)
            b.chunk = L
            out = b.memory(x)
            assert torch.allclose(ref, out, atol=1e-5), (gate, T, L, float((ref - out).abs().max()))
            # gradients agree too
            x1 = x.clone().requires_grad_(True); x2 = x.clone().requires_grad_(True)
            b.chunk = 0; b.memory(x1).sum().backward()
            b.chunk = L; b.memory(x2).sum().backward()
            assert torch.allclose(x1.grad, x2.grad, atol=1e-4)


def test_chunked_head_refuses_unsupported_settings():
    from flybrain.model.sparsela import Block, KeyCode
    b = Block(8, KeyCode("learned", 8, 8, 4, 4, 1.0, 0), 1.0, d_ff=1, normalize=True)
    b.chunk = 4
    import pytest
    with pytest.raises(ValueError):
        b.memory(torch.randn(1, 12, 8))


def _causal_ok(m, T=32, B=2):
    torch.manual_seed(0)
    x = torch.randint(0, 256, (B, T))
    y = x.clone(); y[:, T // 2:] = (y[:, T // 2:] + 1) % 256
    m.eval()
    a, b = m(x)[0], m(y)[0]
    return torch.allclose(a[:, : T // 2], b[:, : T // 2], atol=1e-5) and not torch.allclose(a[:, T // 2:], b[:, T // 2:], atol=1e-3)


def test_interleaved_heads_count_and_causality():
    """mem_every=k inserts a head after every k-th backbone layer (except the last, which the post head covers)."""
    from flybrain.model.hybrid import GRUHybrid
    for bb in ("mingru", "gla"):
        torch.manual_seed(0)
        m = GRUHybrid(d_model=16, key="sparse", where="post", backbone=bb, n_layers=4, d_ff=32, n_key=64, top_k=4, key_conv=2, normalize=True, gate=True, mem_every=2)
        assert m.n_heads == 2 and len(m.mems) == 1
        assert _causal_ok(m), bb
    torch.manual_seed(0)
    m3 = GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=6, d_ff=32, n_key=64, top_k=4, mem_every=2)
    assert m3.n_heads == 3
    # interleaved heads are real: output differs from the single-head model with the same seed
    torch.manual_seed(0)
    m1 = GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4, key_conv=2, normalize=True, gate=True)
    torch.manual_seed(0)
    m2 = GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4, key_conv=2, normalize=True, gate=True, mem_every=2)
    assert m1.n_heads == 1 and m1.flops_per_step() < m2.flops_per_step()
    x = torch.randint(0, 256, (1, 24)); m1.eval(); m2.eval()
    assert not torch.allclose(m1(x)[0], m2(x)[0], atol=1e-3)


def test_interleaved_heads_use_distinct_keys_and_support_chunked_form():
    from flybrain.model.hybrid import GRUHybrid
    torch.manual_seed(0)
    par = GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4, key_conv=2, normalize=True, gate=True, decay=1.0, mem_every=2, chunk=0)
    torch.manual_seed(0)
    chk = GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4, key_conv=2, normalize=True, gate=True, decay=1.0, mem_every=2, chunk=8)
    chk.load_state_dict(par.state_dict())
    assert all(h.chunk == 8 for h in chk.all_heads()) and all(h.chunk == 0 for h in par.all_heads())
    heads = par.all_heads()
    assert not torch.equal(heads[0].code.proj, heads[1].code.proj)  # different random address per head
    x = torch.randint(0, 256, (2, 40)); par.eval(); chk.eval()
    assert torch.allclose(par(x)[0], chk(x)[0], atol=1e-4)


def test_interleaved_heads_rejected_for_unlayered_backbones():
    import pytest
    from flybrain.model.hybrid import GRUHybrid
    for bb in ("gru", "transformer"):
        with pytest.raises(ValueError):
            GRUHybrid(d_model=16, hidden=16, key="sparse", where="post", backbone=bb, n_layers=2, d_ff=32, n_key=64, top_k=4, mem_every=1)
    with pytest.raises(ValueError):  # interleaving without a post head has no meaning here
        GRUHybrid(d_model=16, key="sparse", where="pre", backbone="mingru", n_layers=2, d_ff=32, n_key=64, top_k=4, mem_every=1)


def test_ntk_rope_scaling_matches_lowest_frequency_at_scaled_length():
    """NTK-aware scaling: base' = base * s^(Dh/(Dh-2)) so the lowest-frequency pair at position s*L has the angle it had at L."""
    import math
    from flybrain.model.hybrid import TransformerBackbone, rope
    m = TransformerBackbone(32, 1, 64, n_heads=2, rope=True)
    dh = 16; base = 10000.0; s = 4.0
    m.set_rope_scale(s)
    assert all(abs(a.rope_base - base * s ** (dh / (dh - 2))) < 1e-3 for a in m.attn)
    half = dh // 2; low = base ** (-(half - 1) / half); low2 = m.attn[0].rope_base ** (-(half - 1) / half)
    assert abs(low * 512 - low2 * 2048) < 1e-6 * low * 512
    # rope() takes the base through: same angle at 4x the position with the scaled base
    x = torch.randn(1, 1, 2049, dh); x[:, :, 2048] = x[:, :, 512]  # same vector at position 512 (plain) and 2048 (scaled)
    a = rope(x[:, :, :513])[:, :, 512]; b = rope(x[:, :, :2049], base=m.attn[0].rope_base)[:, :, 2048]
    assert torch.allclose(a[..., half - 1], b[..., half - 1], atol=1e-4) and torch.allclose(a[..., -1], b[..., -1], atol=1e-4)
    assert not torch.allclose(a[..., 0], b[..., 0], atol=1e-2)  # the highest frequency is not preserved (it is 4x wrapped)
    m.set_rope_scale(1.0); assert all(a.rope_base == base for a in m.attn)


def test_expand_slots_keeps_trained_rows_and_runs_chunked():
    """Eval-time slot expansion: trained address rows and thresholds are kept, new rows are fresh random, new thetas = trained mean."""
    from flybrain.model.hybrid import GRUHybrid
    torch.manual_seed(0)
    m = GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4, key_conv=2, normalize=True, gate=True, decay=1.0, mem_every=2, chunk=8)
    with torch.no_grad():
        for h in m.all_heads(): h.code.theta.copy_(torch.linspace(0.5, 1.5, 64))
    proj0 = [h.code.proj.clone() for h in m.all_heads()]; theta0 = [h.code.theta.clone() for h in m.all_heads()]
    m.expand_slots(64)  # no-op
    assert all(h.code.n_key == 64 and torch.equal(h.code.proj, p0) for h, p0 in zip(m.all_heads(), proj0))
    m.expand_slots(256)
    for h, p0, t0 in zip(m.all_heads(), proj0, theta0):
        assert h.code.n_key == 256 and h.code.proj.shape == (256, 16) and h.code.theta.shape == (256,)
        assert torch.equal(h.code.proj[:64], p0) and torch.equal(h.code.theta[:64], t0)
        assert torch.allclose(h.code.theta[64:], t0.mean().expand(192))
        assert (h.code.proj[64:].abs().sum(-1) == 4).all()  # fresh rows have `claws` nonzeros
    assert not torch.equal(m.all_heads()[0].code.proj[64:], m.all_heads()[1].code.proj[64:])  # heads get different new rows
    x = torch.randint(0, 256, (2, 40)); m.eval()
    y, _ = m(x); assert y.shape == (2, 40, 256) and torch.isfinite(y).all()
    import pytest
    with pytest.raises(ValueError):
        m.expand_slots(128)  # cannot shrink


def test_sparse_code_index_form_matches_dense_and_is_slot_count_independent_in_memory():
    """sparse_code must equal the dense relu(z - theta).topk, values and grads, while never materialising [B, T, n_key]."""
    from flybrain.model.sparsela import KeyCode
    torch.manual_seed(0)
    c = KeyCode("sparse", 16, 512, 4, 4, 0.5, 3)
    with torch.no_grad(): c.theta.copy_(torch.rand(512))
    x = torch.randn(2, 37, 16, requires_grad=True)
    z = x.matmul(c.proj.t()); dense = torch.relu(z - c.theta); dval, didx = dense.topk(4, dim=-1)
    idx, val = c.sparse_code(x)
    assert torch.equal(idx, didx) and torch.allclose(val, dval, atol=1e-6)
    (val.sum()).backward(); gx, gt = x.grad.clone(), c.theta.grad.clone()
    x.grad = None; c.theta.grad = None
    (dval.sum()).backward()
    assert torch.allclose(gx, x.grad, atol=1e-6) and torch.allclose(gt, c.theta.grad, atol=1e-6)
    # a slice length of 1 exercises the slicing path and must give the same answer
    c.code_slice = 1; idx2, val2 = c.sparse_code(x)
    assert torch.equal(idx2, idx) and torch.allclose(val2, val, atol=1e-6)


def test_key_convs_per_head_cycle_over_heads():
    from flybrain.model.hybrid import GRUHybrid
    m = GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4, key_conv=2, mem_every=2, key_convs=(2, 4))
    assert [h.key_conv for h in m.all_heads()] == [2, 4]  # post head first, then the interleaved one
    m3 = GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=6, d_ff=32, n_key=64, top_k=4, key_conv=2, mem_every=2, key_convs=(2, 4))
    assert [h.key_conv for h in m3.all_heads()] == [2, 4, 2]
    x = torch.randint(0, 256, (1, 24)); m3.eval(); assert torch.isfinite(m3(x)[0]).all()


def test_count_head_reads_next_byte_histogram_and_is_causal():
    """The count memory writes one-hot next bytes under the (bigram) context key; on a repeated random sequence its read on the
    second copy must put most of its mass on the true next byte, and it must be causal and produce a normalised mixture."""
    from flybrain.model.hybrid import CountHead, GRUHybrid
    torch.manual_seed(0)
    emb = torch.nn.Embedding(256, 32)
    r = torch.randint(0, 256, (1, 300)); toks = torch.cat([r, r], dim=1)          # 600 tokens, second half repeats the first
    ch = CountHead(32, n_key=4096, claws=4, top_k=4, theta=0.5, seed=1, key_conv=2, chunk=64)
    ch.eval()
    log_pmem, g = ch(emb(toks), toks)
    assert log_pmem.shape == (1, 600, 256) and g.shape == (1, 600, 1)
    pred = log_pmem[0, 300:599].argmax(-1); truth = toks[0, 301:600]
    assert (pred == truth).float().mean() > 0.6  # untrained random key: ~0.74 vs chance 0.004 (address overlap between bigrams)
    assert torch.allclose(log_pmem.exp().sum(-1), torch.ones(1, 600), atol=1e-3)
    # full model: mixture is a normalised log-distribution and causal
    m = GRUHybrid(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=2, d_ff=32, n_key=64, top_k=4, key_conv=2, normalize=True, gate=True, chunk=8, count_head=True)
    assert _causal_ok(m)
    x = torch.randint(0, 256, (2, 32)); m.eval(); y, _ = m(x)
    assert torch.allclose(y.logsumexp(-1), torch.zeros(2, 32), atol=1e-4)
    assert any(n.startswith("count.") for n, _ in m.named_parameters())
