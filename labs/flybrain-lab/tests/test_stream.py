import torch


def test_new_document_keeps_tables_and_drops_everything_else():
    from flybrain.model.state import AgentState, AttnCache, HeadState
    S, K = torch.randn(2, 8, 4), torch.rand(2, 8)
    st = AgentState(
        h=[torch.randn(2, 4, dtype=torch.float64), torch.randn(2, 4, dtype=torch.float64)],
        heads=[HeadState(S, K, torch.zeros(2, 3, dtype=torch.long), torch.rand(2, 3), torch.randn(2, 1, 4)), HeadState(S.clone(), K.clone())],
        attn=[AttnCache(torch.randn(2, 1, 5, 4), torch.randn(2, 1, 5, 4), 37)],
    )
    assert st.batch == 2
    nd = st.new_document()
    assert nd.h == [None, None]
    assert torch.equal(nd.heads[0].S, S) and torch.equal(nd.heads[0].Ksum, K)
    assert nd.heads[0].last_idx is None and nd.heads[0].last_val is None
    assert torch.equal(nd.heads[0].conv_tail, torch.zeros(2, 1, 4)) and nd.heads[1].conv_tail is None
    assert nd.attn[0].k is None and nd.attn[0].v is None and nd.attn[0].offset == 0
    assert st.heads[0].last_idx is not None and st.attn[0].offset == 37      # the original is untouched
    assert AgentState(h=[None], heads=[], attn=[]).batch is None


def test_mingru_scan_carry_matches_one_pass():
    from flybrain.model.hybrid import MinGRULayer
    torch.manual_seed(0)
    layer = MinGRULayer(12, 24).eval()
    x = torch.randn(2, 70, 12)
    with torch.no_grad():
        ref = layer(x)
        for cut in (1, 31, 32, 33):                      # around the 32-position scan chunk
            a, h = layer.scan(x[:, :cut])
            b, _ = layer.scan(x[:, cut:], h)
            assert torch.allclose(torch.cat([a, b], 1), ref, atol=1e-5), cut
            assert h.dtype == torch.float64 and h.shape == (2, 12)
        h = None; outs = []
        for t in range(70):                              # byte by byte
            y, h = layer.scan(x[:, t : t + 1], h)
            outs.append(y)
        assert torch.allclose(torch.cat(outs, 1), ref, atol=1e-5)
        full, h_full = layer.scan(x)
        assert h_full.untyped_storage().nbytes() == h_full.numel() * 8   # the carry owns its storage; a view would pin the whole scan buffer
        assert torch.equal(full, ref)                    # forward is scan with no carry
        assert torch.allclose(h, h_full, atol=1e-8)


def test_mingru_scan_leaves_the_carry_alone_and_an_empty_segment_hands_it_back():
    from flybrain.model.hybrid import MinGRULayer
    torch.manual_seed(0)
    layer = MinGRULayer(12, 24).eval()
    x = torch.randn(2, 60, 12)
    with torch.no_grad():
        _, h = layer.scan(x[:, :17])
        h_before = h.clone()
        for seg in (x[:, 17:40], x[:, 17:]):             # one chunk (23 positions) and two chunks (43): the carry also feeds the loop over chunks
            y1, h1 = layer.scan(seg, h)
            y2, h2 = layer.scan(seg, h)                  # the same carry again gives the same answer, so scan did not modify it
            assert torch.equal(h, h_before) and torch.equal(y1, y2) and torch.equal(h1, h2)
        y0, h_same = layer.scan(x[:, :0], h)             # an empty segment: no output, and the carry comes back unchanged
        assert y0.shape == (2, 0, 12) and h_same.dtype == torch.float64 and torch.equal(h_same, h_before)
        _, h_zero = layer.scan(x[:, :0])
        assert torch.equal(h_zero, torch.zeros(2, 12, dtype=torch.float64))
        assert layer(x[:, :0]).shape == (2, 0, 12)       # forward on an empty sequence still returns an empty tensor, as it always did


def test_mingru_scan_carry_is_exact_in_float64():
    from flybrain.model.hybrid import MinGRULayer
    torch.manual_seed(0)
    layer = MinGRULayer(12, 24).double().eval()
    x = torch.randn(2, 70, 12, dtype=torch.float64)
    close = lambda p, q: torch.allclose(p, q, atol=1e-12, rtol=0)   # a carry rounded anywhere, even through float32, is off by 1e-8, not 1e-16
    with torch.no_grad():
        ref, h_ref = layer.scan(x)
        h = None; outs = []
        for t in range(70):                              # byte by byte, in float64
            y, h = layer.scan(x[:, t : t + 1], h)
            outs.append(y)
        assert close(torch.cat(outs, 1), ref) and close(h, h_ref)


def test_mingru_scan_carry_owns_its_storage():
    from flybrain.model.hybrid import MinGRULayer
    torch.manual_seed(0)
    layer = MinGRULayer(12, 24).eval()
    with torch.no_grad():
        for B in (1, 2):                                 # at B = 1 a .contiguous() is a no-op and would still pin the whole scan buffer
            for T in (1, 33, 70):
                _, h = layer.scan(torch.randn(B, T, 12))
                assert h.untyped_storage().nbytes() == h.numel() * 8, (B, T)


def test_local_attention_stream_matches_reference_across_segmentations():
    from flybrain.model.hybrid import LocalAttention
    from flybrain.model.state import AttnCache
    torch.manual_seed(0)
    la = LocalAttention(16, n_heads=2, window=8).eval()
    x = torch.randn(2, 45, 16)
    with torch.no_grad():
        ref = la.reference(x)
        for rebase in (65536, 16):                        # 16: positions wrap mid-stream; only relative offsets may matter
            la.rebase = rebase
            for seg in (1, 3, 8, 9, 45):
                cache = AttnCache(); outs = []
                for c0 in range(0, 45, seg):
                    y, cache = la.stream(x[:, c0 : c0 + seg], cache)
                    outs.append(y)
                out = torch.cat(outs, 1)
                assert torch.allclose(out, ref, atol=1e-5), (rebase, seg, float((out - ref).abs().max()))
                assert cache.offset == 45 and cache.k.size(2) == 7


def test_local_attention_stream_matches_forward_not_only_the_reference():
    from flybrain.model.hybrid import LocalAttention
    from flybrain.model.state import AttnCache
    torch.manual_seed(0)
    la = LocalAttention(16, n_heads=2, window=8).eval()
    x = torch.randn(2, 45, 16)
    with torch.no_grad():
        fwd = la(x)                                       # the block-wise pass the models run, compared directly (the test above uses the banded reference)
        for rebase in (65536, 16):
            la.rebase = rebase
            for seg in (1, 3, 8, 9, 45):
                cache = AttnCache(); outs = []
                for c0 in range(0, 45, seg):
                    y, cache = la.stream(x[:, c0 : c0 + seg], cache)
                    outs.append(y)
                out = torch.cat(outs, 1)
                assert torch.allclose(out, fwd, atol=1e-5), (rebase, seg, float((out - fwd).abs().max()))


def test_local_attention_stream_leaves_the_cache_it_is_given_alone():
    from flybrain.model.hybrid import LocalAttention
    from flybrain.model.state import AttnCache
    torch.manual_seed(0)
    la = LocalAttention(16, n_heads=2, window=8).eval()
    x = torch.randn(2, 45, 16)
    with torch.no_grad():
        _, warm = la.stream(x[:, :20], AttnCache())      # a cache of 7 positions at offset 20
        k0, v0 = warm.k.clone(), warm.v.clone()
        seg = x[:, 20:]                                  # 25 positions: three full blocks and a partial one, each block's cache feeds the next
        y1, c1 = la.stream(seg, warm)
        y2, c2 = la.stream(seg, warm)                    # the same cache object and the same segment again
        assert torch.equal(y1, y2) and torch.equal(c1.k, c2.k) and torch.equal(c1.v, c2.v) and c1.offset == c2.offset == 45
        assert warm.offset == 20 and torch.equal(warm.k, k0) and torch.equal(warm.v, v0)
        assert c1 is not warm and c2 is not warm and c1 is not c2     # a new cache each time
        fresh = AttnCache()                              # a fresh cache too: its first block has nothing cached yet
        ya, ca = la.stream(seg, fresh); yb, _ = la.stream(seg, fresh)
        assert torch.equal(ya, yb) and ca is not fresh
        assert fresh.k is None and fresh.v is None and fresh.offset == 0


def test_local_attention_stream_empty_segment_gives_no_output_and_hands_the_cache_back():
    from flybrain.model.hybrid import LocalAttention
    from flybrain.model.state import AttnCache
    torch.manual_seed(0)
    la = LocalAttention(16, n_heads=2, window=8).eval()
    x = torch.randn(2, 45, 16)
    with torch.no_grad():
        ref = la.reference(x)
        empty = la.reference(x[:, :0])                   # forward raises on a zero-length input; the banded reference returns [B, 0, d]
        fresh = AttnCache()
        y0, c0 = la.stream(x[:, :0], fresh)              # empty at the start of a stream, with a fresh cache
        assert y0.shape == empty.shape == (2, 0, 16) and y0.dtype == empty.dtype
        assert c0 is not fresh and c0.k is None and c0.v is None and c0.offset == 0
        y1, c1 = la.stream(x[:, :20], c0)
        k1, v1 = c1.k.clone(), c1.v.clone()
        ym, cm = la.stream(x[:, 20:20], c1)              # empty in the middle of a stream
        assert ym.shape == (2, 0, 16) and ym.dtype == empty.dtype
        assert cm is not c1 and cm.offset == 20 and torch.equal(cm.k, k1) and torch.equal(cm.v, v1)     # a new cache object, the same state
        assert c1.offset == 20 and torch.equal(c1.k, k1) and torch.equal(c1.v, v1)     # the cache it was given is untouched too
        y2, c2 = la.stream(x[:, 20:], cm)
        assert torch.allclose(torch.cat([y0, y1, ym, y2], 1), ref, atol=1e-5)          # the outputs around the empty segments still match
        assert c2.offset == 45 and c2.k.size(2) == 7


def _fresh_head_state(b, B):
    from flybrain.model.state import HeadState
    d = b.val.in_features
    tail = torch.zeros(B, b.key_conv - 1, d) if b.key_conv > 1 else None
    return HeadState(S=torch.zeros(B, b.code.n_key, d), Ksum=torch.zeros(B, b.code.n_key), conv_tail=tail)


def test_head_with_state_matches_the_parallel_form_across_segmentations():
    from flybrain.model.sparsela import Block, KeyCode
    for decay in (1.0, 0.9):
        for gate in (False, True):
            for key_conv in (2, 4):
                torch.manual_seed(0)
                b = Block(24, KeyCode("sparse", 24, 96, 4, 4, 1.0, 0), decay, d_ff=1, key_conv=key_conv, normalize=True, gate=gate)
                if gate:
                    with torch.no_grad():
                        b.gate_tau.fill_(0.3)
                x = torch.randn(3, 37, 24)
                with torch.no_grad():
                    ref = b.memory(x)                    # chunk 0: the T x T parallel form
                    b.chunk = 8
                    for cuts in ([37], [1, 36], [8, 8, 8, 13], [5, 11, 21], [1] * 37):
                        st = _fresh_head_state(b, 3); outs = []; c0 = 0
                        for n in cuts:
                            y, st = b.memory(x[:, c0 : c0 + n], state=st)
                            outs.append(y); c0 += n
                        out = torch.cat(outs, 1)
                        assert torch.allclose(out, ref, atol=1e-5), (decay, gate, key_conv, cuts, float((out - ref).abs().max()))


def test_head_state_is_not_mutated_and_the_stateless_call_is_unchanged():
    import pytest
    from flybrain.model.sparsela import Block, KeyCode
    torch.manual_seed(0)
    b = Block(24, KeyCode("sparse", 24, 96, 4, 4, 1.0, 0), 1.0, d_ff=1, key_conv=2, normalize=True, gate=True)
    x = torch.randn(2, 20, 24)
    with torch.no_grad():
        b.chunk = 8
        st = _fresh_head_state(b, 2); S0 = st.S.clone()
        y, st2 = b.memory(x, state=st)
        assert torch.equal(st.S, S0) and st.last_idx is None           # the state passed in is untouched
        assert st2.last_idx.shape == (2, 4) and float(st2.Ksum.sum()) > 0 and st2.conv_tail.shape == (2, 1, 24)
        assert torch.allclose(y, b.memory(x), atol=1e-6)               # a fresh state is the stateless call
        assert isinstance(b.memory(x), torch.Tensor)                   # no state: same return type as before
        b.chunk = 0
        with pytest.raises(ValueError):                                # the parallel form has no tables to carry
            b.memory(x, state=_fresh_head_state(b, 2))


def test_quantised_state_survives_single_byte_segments():
    from flybrain.model.sparsela import Block, KeyCode, fake_quant
    torch.manual_seed(0)
    b = Block(24, KeyCode("sparse", 24, 96, 4, 4, 1.0, 0), 1.0, d_ff=1, key_conv=2, normalize=True, gate=True)
    b.chunk = 8; b.state_bits = 16
    x = torch.randn(1, 12, 24)
    with torch.no_grad():
        st = _fresh_head_state(b, 1)
        for t in range(12):
            y, st = b.memory(x[:, t : t + 1], state=st)
        assert torch.isfinite(y).all()
        assert torch.allclose(st.S, fake_quant(st.S.reshape(-1, 24), 16).view_as(st.S), atol=1e-6)   # what is carried is on the 16-bit grid


def _head_fields(st):
    return [st.S, st.Ksum, st.last_idx, st.last_val, st.conv_tail]


def _same_fields(a, b):
    return all((p is None and q is None) or (p is not None and q is not None and p.dtype == q.dtype and torch.equal(p, q)) for p, q in zip(a, b))


def test_head_state_passed_in_survives_repeated_calls_bit_for_bit():
    from flybrain.model.sparsela import Block, KeyCode
    for key_conv in (2, 4):
        for decay in (1.0, 0.9):
            for state_bits in (0, 16):
                torch.manual_seed(0)
                b = Block(24, KeyCode("sparse", 24, 96, 4, 4, 1.0, 0), decay, d_ff=1, key_conv=key_conv, normalize=True, gate=True)
                b.chunk = 8; b.state_bits = state_bits
                x = torch.randn(2, 61, 24)
                with torch.no_grad():
                    _, st = b.memory(x[:, :13], state=_fresh_head_state(b, 2))      # a state that is not fresh: the one returned after a first segment
                    assert all(t is not None for t in _head_fields(st))
                    kept = [t.clone() for t in _head_fields(st)]
                    seg = x[:, 13:53]                                              # five chunks, so every chunk's carry feeds the next
                    y1, s1 = b.memory(seg, state=st)
                    y2, s2 = b.memory(seg, state=st)                               # the same state object, the same segment
                    assert torch.equal(y1, y2) and _same_fields(_head_fields(s1), _head_fields(s2)), (key_conv, decay, state_bits)
                    assert _same_fields(_head_fields(st), kept), (key_conv, decay, state_bits)     # all five fields of the state passed in are bit-equal to their clones
                    assert s1 is not st and s2 is not st and s1 is not s2          # a new state each time


def test_head_state_owns_its_storage():
    from flybrain.model.sparsela import Block, KeyCode
    for B in (1, 2):                                                               # at B = 1 a .contiguous() is a no-op and a view would still pin the whole per-call tensor
        for key_conv, state_bits in ((1, 0), (2, 0), (4, 0), (2, 16)):
            torch.manual_seed(0)
            b = Block(24, KeyCode("sparse", 24, 96, 4, 4, 1.0, 0), 0.9, d_ff=1, key_conv=key_conv, normalize=True, gate=True)
            b.chunk = 8; b.state_bits = state_bits
            with torch.no_grad():
                _, st = b.memory(torch.randn(B, 40, 24), state=_fresh_head_state(b, B))   # five chunks
            for name, t in zip(("S", "Ksum", "last_idx", "last_val", "conv_tail"), _head_fields(st)):
                if t is not None:
                    held = t.untyped_storage().nbytes()                            # the bytes this tensor keeps alive
                    assert held == t.numel() * t.element_size(), (B, key_conv, state_bits, name, held)


def test_head_empty_segment_gives_no_output_and_hands_the_state_back():
    import pytest
    from flybrain.model.sparsela import Block, KeyCode
    for key_conv in (1, 2, 4):                                                     # 1: no conv; 2 and 4: the conv's left context is longer than an empty segment
        torch.manual_seed(0)
        b = Block(24, KeyCode("sparse", 24, 96, 4, 4, 1.0, 0), 0.9, d_ff=1, key_conv=key_conv, normalize=True, gate=True)
        x = torch.randn(2, 37, 24)
        with torch.no_grad():
            ref = b.memory(x)                                                      # chunk 0: the T x T parallel form
            b.chunk = 8
            fresh = _fresh_head_state(b, 2)
            y0, s0 = b.memory(x[:, :0], state=fresh)                               # empty at the start of a stream, with a fresh state
            assert y0.shape == (2, 0, 24) and y0.dtype == ref.dtype
            assert s0 is not fresh and _same_fields(_head_fields(s0), _head_fields(_fresh_head_state(b, 2)))
            assert _same_fields(_head_fields(fresh), _head_fields(_fresh_head_state(b, 2)))      # the state passed in is untouched
            y1, s1 = b.memory(x[:, :20], state=s0)
            kept = [None if t is None else t.clone() for t in _head_fields(s1)]
            ym, sm = b.memory(x[:, 20:20], state=s1)                               # empty in the middle of a stream
            assert ym.shape == (2, 0, 24) and ym.dtype == ref.dtype
            assert sm is not s1 and _same_fields(_head_fields(sm), kept)           # a new state with the same values
            assert _same_fields(_head_fields(s1), kept)                            # the one passed in is untouched
            assert not any(p.untyped_storage().data_ptr() == q.untyped_storage().data_ptr() for p, q in zip(_head_fields(sm), _head_fields(s1)) if p is not None)   # and shares no storage with it
            y2, s2 = b.memory(x[:, 20:], state=sm)
            assert torch.allclose(torch.cat([y0, y1, ym, y2], 1), ref, atol=1e-5), key_conv   # the outputs around the empty segments still match
            b.chunk = 0
            with pytest.raises(ValueError):                                        # an empty segment meets the same checks as any other
                b.memory(x[:, :0], state=_fresh_head_state(b, 2))


def _hybrid(**kw):
    from flybrain.model.hybrid import GRUHybrid
    torch.manual_seed(0)
    args = dict(d_model=16, key="sparse", where="post", backbone="mingru", n_layers=4, d_ff=32, n_key=64, top_k=4, key_conv=2,
                normalize=True, gate=True, chunk=8, mem_every=2, local_attn=8, key_convs=(2, 4))
    args.update(kw)
    return GRUHybrid(**args).eval()


def _stream_all(m, x, cuts):
    st = m.init_state(x.size(0)); outs = []; c0 = 0
    for n in cuts:
        y, st = m.stream(x[:, c0 : c0 + n], st)
        outs.append(y); c0 += n
    return torch.cat(outs, 1), st


def test_hybrid_stream_matches_forward_across_segmentations():
    m = _hybrid()
    x = torch.randint(0, 256, (2, 61))
    with torch.no_grad():
        ref, _ = m(x)
        for cuts in ([61], [1, 60], [8, 8, 45], [7, 9, 13, 32], [1] * 61):
            out, st = _stream_all(m, x, cuts)
            assert torch.allclose(out, ref, atol=1e-4), (cuts, float((out - ref).abs().max()))
    assert len(st.heads) == 2 and len(st.attn) == 1 and st.attn[0].offset == 61 and st.batch == 2


def test_stream_covers_the_zero_heads_cell_and_the_heads_only_cell():
    x = torch.randint(0, 256, (2, 40))
    for kw in (dict(where="none", mem_every=0, key_convs=None), dict(local_attn=0)):
        m = _hybrid(**kw)
        with torch.no_grad():
            ref, _ = m(x)
            out, _ = _stream_all(m, x, [3, 17, 20])
        assert torch.allclose(out, ref, atol=1e-4), kw
    assert _hybrid(where="none", mem_every=0, key_convs=None).init_state(1).heads == []


def test_stream_refuses_what_it_cannot_do():
    import pytest
    from flybrain.model.hybrid import GRUHybrid
    with pytest.raises(ValueError):
        GRUHybrid(d_model=16, backbone="transformer", where="none", n_layers=2, d_ff=32).init_state(1)
    with pytest.raises(ValueError):
        _hybrid(chunk=0).init_state(1)
    with pytest.raises(ValueError):
        _hybrid(count_head=True).init_state(1)
    m = _hybrid()
    with pytest.raises(ValueError):                        # a state for 2 agents, tokens for 3
        m.stream(torch.randint(0, 256, (3, 5)), m.init_state(2))


def test_new_document_resets_the_reader_and_keeps_the_tables():
    m = _hybrid()
    a = torch.randint(0, 256, (1, 30)); b = torch.randint(0, 256, (1, 26))
    with torch.no_grad():
        _, st_a = m.stream(a, m.init_state(1))
        # with the tables emptied as well, a new document is exactly a fresh forward
        nd = st_a.new_document()
        for head, empty in zip(nd.heads, m.init_state(1).heads):
            head.S, head.Ksum = empty.S, empty.Ksum
        out, _ = m.stream(b, nd)
        ref, _ = m(b)
        assert torch.allclose(out, ref, atol=1e-4)
        # the lowest head's keys depend on no memory read, so its table is additive across documents
        _, st_ab = m.stream(b, st_a.new_document())
        _, st_b = m.stream(b, m.init_state(1))
        assert torch.allclose(st_ab.heads[1].S, st_a.heads[1].S + st_b.heads[1].S, atol=1e-5)
        assert not torch.allclose(st_ab.heads[0].S, st_a.heads[0].S + st_b.heads[0].S, atol=1e-5)   # the top head reads the lower one


def test_stream_runs_under_bf16_autocast_close_to_fp32():
    m = _hybrid()
    x = torch.randint(0, 256, (2, 33))
    with torch.no_grad():
        ref, _ = _stream_all(m, x, [9, 24])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out, _ = _stream_all(m, x, [9, 24])
    assert (out.float() - ref).abs().max() < 0.5          # the same budget the repo's bf16 forward test uses


def _state_snapshot(st):
    """Every tensor of an AgentState cloned (None where a field is empty), in a fixed order, and the attention offsets."""
    cl = lambda t: None if t is None else t.clone()
    return [cl(t) for t in st.h] + [cl(t) for hd in st.heads for t in _head_fields(hd)] + [cl(t) for c in st.attn for t in (c.k, c.v)], [c.offset for c in st.attn]


def _same_state(st, snap):
    """st is bit-equal to a snapshot: the same tensors (dtype and values, None where it was None) and the same offsets."""
    now = _state_snapshot(st)
    return len(now[0]) == len(snap[0]) and now[1] == snap[1] and _same_fields(now[0], snap[0])


def test_stream_empty_segment_gives_no_logits_and_hands_the_state_back():
    m = _hybrid()
    x = torch.randint(0, 256, (2, 61))
    with torch.no_grad():
        ref, _ = m(x)
        fresh = m.init_state(2); fresh_kept = _state_snapshot(fresh)
        y0, s0 = m.stream(x[:, :0], fresh)               # empty at the very start, with a fresh state
        assert y0.shape == (2, 0, 256) and y0.dtype == ref.dtype and s0.batch == 2
        assert s0 is not fresh and _same_state(fresh, fresh_kept)         # a new state, and the one passed in is untouched
        ya, sa = m.stream(x, fresh); yb, sb = m.stream(x, s0)
        assert torch.equal(ya, yb) and _same_state(sa, _state_snapshot(sb))   # the state it hands back reads exactly like a fresh one
        y1, s1 = m.stream(x[:, :25], s0)
        kept = _state_snapshot(s1)
        ym, sm = m.stream(x[:, 25:25], s1)               # empty in the middle of a stream
        assert ym.shape == (2, 0, 256) and ym.dtype == ref.dtype
        assert sm is not s1 and _same_state(sm, kept) and _same_state(s1, kept)   # a new state with the same values; the one passed in is untouched
        y2, s2 = m.stream(x[:, 25:], sm)
        assert torch.allclose(torch.cat([y0, y1, ym, y2], 1), ref, atol=1e-4)     # everything read around the empty segments still matches forward
        y3, s3 = m.stream(x[:, 25:], s1)                 # and the rest reads exactly as it does when the empty segment is left out
        assert torch.equal(y2, y3) and _same_state(s2, _state_snapshot(s3)) and s2.attn[0].offset == 61


def test_stream_leaves_the_state_it_is_given_alone_bit_for_bit():
    m = _hybrid()
    x = torch.randint(0, 256, (2, 61))
    with torch.no_grad():
        _, st = m.stream(x[:, :13], m.init_state(2))     # a state that is not fresh: the one returned after a first segment
        kept = _state_snapshot(st)
        assert all(t is not None for t in kept[0])       # nothing is empty, so every field is checked
        seg = x[:, 13:]                                  # 48 positions: six attention windows, six head chunks and two scan chunks
        y1, s1 = m.stream(seg, st)
        y2, s2 = m.stream(seg, st)                       # the same state object, the same segment
        assert torch.equal(y1, y2) and _same_state(s1, _state_snapshot(s2))
        assert _same_state(st, kept)                     # h, all five fields of both heads, k and v of the cache and its offset: bit-equal to their clones
        assert s1 is not st and s2 is not st and s1 is not s2


def test_stream_under_bf16_autocast_stays_within_its_noise_of_forward_under_the_same_autocast():
    m = _hybrid()
    x = torch.randint(0, 256, (2, 33))
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        ref, _ = m(x)
        for cuts in ([9, 24], [1] * 33):
            out, _ = _stream_all(m, x, cuts)
            diff = float((out.float() - ref.float()).abs().max())
            # measured 0.098 at both cuts; logits reach 2.3 and a bf16 step near 1 is 0.008. forward at T = 33 is itself 0.098 from forward at
            # T = 40 (T = 33 pads its attention blocks), so the gap is bf16 noise of the one-pass form. The bound is 10x the measurement.
            assert diff < 1.0, (cuts, diff)
