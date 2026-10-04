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
