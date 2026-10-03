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
        assert torch.equal(full, ref)                    # forward is scan with no carry
        assert torch.allclose(h, h_full, atol=1e-8)


def test_mingru_scan_leaves_the_carry_alone_and_an_empty_segment_hands_it_back():
    from flybrain.model.hybrid import MinGRULayer
    torch.manual_seed(0)
    layer = MinGRULayer(12, 24).eval()
    x = torch.randn(2, 40, 12)
    with torch.no_grad():
        _, h = layer.scan(x[:, :17])
        h_before = h.clone()
        y1, h1 = layer.scan(x[:, 17:], h)
        y2, h2 = layer.scan(x[:, 17:], h)                # the same carry again gives the same answer, so scan did not modify it
        assert torch.equal(h, h_before) and torch.equal(y1, y2) and torch.equal(h1, h2)
        y0, h_same = layer.scan(x[:, :0], h)             # an empty segment: no output, and the carry comes back unchanged
        assert y0.shape == (2, 0, 12) and h_same.dtype == torch.float64 and torch.equal(h_same, h_before)
        _, h_zero = layer.scan(x[:, :0])
        assert torch.equal(h_zero, torch.zeros(2, 12, dtype=torch.float64))
        assert layer(x[:, :0]).shape == (2, 0, 12)       # forward on an empty sequence still returns an empty tensor, as it always did
