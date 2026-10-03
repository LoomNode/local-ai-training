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
