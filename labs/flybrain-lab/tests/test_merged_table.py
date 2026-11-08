"""One table (design 2026-11-05): a learned prior under each memory head's table, an n-gram term in its address, and the
separate Qwen-style n-gram embedding that arm B uses as the control."""

import math

import pytest
import torch

from flybrain.model.hybrid import GRUHybrid, NgramEmbed, ngram_hash
from flybrain.model.sparsela import KeyCode, sparse_memory_read


def small(**kw):
    torch.manual_seed(0)
    base = dict(d_model=32, key="sparse", where="post", n_key=64, top_k=4, backbone="mingru", n_layers=4, d_ff=32,
                normalize=True, key_conv=2, decay=1.0, seed=0, gate=True, chunk=8, mem_every=2, local_attn=8)
    base.update(kw)
    return GRUHybrid(**base)


def code(d=16, n_key=64):
    torch.manual_seed(0)
    return KeyCode("sparse", d, n_key, claws=4, top_k=4, theta=0.0, seed=3)


def test_ngram_hash_is_causal_and_order_sensitive():
    toks = torch.tensor([[1, 2, 3, 4, 2, 3]])
    h2 = ngram_hash(toks, 2, 1000, salt=7)
    assert h2.shape == toks.shape and int(h2.min()) >= 0 and int(h2.max()) < 1000
    assert h2[0, 2] == h2[0, 5]                    # both end in the bigram (2, 3)
    assert h2[0, 1] != h2[0, 2] or h2[0, 1] != h2[0, 3]
    changed = toks.clone(); changed[0, 4] = 9      # a later byte moves only the hashes that include it
    h2c = ngram_hash(changed, 2, 1000, salt=7)
    assert torch.equal(h2[0, :4], h2c[0, :4])
    assert not torch.equal(ngram_hash(toks, 2, 1000, salt=7), ngram_hash(toks, 2, 1000, salt=8))


def test_a_vanishing_prior_reads_like_no_prior():
    c = code(); torch.manual_seed(1)
    xk, v = torch.randn(2, 20, 16), torch.randn(2, 20, 16)
    plain = sparse_memory_read(c, xk, v, 8, 1.0, True, None)
    tiny = sparse_memory_read(c, xk, v, 8, 1.0, True, None, prior=(1e-12 * torch.randn(64, 16), torch.full((64,), 1e-12)))
    assert torch.allclose(plain, tiny, atol=1e-6)


def test_the_prior_is_what_position_zero_reads():
    c = code(); torch.manual_seed(1)
    xk, v = torch.randn(1, 5, 16), torch.randn(1, 5, 16)
    V0, m0 = torch.randn(64, 16), torch.rand(64) + 0.5
    out = sparse_memory_read(c, xk, v, 8, 1.0, True, None, prior=(m0[:, None] * V0, m0))
    idx, val = c.sparse_code(xk)
    val = val / (val.norm(dim=-1, keepdim=True) + 1e-6)
    q_i, q_v = idx[0, 0], val[0, 0]
    want = (q_v[:, None] * m0[q_i, None] * V0[q_i]).sum(0) / ((q_v * m0[q_i]).sum() + 1e-3)
    assert torch.allclose(out[0, 0], want, atol=1e-5)


def test_the_address_offset_moves_reads_and_writes_together():
    c = code(); torch.manual_seed(1)
    xk, v = torch.randn(1, 24, 16), torch.randn(1, 24, 16)
    same = torch.full((1, 24), 5, dtype=torch.long)
    # one constant offset is a relabelling of slots: with no prior, the reads are unchanged
    assert torch.allclose(sparse_memory_read(c, xk, v, 8, 1.0, True, None),
                          sparse_memory_read(c, xk, v, 8, 1.0, True, None, idx_offset=same), atol=1e-6)
    # offsets that differ by position separate what was written under another n-gram
    varied = torch.arange(24)[None] * 17
    assert not torch.allclose(sparse_memory_read(c, xk, v, 8, 1.0, True, None),
                              sparse_memory_read(c, xk, v, 8, 1.0, True, None, idx_offset=varied), atol=1e-4)


def test_the_prior_learns():
    m = small(mem_prior=True)
    toks = torch.randint(0, 256, (2, 32))
    logits, _ = m(toks)
    logits.logsumexp(-1).mean().backward()
    heads = m.all_heads()
    assert len(heads) == 2 and all(h.prior_V.grad is not None and h.prior_V.grad.abs().sum() > 0 for h in heads)
    assert all(h.prior_r.grad is not None for h in heads)
    assert math.isclose(float(torch.nn.functional.softplus(heads[0].prior_r[0])), math.log(2), rel_tol=1e-5)


@pytest.mark.parametrize("kw", [dict(mem_prior=True), dict(mem_prior=True, ngram_addr=(2, 3)), dict(ngram_embed=64)])
def test_new_arms_stay_causal(kw):
    m = small(**kw).eval()
    toks = torch.randint(0, 256, (1, 40))
    later = toks.clone(); later[0, 30:] = torch.randint(0, 256, (10,))
    with torch.no_grad():
        a, _ = m(toks); b, _ = m(later)
    assert torch.allclose(a[0, :30], b[0, :30], atol=1e-5)


def test_ngram_embed_starts_silent_and_sits_after_layer_two():
    e = NgramEmbed(32, rows=64, orders=(2, 3), seed=0)
    toks = torch.randint(0, 256, (2, 10))
    assert torch.count_nonzero(e(toks)) == 0                 # zero-initialised output projection
    m = small(ngram_embed=64)
    assert m.ngram_mid is not None and m.ngram_after == 2 and m.ngram is None   # `ngram` is the input-side channel (--ngram-emb)
    assert sum(p.numel() for p in m.ngram_mid.parameters()) == 2 * 64 * 32 + 32 * 32


def test_a_pre_merge_checkpoint_of_the_mid_stack_ngram_embed_loads():
    # before the merge with feat/arm-b the mid-stack embedding was the attribute `ngram`: its keys were ngram.tables.* / ngram.proj.*
    m = small(ngram_embed=64)
    m.ngram_mid.proj.weight.data.normal_(0, 0.1)
    old = {("ngram." + k[len("ngram_mid."):] if k.startswith("ngram_mid.") else k): v for k, v in m.state_dict().items()}
    assert "ngram.tables.0.weight" in old and "ngram.proj.weight" in old
    r = small(ngram_embed=64)
    r.load_state_dict(old)
    toks = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        assert torch.equal(m.eval()(toks)[0], r.eval()(toks)[0])


def test_param_budgets_match_within_three_percent_at_20m():
    a = GRUHybrid(d_model=512, n_key=4096, backbone="mingru", n_layers=8, d_ff=1024, chunk=256, mem_every=2, local_attn=512,
                  key_convs=(2, 4), gate=True)
    b = GRUHybrid(d_model=512, n_key=4096, backbone="mingru", n_layers=8, d_ff=1024, chunk=256, mem_every=2, local_attn=512,
                  key_convs=(2, 4), gate=True, ngram_embed=8192)
    c = GRUHybrid(d_model=512, n_key=4096, backbone="mingru", n_layers=8, d_ff=1024, chunk=256, mem_every=2, local_attn=512,
                  key_convs=(2, 4), gate=True, mem_prior=True)
    n = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    extra_b, extra_c = n(b) - n(a), n(c) - n(a)
    assert abs(extra_b - extra_c) / extra_c < 0.03


def test_streaming_refuses_the_new_parts():
    for kw in (dict(mem_prior=True), dict(ngram_addr=(2,)), dict(ngram_embed=64)):
        with pytest.raises(ValueError):
            small(**kw).init_state(1)


def test_scoring_rebuilds_the_new_arms_from_a_checkpoint():
    from scripts.mask_sweep import rebuild
    a = dict(d_model=32, gru_hidden=16, n_key=64, n_layers=4, d_ff=32, top_k=4, claws=4, theta=1.0, learned_key_dim=64,
             normalize=True, key_conv=2, decay=1.0, seed=0, gate=True, seq_len=64, chunk=8, mem_every=2, local_attn=8,
             key_convs="2,4", mem_prior=True, ngram_addr="2,3", ngram_embed=64)
    m = small(mem_prior=True, ngram_addr=(2, 3), ngram_embed=64, key_convs=(2, 4))
    r = rebuild({"args": a, "arm": "mingru_sparse_post"})
    r.load_state_dict(m.state_dict())
    toks = torch.randint(0, 256, (1, 24))
    with torch.no_grad():
        assert torch.allclose(m.eval()(toks)[0], r.eval()(toks)[0], atol=1e-5)
