"""Coincidence block as a residual sequence layer: expand → AND → top-k → thin → residual."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from flybrain.baselines.gru import matched_gru
from flybrain.model.seqmixer import CoincidenceLM, DenseFFNLM, matched_dense_ffn


def _lm(**kw) -> CoincidenceLM:
    base = dict(d_model=16, n_expand=64, claws=4, top_k=8, n_read=16, seed=0)
    base.update(kw)
    return CoincidenceLM(**base)


def test_expand_rows_have_unique_signed_claws():
    m = _lm()
    e = m.expand
    assert e.shape == (64, 16)
    assert torch.all((e != 0).sum(dim=1) == 4)
    assert set(e.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert not e.requires_grad


def test_mixer_activation_is_topk_sparse():
    m = _lm(top_k=8)
    x = torch.randint(0, 256, (3, 5))
    with torch.no_grad():
        _, _, act = m.forward_with_activation(x)
    assert act.shape == (3, 5, 64)
    nnz = (act != 0).sum(dim=-1)
    assert torch.all(nnz <= 8)


def test_forward_shape_and_state_feeds_back():
    m = _lm()
    x = torch.randint(0, 256, (2, 7))
    logits, state = m(x)
    assert logits.shape == (2, 7, 256)
    assert state.shape == (2, 16)
    assert m.persistent_state_scalars(batch=2) == 32
    logits2, _ = m(x, state)
    assert not torch.allclose(logits, logits2)


def test_dense_control_matches_scalars_and_has_no_expand():
    m = _lm()
    n_m = sum(p.numel() for p in m.parameters() if p.requires_grad)
    d = matched_dense_ffn(m)
    assert isinstance(d, DenseFFNLM)
    n_d = sum(p.numel() for p in d.parameters() if p.requires_grad)
    assert abs(n_d - n_m) / n_m < 0.05
    assert not hasattr(d, "expand")
    g = matched_gru(m)
    n_g = sum(p.numel() for p in g.parameters() if p.requires_grad)
    assert abs(n_g - n_m) / n_m < 0.05


def test_sparse_expand_flops_below_dense():
    m = _lm()
    d = matched_dense_ffn(m)
    assert m.flops_per_step(batch=1) < d.flops_per_step(batch=1)


def test_train_step_drops_loss():
    torch.manual_seed(0)
    m = _lm()
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    x = torch.randint(0, 8, (8, 16))
    y = x.roll(-1, dims=1)
    first = None
    for _ in range(30):
        logits, _ = m(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        if first is None:
            first = float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < first


def test_flop_matched_controls_land_within_five_percent():
    from flybrain.model.seqmixer import flop_matched_dense_ffn, flop_matched_gru

    m = CoincidenceLM(d_model=48, n_expand=1024, claws=4, top_k=64, n_read=48, seed=0)
    f = m.flops_per_step(batch=1)
    d = flop_matched_dense_ffn(m)
    g = flop_matched_gru(m)
    assert abs(d.flops_per_step(1) - f) / f < 0.05
    assert abs(g.flops_per_step(1) - f) / f < 0.05
    n_m = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_d = sum(p.numel() for p in d.parameters() if p.requires_grad)
    assert n_d < n_m  # FLOP parity gives the dense arm fewer scalars, by design
