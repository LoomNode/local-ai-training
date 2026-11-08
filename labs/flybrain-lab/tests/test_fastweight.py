"""KC→MBON-style fast weights on the coincidence block: Hebbian, modulated, in-sequence."""

from __future__ import annotations

import torch

from flybrain.data.task import copy_batch
from flybrain.model.fastweight import FastWeightLM, flop_matched_dense_fw


def _lm(**kw) -> FastWeightLM:
    base = dict(d_model=16, n_expand=64, claws=4, top_k=8, n_read=16, seed=0, code="sparse", fast=True)
    base.update(kw)
    return FastWeightLM(**base)


def test_fast_weights_start_empty_so_first_step_matches_slow_model():
    a = _lm(fast=True)
    b = _lm(fast=False)
    b.load_state_dict(a.state_dict(), strict=False)
    x = torch.randint(0, 256, (2, 1))
    with torch.no_grad():
        la, _ = a(x)
        lb, _ = b(x)
    assert torch.allclose(la, lb)


def test_fast_weights_change_later_logits():
    a = _lm(fast=True)
    b = _lm(fast=False)
    b.load_state_dict(a.state_dict(), strict=False)
    x = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        la, _ = a(x)
        lb, _ = b(x)
    assert torch.allclose(la[:, 0], lb[:, 0])
    assert not torch.allclose(la[:, -1], lb[:, -1])


def test_fast_state_is_returned_and_feeds_back():
    m = _lm()
    x = torch.randint(0, 256, (2, 6))
    logits, state = m(x)
    assert logits.shape == (2, 6, 256)
    assert isinstance(state, tuple) and len(state) == 2
    h, F = state
    assert h.shape == (2, 16) and F.shape == (2, 16, 64)
    assert F.abs().sum() > 0
    logits2, _ = m(x, state)
    assert not torch.allclose(logits, logits2)


def test_dense_fw_control_is_flop_matched_and_wider_costs_more():
    m = _lm(d_model=48, n_expand=1024, top_k=64, n_read=48)
    d = flop_matched_dense_fw(m)
    f = m.flops_per_step(1)
    assert d.code == "dense"
    assert abs(d.flops_per_step(1) - f) / f < 0.05
    assert d.n_expand < m.n_expand


def test_copy_batch_mask_scores_second_occurrence_only():
    x, y, mask = copy_batch(4, seq_len=32, span=8, seed=0)
    assert x.shape == y.shape == mask.shape == (4, 32)
    assert torch.all(y[:, :-1] == x[:, 1:])
    assert torch.all(mask.sum(dim=1) == 7)
    for b in range(4):
        idx = mask[b].nonzero().view(-1)
        p2 = int(idx[0])
        second = x[b, p2 : p2 + 8]
        earlier = [x[b, p : p + 8] for p in range(0, p2 - 8 + 1)]
        assert any(torch.equal(w, second) for w in earlier)
        assert torch.all(y[b, idx] == x[b, p2 + 1 : p2 + 8])


def test_gla_arm_has_learned_key_and_query_and_is_flop_matched():
    from flybrain.model.fastweight import flop_matched_gla

    m = _lm(d_model=48, n_expand=1024, top_k=64, n_read=48)
    g = flop_matched_gla(m)
    assert g.code == "learned"
    assert g.expand.requires_grad and g.query.weight.requires_grad
    f = m.flops_per_step(1)
    assert abs(g.flops_per_step(1) - f) / f < 0.05
    x = torch.randint(0, 256, (2, 5))
    logits, (h, F) = g(x)
    assert logits.shape == (2, 5, 256) and F.shape == (2, 48, g.n_expand)


def test_constant_gate_has_no_modulator_params():
    m = _lm(gate="const")
    assert not hasattr(m, "mod")
    x = torch.randint(0, 256, (2, 5))
    logits, _ = m(x)
    assert logits.shape == (2, 5, 256)
