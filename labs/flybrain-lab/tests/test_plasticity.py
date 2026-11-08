"""V1 structural plasticity: prune/grow stay sparse and sign-locked."""

from __future__ import annotations

import torch
from torch import nn

from flybrain.data.connectome import degree_matched_shuffle, synthetic_flylike
from flybrain.model.brain import FlyLanguageModel
from flybrain.plasticity.rewire import grow_random, prune_weakest


def _mean_model(n: int = 40, seed: int = 0) -> FlyLanguageModel:
    g = synthetic_flylike(n=n, seed=seed)
    return FlyLanguageModel(g, state_dim=4, readout="mean")


def test_prune_drops_weak_edges_and_keeps_sign_lock():
    m = _mean_model(n=40, seed=3)
    before = m.n_synapses
    signs_before = { (int(s), int(d)): float(sg) for s, d, sg in zip(m.src, m.dst, m.sign) }
    dropped = prune_weakest(m, drop=8)
    assert dropped == 8
    assert m.n_synapses == before - 8
    remaining = list(zip(m.src.tolist(), m.dst.tolist()))
    assert len(set(remaining)) == m.n_synapses
    w = m.synapse_weights().detach()
    assert torch.equal(torch.sign(w), m.sign)
    for s, d, sg in zip(m.src.tolist(), m.dst.tolist(), m.sign.tolist()):
        assert signs_before[(s, d)] == sg
    for _, p in m.named_parameters():
        assert list(p.shape) != [40, 40]


def test_prune_then_backward_still_works():
    m = _mean_model(n=32, seed=1)
    prune_weakest(m, drop=5)
    tokens = torch.randint(0, 256, (2, 6))
    logits, state = m(tokens)
    loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 256), tokens[:, 1:].reshape(-1))
    loss.backward()
    assert m.synapse_param.grad is not None
    assert state.shape == (2, 32, 4)


def test_grow_increases_edges_within_sparse_budget():
    m = _mean_model(n=36, seed=4)
    before = m.n_synapses
    budget = before + 20
    added = grow_random(m, n_new=10, seed=7, max_edges=budget)
    assert added == 10
    assert m.n_synapses == before + 10
    assert m.n_synapses <= budget
    assert m.n_synapses < 36 * 36
    pairs = list(zip(m.src.tolist(), m.dst.tolist()))
    assert len(pairs) == len(set(pairs))
    assert all(s != d for s, d in pairs)
    w = m.synapse_weights().detach()
    assert torch.equal(torch.sign(w), m.sign)
    for _, p in m.named_parameters():
        assert list(p.shape) != [36, 36]


def test_grow_respects_max_edges():
    m = _mean_model(n=32, seed=2)
    before = m.n_synapses
    added = grow_random(m, n_new=50, seed=1, max_edges=before + 3)
    assert added == 3
    assert m.n_synapses == before + 3


def test_shuffle_can_prune_and_grow_like_fly():
    g = synthetic_flylike(n=36, seed=8)
    sh = degree_matched_shuffle(g, seed=2)
    fly = FlyLanguageModel(g, state_dim=4, readout="mean")
    rnd = FlyLanguageModel(sh, state_dim=4, readout="mean")
    assert fly.n_synapses == rnd.n_synapses
    prune_weakest(fly, drop=4)
    prune_weakest(rnd, drop=4)
    grow_random(fly, n_new=4, seed=0, max_edges=10_000)
    grow_random(rnd, n_new=4, seed=0, max_edges=10_000)
    assert fly.n_synapses == rnd.n_synapses
