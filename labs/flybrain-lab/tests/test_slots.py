"""V2 neuron slots: activate/deactivate inside a preallocated capacity."""

from __future__ import annotations

import torch

from flybrain.data.connectome import synthetic_flylike
from flybrain.model.brain import FlyLanguageModel


def test_capacity_is_preallocated_and_live_count_changes_in_place():
    g = synthetic_flylike(n=24, seed=1)
    extra = 6
    m = FlyLanguageModel(g, state_dim=4, readout="mean", capacity=g.n + extra)
    assert m.capacity == g.n + extra
    assert m.n_live == g.n
    reserved = m.capacity - 1
    assert not bool(m.live_mask[reserved])
    m.activate_slot(reserved)
    assert m.n_live == g.n + 1
    assert m.capacity == g.n + extra
    m.deactivate_slot(reserved)
    assert m.n_live == g.n
    assert m.capacity == g.n + extra


def test_inactive_slots_do_not_update():
    g = synthetic_flylike(n=24, seed=2)
    m = FlyLanguageModel(g, state_dim=4, readout="mean", capacity=g.n + 4)
    idx = m.capacity - 1
    tokens = torch.randint(0, 256, (1, 4))
    _, h0 = m(tokens)
    assert torch.allclose(h0[:, idx, :], torch.zeros_like(h0[:, idx, :]))
    m.activate_slot(idx)
    _, h1 = m(tokens)
    # GRU bias moves an active slot even with no edges.
    assert not torch.allclose(h1[:, idx, :], torch.zeros_like(h1[:, idx, :]))
    frozen = h1[:, idx, :].detach().clone()
    m.deactivate_slot(idx)
    more = torch.randint(0, 256, (1, 4))
    _, h2 = m(more, state=h1)
    assert torch.allclose(h2[:, idx, :], frozen)


def test_deactivate_stops_a_live_neuron_from_affecting_others():
    g = synthetic_flylike(n=28, seed=3)
    m = FlyLanguageModel(g, state_dim=4, readout="mean")
    src = int(m.src[0])
    tokens = torch.randint(0, 256, (1, 3))
    _, h_on = m(tokens)
    m.deactivate_slot(src)
    _, h_off = m(tokens)
    assert not torch.allclose(h_on, h_off)
