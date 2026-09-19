"""Shared-rule nets: program size must not secretly grow with N; state is not parameters."""

from __future__ import annotations

import torch

from biointel.accounting import count_resources
from biointel.graphs import erdos_renyi
from biointel.network import SharedRuleNet, unique_gru


def _net(n: int, neuron: str, synapse: str = "shared") -> SharedRuleNet:
    g = erdos_renyi(n=n, p=0.2, seed=1)
    return SharedRuleNet(
        graph=g,
        neuron=neuron,
        synapse=synapse,
        hidden=8,
        input_dim=4,
        output_dim=4,
        n_input=4,
        n_output=4,
    )


def test_shared_program_param_count_does_not_grow_with_neuron_count():
    small = _net(16, "recurrent")
    large = _net(64, "recurrent")
    r_small = count_resources(
        small,
        persistent_state_scalars=small.persistent_state_scalars(batch=1),
        neuron_count=small.graph.n,
        synapse_count=small.graph.n_edges,
    )
    r_large = count_resources(
        large,
        persistent_state_scalars=large.persistent_state_scalars(batch=1),
        neuron_count=large.graph.n,
        synapse_count=large.graph.n_edges,
    )
    assert r_small.trainable_scalars == r_large.trainable_scalars
    assert r_large.persistent_state_scalars > r_small.persistent_state_scalars
    assert r_large.neuron_count == 64
    assert r_small.neuron_count == 16


def test_unique_synapse_params_grow_with_edges():
    shared = _net(32, "point", synapse="shared")
    unique = _net(32, "point", synapse="unique")
    assert unique.n_edges == shared.n_edges
    n_shared = sum(p.numel() for p in shared.parameters() if p.requires_grad)
    n_unique = sum(p.numel() for p in unique.parameters() if p.requires_grad)
    assert n_unique == n_shared + unique.n_edges


def test_recurrent_uses_previous_hidden_point_does_not():
    # Empty wiring: the only memory channel is a neuron's own state.
    g = erdos_renyi(n=12, p=0.0, seed=1)
    rec = SharedRuleNet(graph=g, neuron="recurrent", hidden=8, input_dim=4, output_dim=4)
    point = SharedRuleNet(graph=g, neuron="point", hidden=8, input_dim=4, output_dim=4)
    x0 = torch.zeros(2, 4)
    x1 = torch.zeros(2, 4)
    x1[:, 0] = 1.0
    rec.reset_state(2)
    rec.step(x1)
    rec.step(x0)
    h_rec = rec.state.detach().clone()
    rec.reset_state(2)
    rec.step(x0)
    rec.step(x0)
    h_rec_blank = rec.state.detach().clone()
    assert not torch.allclose(h_rec, h_rec_blank)

    point.reset_state(2)
    point.step(x1)
    point.step(x0)
    h_point = point.state.detach().clone()
    point.reset_state(2)
    point.step(x0)
    point.step(x0)
    h_point_blank = point.state.detach().clone()
    # Point F ignores own previous h, so after a zero-input step both histories match.
    assert torch.allclose(h_point, h_point_blank, atol=1e-5)


def test_dendritic_has_two_compartment_aggregates():
    net = _net(20, "dendritic")
    assert net.n_compartments == 2
    net.reset_state(3)
    logits = net.forward_sequence(torch.randn(3, 5, 4))
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()


def test_forward_sequence_shape_and_unique_gru_baseline():
    net = _net(16, "recurrent")
    xs = torch.randn(5, 7, 4)
    out = net.forward_sequence(xs)
    assert out.shape == (5, 4)

    gru = unique_gru(input_dim=4, hidden=8, output_dim=4)
    gout = gru.forward_sequence(xs)
    assert gout.shape == (5, 4)
    n_gru = sum(p.numel() for p in gru.parameters() if p.requires_grad)
    n_shared = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # Unique GRU is allowed to be bigger; the comparison is the point.
    assert n_gru > 0 and n_shared > 0
