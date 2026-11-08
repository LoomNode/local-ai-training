"""Two-timescale learning: synapse adapters are state; the shared program stays frozen."""

from __future__ import annotations

import torch

from biointel.accounting import count_resources
from biointel.graphs import erdos_renyi
from biointel.network import SharedRuleNet
from biointel.plasticity import derangement, inner_adapt, program_tensors
from biointel.task import delayed_identity_batch, train_delayed_identity


def _stateful_net(n: int = 16, hidden: int = 8) -> SharedRuleNet:
    g = erdos_renyi(n=n, p=0.3, seed=4)
    return SharedRuleNet(
        graph=g,
        neuron="recurrent",
        synapse="stateful",
        hidden=hidden,
        input_dim=4,
        output_dim=4,
    )


def test_stateful_synapses_are_state_not_trainable_scalars():
    shared = SharedRuleNet(
        graph=erdos_renyi(n=16, p=0.3, seed=4),
        neuron="recurrent",
        synapse="shared",
        hidden=8,
        input_dim=4,
        output_dim=4,
    )
    stateful = _stateful_net()
    n_shared = sum(p.numel() for p in shared.parameters() if p.requires_grad)
    n_stateful = sum(p.numel() for p in stateful.parameters() if p.requires_grad)
    assert n_stateful == n_shared
    readout_adapter = stateful.output_dim * stateful.hidden
    assert stateful.persistent_state_scalars(batch=1) == 16 * 8 + stateful.n_edges + readout_adapter
    report = count_resources(
        stateful,
        persistent_state_scalars=stateful.persistent_state_scalars(batch=1),
        neuron_count=stateful.graph.n,
        synapse_count=stateful.graph.n_edges,
    )
    assert report.trainable_scalars == n_shared
    assert report.persistent_state_scalars > 16 * 8


def test_inner_adapt_does_not_change_the_shared_program():
    net = _stateful_net()
    before = program_tensors(net)
    inner_adapt(net, n_classes=4, delay=3, steps=5, perm=derangement(4, seed=1), lr=0.05, seed=0)
    after = program_tensors(net)
    assert len(before) == len(after)
    for a, b in zip(before, after, strict=True):
        assert torch.equal(a, b)
    assert net.synapse_state is not None
    assert not torch.allclose(net.synapse_state, torch.ones_like(net.synapse_state))


def test_inner_loop_recovers_after_label_permutation_without_backprop_on_theta():
    torch.manual_seed(0)
    net = _stateful_net(n=16, hidden=12)
    train_delayed_identity(net, n_classes=4, delay=3, steps=80, batch_size=32, lr=0.05, seed=0)
    perm = derangement(4, seed=2)
    adapted = inner_adapt(
        net,
        n_classes=4,
        delay=3,
        steps=60,
        perm=perm,
        lr=0.05,
        batch_size=32,
        seed=3,
    )
    assert adapted["shift_adapt_acc"] >= adapted["shift_frozen_acc"] + 0.15
    assert adapted["program_changed"] is False
    x, y = delayed_identity_batch(8, 4, 3, seed=11)
    y = perm[y]
    logits = net.forward_sequence(x)
    assert logits.shape == (8, 4)
