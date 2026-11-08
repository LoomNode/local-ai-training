"""Delayed-identity task and a tiny train smoke that must beat chance with recurrence."""

from __future__ import annotations

import torch

from biointel.accounting import count_resources
from biointel.graphs import erdos_renyi
from biointel.network import SharedRuleNet
from biointel.task import delayed_identity_batch, evaluate_accuracy, train_delayed_identity


def test_delayed_identity_batch_shapes_and_labels():
    x, y = delayed_identity_batch(batch=16, n_classes=4, delay=6, seed=0)
    assert x.shape == (16, 7, 4)  # 1 cue step + 6 delay
    assert y.shape == (16,)
    assert y.min() >= 0 and y.max() < 4
    # Cue lives only on step 0.
    assert (x[:, 1:] == 0).all()
    assert (x[:, 0].sum(dim=-1) == 1).all()


def test_untrained_accuracy_is_near_chance():
    g = erdos_renyi(n=16, p=0.25, seed=0)
    net = SharedRuleNet(graph=g, neuron="recurrent", hidden=8, input_dim=4, output_dim=4)
    acc = evaluate_accuracy(net, n_classes=4, delay=5, batches=8, batch_size=32, seed=1)
    assert 0.1 <= acc <= 0.45  # chance is 0.25; untrained should not solve it


def test_recurrent_train_smoke_beats_chance():
    torch.manual_seed(0)
    g = erdos_renyi(n=16, p=0.3, seed=2)
    net = SharedRuleNet(graph=g, neuron="recurrent", hidden=12, input_dim=4, output_dim=4)
    result = train_delayed_identity(
        net,
        n_classes=4,
        delay=4,
        steps=80,
        batch_size=32,
        lr=0.05,
        seed=0,
    )
    acc = evaluate_accuracy(net, n_classes=4, delay=4, batches=8, batch_size=32, seed=99)
    assert acc >= 0.55
    assert result["steps"] == 80
    report = count_resources(
        net,
        persistent_state_scalars=net.persistent_state_scalars(batch=1),
        neuron_count=net.graph.n,
        synapse_count=net.graph.n_edges,
        flops_per_step=net.flops_per_step(batch=1),
    )
    payload = {**result, **report.as_dict()}
    assert payload["trainable_scalars"] > 0
    assert payload["persistent_state_scalars"] == 16 * 12
    assert payload["flops_per_step"] > 0
    assert torch.isfinite(torch.tensor(result["loss"]))
