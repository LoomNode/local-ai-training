"""Local G_φ: lifetime updates with no BPTT; resetting adapters restores the program's task."""

from __future__ import annotations

import torch

from biointel.graphs import erdos_renyi
from biointel.network import SharedRuleNet, unique_gru
from biointel.plasticity import derangement, evaluate_accuracy_perm, local_adapt, program_tensors, reset_adapters
from biointel.task import evaluate_accuracy, last_cue_batch, train_delayed_identity, train_task


def _net() -> SharedRuleNet:
    torch.manual_seed(0)
    g = erdos_renyi(n=16, p=0.3, seed=4)
    return SharedRuleNet(
        graph=g,
        neuron="recurrent",
        synapse="stateful",
        hidden=12,
        input_dim=4,
        output_dim=4,
    )


def test_last_cue_batch_puts_the_symbol_on_the_final_step():
    x, y = last_cue_batch(batch=8, n_classes=4, delay=5, seed=0)
    assert x.shape == (8, 6, 4)
    assert (x[:, :-1] == 0).all()
    assert (x[:, -1].sum(dim=-1) == 1).all()
    assert torch.equal(x[:, -1].argmax(dim=-1), y)


def test_local_adapt_does_not_call_backward():
    net = _net()
    hits = {"n": 0}
    orig = torch.Tensor.backward

    def wrapped(*args, **kwargs):
        hits["n"] += 1
        return orig(*args, **kwargs)

    torch.Tensor.backward = wrapped  # type: ignore[method-assign]
    try:
        local_adapt(net, cue="last", n_classes=4, delay=3, steps=3, lr=0.05, seed=0)
    finally:
        torch.Tensor.backward = orig  # type: ignore[method-assign]
    assert hits["n"] == 0


def test_local_adapt_then_reset_keeps_the_program_and_restores_first_cue():
    net = _net()
    train_delayed_identity(net, n_classes=4, delay=3, steps=80, batch_size=32, lr=0.05, seed=0)
    acc_a0 = evaluate_accuracy(net, n_classes=4, delay=3, batches=6, batch_size=32, seed=20, cue="first")
    perm = derangement(4, seed=2)
    acc_b0 = evaluate_accuracy_perm(net, perm, n_classes=4, delay=3, batches=6, batch_size=32, seed=20)
    before = program_tensors(net)
    result = local_adapt(net, cue="first", perm=perm, n_classes=4, delay=3, steps=80, lr=0.2, seed=1)
    after = program_tensors(net)
    for a, b in zip(before, after, strict=True):
        assert torch.equal(a, b)
    acc_b1 = evaluate_accuracy_perm(net, perm, n_classes=4, delay=3, batches=6, batch_size=32, seed=20)
    assert acc_b1 >= acc_b0 + 0.2
    assert result["used_backward"] is False
    reset_adapters(net)
    acc_a_reset = evaluate_accuracy(net, n_classes=4, delay=3, batches=6, batch_size=32, seed=20, cue="first")
    assert acc_a_reset >= acc_a0 - 0.05


def test_gru_finetune_on_permuted_labels_overwrites_identity():
    torch.manual_seed(0)
    gru = unique_gru(4, 12, 4)
    train_task(gru, cue="first", n_classes=4, delay=3, steps=80, batch_size=32, lr=0.05, seed=0)
    acc_a0 = evaluate_accuracy(gru, n_classes=4, delay=3, batches=6, batch_size=32, seed=20, cue="first")
    perm = derangement(4, seed=2)
    train_task(gru, cue="first", perm=perm, n_classes=4, delay=3, steps=80, batch_size=32, lr=0.05, seed=1)
    acc_a1 = evaluate_accuracy(gru, n_classes=4, delay=3, batches=6, batch_size=32, seed=20, cue="first")
    assert acc_a0 >= 0.7
    assert acc_a1 <= acc_a0 - 0.3
