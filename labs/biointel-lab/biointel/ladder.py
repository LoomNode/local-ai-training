"""Ablation ladder with identical resource keys on every arm."""

from __future__ import annotations

from typing import Any

from biointel.accounting import count_resources
from biointel.graphs import degree_matched_shuffle, erdos_renyi
from biointel.network import SharedRuleNet, unique_gru
from biointel.task import evaluate_accuracy, train_delayed_identity

LADDER_ARMS = (
    "unique_gru",
    "shared_point",
    "shared_recurrent",
    "shared_dendritic",
    "shared_recurrent_shuffled",
)

_NEURON = {
    "shared_point": "point",
    "shared_recurrent": "recurrent",
    "shared_dendritic": "dendritic",
    "shared_recurrent_shuffled": "recurrent",
}


def run_ladder(*, tiny: bool = True, seed: int = 0) -> list[dict[str, Any]]:
    n = 12 if tiny else 32
    hidden = 8 if tiny else 16
    steps = 4 if tiny else 80
    delay = 3 if tiny else 4
    n_classes = 4
    batch_size = 16 if tiny else 32
    g = erdos_renyi(n=n, p=0.25, seed=seed)
    g_shuf = degree_matched_shuffle(g, seed=seed + 1)
    rows: list[dict[str, Any]] = []
    for arm in LADDER_ARMS:
        if arm == "unique_gru":
            net = unique_gru(n_classes, hidden, n_classes)
            train = train_delayed_identity(
                net, n_classes=n_classes, delay=delay, steps=steps, batch_size=batch_size, seed=seed
            )
            eval_acc = evaluate_accuracy(
                net, n_classes=n_classes, delay=delay, batches=2 if tiny else 8, batch_size=batch_size, seed=seed + 1000
            )
            report = count_resources(
                net,
                persistent_state_scalars=net.persistent_state_scalars(batch=1),
                neuron_count=0,
                synapse_count=0,
                flops_per_step=net.flops_per_step(batch=1),
            )
            rows.append(
                {
                    "arm": arm,
                    "neuron": "gru",
                    "synapse": "unique",
                    "graph": "none",
                    **report.as_dict(),
                    "train_acc": train["train_acc"],
                    "eval_acc": eval_acc,
                    "loss": train["loss"],
                }
            )
            continue
        graph = g_shuf if arm.endswith("shuffled") else g
        net = SharedRuleNet(
            graph=graph,
            neuron=_NEURON[arm],  # type: ignore[arg-type]
            hidden=hidden,
            input_dim=n_classes,
            output_dim=n_classes,
        )
        train = train_delayed_identity(
            net, n_classes=n_classes, delay=delay, steps=steps, batch_size=batch_size, seed=seed
        )
        eval_acc = evaluate_accuracy(
            net, n_classes=n_classes, delay=delay, batches=2 if tiny else 8, batch_size=batch_size, seed=seed + 1000
        )
        report = count_resources(
            net,
            persistent_state_scalars=net.persistent_state_scalars(batch=1),
            neuron_count=net.graph.n,
            synapse_count=net.graph.n_edges,
            flops_per_step=net.flops_per_step(batch=1),
        )
        rows.append(
            {
                "arm": arm,
                "neuron": _NEURON[arm],
                "synapse": "shared",
                "graph": "degree_matched_shuffle" if arm.endswith("shuffled") else "erdos_renyi",
                **report.as_dict(),
                "train_acc": train["train_acc"],
                "eval_acc": eval_acc,
                "loss": train["loss"],
            }
        )
    return rows
