#!/usr/bin/env python3
"""Two-timescale shift: remap labels, adapt adapters, leave the shared program frozen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biointel.accounting import count_resources
from biointel.graphs import erdos_renyi
from biointel.network import SharedRuleNet, unique_gru
from biointel.plasticity import derangement, evaluate_accuracy_perm, inner_adapt
from biointel.task import delayed_identity_batch, evaluate_accuracy, train_delayed_identity

RESULTS = ROOT / "results"


def _finetune_perm(net, perm, *, n_classes: int, delay: int, steps: int, batch_size: int, lr: float, seed: int) -> float:
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for step in range(steps):
        x, y = delayed_identity_batch(batch_size, n_classes, delay, seed=seed + step)
        y = perm[y]
        logits = net.forward_sequence(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return evaluate_accuracy_perm(net, perm, n_classes=n_classes, delay=delay, batches=8, batch_size=batch_size, seed=seed + 50)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="")
    args = p.parse_args()
    tiny = args.tiny
    n = 12 if tiny else 16
    hidden = 8 if tiny else 12
    outer_steps = 8 if tiny else 80
    inner_steps = 8 if tiny else 60
    delay = 3
    n_classes = 4
    batch_size = 16 if tiny else 32
    lr = 0.05
    perm = derangement(n_classes, seed=args.seed + 2)

    g = erdos_renyi(n=n, p=0.3, seed=4)
    plastic = SharedRuleNet(graph=g, neuron="recurrent", synapse="stateful", hidden=hidden, input_dim=n_classes, output_dim=n_classes)
    gru = unique_gru(n_classes, hidden, n_classes)

    outer_p = train_delayed_identity(plastic, n_classes=n_classes, delay=delay, steps=outer_steps, batch_size=batch_size, lr=lr, seed=args.seed)
    outer_g = train_delayed_identity(gru, n_classes=n_classes, delay=delay, steps=outer_steps, batch_size=batch_size, lr=lr, seed=args.seed)
    id_acc_p = evaluate_accuracy(plastic, n_classes=n_classes, delay=delay, batches=6, batch_size=batch_size, seed=args.seed + 1000)
    id_acc_g = evaluate_accuracy(gru, n_classes=n_classes, delay=delay, batches=6, batch_size=batch_size, seed=args.seed + 1000)

    adapted = inner_adapt(
        plastic,
        n_classes=n_classes,
        delay=delay,
        steps=inner_steps,
        perm=perm,
        lr=lr,
        batch_size=batch_size,
        seed=args.seed + 3,
    )
    gru_frozen = evaluate_accuracy_perm(gru, perm, n_classes=n_classes, delay=delay, batches=6, batch_size=batch_size, seed=args.seed + 53)
    gru_ft = _finetune_perm(
        gru, perm, n_classes=n_classes, delay=delay, steps=inner_steps, batch_size=batch_size, lr=lr, seed=args.seed + 3
    )

    report = count_resources(
        plastic,
        persistent_state_scalars=plastic.persistent_state_scalars(batch=1),
        neuron_count=plastic.graph.n,
        synapse_count=plastic.graph.n_edges,
        flops_per_step=plastic.flops_per_step(batch=1),
    )
    payload = {
        "tiny": tiny,
        "seed": args.seed,
        "n": n,
        "hidden": hidden,
        "outer_steps": outer_steps,
        "inner_steps": inner_steps,
        "perm": perm.tolist(),
        "plastic": {
            **report.as_dict(),
            "identity_train_acc": outer_p["train_acc"],
            "identity_eval_acc": id_acc_p,
            **adapted,
        },
        "unique_gru": {
            "trainable_scalars": sum(p.numel() for p in unique_gru(n_classes, hidden, n_classes).parameters()),
            "identity_train_acc": outer_g["train_acc"],
            "identity_eval_acc": id_acc_g,
            "shift_frozen_acc": gru_frozen,
            "shift_finetune_acc": gru_ft,
            "finetune_updates_theta": True,
        },
    }
    name = args.run_name or ("shift_tiny" if tiny else "shift_cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    print(
        f"plastic identity={id_acc_p:.3f} shift_frozen={adapted['shift_frozen_acc']:.3f} "
        f"shift_adapt={adapted['shift_adapt_acc']:.3f} theta_changed={adapted['program_changed']}"
    )
    print(f"gru     identity={id_acc_g:.3f} shift_frozen={gru_frozen:.3f} shift_finetune={gru_ft:.3f} (updates θ)")


if __name__ == "__main__":
    main()
