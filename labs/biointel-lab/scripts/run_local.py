#!/usr/bin/env python3
"""Local G_φ (no BPTT) vs GRU fine-tune on a label derangement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biointel.graphs import erdos_renyi
from biointel.network import SharedRuleNet, unique_gru
from biointel.plasticity import derangement, evaluate_accuracy_perm, local_adapt, reset_adapters
from biointel.task import evaluate_accuracy, train_delayed_identity, train_task

RESULTS = ROOT / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="")
    args = p.parse_args()
    tiny = args.tiny
    n = 12 if tiny else 16
    hidden = 8 if tiny else 12
    outer = 8 if tiny else 80
    inner = 8 if tiny else 80
    delay = 3
    n_classes = 4
    batch = 16 if tiny else 32
    perm = derangement(n_classes, seed=args.seed + 2)

    g = erdos_renyi(n=n, p=0.3, seed=4)
    plastic = SharedRuleNet(
        graph=g, neuron="recurrent", synapse="stateful", hidden=hidden, input_dim=n_classes, output_dim=n_classes
    )
    gru = unique_gru(n_classes, hidden, n_classes)

    train_delayed_identity(plastic, n_classes=n_classes, delay=delay, steps=outer, batch_size=batch, lr=0.05, seed=args.seed)
    train_task(gru, cue="first", n_classes=n_classes, delay=delay, steps=outer, batch_size=batch, lr=0.05, seed=args.seed)

    a0_p = evaluate_accuracy(plastic, n_classes=n_classes, delay=delay, batches=6, batch_size=batch, seed=20)
    a0_g = evaluate_accuracy(gru, n_classes=n_classes, delay=delay, batches=6, batch_size=batch, seed=20)
    b0_p = evaluate_accuracy_perm(plastic, perm, n_classes=n_classes, delay=delay, batches=6, batch_size=batch, seed=20)

    adapted = local_adapt(
        plastic, cue="first", perm=perm, n_classes=n_classes, delay=delay, steps=inner, lr=0.2, batch_size=batch, seed=1
    )
    a_on = evaluate_accuracy(plastic, n_classes=n_classes, delay=delay, batches=6, batch_size=batch, seed=20)
    b_on = evaluate_accuracy_perm(plastic, perm, n_classes=n_classes, delay=delay, batches=6, batch_size=batch, seed=20)
    reset_adapters(plastic)
    a_reset = evaluate_accuracy(plastic, n_classes=n_classes, delay=delay, batches=6, batch_size=batch, seed=20)

    train_task(gru, cue="first", perm=perm, n_classes=n_classes, delay=delay, steps=inner, batch_size=batch, lr=0.05, seed=1)
    a_g1 = evaluate_accuracy(gru, n_classes=n_classes, delay=delay, batches=6, batch_size=batch, seed=20)
    b_g1 = evaluate_accuracy_perm(gru, perm, n_classes=n_classes, delay=delay, batches=6, batch_size=batch, seed=20)

    payload = {
        "tiny": tiny,
        "seed": args.seed,
        "perm": perm.tolist(),
        "plastic": {
            "identity_before": a0_p,
            "shift_frozen": b0_p,
            "shift_after_local": b_on,
            "identity_with_adapters": a_on,
            "identity_after_reset": a_reset,
            "used_backward": adapted["used_backward"],
            "program_changed": adapted["program_changed"],
        },
        "unique_gru": {
            "identity_before": a0_g,
            "identity_after_finetune": a_g1,
            "shift_after_finetune": b_g1,
            "finetune_updates_theta": True,
        },
    }
    name = args.run_name or ("local_tiny" if tiny else "local_cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    pl = payload["plastic"]
    print(
        f"plastic A={pl['identity_before']:.3f} B_frozen={pl['shift_frozen']:.3f} "
        f"B_local={pl['shift_after_local']:.3f} A_adapters={pl['identity_with_adapters']:.3f} "
        f"A_reset={pl['identity_after_reset']:.3f} backward={pl['used_backward']} theta_changed={pl['program_changed']}"
    )
    ug = payload["unique_gru"]
    print(
        f"gru     A={ug['identity_before']:.3f} A_after_ft={ug['identity_after_finetune']:.3f} "
        f"B_after_ft={ug['shift_after_finetune']:.3f} (updates θ)"
    )


if __name__ == "__main__":
    main()
