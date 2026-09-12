#!/usr/bin/env python3
"""V1 smoke: prune then grow on fly vs shuffle; stay sparse and sign-locked."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.data.connectome import degree_matched_shuffle, synthetic_flylike
from flybrain.model.brain import FlyLanguageModel
from flybrain.plasticity.rewire import grow_random, prune_weakest

RESULTS = ROOT / "results"


def _arm(name: str, model: FlyLanguageModel, drop: int, grow: int, seed: int) -> dict:
    e0 = model.n_synapses
    dropped = prune_weakest(model, drop=drop)
    e1 = model.n_synapses
    added = grow_random(model, n_new=grow, seed=seed, max_edges=e0 + grow)
    e2 = model.n_synapses
    w = model.synapse_weights().detach()
    sign_ok = bool(torch.equal(torch.sign(w), model.sign))
    dense = any(list(p.shape) == [model.connectome.n, model.connectome.n] for p in model.parameters())
    tokens = torch.randint(0, 256, (2, 8))
    logits, _ = model(tokens)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), tokens[:, 1:].reshape(-1))
    loss.backward()
    return {
        "arm": name,
        "edges_before": e0,
        "dropped": dropped,
        "edges_after_prune": e1,
        "added": added,
        "edges_after_grow": e2,
        "sign_locked": sign_ok,
        "dense_NxN": dense,
        "loss": float(loss.detach()),
        "trainable_scalars": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def main() -> None:
    g = synthetic_flylike(n=64, seed=0)
    sh = degree_matched_shuffle(g, seed=1)
    fly = FlyLanguageModel(g, state_dim=4, readout="mean")
    rnd = FlyLanguageModel(sh, state_dim=4, readout="mean")
    rows = [
        _arm("fly", fly, drop=16, grow=16, seed=0),
        _arm("shuffle", rnd, drop=16, grow=16, seed=0),
    ]
    payload = {
        "run_name": "v1_prune_grow",
        "n": 64,
        "note": "V1 structural plasticity smoke; not a language result",
        "arms": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "v1_prune_grow.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    for r in rows:
        print(
            f"{r['arm']:8} E {r['edges_before']} -> prune {r['edges_after_prune']} "
            f"-> grow {r['edges_after_grow']} sign_locked={r['sign_locked']} "
            f"dense={r['dense_NxN']} loss={r['loss']:.3f}"
        )


if __name__ == "__main__":
    main()
