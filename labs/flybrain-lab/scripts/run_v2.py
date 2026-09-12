#!/usr/bin/env python3
"""V2 smoke: reserved neuron slots activate/deactivate without resizing capacity."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.data.connectome import synthetic_flylike
from flybrain.model.brain import FlyLanguageModel

RESULTS = ROOT / "results"


def main() -> None:
    g = synthetic_flylike(n=32, seed=0)
    extra = 8
    m = FlyLanguageModel(g, state_dim=4, readout="mean", capacity=g.n + extra)
    tokens = torch.randint(0, 256, (1, 6))
    _, h0 = m(tokens)
    reserved = m.capacity - 1
    inactive_norm = float(h0[:, reserved, :].detach().abs().sum())
    m.activate_slot(reserved)
    live_after_on = m.n_live
    cap = m.capacity
    _, h1 = m(tokens)
    active_norm = float(h1[:, reserved, :].detach().abs().sum())
    frozen = h1[:, reserved, :].detach().clone()
    m.deactivate_slot(reserved)
    _, h2 = m(tokens, state=h1)
    frozen_ok = bool(torch.allclose(h2[:, reserved, :], frozen))
    payload = {
        "run_name": "v2_slots",
        "n": g.n,
        "capacity": cap,
        "live_init": g.n,
        "live_after_activate": live_after_on,
        "live_after_deactivate": m.n_live,
        "capacity_unchanged": cap == g.n + extra and cap == m.capacity,
        "inactive_slot_norm": inactive_norm,
        "activated_slot_norm": active_norm,
        "deactivated_state_frozen": frozen_ok,
        "note": "V2 slot-pool smoke; not a language result",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "v2_slots.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    print(
        f"live {payload['live_init']} -> {payload['live_after_activate']} -> {payload['live_after_deactivate']} "
        f"capacity={cap} frozen={frozen_ok} inactive_norm={inactive_norm:.4f} active_norm={active_norm:.4f}"
    )


if __name__ == "__main__":
    main()
