#!/usr/bin/env python3
"""V3 plumbing episode: playbook actions in a contained sandbox; brain holds H_t."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.data.connectome import synthetic_flylike
from flybrain.env.loop import TerminalLoop
from flybrain.env.sandbox import Sandbox
from flybrain.model.brain import FlyLanguageModel

RESULTS = ROOT / "results"
PLAYBOOK = ["echo v3-ok > notes.txt", "cat notes.txt", "ls"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--run-name", default="v3_episode")
    p.add_argument("--n", type=int, default=32)
    args = p.parse_args()
    root = (args.root or (RESULTS / args.run_name / "sandbox")).resolve()
    if root.exists():
        # fresh episode directory
        pass
    box = Sandbox(root)
    g = synthetic_flylike(n=args.n, seed=0)
    model = FlyLanguageModel(g, state_dim=4, readout="mean")
    loop = TerminalLoop(model, box)
    obs = box.reset()
    state = None
    steps = []
    print(f"sandbox_root={box.root}")
    print(f"OBS0:{obs!r}")
    for action in PLAYBOOK:
        result = loop.step(action, state=state, observation=obs)
        state = result.state
        obs = result.observation
        ht = float(state.detach().abs().mean())
        print(f"ACTION:{action}")
        print(f"DECODED:{result.decoded!r}")
        print(f"OBS:{obs!r}")
        print(f"REFUSED:{result.refused} Ht_mean={ht:.6f}")
        steps.append(
            {
                "action": action,
                "decoded": result.decoded,
                "observation": obs,
                "refused": result.refused,
                "ht_mean": ht,
            }
        )
    notes = box.root / "notes.txt"
    payload = {
        "run_name": args.run_name,
        "sandbox_root": str(box.root),
        "sandbox_kind": "in-process-relative-allowlist",
        "playbook": PLAYBOOK,
        "steps": steps,
        "notes_exists": notes.is_file(),
        "notes_body": notes.read_text() if notes.is_file() else "",
        "ht_final_mean": float(state.detach().abs().mean()) if state is not None else 0.0,
        "note": "V3 loop plumbing. Playbook writes the file because V0 cannot type commands. Not a coding-agent result.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.run_name}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"FILE:{notes} exists={notes.is_file()} body={payload['notes_body']!r}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
