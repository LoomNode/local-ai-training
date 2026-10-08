"""Steps for an arm at the base arm's calibrated cost (Kyle, 2026-10-07: "we should stop using literal time passage as a metric ...
congestion is gonna prevent good reads"). Both calibrations come from one quiet session (400 steps each, --val-fixed 8 --eval-every
100, nothing else of ours on the box); the pace is seconds per step over steps 100 to 400, so the first step's overhead (lazy init,
torch.compile) drops out. steps = <base steps> * base pace / arm pace: the arm gets the steps that cost what the base's steps cost,
by the ratio, which cancels whatever the box was doing to both. usage: budget_ratio.py <arm calibration json> <base calibration json> <base steps> [max steps]
(the cap, 60,000 by default, catches a broken calibration; the 1- and 0-layer arms of bet A legitimately need 71k and 165k steps and pass 200000)"""
import json
import sys


def pace(path):
    d = json.load(open(path))
    t = {r["step"]: r["train_seconds"] for r in d["arms"][0]["curve_fixed"]}
    return (t[400] - t[100]) / 300


arm, base, base_steps = pace(sys.argv[1]), pace(sys.argv[2]), int(sys.argv[3])
max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
steps = int(round(base_steps * base / arm))
print(f"arm {arm:.4f} s/step, base {base:.4f}: ratio {arm / base:.3f}, {steps} steps for the base's {base_steps}", file=sys.stderr)
if not 500 <= steps <= max_steps:
    sys.exit(f"implausible step count {steps}")
print(steps)
