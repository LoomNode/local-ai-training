"""Steps for a training-seconds budget, from a 400-step calibration run with --val-fixed N --eval-every 100.
usage: budget.py <calibration json> <target seconds> <drift factor>
train_seconds at step 100 carries the first step's overhead (lazy init, torch.compile); steps 100 to 400 give the pace.
The drift factor covers a long run being slower per step than its first minutes (the spine runs: 4 to 8 %).
Prints the step count on stdout and the working on stderr."""
import json
import sys

d = json.load(open(sys.argv[1]))
target, drift = float(sys.argv[2]), float(sys.argv[3])
t = {r["step"]: r["train_seconds"] for r in d["arms"][0]["curve_fixed"]}
pace = (t[400] - t[100]) / 300
overhead = t[100] - 100 * pace
steps = int((target - overhead) / (pace * drift))
print(f"calibration: {pace:.4f} s/step, overhead {overhead:.1f} s, drift factor {drift}: {steps} steps for {target:.0f} s", file=sys.stderr)
if not 500 <= steps <= 60000:
    sys.exit(f"implausible step count {steps}")
print(steps)
