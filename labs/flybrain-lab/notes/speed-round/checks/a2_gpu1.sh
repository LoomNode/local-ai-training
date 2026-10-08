#!/usr/bin/env bash
# Bet A, second pass (2026-10-08): the three arms the first pass lost. l1b and l0b were refused by budget_ratio.py's 60,000-step cap
# (their calibrated budgets are 71,202 and 165,309 steps; session 2's cal3 JSONs stand); l4n's calibration died under torch.compile
# with --ngram-sparse, so it gets a mini-session of its own here (base, arm, base again, drift <= 3 %) and the dense table.
# Runs on GPU 1 after the C/E chain (tables-c-gpu1), then scores all three.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-a2-gpu1.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/a_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/a2_gpu1.log") 2>&1
echo "bet A pass 2 gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
while systemctl --user is-active --quiet tables-calsession3.service || systemctl --user is-active --quiet tables-c-gpu1.service; do sleep 60; done
sleep 10
preflight
MAXSTEPS=200000; sweep l1b_mt_r l0b_mt_r; MAXSTEPS=60000

cal() {  # cal <name> <flags...>: 400 steps, the yardstick line only
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "=== $name $(date +%T)  load $(cut -d' ' -f1 /proc/loadavg)  gpu0 $(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON "$@" --steps 400 --val-fixed 8 --eval-every 100 2>&1 | grep --line-buffered -v "^  " | tail -n 3 | grep -E "yardstick|Error"
}
echo; echo "=== mini calibration session for l4n $(date +%T)"
cal cal4b_base $BASE
cal cal4b_l4n_mt_r ${FLAGS[l4n_mt_r]}
cal cal4b_base_end $BASE
"$PY" - <<'PY' || exit 1
import json
def pace(p):
    d = json.load(open(p)); t = {r["step"]: r["train_seconds"] for r in d["arms"][0]["curve_fixed"]}; return (t[400] - t[100]) / 300
b0, b1 = pace("results/cal4b_base.json"), pace("results/cal4b_base_end.json")
drift = b1 / b0 - 1
print(f"base {b0:.4f} s/step at the start, {b1:.4f} at the end: drift {drift:+.1%}")
pa = pace("results/cal4b_l4n_mt_r.json")
out = {"base_start": b0, "base_end": b1, "arms": {"l4n_mt_r": {"pace": pa, "ratio": pa / b0, "steps": int(round(14331 * b0 / pa))}}}
print(f"l4n_mt_r {pa:.4f} s/step  ratio {pa / b0:.3f}  steps {out['arms']['l4n_mt_r']['steps']}")
json.dump(out, open("/games/ailab/lab/flybrain-lab/.worktrees/speed/results/tables/calsession4b_2026-10-08.json", "w"), indent=1)
if abs(drift) > 0.03:
    raise SystemExit("drift over 3 %: the l4n arm is not run")
PY
CAL=cal4b; sweep l4n_mt_r
echo; echo "=== scoring $(date +%T)"
for a in l1b_mt_r l0b_mt_r l4n_mt_r; do score "$a"; done
echo "bet A pass 2 gpu1 done $(date '+%F %T')"
