#!/usr/bin/env bash
# Bet A second pass, continued (2026-10-09): a2_gpu1.sh trained l1b and l0b but died before scoring them, because the l4n calibration
# died a third time: ngram_index overflowed int64 at order 9 (fixed at 0b278d5, the run worktree moved there) and its mini-session
# drifted -4.0 % anyway. This scores l1b and l0b first, then gives l4n a fresh mini-session (cal4c: base, arm, base again, drift <= 3 %),
# runs it at the base's calibrated cost and scores it. Runs on GPU 1 alone.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-a3-gpu1.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/a_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/a3_gpu1.log") 2>&1
echo "bet A pass 3 gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
preflight
echo; echo "=== scoring l1b l0b $(date +%T)"
for a in l1b_mt_r l0b_mt_r; do score "$a"; done

cal() {  # cal <name> <flags...>: 400 steps; the yardstick line here, the whole output in $OUT/<name>.out (the filter hid two deaths)
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "=== $name $(date +%T)  load $(cut -d' ' -f1 /proc/loadavg)  gpu0 $(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON "$@" --steps 400 --val-fixed 8 --eval-every 100 2>&1 | tee "$OUT/$name.out" | grep --line-buffered -v "^  " | tail -n 3 | grep -E "yardstick" || echo "no yardstick line for $name: read $OUT/$name.out"
}
echo; echo "=== mini calibration session for l4n $(date +%T)"
cal cal4c_base $BASE
cal cal4c_l4n_mt_r ${FLAGS[l4n_mt_r]}
cal cal4c_base_end $BASE
"$PY" - <<'PY' || exit 1
import json
def pace(p):
    d = json.load(open(p)); t = {r["step"]: r["train_seconds"] for r in d["arms"][0]["curve_fixed"]}; return (t[400] - t[100]) / 300
b0, b1 = pace("results/cal4c_base.json"), pace("results/cal4c_base_end.json")
drift = b1 / b0 - 1
print(f"base {b0:.4f} s/step at the start, {b1:.4f} at the end: drift {drift:+.1%}")
pa = pace("results/cal4c_l4n_mt_r.json")
out = {"base_start": b0, "base_end": b1, "arms": {"l4n_mt_r": {"pace": pa, "ratio": pa / b0, "steps": int(round(14331 * b0 / pa))}}}
print(f"l4n_mt_r {pa:.4f} s/step  ratio {pa / b0:.3f}  steps {out['arms']['l4n_mt_r']['steps']}")
json.dump(out, open("/games/ailab/lab/flybrain-lab/.worktrees/speed/results/tables/calsession4c_2026-10-09.json", "w"), indent=1)
if abs(drift) > 0.03:
    raise SystemExit("drift over 3 %: the l4n arm is not run")
PY
CAL=cal4c; sweep l4n_mt_r
echo; echo "=== scoring l4n $(date +%T)"
score l4n_mt_r
echo "bet A pass 3 gpu1 done $(date '+%F %T')"
