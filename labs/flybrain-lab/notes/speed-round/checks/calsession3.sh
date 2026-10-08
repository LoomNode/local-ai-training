#!/usr/bin/env bash
# Calibration session 3 on GPU 1 (the protocol of calsession.sh): 400 steps of the base and of every bet-C arm (and the E arm) back to back with
# nothing else of ours on the box, base first and last, after the bet-A arms (tables-a-gpu1) have finished; the E arm joins if its
# features exist. Writes results/tables/calsession3_2026-10-08.json; c_gpu1.sh waits for this unit.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/c_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/calsession3.log") 2>&1
echo "calibration session 3 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
while systemctl --user is-active --quiet tables-a-gpu1.service || systemctl --user is-active --quiet tables-calsession2.service; do sleep 30; done

ARMS=$(arms)
sleep 15
preflight
mine=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c flybrain-lab)
[ "$mine" = 0 ] || { echo "something of ours is still on a card"; exit 1; }
cat "$MEM"/feat/*.npy "$MEM"/feat_all28/*.npy > /dev/null; [ -d "$MEM/feat_all28_paq" ] && cat "$MEM"/feat_all28_paq/*.npy > /dev/null
cal() {  # cal <name> <flags...>
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "=== $name $(date +%T)  load $(cut -d' ' -f1 /proc/loadavg)  gpu0 $(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON "$@" --steps 400 --val-fixed 8 --eval-every 100 2>&1 | grep --line-buffered -v "^  " | grep yardstick
}
cal cal4_base $BASE
for a in $ARMS; do cal "cal4_$a" ${FLAGS[$a]}; done
cal cal4_base_end $BASE
echo; echo "=== paces $(date +%T)"
"$PY" - <<'PY'
import json, glob, os
def pace(p):
    d = json.load(open(p)); t = {r["step"]: r["train_seconds"] for r in d["arms"][0]["curve_fixed"]}; return (t[400] - t[100]) / 300
b0, b1 = pace("results/cal4_base.json"), pace("results/cal4_base_end.json")
print(f"base {b0:.4f} s/step at the start, {b1:.4f} at the end: drift {b1 / b0 - 1:+.1%}" + ("  REDO THE SESSION" if abs(b1 / b0 - 1) > 0.03 else ""))
out = {"base_start": b0, "base_end": b1, "arms": {}}
for p in sorted(glob.glob("results/cal4_*.json")):
    n = os.path.basename(p)[5:-5]
    if n in ("base", "base_end"): continue
    pa = pace(p); out["arms"][n] = {"pace": pa, "ratio": pa / b0, "steps": int(round(14331 * b0 / pa))}
    print(f"{n:14s} {pa:.4f} s/step  ratio {pa / b0:.3f}  steps {out['arms'][n]['steps']}")
json.dump(out, open("/games/ailab/lab/flybrain-lab/.worktrees/speed/results/tables/calsession3_2026-10-08.json", "w"), indent=1)
PY
echo "calibration session 3 done $(date '+%F %T')"
