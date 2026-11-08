#!/usr/bin/env bash
# One quiet calibration session on GPU 1 (Kyle, 2026-10-07: "i think we should stop using literal time passage as a metric then. it
# seems like congestion is gonna prevent good reads"): 400 steps of every arm back to back with nothing else of ours on the box, the
# base arm first and last. The arms are then budgeted by the ratio of paces (budget_ratio.py), not by the clock. Waits for the running
# 12x384 arm to finish, then stops the old queue units (they are restarted from the new scripts, which wait for this unit).
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$S/tier1_common.sh"
declare -A T1=(); for k in "${!FLAGS[@]}"; do T1[$k]=${FLAGS[$k]}; done
source "$S/ideas_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/calsession.log") 2>&1
echo "calibration session start $(date '+%F %T') at $(git rev-parse --short HEAD)"
while [ ! -f results/s12_tf_mt_r.json ] && systemctl --user is-active --quiet tables-tier1b-gpu1.service; do sleep 20; done
systemctl --user stop tables-tier1b-gpu1.service tables-tier1b-gpu0.service tables-ideas-gpu1.service tables-ideas-gpu0.service 2>/dev/null
sleep 15
preflight
mine=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -c flybrain-lab)
[ "$mine" = 0 ] || { echo "something of ours is still on a card"; exit 1; }
cat "$MEM"/feat/*.npy "$MEM"/feat_all/*.npy "$MEM"/feat_all28/*.npy "$MEM"/feat_paq/*.npy "$MEM"/feat_top2/*.npy > /dev/null
BASE="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32 --mem-feat $MEM/feat --mem-orders 8,16,32"
ALLF="$TFI --mem-feat $MEM/feat_all --mem-orders $ALL18"
cal() {  # cal <name> <flags...>
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "=== $name $(date +%T)  load $(cut -d' ' -f1 /proc/loadavg)  gpu0 $(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON "$@" --steps 400 --val-fixed 8 --eval-every 100 2>&1 | grep --line-buffered -v "^  " | grep yardstick
}
cal cal2_base $BASE
cal cal2_allf $ALLF
for a in s12_tf_mt_r s6_tf_mt_r w4k_tf_mt_r hyb_mt_r gla_mt_r; do cal "cal2_$a" ${T1[$a]}; done
for a in paq_mt_r top2_mt_r l4_mt_r l6_mt_r l4b_mt_r; do cal "cal2_$a" ${FLAGS[$a]}; done
cal cal2_base_end $BASE
echo; echo "=== paces $(date +%T)"
"$PY" - <<'PY'
import json, glob, os
def pace(p):
    d = json.load(open(p)); t = {r["step"]: r["train_seconds"] for r in d["arms"][0]["curve_fixed"]}; return (t[400] - t[100]) / 300
b0, b1 = pace("results/cal2_base.json"), pace("results/cal2_base_end.json")
print(f"base {b0:.4f} s/step at the start, {b1:.4f} at the end: drift {b1 / b0 - 1:+.1%}" + ("  REDO THE SESSION" if abs(b1 / b0 - 1) > 0.03 else ""))
out = {"base_start": b0, "base_end": b1, "arms": {}}
for p in sorted(glob.glob("results/cal2_*.json")):
    n = os.path.basename(p)[5:-5]
    if n in ("base", "base_end"): continue
    pa = pace(p); out["arms"][n] = {"pace": pa, "ratio": pa / b0, "steps": int(round(14331 * b0 / pa))}
    print(f"{n:14s} {pa:.4f} s/step  ratio {pa / b0:.3f}  steps {out['arms'][n]['steps']}")
json.dump(out, open("/games/ailab/lab/flybrain-lab/.worktrees/speed/results/tables/calsession_2026-10-07.json", "w"), indent=1)
PY
echo "calibration session done $(date '+%F %T')"
