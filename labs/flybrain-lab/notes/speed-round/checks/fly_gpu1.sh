#!/usr/bin/env bash
# Tests 1 and 2 from the fly's circuit (Kyle, 2026-10-09: "queue tests 1 and 2 aswell"), on GPU 1 after the consolidation pass and
# the CPU builders (fly_key_tables.py -> feat_all28_fly; rep_tables.py -> feat_all28rep). Three hour-arms at 4 layers on top of the
# 18 inputs and 2^28 two-way tables (l4b, 0.652), each budgeted by its own mini-session (cal6: base, the three arms, base again):
#   l4fly_r   + the fly key (3001): 8 masks of 6 of the last 16 bytes, read by agreement (test 1)
#   l4dir_r   + the project tier (2008, 2016): orders 8 and 16 keyed with the file's directory (test 2, item 6)
#   l4rep_r   the byte orders rebuilt with a repetition-gated write, same slots (test 2, the fly's spaced admission)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-fly-gpu1.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/a_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/fly_gpu1.log") 2>&1
echo "fly tests 1+2 gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
while systemctl --user is-active --quiet tables-cons-gpu1.service || systemctl --user is-active --quiet tables-build-tiers.service; do sleep 60; done
sleep 10
for d in feat_all28_fly feat_all28rep; do [ -f "$MEM/$d/features.json" ] || { echo "$d was not built"; exit 1; }; done
preflight
FLAGS[l4fly_r]="$TFI --mem-feat $MEM/feat_all28_fly --mem-orders $ALL18,3001 --n-layers 4"
FLAGS[l4dir_r]="$TFI --mem-feat $MEM/feat_all28_fly --mem-orders $ALL18,2008,2016 --n-layers 4"
FLAGS[l4rep_r]="$TFI --mem-feat $MEM/feat_all28rep --mem-orders $ALL18 --n-layers 4"
FEAT[l4fly_r]=feat_all28_fly; FEAT[l4dir_r]=feat_all28_fly; FEAT[l4rep_r]=feat_all28rep
cat "$MEM"/feat_all28_fly/pred_n3001.npy "$MEM"/feat_all28_fly/conf_n3001.npy "$MEM"/feat_all28rep/pred_n8.npy > /dev/null

cal() {  # cal <name> <flags...>: 400 steps; the yardstick line here, the whole output in $OUT/<name>.out
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "=== $name $(date +%T)  load $(cut -d' ' -f1 /proc/loadavg)  gpu0 $(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON "$@" --steps 400 --val-fixed 8 --eval-every 100 2>&1 | tee "$OUT/$name.out" | grep --line-buffered -v "^  " | tail -n 3 | grep -E "yardstick" || echo "no yardstick line for $name: read $OUT/$name.out"
}
echo; echo "=== calibration session 6 $(date +%T)"
cal cal6_base $BASE
for a in l4fly_r l4dir_r l4rep_r; do cal "cal6_$a" ${FLAGS[$a]}; done
cal cal6_base_end $BASE
"$PY" - <<'PY' || exit 1
import json
def pace(p):
    d = json.load(open(p)); t = {r["step"]: r["train_seconds"] for r in d["arms"][0]["curve_fixed"]}; return (t[400] - t[100]) / 300
b0, b1 = pace("results/cal6_base.json"), pace("results/cal6_base_end.json")
drift = b1 / b0 - 1
print(f"base {b0:.4f} s/step at the start, {b1:.4f} at the end: drift {drift:+.1%}")
out = {"base_start": b0, "base_end": b1, "arms": {}}
for a in ["l4fly_r", "l4dir_r", "l4rep_r"]:
    pa = pace(f"results/cal6_{a}.json"); out["arms"][a] = {"pace": pa, "ratio": pa / b0, "steps": int(round(14331 * b0 / pa))}
    print(f"{a} {pa:.4f} s/step  ratio {pa / b0:.3f}  steps {out['arms'][a]['steps']}")
json.dump(out, open("/games/ailab/lab/flybrain-lab/.worktrees/speed/results/tables/calsession6_2026-10-09.json", "w"), indent=1)
if abs(drift) > 0.03:
    raise SystemExit("drift over 3 %: the arms are not run")
PY
CAL=cal6; MAXSTEPS=60000; sweep l4fly_r l4dir_r l4rep_r
echo; echo "=== scoring $(date +%T)"
for a in l4fly_r l4dir_r l4rep_r; do score "$a"; done
echo "fly tests 1+2 gpu1 done $(date '+%F %T')"
