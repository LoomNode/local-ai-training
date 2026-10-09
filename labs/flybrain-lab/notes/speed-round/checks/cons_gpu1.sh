#!/usr/bin/env bash
# Consolidation, first pass (test 3 from the fly's circuit; Kyle, 2026-10-09: "the tables are like bulky speed training ... regular
# training would just be like a quant/compression tech"). The probe (checks/ablate_inputs.py) says ordinary training does no such
# compression: the 4-layer table model is 0.652 with its table and 2.23 without it. Three hour-runs at 4 layers, GPU 1, code deafa86:
#   l4plainm_r  4 layers, the in-window match lookup, no corpus table: the control for "weights + lookup" (its own mini-session, cal5)
#   l4drop5_r   l4b's model and inputs, --mem-dropout 0.5: the table withdrawn at half the positions all through training
#   l4ramp_r    the same, --mem-dropout 1.0 --mem-dropout-ramp: withdrawn at a fraction rising from 0 to 1 over the run (weaning)
# The dropout arms get l4b's calibrated budget (25,562 steps: the same work per step, a mask over the inputs; the run's first-evaluation
# pace is checked against l4b's). Every arm is then scored with its memories withdrawn (ablate_inputs.py): the question is whether a
# model that learned with the table and was weaned off it beats one that never had it, scored the same way.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-cons-gpu1.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/a_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/cons_gpu1.log") 2>&1
echo "consolidation gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
preflight
L4B="$TFI --mem-feat $MEM/feat_all28 --mem-orders $ALL18 --n-layers 4"
FLAGS[l4plainm_r]="$TFI --n-layers 4"
FLAGS[l4drop5_r]="$L4B --mem-dropout 0.5"
FLAGS[l4ramp_r]="$L4B --mem-dropout 1.0 --mem-dropout-ramp"

cal() {  # cal <name> <flags...>: 400 steps; the yardstick line here, the whole output in $OUT/<name>.out
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "=== $name $(date +%T)  load $(cut -d' ' -f1 /proc/loadavg)  gpu0 $(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON "$@" --steps 400 --val-fixed 8 --eval-every 100 2>&1 | tee "$OUT/$name.out" | grep --line-buffered -v "^  " | tail -n 3 | grep -E "yardstick" || echo "no yardstick line for $name: read $OUT/$name.out"
}
run_fixed() {  # run_fixed <name> <steps>: the arm at a step count set here
  name=$1; steps=$2
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "$UNIT" > "results/claim_$name"
  echo; echo "=== $name $(date +%T): $steps steps"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON ${FLAGS[$name]} --steps "$steps" --save --val-fixed 128 --eval-every 2000 2>&1 | grep --line-buffered -v "^  "
}

echo; echo "=== mini calibration session for l4plainm $(date +%T)"
cal cal5_base $BASE
cal cal5_l4plainm_r ${FLAGS[l4plainm_r]}
cal cal5_base_end $BASE
"$PY" - <<'PY' || exit 1
import json
def pace(p):
    d = json.load(open(p)); t = {r["step"]: r["train_seconds"] for r in d["arms"][0]["curve_fixed"]}; return (t[400] - t[100]) / 300
b0, b1 = pace("results/cal5_base.json"), pace("results/cal5_base_end.json")
drift = b1 / b0 - 1
print(f"base {b0:.4f} s/step at the start, {b1:.4f} at the end: drift {drift:+.1%}")
pa = pace("results/cal5_l4plainm_r.json")
out = {"base_start": b0, "base_end": b1, "arms": {"l4plainm_r": {"pace": pa, "ratio": pa / b0, "steps": int(round(14331 * b0 / pa))}}}
print(f"l4plainm_r {pa:.4f} s/step  ratio {pa / b0:.3f}  steps {out['arms']['l4plainm_r']['steps']}")
json.dump(out, open("/games/ailab/lab/flybrain-lab/.worktrees/speed/results/tables/calsession5_2026-10-09.json", "w"), indent=1)
if abs(drift) > 0.03:
    raise SystemExit("drift over 3 %: the control is not run")
PY
CAL=cal5; MAXSTEPS=60000; sweep l4plainm_r
run_fixed l4drop5_r 25562
run_fixed l4ramp_r 25562
echo; echo "=== scoring with the memories withdrawn $(date +%T)"
"$PY" "$S/ablate_inputs.py" --feat "$MEM/feat_all28" --out "$OUT/ablate_cons.json" l4plainm=results/ckpt/l4plainm_r_transformer.pt l4drop5=results/ckpt/l4drop5_r_transformer.pt l4ramp=results/ckpt/l4ramp_r_transformer.pt 2>&1 | grep -v "^  " || echo "scoring failed"
echo "consolidation gpu1 done $(date '+%F %T')"
