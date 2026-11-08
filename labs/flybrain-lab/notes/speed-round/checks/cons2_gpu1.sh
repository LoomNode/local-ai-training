#!/usr/bin/env bash
# Consolidation, second pass (test 3 from the fly's circuit): the table model as the teacher. The first pass (checks/cons_gpu1.sh) withdrew
# the table while training and moved nothing into the weights (weaned: 0.799 with the table gone, against 0.784 for a model that never had
# one). The fly's slow store is written by the fast store's output, not by using the fast store less (gamma MBON output gates alpha/beta
# plasticity; CLS: replay teaches the cortex; MLP Memory: a module trained to the retriever's distribution). So: the plain 4-layer model
# (the control's architecture, in-window lookup, no corpus table) trained on the bytes and on the KL to the 0.652 table model's distribution
# over the same windows (--distill-from, w 0.5, T 1; the teacher reads its table, the student cannot). One arm, GPU 1, after tests 1-2:
#   l4dist_r  at the control's 28,438 steps (the mechanism, at equal steps), with a checkpoint at its own calibrated step count (cal7:
#             base, the arm, base again; the teacher's forward makes the step dearer) for the reading at equal cost.
# Scored with the memories withdrawn as the first pass was (ablate_inputs.py): does a model taught by the table model beat the control?
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-cons2b-gpu1.service
CODE=${CODE:?set CODE to the spike/speed-int commit that holds --distill-from}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/a_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/cons2_gpu1.log") 2>&1
echo "consolidation 2 gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD), wants $CODE"
while systemctl --user is-active --quiet tables-fly-gpu1.service; do sleep 60; done
sleep 10
if [ "$(git rev-parse --short HEAD)" != "$(git rev-parse --short "$CODE")" ]; then   # nothing runs from this worktree now: pin it to the code
  for u in $(systemctl --user list-units --type=service --state=active --plain --no-legend 'tables-*' | cut -d' ' -f1); do
    [ "$u" != "$UNIT" ] && { echo "$u is active: not moving the run worktree"; exit 1; }
  done
  git checkout --detach "$CODE" || { echo "checkout of $CODE failed"; exit 1; }
  echo "run worktree now at $(git rev-parse --short HEAD)"
fi
preflight
TEACHER=results/ckpt/l4b_mt_r_transformer.pt
[ -f "$TEACHER" ] || { echo "no teacher at $TEACHER"; exit 1; }
FLAGS[l4dist_r]="$TFI --n-layers 4 --distill-from $TEACHER --distill-weight 0.5 --distill-temp 1.0"
cat "$MEM"/feat_all28/pred_n8.npy > /dev/null

cal() {  # cal <name> <flags...>: 400 steps; the yardstick line here, the whole output in $OUT/<name>.out
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "=== $name $(date +%T)  load $(cut -d' ' -f1 /proc/loadavg)  gpu0 $(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON "$@" --steps 400 --val-fixed 8 --eval-every 100 2>&1 | tee "$OUT/$name.out" | grep --line-buffered -v "^  " | tail -n 3 | grep -E "yardstick" || echo "no yardstick line for $name: read $OUT/$name.out"
}
run_fixed() {  # run_fixed <name> <steps> [more flags]: the arm at a step count set here
  name=$1; steps=$2; shift 2
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "$UNIT" > "results/claim_$name"
  echo; echo "=== $name $(date +%T): $steps steps $*"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON ${FLAGS[$name]} --steps "$steps" --save --val-fixed 128 --eval-every 2000 "$@" 2>&1 | grep --line-buffered -v "^  "
}

# Session 7 (13:40-13:47) completed and drifted +4.3 % (0.2232 -> 0.2328 s/step; the game on GPU 0 at 100 %, load 3); its files stay as
# the record (cal7_*.json, cal7_*.out). The first version of this block put a here-document inside $(...) with the ")" on the same line,
# which bash cannot parse, so the unit died here at 13:47 before the arm ran. Session 8 is the one the arm is budgeted from: a quiet
# session (drift within 3 %) sets the budget from the start base as every session before it; a drifted one sets it from the mean of
# the two base paces and says so (the budget is then uncertain by half the drift; the arm's main reading, at the control's steps, does
# not depend on it), rather than leaving the pass unrun on a box that stays noisy.
echo; echo "=== mini calibration session 8 $(date +%T)"
cal cal8_base $BASE
cal cal8_l4dist_r ${FLAGS[l4dist_r]}
cal cal8_base_end $BASE
"$PY" - "$OUT" <<'PY' || exit 1
import json, sys
out_dir = sys.argv[1]
def pace(p):
    d = json.load(open(p)); t = {r["step"]: r["train_seconds"] for r in d["arms"][0]["curve_fixed"]}; return (t[400] - t[100]) / 300
b0, b1 = pace("results/cal8_base.json"), pace("results/cal8_base_end.json")
drift = b1 / b0 - 1
pa = pace("results/cal8_l4dist_r.json")
base = b0 if abs(drift) <= 0.03 else (b0 + b1) / 2
out = {"base_start": b0, "base_end": b1, "drift": drift, "base_used": base, "arms": {"l4dist_r": {"pace": pa, "ratio": pa / base, "steps": int(round(14331 * base / pa))}}}
json.dump(out, open(f"{out_dir}/calsession8_2026-10-09.json", "w"), indent=1)
print(f"base {b0:.4f} s/step at the start, {b1:.4f} at the end: drift {drift:+.1%}", file=sys.stderr)
print(f"l4dist_r {pa:.4f} s/step  ratio {pa / base:.3f}  steps {out['arms']['l4dist_r']['steps']}", file=sys.stderr)
if abs(drift) > 0.03:
    print(f"DRIFT OVER 3 %: the budget is set from the mean base pace and is uncertain by about {abs(drift) / 2:.1%}; the run's pace check decides", file=sys.stderr)
open(f"{out_dir}/cal8_steps", "w").write(str(out["arms"]["l4dist_r"]["steps"]))
PY
CALSTEPS=$(cat "$OUT/cal8_steps")
[ -n "$CALSTEPS" ] || { echo "no calibrated step count"; exit 1; }
echo "calibrated step count $CALSTEPS; the arm runs the control's 28,438 with a checkpoint at $CALSTEPS"
run_fixed l4dist_r 28438 --ckpt-every "$CALSTEPS"
echo; echo "=== scoring with the memories withdrawn $(date +%T)"
"$PY" "$S/ablate_inputs.py" --feat "$MEM/feat_all28" --out "$OUT/ablate_cons2.json" l4dist=results/ckpt/l4dist_r_transformer.pt "l4dist_cal=results/ckpt/l4dist_r_transformer.step$CALSTEPS.pt" 2>&1 | grep -v "^  " || echo "scoring failed"
echo "consolidation 2 gpu1 done $(date '+%F %T')"
