#!/usr/bin/env bash
# After the pre-train queue (Kyle, 2026-10-10 evening: "anything else we could queue?"): the depth question the big train hinges on,
# now that 12 layers beat 8 at four hours (0.591 against 0.598, one seed) and width is settled at 512.
#   seed 1 of 12x512 at the 4x budget: the 0.007 over 8 layers is one seed and the 2x seed line is 0.004-0.005; the same 37,124 steps
#   16 layers of 512 at 4x: one mini calibration first (cal10: base, the arm, base again, 400 steps each; the 4x budget is
#           4 x round(14,331 x base pace / arm pace), the base's hour being 14,331 steps), then the arm at that budget.
# Both scored on the dev windows against l12x4_mt_r (0.591) and l8x4_mt_r (0.598). Waits for tables-pre2-gpu1 (the restart test) to end.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-post-gpu1.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/post_gpu1.log") 2>&1
echo "post queue gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
while systemctl --user is-active --quiet tables-pre2-gpu1.service; do sleep 120; done
echo "pre2 queue ended $(date '+%F %T')"
BASE="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32 --mem-feat $MEM/feat --mem-orders 8,16,32"
FLAGS[l12x4s1_mt_r]="$TFI --n-layers 12 --seed 1"
STEPS[l12x4s1_mt_r]=37124
FLAGS[l16x4_mt_r]="$TFI --n-layers 16"
declare -A FEAT=([l12x4s1_mt_r]=feat_all28 [l16x4_mt_r]=feat_all28)
score2() {  # score2 <name>: as scale_common's score, with the arm's own feature directory
  name=$1; short=${name%_r}
  [ -f "$OUT/k_$short.json" ] && return 0
  "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_$short=results/ckpt/${name}_transformer.pt@$MEM/${FEAT[$name]}" || echo "scoring $name failed"
}
cal() {  # cal <name> <flags...>: 400 steps; the yardstick line here, the whole output in $OUT/<name>.out
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo "=== $name $(date +%T)  load $(cut -d' ' -f1 /proc/loadavg)  gpu0 $(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON "$@" --steps 400 --val-fixed 8 --eval-every 100 2>&1 | tee "$OUT/$name.out" | grep --line-buffered -v "^  " | tail -n 3 | grep -E "yardstick" || echo "no yardstick line for $name: read $OUT/$name.out"
}
preflight
cat "$MEM"/feat_all28/pred_n8.npy "$MEM"/feat/pred_n8.npy > /dev/null
echo; echo "=== mini calibration session 10: 16 layers of 512 $(date +%T)"
cal cal10_base $BASE
cal cal10_l16_mt_r ${FLAGS[l16x4_mt_r]}
cal cal10_base_end $BASE
"$PY" - "$OUT" <<'PY' || exit 1
import json, sys
out_dir = sys.argv[1]
def pace(p):
    d = json.load(open(p)); a = d["arms"]; a = a[0] if isinstance(a, list) else next(iter(a.values()))
    t = {r["step"]: r["train_seconds"] for r in a["curve_fixed"]}; return (t[400] - t[100]) / 300
b0, b1 = pace("results/cal10_base.json"), pace("results/cal10_base_end.json")
drift = b1 / b0 - 1
pa = pace("results/cal10_l16_mt_r.json")
base = b0 if abs(drift) <= 0.03 else (b0 + b1) / 2
steps1 = int(round(14331 * base / pa))
out = {"base_start": b0, "base_end": b1, "drift": drift, "base_used": base, "arms": {"l16_mt_r": {"pace": pa, "ratio": pa / base, "steps": steps1, "steps_4x": 4 * steps1}}}
json.dump(out, open(f"{out_dir}/calsession10_2026-10-10.json", "w"), indent=1)
print(f"base {b0:.4f} s/step at the start, {b1:.4f} at the end: drift {drift:+.1%}", file=sys.stderr)
print(f"l16_mt_r {pa:.4f} s/step  ratio {pa / base:.3f}  steps {steps1} at 1x, {4 * steps1} at 4x", file=sys.stderr)
if abs(drift) > 0.03:
    print(f"DRIFT OVER 3 %: the budget is set from the mean base pace and is uncertain by about {abs(drift) / 2:.1%}; the run's pace check decides", file=sys.stderr)
open(f"{out_dir}/cal10_l16x4_steps", "w").write(str(4 * steps1))
PY
STEPS[l16x4_mt_r]=$(cat "$OUT/cal10_l16x4_steps")
[ -n "${STEPS[l16x4_mt_r]}" ] || { echo "no calibrated step count for 16 layers"; exit 1; }
sweep l12x4s1_mt_r
echo; echo "=== scoring l12x4s1_mt_r $(date +%T)"
score2 l12x4s1_mt_r
sweep l16x4_mt_r
echo; echo "=== scoring l16x4_mt_r $(date +%T)"
score2 l16x4_mt_r
echo "post queue gpu1 done $(date '+%F %T')"
