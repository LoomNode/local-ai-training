#!/usr/bin/env bash
# Before the big train (Kyle, 2026-10-10: "go ahead with 1, 3 and 4"), GPU 1's share:
#   item 4  batch size: the 8-layer arm timed at batch 8 / 16 / 32 for 400 steps each (bt_b8, bt_b16, bt_b32); bytes per second from
#           steps 100-400; if 16 or 32 is a tenth faster per byte, a 2x arm at equal bytes follows (not here)
#   item 3  the last shape question needs one new calibration: 12 layers of 512 (cal9: base, the arm, base again, 400 steps each;
#           the 4x budget is 4 x round(14,331 x base pace / arm pace), the base's hour being 14,331 steps); then the arm, l12x4_mt_r,
#           at that budget, scored on the dev windows. The 12x384 and 6x768 arms run on GPU 0 (pre_gpu0.sh) at 4 x their 2026-10-07 budgets.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-pre-gpu1.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/pre_gpu1.log") 2>&1
echo "pre-train tests gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
BASE="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32 --mem-feat $MEM/feat --mem-orders 8,16,32"
FLAGS[l12x4_mt_r]="$TFI --n-layers 12"
declare -A FEAT=([l12x4_mt_r]=feat_all28)
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
echo; echo "=== item 4: batch timing, 8 layers $(date +%T)"
cal bt_b8 $TFI --n-layers 8 --batch 8
cal bt_b16 $TFI --n-layers 8 --batch 16
cal bt_b32 $TFI --n-layers 8 --batch 32
"$PY" - "$OUT" <<'PY' || echo "batch timing summary failed"
import json, sys
out_dir = sys.argv[1]
def pace(n):
    d = json.load(open(f"results/{n}.json")); a = d["arms"]; a = a[0] if isinstance(a, list) else next(iter(a.values()))
    t = {r["step"]: r["train_seconds"] for r in a["curve_fixed"]}; return (t[400] - t[100]) / 300
res = {}
for b in (8, 16, 32):
    p = pace(f"bt_b{b}"); res[b] = {"pace": p, "bytes_per_s": b * 2048 / p}
ref = res[8]["bytes_per_s"]
for b in (8, 16, 32):
    print(f"batch {b}: {res[b]['pace']:.4f} s/step, {res[b]['bytes_per_s'] / 1e3:.1f} kB/s, {res[b]['bytes_per_s'] / ref - 1:+.1%} per byte against batch 8", file=sys.stderr)
json.dump(res, open(f"{out_dir}/batch_timing.json", "w"), indent=1)
PY
echo; echo "=== mini calibration session 9: 12 layers of 512 $(date +%T)"
cal cal9_base $BASE
cal cal9_l12_mt_r ${FLAGS[l12x4_mt_r]}
cal cal9_base_end $BASE
"$PY" - "$OUT" <<'PY' || exit 1
import json, sys
out_dir = sys.argv[1]
def pace(p):
    d = json.load(open(p)); a = d["arms"]; a = a[0] if isinstance(a, list) else next(iter(a.values()))
    t = {r["step"]: r["train_seconds"] for r in a["curve_fixed"]}; return (t[400] - t[100]) / 300
b0, b1 = pace("results/cal9_base.json"), pace("results/cal9_base_end.json")
drift = b1 / b0 - 1
pa = pace("results/cal9_l12_mt_r.json")
base = b0 if abs(drift) <= 0.03 else (b0 + b1) / 2
steps1 = int(round(14331 * base / pa))
out = {"base_start": b0, "base_end": b1, "drift": drift, "base_used": base, "arms": {"l12_mt_r": {"pace": pa, "ratio": pa / base, "steps": steps1, "steps_4x": 4 * steps1}}}
json.dump(out, open(f"{out_dir}/calsession9_2026-10-10.json", "w"), indent=1)
print(f"base {b0:.4f} s/step at the start, {b1:.4f} at the end: drift {drift:+.1%}", file=sys.stderr)
print(f"l12_mt_r {pa:.4f} s/step  ratio {pa / base:.3f}  steps {steps1} at 1x, {4 * steps1} at 4x", file=sys.stderr)
if abs(drift) > 0.03:
    print(f"DRIFT OVER 3 %: the budget is set from the mean base pace and is uncertain by about {abs(drift) / 2:.1%}; the run's pace check decides", file=sys.stderr)
open(f"{out_dir}/cal9_l12x4_steps", "w").write(str(4 * steps1))
PY
STEPS[l12x4_mt_r]=$(cat "$OUT/cal9_l12x4_steps")
[ -n "${STEPS[l12x4_mt_r]}" ] || { echo "no calibrated step count"; exit 1; }
sweep l12x4_mt_r
echo; echo "=== scoring l12x4_mt_r $(date +%T)"
score2 l12x4_mt_r
echo "pre-train tests gpu1 done $(date '+%F %T')"
