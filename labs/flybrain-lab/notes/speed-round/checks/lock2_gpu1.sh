#!/usr/bin/env bash
# The second lock queue (2026-10-10 morning, GPU 1), after the overnight one: the project tier paid 0.008 / 0.014 at the hour with 4 layers
# and nothing at two hours with 6 (l6x2dir_mt_r 0.617 against 0.619 / 0.615). Two arms say what that is, each with per-step work already
# budgeted in a quiet session, so no new calibration:
#   l4b_s1_r      seed 1 of l4b (4 layers, 18 inputs, 2^28 tables) at the hour: l4b_mt_r's 25,562 steps; makes the tier's hour gain a
#                 paired two-seed number (l4dir_s1_r 0.638 against it)
#   l4x2dir_mt_r  4 layers + the tier at twice the cost: 49,877 steps = l4x2's 51,124 / 1.025 (the tier's step ratio from cal6 at 4 layers);
#                 against l4x2's 0.623 / 0.619: if it pays, the depth removed the gain at 6 layers; if not, the budget did
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-lock2-gpu1.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/lock2_gpu1.log") 2>&1
echo "lock tests 2 gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
# argparse keeps the last value: the tier arm's --mem-feat/--mem-orders and the seed override TFI's and COMMON's
FLAGS[l4b_s1_r]="$TFI --n-layers 4 --seed 1"
FLAGS[l4x2dir_mt_r]="$TFI --mem-feat $MEM/feat_all28_fly --mem-orders $ALL18,2008,2016 --n-layers 4"
STEPS[l4b_s1_r]=25562; STEPS[l4x2dir_mt_r]=49877
declare -A FEAT=([l4b_s1_r]=feat_all28 [l4x2dir_mt_r]=feat_all28_fly)
score2() {  # score2 <name>: as scale_common's score, with the arm's own feature directory
  name=$1; short=${name%_r}
  [ -f "$OUT/k_$short.json" ] && return 0
  "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_$short=results/ckpt/${name}_transformer.pt@$MEM/${FEAT[$name]}" || echo "scoring $name failed"
}
preflight
for d in feat_all28 feat_all28_fly; do [ -f "$MEM/$d/features.json" ] || { echo "$d is not built"; exit 1; }; done
cat "$MEM"/feat_all28/pred_n8.npy "$MEM"/feat_all28_fly/pred_n2008.npy > /dev/null
for a in l4b_s1_r l4x2dir_mt_r; do
  sweep "$a"
  echo; echo "=== scoring $a $(date +%T)"
  score2 "$a"
done
echo "lock tests 2 gpu1 done $(date '+%F %T')"
