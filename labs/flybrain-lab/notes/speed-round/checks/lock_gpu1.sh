#!/usr/bin/env bash
# The lock tests (Kyle, 2026-10-09: "queue the next todo items so we have something to view"), on GPU 1 after the fly items and the
# consolidation passes. Each arm's per-step work is identical to an arm already budgeted in a quiet session, so no new calibration
# (the box is noisy with the game on GPU 0: cal7 and cal8 drifted 4-5 %):
#   l4dir_s1_r   seed 1 of the project tier (0.644 at seed 0): l4dir_r's flags, cal6's 24,939 steps
#   l6x4_mt_r    the depth curve at four times the cost, 6 layers: 70,688 steps (2 x l6x2's 35,344)
#   l8x4_mt_r    the same, 8 layers: 54,240 steps (2 x l8x2's 27,120)
#   l8x2s1_mt_r  seed 1 of the 8-layer arm at 2x: 27,120 steps
# The 4x arms keep the 2x arms' inputs (no tier) so the depth curve stays clean. Scored on the dev windows after each arm.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-lock-gpu1.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/lock_gpu1.log") 2>&1
echo "lock tests gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
# argparse keeps the last value: the tier arm's --mem-feat/--mem-orders and the seeds override TFI's and COMMON's
FLAGS[l4dir_s1_r]="$TFI --mem-feat $MEM/feat_all28_fly --mem-orders $ALL18,2008,2016 --n-layers 4 --seed 1"
FLAGS[l6x4_mt_r]="$TFI --n-layers 6"
FLAGS[l8x4_mt_r]="$TFI --n-layers 8"
FLAGS[l8x2s1_mt_r]="$TFI --n-layers 8 --seed 1"
STEPS[l4dir_s1_r]=24939; STEPS[l6x4_mt_r]=70688; STEPS[l8x4_mt_r]=54240; STEPS[l8x2s1_mt_r]=27120
declare -A FEAT=([l4dir_s1_r]=feat_all28_fly [l6x4_mt_r]=feat_all28 [l8x4_mt_r]=feat_all28 [l8x2s1_mt_r]=feat_all28)
score2() {  # score2 <name>: as scale_common's score, with the arm's own feature directory
  name=$1; short=${name%_r}
  [ -f "$OUT/k_$short.json" ] && return 0
  "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_$short=results/ckpt/${name}_transformer.pt@$MEM/${FEAT[$name]}" || echo "scoring $name failed"
}
preflight
for d in feat_all28 feat_all28_fly; do [ -f "$MEM/$d/features.json" ] || { echo "$d is not built"; exit 1; }; done
cat "$MEM"/feat_all28/pred_n8.npy "$MEM"/feat_all28_fly/pred_n2008.npy > /dev/null
for a in l4dir_s1_r l6x4_mt_r l8x4_mt_r l8x2s1_mt_r; do
  sweep "$a"
  echo; echo "=== scoring $a $(date +%T)"
  score2 "$a"
done
echo "lock tests gpu1 done $(date '+%F %T')"
