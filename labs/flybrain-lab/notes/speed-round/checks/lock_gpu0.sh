#!/usr/bin/env bash
# The lock tests' second card (Kyle, 2026-10-09 23:50: "you can use gpu 0 overnight now"): waits for an idle card (three minutes without
# a compute process and under 2,500 MiB), then runs under the guard (probe_guard.sh stops this unit when a game or another process
# takes the card; a run lost that way is retaken by tables-lock-gpu1 through its stale claim). Takes the 4x depth point's 8-layer arm
# off GPU 1's list so both 4x arms run tonight, then one arm of its own:
#   l8x4_mt_r      8 layers, 2^28 tables, four hours: 54,240 steps (2 x l8x2's 27,120)
#   l6x2dir_mt_r   6 layers + the project tier at two hours: does the tier's gain (0.008 at 4 layers, one hour) hold at the lock's
#                  likely depth and budget? 34,482 steps = l6x2's 35,344 / 1.025, the tier's per-step cost from cal6 (0.575 / 0.561 at
#                  4 layers; at 6 the tier's share is smaller, so this budget is, if anything, short), scored with feat_all28_fly.
# GPU 1 (lock_gpu1.sh) keeps l4dir_s1_r, l6x4_mt_r and l8x2s1_mt_r; its sweep skips l8x4 once this card has it.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-lock-gpu0.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/lock_gpu0.log") 2>&1
echo "lock tests gpu0 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
FLAGS[l8x4_mt_r]="$TFI --n-layers 8"
FLAGS[l6x2dir_mt_r]="$TFI --mem-feat $MEM/feat_all28_fly --mem-orders $ALL18,2008,2016 --n-layers 6"
STEPS[l8x4_mt_r]=54240; STEPS[l6x2dir_mt_r]=34482
declare -A FEAT=([l8x4_mt_r]=feat_all28 [l6x2dir_mt_r]=feat_all28_fly)
score2() {  # score2 <name>: as scale_common's score, with the arm's own feature directory
  name=$1; short=${name%_r}
  [ -f "$OUT/k_$short.json" ] && return 0
  "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_$short=results/ckpt/${name}_transformer.pt@$MEM/${FEAT[$name]}" || echo "scoring $name failed"
}
idle=0
while [ "$idle" -lt 6 ]; do
  sleep 30
  n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$n" = 0 ] && [ "$used" -lt 2500 ]; then idle=$((idle + 1)); else idle=0; fi
done
echo "card idle for three minutes at $(date '+%F %T')"
preflight
systemd-run --user --unit tables-lock-guard /bin/bash "$S/probe_guard.sh" tables-lock-gpu0.service "$OUT/guard.log"
cat "$MEM"/feat_all28/pred_n8.npy "$MEM"/feat_all28_fly/pred_n2008.npy > /dev/null
for a in l8x4_mt_r l6x2dir_mt_r; do
  sweep "$a"
  echo; echo "=== scoring $a $(date +%T)"
  score2 "$a"
done
echo "lock tests gpu0 done $(date '+%F %T')"
