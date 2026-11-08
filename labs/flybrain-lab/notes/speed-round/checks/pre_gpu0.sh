#!/usr/bin/env bash
# Before the big train (Kyle, 2026-10-10: "go ahead with 1, 3 and 4"), GPU 0's share, under the guard (probe_guard.sh stops this unit
# when a game or another process takes the card): item 3, width at four times the hour budget, against 8x512's 0.598 at 4x:
#   s12x4_tf_mt_r  12 layers of 384: 52,588 steps = 4 x 13,147 (the 2026-10-07 session's ratio 1.090; per-step work does not depend on
#                  the table's size, so the 1x budget carries over to the 2^28 tables)
#   s6x4_tf_mt_r   6 layers of 768: 38,264 steps = 4 x 9,566 (ratio 1.498)
# The 12x512 arm needs its own calibration and runs on GPU 1 (pre_gpu1.sh). A run lost to a yield is retaken by whichever card is free.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-pre-gpu0.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/pre_gpu0.log") 2>&1
echo "pre-train tests gpu0 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
FLAGS[s12x4_tf_mt_r]="$TFI --d-model 384 --n-layers 12 --d-ff 768"
FLAGS[s6x4_tf_mt_r]="$TFI --d-model 768 --n-layers 6 --d-ff 1536"
STEPS[s12x4_tf_mt_r]=52588; STEPS[s6x4_tf_mt_r]=38264
declare -A FEAT=([s12x4_tf_mt_r]=feat_all28 [s6x4_tf_mt_r]=feat_all28)
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
systemd-run --user --unit tables-pre-guard /bin/bash "$S/probe_guard.sh" tables-pre-gpu0.service "$OUT/guard.log"
cat "$MEM"/feat_all28/pred_n8.npy > /dev/null
for a in s12x4_tf_mt_r s6x4_tf_mt_r; do
  sweep "$a"
  echo; echo "=== scoring $a $(date +%T)"
  score2 "$a"
done
echo "pre-train tests gpu0 done $(date '+%F %T')"
