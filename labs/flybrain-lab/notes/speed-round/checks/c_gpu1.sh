#!/usr/bin/env bash
# Bets C and E on GPU 1 after calibration session 3: the arms at their calibrated budgets, then the dev scores.
# GPU 1 only: Kyle is using GPU 0 today (2026-10-08, "i'm gonna use gpu 0 soon").
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-c-gpu1.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/c_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/c_gpu1.log") 2>&1
echo "bets C and E gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
while systemctl --user is-active --quiet tables-calsession3.service || [ ! -f results/cal4_base_end.json ]; do sleep 30; done
ARMS=$(arms)
sleep 10
preflight
sweep $ARMS
echo; echo "=== scoring $(date +%T)"
for a in $ARMS; do score "$a"; done
echo "bets C and E gpu1 done $(date '+%F %T')"
