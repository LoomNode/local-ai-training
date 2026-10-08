#!/usr/bin/env bash
# Budget-scaling pair on GPU 1: the depth curve (8 / 6 / 4 layers, 2^28 tables) at twice the base's calibrated cost, sharing the
# list with the GPU 0 unit through claims, then scores whatever has a checkpoint.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-scale-gpu1.service
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh
cd "$WT"
exec > >(tee -a "$OUT/scale_gpu1.log") 2>&1
echo "scale gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
preflight
cat "$MEM"/feat_all28/*.npy > /dev/null
sweep l8x2_mt_r l4x2_mt_r l6x2_mt_r
echo; echo "=== scoring $(date +%T)"
for a in l8x2_mt_r l6x2_mt_r l4x2_mt_r; do score "$a"; done
echo "scale gpu1 done $(date '+%F %T')"
