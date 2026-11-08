#!/usr/bin/env bash
# Seed 1 of the depth curve at twice the cost (6 and 4 layers, 2^28 tables) on GPU 1, then the scores; see scale_common.sh.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-scale-s1-gpu1.service
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh
cd "$WT"
exec > >(tee -a "$OUT/scale_s1_gpu1.log") 2>&1
echo "scale seed 1 gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
preflight
cat "$MEM"/feat_all28/*.npy > /dev/null
sweep l6x2s1_mt_r l4x2s1_mt_r
echo; echo "=== scoring $(date +%T)"
for a in l6x2s1_mt_r l4x2s1_mt_r; do score "$a"; done
echo "scale seed 1 gpu1 done $(date '+%F %T')"
