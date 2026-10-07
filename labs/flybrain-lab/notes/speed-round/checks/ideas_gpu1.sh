#!/usr/bin/env bash
# Idea arms (4: PAQ keys, top-two slots; 3: fewer layers) on GPU 1 after the Tier 1 sweep: sweeps the five arms (sharing the list with
# the GPU 0 unit through claims), then scores them on the dev windows.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-ideas-gpu1.service
source /tmp/claude-1000/-games-ailab/f5e9946e-97cc-48fe-9eb3-0b019140c726/scratchpad/tables/ideas_common.sh
cd "$WT"
exec > >(tee -a "$OUT/ideas_gpu1.log") 2>&1
echo "ideas gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
while systemctl --user is-active --quiet tables-tier1b-gpu1.service; do sleep 30; done
for d in feat_paq feat_top2 feat_all28; do while [ ! -f "$MEM/$d/features.json" ]; do sleep 60; done; done
sleep 10
preflight
cat "$MEM"/feat_paq/*.npy "$MEM"/feat_top2/*.npy > /dev/null
sweep paq_mt_r top2_mt_r l4_mt_r l6_mt_r l4b_mt_r
echo; echo "=== scoring $(date +%T)"
for a in paq_mt_r top2_mt_r l4_mt_r l6_mt_r l4b_mt_r; do score "$a"; done
echo "ideas gpu1 done $(date '+%F %T')"
