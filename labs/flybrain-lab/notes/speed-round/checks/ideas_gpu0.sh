#!/usr/bin/env bash
# Idea arms on GPU 0 (Kyle's card): after the Tier 1 GPU 0 unit, waits for an idle card (three minutes without a compute process and
# under 2,500 MiB), then runs arms under the guard, in the order the GPU 1 unit reaches last.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-ideasb-gpu0.service
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ideas_common.sh
cd "$WT"
exec > >(tee -a "$OUT/ideasb_gpu0.log") 2>&1
echo "ideas gpu0 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
while systemctl --user is-active --quiet tables-calsession.service || [ ! -f results/cal2_base_end.json ]; do sleep 30; done
while systemctl --user is-active --quiet tables-tier1c-gpu0.service; do sleep 60; done
for d in feat_paq feat_top2 feat_all28; do while [ ! -f "$MEM/$d/features.json" ]; do sleep 60; done; done
idle=0
while [ "$idle" -lt 6 ]; do
  sleep 30
  n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$n" = 0 ] && [ "$used" -lt 2500 ]; then idle=$((idle + 1)); else idle=0; fi
done
echo "card idle for three minutes at $(date '+%F %T')"
preflight
systemd-run --user --unit tables-ideasb-guard /bin/bash "$S/probe_guard.sh" tables-ideasb-gpu0.service "$OUT/guard.log"
cat "$MEM"/feat_paq/*.npy "$MEM"/feat_top2/*.npy > /dev/null
sweep l4b_mt_r l6_mt_r l4_mt_r top2_mt_r paq_mt_r
echo "ideas gpu0 done $(date '+%F %T')"
