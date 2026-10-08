#!/usr/bin/env bash
# Budget-scaling arms on GPU 0 (Kyle's card): waits for an idle card (three minutes without a compute process and under 2,500 MiB),
# then runs arms under the guard, taking the list from the end the GPU 1 unit reaches last. A run lost to the guard is retaken by
# GPU 1 through its stale claim.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-scale-gpu0.service
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh
cd "$WT"
exec > >(tee -a "$OUT/scale_gpu0.log") 2>&1
echo "scale gpu0 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
idle=0
while [ "$idle" -lt 6 ]; do
  sleep 30
  n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$n" = 0 ] && [ "$used" -lt 2500 ]; then idle=$((idle + 1)); else idle=0; fi
done
echo "card idle for three minutes at $(date '+%F %T')"
preflight
systemd-run --user --unit tables-scale-guard /bin/bash "$S/probe_guard.sh" tables-scale-gpu0.service "$OUT/guard.log"
cat "$MEM"/feat_all28/*.npy > /dev/null
sweep l6x2_mt_r l4x2_mt_r l8x2_mt_r
echo "scale gpu0 done $(date '+%F %T')"
