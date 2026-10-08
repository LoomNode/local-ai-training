#!/usr/bin/env bash
# Tier 1 backbone arms on GPU 0 (Kyle's card): waits until the card has had no compute process (a game counts) and under 2,500 MiB in
# use for three minutes, then runs arms under the guard (a game stops this unit; the GPU 1 unit takes over what was lost), in the
# order the GPU 1 unit reaches last.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-tier1c-gpu0.service
source /tmp/claude-1000/-games-ailab/f5e9946e-97cc-48fe-9eb3-0b019140c726/scratchpad/tables/tier1_common.sh
cd "$WT"
exec > >(tee -a "$OUT/tier1c_gpu0.log") 2>&1
echo "tier 1 gpu0 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
while systemctl --user is-active --quiet tables-calsession.service || [ ! -f results/cal2_base_end.json ]; do sleep 30; done
idle=0
while [ "$idle" -lt 6 ]; do
  sleep 30
  n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$n" = 0 ] && [ "$used" -lt 2500 ]; then idle=$((idle + 1)); else idle=0; fi
done
echo "card idle for three minutes at $(date '+%F %T')"
preflight
systemd-run --user --unit tables-tier1c-guard /bin/bash "$S/probe_guard.sh" tables-tier1c-gpu0.service "$OUT/guard.log"
cat "$MEM"/feat_all/*.npy > /dev/null
sweep gla_mt_r hyb_mt_r w4k_tf_mt_r s6_tf_mt_r s12_tf_mt_r
echo "tier 1 gpu0 done $(date '+%F %T')"
