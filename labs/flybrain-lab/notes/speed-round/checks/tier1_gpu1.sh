#!/usr/bin/env bash
# Tier 1 backbone arms on GPU 1: waits for the table queue (tables-gpu1-queue3), then sweeps all five arms (an arm another unit is
# running is waited for, a run lost to a guard yield on GPU 0 is taken over), then scores them on the dev windows. The window-4096
# arm is also scored on 4096-byte windows for its own warm number.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-tier1c-gpu1.service
source /tmp/claude-1000/-games-ailab/f5e9946e-97cc-48fe-9eb3-0b019140c726/scratchpad/tables/tier1_common.sh
cd "$WT"
exec > >(tee -a "$OUT/tier1c_gpu1.log") 2>&1
echo "tier 1 gpu1 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
while systemctl --user is-active --quiet tables-calsession.service || [ ! -f results/cal2_base_end.json ]; do sleep 30; done
sleep 10
preflight
cat "$MEM"/feat_all/*.npy > /dev/null
sweep s12_tf_mt_r s6_tf_mt_r w4k_tf_mt_r hyb_mt_r gla_mt_r
echo; echo "=== scoring $(date +%T)"
for a in s12_tf_mt_r s6_tf_mt_r w4k_tf_mt_r hyb_mt_r gla_mt_r; do score "$a"; done
[ -f "$OUT/k_w4k_T4096.json" ] || "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 512 --T 4096 --device cuda:0 --out-dir "$OUT" "k_w4k_T4096=results/ckpt/w4k_tf_mt_r_transformer.pt@$MEM/feat_all" || echo "4096-window scoring failed"
echo "tier 1 gpu1 done $(date '+%F %T')"
