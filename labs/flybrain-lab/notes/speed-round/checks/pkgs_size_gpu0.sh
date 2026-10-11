#!/usr/bin/env bash
# The table's size on the package corpus (item 2 before the big train; the use-time RAM is Kyle's call: 12 / 24 / 48 GB at 2^28 / 2^29 /
# 2^30 two-way). The same checkpoints scored through the three tables on the same windows: l4b (the kernel hour arm) and l12x4 (the best
# four-hour arm), both trained with the kernel's 2^28 features, and l4pk (the hour arm trained on the package corpus with its 2^28
# table), on the corpus's dev windows and on the old kernel dev windows. A zero-training swap of the table under a reader trained on a
# smaller one understates what the bigger table is worth; a reader trained on it would say more (not run). GPU 0 under the guard,
# after tables-pkgs-gpu0 and after the 2^30 build (unit pkgs-assoc30) has ended. The 2^28 scores come from pkgs_gpu0.sh (k_*_pk).
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-pkgs-size-gpu0.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
WT5=/games/ailab/lab/flybrain-lab/.worktrees/speed-run5
cd "$WT"
exec > >(tee -a "$OUT/pkgs_size_gpu0.log") 2>&1
echo "pkgs size gpu0 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
M=/vault/ailab/flybrain-mem/pkgs
C2=/games/ailab/corpora/pkgs.bytes
DEV=5283527978; OLDK=5460445051
score4() {  # score4 <name> <offset> <tag> <featdir> [ckpt]
  name=$1; off=$2; tag=$3; feat=$4; ckpt=${5:-results/ckpt/${name}_transformer.pt}; short=${name%_r}
  [ -f "$OUT/k_${short}_$tag.json" ] && return 0
  "$PY" "$S/score_tables.py" --corpus "$C2" --offset "$off" --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_${short}_$tag=$ckpt@$feat" || echo "scoring $name $tag failed"
}
while systemctl --user is-active --quiet tables-pkgs-gpu0.service; do sleep 120; done
echo "pkgs queue ended $(date '+%F %T')"
t=0
while systemctl --user is-active --quiet pkgs-assoc30.service && [ "$t" -lt 18000 ]; do sleep 300; t=$((t + 300)); done
echo "2^30 build state at $(date '+%F %T'): $(systemctl --user is-active pkgs-assoc30.service)"
idle=0
while [ "$idle" -lt 6 ]; do
  sleep 30
  n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$n" = 0 ] && [ "$used" -lt 2500 ]; then idle=$((idle + 1)); else idle=0; fi
done
echo "card idle for three minutes at $(date '+%F %T')"
preflight
systemd-run --user --unit tables-pkgs-size-guard /bin/bash "$S/probe_guard.sh" tables-pkgs-size-gpu0.service "$OUT/guard.log"
for size in 29 30; do
  F=$M/feat_all$size
  [ -f "$F/features.json" ] || { echo "no $F"; continue; }
  echo; echo "=== through the 2^$size table $(date +%T)"
  for name in l4b_mt_r l12x4_mt_r; do
    score4 "$name" "$DEV" "pk$size" "$F"
    score4 "$name" "$OLDK" "oldk_pk$size" "$F"
  done
  if [ -f "$WT5/results/ckpt/l4pk_mt_r_transformer.pt" ]; then
    score4 l4pk_mt_r "$DEV" "pk$size" "$F" "$WT5/results/ckpt/l4pk_mt_r_transformer.pt"
    score4 l4pk_mt_r "$OLDK" "oldk_pk$size" "$F" "$WT5/results/ckpt/l4pk_mt_r_transformer.pt"
  fi
done
echo "pkgs size gpu0 done $(date '+%F %T')"
