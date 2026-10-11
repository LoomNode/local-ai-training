#!/usr/bin/env bash
# The first run on the package corpus (Kyle, 2026-10-10: "installed plus main", "yeah use vault"): the hour arm (4 layers, l4b's
# 25,562 steps) with the corpus's own 2^28 table (/vault/ailab/flybrain-mem/pkgs/feat_all28, built by pkgs_tables.sh), on GPU 0 under the
# guard after the narrow arm (tables-pre-gpu0). Scored on the corpus's 1,024 fixed dev windows (from dev_offset = train_end, the dev
# file shuffled at the file level so the slice spans many packages) and on the old kernel dev windows through the new table (the old
# held-out region sits after dev and test in pkgs.bytes; its windows start at 5,460,445,051: pkgs/layout.json). The runner's step cost
# is the same per step as on the old corpus (same model, same inputs; the features are memory-mapped from /vault), so no new calibration.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-pkgs-gpu0.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
cd "$WT"
exec > >(tee -a "$OUT/pkgs_gpu0.log") 2>&1
echo "pkgs gpu0 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
PK=/vault/ailab/flybrain-mem/pkgs/feat_all28
C2=/games/ailab/corpora/pkgs.bytes
NEWC="--corpus-file $C2 --train-bytes 5283527978 --val-offset 5283527978"
FLAGS[l4pk_mt_r]="$TFI --mem-feat $PK --n-layers 4 $NEWC"
STEPS[l4pk_mt_r]=25562
score3() {  # score3 <name> <offset> <tag>: the checkpoint through the new table on the windows from <offset> of the package corpus
  name=$1; off=$2; tag=$3; short=${name%_r}
  [ -f "$OUT/k_${short}_$tag.json" ] && return 0
  "$PY" "$S/score_tables.py" --corpus "$C2" --offset "$off" --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_${short}_$tag=results/ckpt/${name}_transformer.pt@$PK" || echo "scoring $name $tag failed"
}
while systemctl --user is-active --quiet tables-pre-gpu0.service; do sleep 120; done
echo "pre queue ended $(date '+%F %T')"
idle=0
while [ "$idle" -lt 6 ]; do
  sleep 30
  n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$n" = 0 ] && [ "$used" -lt 2500 ]; then idle=$((idle + 1)); else idle=0; fi
done
echo "card idle for three minutes at $(date '+%F %T')"
preflight
systemd-run --user --unit tables-pkgs-guard /bin/bash "$S/probe_guard.sh" tables-pkgs-gpu0.service "$OUT/guard.log"
sweep l4pk_mt_r
echo; echo "=== scoring l4pk_mt_r $(date +%T)"
score3 l4pk_mt_r 5283527978 pk
score3 l4pk_mt_r 5460445051 oldk_pk
echo "pkgs gpu0 done $(date '+%F %T')"
