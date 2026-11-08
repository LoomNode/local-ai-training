#!/usr/bin/env bash
# The first numbers on the package corpus (Kyle, 2026-10-10: "installed plus main", "yeah use vault"; the corpus from
# build_corpus_pkgs.py, its 2^28 table at /vault/ailab/flybrain-mem/pkgs/feat_all28 from pkgs_tables.sh), on GPU 0 under the guard after
# the narrow arm (tables-pre-gpu0):
#   1. the corpus probe at zero training: the kernel-trained checkpoints (l4b at the hour, l8x4 and l12x4 at four hours) read through
#      the new table, scored on the corpus's 1,024 fixed dev windows (from dev_offset = train_end; the dev file is shuffled at the file
#      level so the slice spans many packages) and on the old kernel dev windows (the old held-out region sits after dev and test in
#      pkgs.bytes; its windows start at 5,460,445,051: pkgs/layout.json), against their old scores 0.652 / 0.598 / 0.591;
#   2. the hour arm (4 layers, l4b's 25,562 steps, cosine as l4b) trained on the package corpus with its table, from
#      .worktrees/speed-run5 (spike/speed-int: the uint8 corpus loader, without which a 5 GB corpus needs 90 GB of RAM), scored the same
#      two ways. Same model and inputs as on the old corpus, so no new calibration; the features are memory-mapped from /vault.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
UNIT=tables-pkgs-gpu0.service
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
WT5=/games/ailab/lab/flybrain-lab/.worktrees/speed-run5
cd "$WT"
exec > >(tee -a "$OUT/pkgs_gpu0.log") 2>&1
echo "pkgs gpu0 start $(date '+%F %T'); scoring code at $(git rev-parse --short HEAD), the run from $WT5 at $(git -C "$WT5" rev-parse --short HEAD)"
PK=/vault/ailab/flybrain-mem/pkgs/feat_all28
C2=/games/ailab/corpora/pkgs.bytes
DEV=5283527978; OLDK=5460445051
NEWC="--corpus-file $C2 --train-bytes 5283527978 --val-offset $DEV"
L4PK="$TFI --mem-feat $PK --n-layers 4 $NEWC"
score3() {  # score3 <name> <offset> <tag> [ckpt]: the checkpoint through the new table on the windows from <offset> of the package corpus
  name=$1; off=$2; tag=$3; ckpt=${4:-results/ckpt/${name}_transformer.pt}; short=${name%_r}
  [ -f "$OUT/k_${short}_$tag.json" ] && return 0
  "$PY" "$S/score_tables.py" --corpus "$C2" --offset "$off" --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_${short}_$tag=$ckpt@$PK" || echo "scoring $name $tag failed"
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
echo; echo "=== the corpus probe at zero training $(date +%T)"
for name in l4b_mt_r l8x4_mt_r l12x4_mt_r; do
  score3 "$name" "$DEV" pk
  score3 "$name" "$OLDK" oldk_pk
done
echo "$UNIT" > "$WT/results/claim_l4pk_mt_r"
cd "$WT5"
if [ ! -f results/l4pk_mt_r.json ]; then
  echo; echo "=== l4pk_mt_r $(date +%T): 25562 steps (l4b's hour) on the package corpus, from $WT5"
  "$PY" scripts/run_sparsela.py --run-name l4pk_mt_r $COMMON $L4PK --steps 25562 --save --val-fixed 128 --eval-every 2000 2>&1 | grep --line-buffered -v "^  "
fi
cp -n results/l4pk_mt_r.json "$SP/results/" 2>/dev/null
cd "$WT"
echo; echo "=== scoring l4pk_mt_r $(date +%T)"
score3 l4pk_mt_r "$DEV" pk "$WT5/results/ckpt/l4pk_mt_r_transformer.pt"
score3 l4pk_mt_r "$OLDK" oldk_pk "$WT5/results/ckpt/l4pk_mt_r_transformer.pt"
echo "pkgs gpu0 done $(date '+%F %T')"
