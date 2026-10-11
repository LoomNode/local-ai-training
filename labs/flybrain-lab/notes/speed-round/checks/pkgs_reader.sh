#!/usr/bin/env bash
# A reader trained on the package corpus with its 2^SIZE table (the matched number for the table-size decision: the zero-training
# probe swapped bigger tables under readers trained on 2^28 statistics, which understates them). The hour arm as l4pk_mt_r (4 layers,
# l4b's 25,562 steps, cosine), run from .worktrees/speed-run5 (the uint8 corpus loader), scored through its own table on the corpus's
# dev windows and on the old kernel dev windows. usage: SIZE=30 UNIT=tables-pkgs30-gpu1 GUARD=0|1 pkgs_reader.sh
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
: "${SIZE:?SIZE=29 or 30}"; : "${UNIT:?the unit name, for the claim}"; GUARD=${GUARD:-0}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
WT5=/games/ailab/lab/flybrain-lab/.worktrees/speed-run5
cd "$WT"
exec > >(tee -a "$OUT/pkgs_reader$SIZE.log") 2>&1
NAME=l4pk${SIZE}_mt_r
echo "$NAME start $(date '+%F %T'); scoring code at $(git rev-parse --short HEAD), the run from $WT5 at $(git -C "$WT5" rev-parse --short HEAD)"
PK=/vault/ailab/flybrain-mem/pkgs/feat_all$SIZE
C2=/games/ailab/corpora/pkgs.bytes
DEV=5283527978; OLDK=5460445051
FLAGS="$TFI --mem-feat $PK --n-layers 4 --corpus-file $C2 --train-bytes 5283527978 --val-offset $DEV"
score3() {  # score3 <offset> <tag>
  off=$1; tag=$2; short=${NAME%_r}
  [ -f "$OUT/k_${short}_$tag.json" ] && return 0
  "$PY" "$S/score_tables.py" --corpus "$C2" --offset "$off" --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_${short}_$tag=$WT5/results/ckpt/${NAME}_transformer.pt@$PK" || echo "scoring $NAME $tag failed"
}
if [ "$GUARD" = 1 ]; then
  idle=0
  while [ "$idle" -lt 6 ]; do
    sleep 30
    n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
    used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "$n" = 0 ] && [ "$used" -lt 2500 ]; then idle=$((idle + 1)); else idle=0; fi
  done
  echo "card idle for three minutes at $(date '+%F %T')"
  systemd-run --user --unit "${UNIT%.service}-guard" /bin/bash "$S/probe_guard.sh" "$UNIT" "$OUT/guard.log"
fi
preflight
cat "$PK"/pred_n8.npy > /dev/null &
echo "$UNIT" > "$WT/results/claim_$NAME"
cd "$WT5"
if [ ! -f "results/$NAME.json" ]; then
  echo; echo "=== $NAME $(date +%T): 25562 steps (l4b's hour) on the package corpus with the 2^$SIZE table"
  "$PY" scripts/run_sparsela.py --run-name "$NAME" $COMMON $FLAGS --steps 25562 --save --val-fixed 128 --eval-every 2000 2>&1 | grep --line-buffered -v "^  "
fi
cp -n "results/$NAME.json" "$SP/results/" 2>/dev/null
cd "$WT"
echo; echo "=== scoring $NAME $(date +%T)"
score3 "$DEV" pk
score3 "$OLDK" oldk_pk
echo "$NAME done $(date '+%F %T')"
