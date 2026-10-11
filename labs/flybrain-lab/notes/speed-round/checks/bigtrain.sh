#!/usr/bin/env bash
# The big train (Kyle, 2026-10-11: "go"; "we should train till we get specific outcomes, not till certain steps"): the locked reader on the
# package corpus, from the pinned worktree .worktrees/big1 (a5e9e6b on spike/speed-int: the uint8 loader, --wsd-cooldown, --resume).
#   inputs: the match lookup 8/16/32 and the package corpus's 2^28 table (the reader trained on the small table reads any table best:
#           SPEED.md "The package corpus"), the 18 orders through the fused channel; 512 wide, batch 8, windows of 2048
#   schedule: warmup-stable-decay with no end: --steps 10,000,000 and --wsd-cooldown 1 is a constant rate after the warmup; a snapshot
#           is a separate short run resumed from a checkpoint with STEPS=<that step + K> COOLDOWN=K NAME=<name>_snap<step>
#   checkpoints every 2,000 steps (bigscore.sh scores each on the clean dev and kernel windows and prunes to the last three + one per 10,000)
#   resume: RESUME_FROM=auto picks the latest checkpoint of NAME, so the GPU 0 copy restarts where the guard stopped it
# usage: NAME=big12 LAYERS=12 UNIT=big12-gpu1.service CUDA_VISIBLE_DEVICES=<uuid> bigtrain.sh   (env: STEPS COOLDOWN RESUME_FROM CKPT_EVERY SEED)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"; : "${NAME:?}"; : "${LAYERS:?}"; : "${UNIT:?}"
# read the environment BEFORE scale_common.sh: it declares an associative array named STEPS, which hides an exported STEPS (a dry run
# meant for 200 steps ran open-ended on 2026-10-11 because this line came after the source)
NSTEPS=${STEPS:-10000000}; COOLDOWN=${COOLDOWN:-1}; RESUME_FROM=${RESUME_FROM:-auto}; CKPT_EVERY=${CKPT_EVERY:-2000}; SEED=${SEED:-0}
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
WT1=/games/ailab/lab/flybrain-lab/.worktrees/big1
mkdir -p "$OUT/big" "$WT1/results/ckpt"
cd "$WT1"
exec >> "$OUT/big/$NAME.log" 2>&1
PK=/vault/ailab/flybrain-mem/pkgs/feat_all28
COMMON_WSD=${COMMON//--cosine/}
FLAGS="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32 --fused-channels --mem-feat $PK --mem-orders $ALL18 --n-layers $LAYERS --d-model 512 --d-ff 1024 --seed $SEED --corpus-file /games/ailab/corpora/pkgs.bytes --train-bytes 5283527978 --val-offset 5283527978"
resume=""
if [ "$RESUME_FROM" = auto ]; then
  last=$(ls -v results/ckpt/${NAME}_transformer.step*.pt 2>/dev/null | tail -n 1)
  [ -n "$last" ] && resume="--resume $last"
elif [ "$RESUME_FROM" != none ]; then
  resume="--resume $RESUME_FROM"
fi
echo "$UNIT" > "$WT/results/claim_$NAME"                        # the board reads claims from the run worktree
echo; echo "=== $NAME $(date +%T): $NSTEPS steps ($LAYERS layers, cooldown $COOLDOWN, ${resume:-fresh}, code $(git rev-parse --short HEAD), $(date '+%F'))"
"$PY" scripts/run_sparsela.py --run-name "$NAME" $COMMON_WSD $FLAGS --steps "$NSTEPS" --wsd-cooldown "$COOLDOWN" --ckpt-every "$CKPT_EVERY" --save --val-fixed 128 --eval-every 2000 $resume 2>&1 | grep --line-buffered -v "^  "
echo "=== $NAME exited $(date '+%F %T') status ${PIPESTATUS[0]}"
rm -f "$WT/results/claim_$NAME"
