#!/usr/bin/env bash
# Spine runs (group-learning spec 1.6): 16M, 16k steps, the run_20m.sh recipe under bf16 autocast, metered, with the dev fixed-window eval.
#   hybrid    : 4 memory heads (--mem-every 2) + one 512-window attention layer + key widths 2,4   (run_capacity.sh depth4local)
#   tf        : RoPE transformer, same d / layers / d_ff                                           (run_20m.sh transformer arm)
#   zeroheads : the same recurrence + local attention with no memory head (the Samba cell)
# usage: scripts/run_spine.sh <device> <hybrid|tf|zeroheads> <seed> [steps] [tag]
#   tag goes into the run name (spine16_<arm>_<tag>_s<seed>): an equal-seconds rerun with other steps is a new name, not a skip.
# DRY_RUN=1 prints "<run name> <flags>" and exits; tests/test_spine_script.py parses it with the runner's own argparse.
set -u -o pipefail
DEV=${1:-cuda:1}; ARM=${2:-hybrid}; S=${3:-0}; STEPS=${4:-16000}; TAG=${5:-}
C=${CORPORA:-$(cd "$(dirname "$0")/.." && pwd)/corpora}
cd "$(dirname "$0")/.."
FILE=$C/linux_full.bytes
[ -f "$FILE" ] || { echo "corpus not found: $FILE"; echo "build it with scripts/build_corpus.py, or set CORPORA=<dir>"; exit 2; }
COMMON="--task code --corpus-file $FILE --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --steps $STEPS --save --seed $S --device $DEV --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-fixed 128 --val-offset 202117357 --eval-every 1000"
case $ARM in
  hybrid)    EXTRA="--arms mingru_sparse_post --gate --mem-every 2 --local-attn 512 --key-convs 2,4" ;;
  tf)        EXTRA="--arms transformer --rope" ;;
  zeroheads) EXTRA="--arms mingru --local-attn 512" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
NAME=spine16_${ARM}${TAG:+_$TAG}_s$S
[ -n "${DRY_RUN:-}" ] && { echo "$NAME $COMMON $EXTRA"; exit 0; }
[ -f results/$NAME.json ] && { echo "skip $NAME"; exit 0; }
echo "=== $NAME"
uv run python scripts/run_sparsela.py --run-name "$NAME" $COMMON $EXTRA 2>&1 | grep --line-buffered -v "^  "
