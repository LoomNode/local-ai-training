#!/usr/bin/env bash
# Width ladder: plain backbone vs backbone + sparse head (output placement), Run 28 protocol at each width.
# usage: scripts/run_ladder.sh <backbone> <device> [widths] [seeds] [corpora]
set -u
BB=${1:-mingru}; DEV=${2:-cuda:0}; WIDTHS=${3:-"64 128 256 384"}; SEEDS=${4:-"0 1"}; CORPORA=${5:-"code cpp wiki"}
C=corpora
cd "$(dirname "$0")/.."
declare -A FILE=([code]=$C/code.bytes [cpp]=$C/cpp.bytes [wiki]=/games/ailab/lab/scrt-lab/data/wiki.train.raw)
for D in $WIDTHS; do for CORPUS in $CORPORA; do for S in $SEEDS; do
  NAME=ladder_${BB}_${CORPUS}_d${D}_s${S}
  [ -f results/$NAME.json ] && { echo "skip $NAME"; continue; }
  echo "=== $NAME"
  uv run python scripts/run_sparsela.py --task code --corpus-file "${FILE[$CORPUS]}" --device "$DEV" \
    --arms "$BB,${BB}_sparse_post" --normalize --key-conv 2 --top-k 4 --n-key 1024 --lr 0.001 --decay 1.0 \
    --d-model "$D" --d-ff $((2*D)) --n-layers 4 --steps 3000 --batch 16 --seq-len 512 \
    --train-bytes 9000000 --val-bytes 1000000 --seed "$S" --run-name "$NAME" 2>&1 | grep -v "^  "
done; done; done
