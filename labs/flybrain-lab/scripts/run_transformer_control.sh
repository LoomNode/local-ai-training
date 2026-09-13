#!/usr/bin/env bash
# Transformer negative control: Run 28 protocol, d=128, plain / +sparse (pre, post) / +learned-key head.
set -u
DEV=${1:-cuda:1}; SEEDS=${2:-"0 1"}; CORPORA=${3:-"code cpp wiki"}
C=corpora
cd "$(dirname "$0")/.."
declare -A FILE=([code]=$C/code.bytes [cpp]=$C/cpp.bytes [wiki]=/games/ailab/lab/scrt-lab/data/wiki.train.raw)
for CORPUS in $CORPORA; do for S in $SEEDS; do
  NAME=bb_transformer_${CORPUS}_s${S}
  [ -f results/$NAME.json ] && { echo "skip $NAME"; continue; }
  echo "=== $NAME"
  uv run python scripts/run_sparsela.py --task code --corpus-file "${FILE[$CORPUS]}" --device "$DEV" \
    --arms transformer,transformer_sparse_pre,transformer_sparse_post,transformer_learned_post \
    --normalize --key-conv 2 --top-k 4 --n-key 1024 --lr 0.001 --decay 1.0 \
    --d-model 128 --d-ff 256 --n-layers 4 --steps 3000 --batch 16 --seq-len 512 \
    --train-bytes 9000000 --val-bytes 1000000 --seed "$S" --run-name "$NAME" 2>&1 | grep -v "^  "
done; done
