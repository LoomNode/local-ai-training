#!/usr/bin/env bash
# Does the gain keep growing with context? Tuned recipe, d=256, seq 4096, batch 4 (same tokens/step as Run 34), index-form head.
# usage: scripts/run_ctx4096.sh <device> <corpus: cpython|enwik8> <seed>
set -u
DEV=${1:-cuda:0}; CORPUS=${2:-cpython}; S=${3:-0}
C=${CORPORA:-$(cd "$(dirname "$0")/.." && pwd)/corpora}
cd "$(dirname "$0")/.."
if [ "$CORPUS" = cpython ]; then TASK=code; FILE=$C/cpython.bytes; TB=40000000; else TASK=wiki; FILE=$C/enwik8.bytes; TB=90000000; fi
COMMON="--task $TASK --corpus-file $FILE --train-bytes $TB --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.001 --decay 1.0 --d-model 256 --d-ff 512 --n-layers 4 --batch 4 --seq-len 4096 --steps 8000 --val-bytes 2000000 --save --seed $S --device $DEV --warmup 500 --cosine --wd 0.1 --chunk 256"
run() { NAME=$1; shift; [ -f results/$NAME.json ] && { echo "skip $NAME"; return; }; echo "=== $NAME"; uv run python scripts/run_sparsela.py --run-name "$NAME" $COMMON "$@" 2>&1 | grep -v "^  "; }
run ctx4096_${CORPUS}_gate_s$S  --arms mingru_sparse_post --gate
run ctx4096_${CORPUS}_s$S       --arms mingru,transformer --rope
