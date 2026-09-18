#!/usr/bin/env bash
# 20M-class run: d=512, 8 layers, d_ff 1024 (minGRU ~15M, transformer ~17M), seq 2048, 16k steps, tuned recipe, index-form head.
# Train on the Linux kernel (200 MB) or enwik8 (90 MB); CPython is the held-out code corpus (scripts/cross_eval.py).
# usage: scripts/run_20m.sh <device> <corpus: linux|enwik8> <seed> [steps]
set -u
DEV=${1:-cuda:0}; CORPUS=${2:-linux}; S=${3:-0}; STEPS=${4:-16000}
C=${CORPORA:-$(cd "$(dirname "$0")/.." && pwd)/corpora}
cd "$(dirname "$0")/.."
if [ "$CORPUS" = linux ]; then TASK=code; FILE=$C/linux_full.bytes; TB=200000000; else TASK=wiki; FILE=$C/enwik8.bytes; TB=90000000; fi
COMMON="--task $TASK --corpus-file $FILE --train-bytes $TB --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --steps $STEPS --val-bytes 2000000 --save --seed $S --device $DEV --warmup 1000 --cosine --wd 0.1 --chunk 256"
run() { NAME=$1; shift; [ -f results/$NAME.json ] && { echo "skip $NAME"; return; }; echo "=== $NAME"; uv run python scripts/run_sparsela.py --run-name "$NAME" $COMMON "$@" 2>&1 | grep -v "^  "; }
run m20_${CORPUS}_gate_s$S  --arms mingru_sparse_post --gate
run m20_${CORPUS}_s$S       --arms mingru,transformer --rope
