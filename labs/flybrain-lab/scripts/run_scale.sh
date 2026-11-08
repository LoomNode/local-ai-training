#!/usr/bin/env bash
# Scale run: d=256, seq 2048, 4096-unit head, CPython (45 MB) then enwik8 (100 MB). One seed per invocation.
# usage: scripts/run_scale.sh <device> <seed>
set -u
DEV=${1:-cuda:0}; S=${2:-0}
C=${CORPORA:-$(cd "$(dirname "$0")/.." && pwd)/corpora}
cd "$(dirname "$0")/.."
COMMON="--normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.001 --decay 1.0 --d-model 256 --d-ff 512 --n-layers 4 --batch 8 --seq-len 2048 --steps 8000 --val-bytes 2000000 --save --seed $S --device $DEV"
run() { NAME=$1; shift; [ -f results/$NAME.json ] && { echo "skip $NAME"; return; }; echo "=== $NAME"; uv run python scripts/run_sparsela.py --run-name "$NAME" $COMMON "$@" 2>&1 | grep -v "^  "; }
run scale_cpython_s$S       --task code --corpus-file $C/cpython.bytes --train-bytes 40000000 --arms mingru,mingru_sparse_post,transformer
run scale_cpython_gate_s$S  --task code --corpus-file $C/cpython.bytes --train-bytes 40000000 --arms mingru_sparse_post --gate
run scale_enwik8_s$S        --task wiki --corpus-file $C/enwik8.bytes  --train-bytes 90000000 --arms mingru,mingru_sparse_post,transformer
run scale_enwik8_gate_s$S   --task wiki --corpus-file $C/enwik8.bytes  --train-bytes 90000000 --arms mingru_sparse_post --gate
