#!/usr/bin/env bash
# Run 39b: slots vs window, trimmed to the informative cells after the width ladder showed <=2 writes/slot is a wash.
# tuned d=256 CPython setting (8000 steps, tokens/step 16384 held equal), head arm only. Existing cells: T=2048 nk=4096 (1.084), T=4096 nk=4096 (1.042).
#   T=1024 nk=1024  ( 4 writes/slot)   T=2048 nk=1024  ( 8)   T=4096 nk=1024  (16)   T=4096 nk=16384 (0.5, headroom at the longest window)
# usage: scripts/run_slot_window.sh <device>
set -u
DEV=${1:-cuda:1}
C=corpora
cd "$(dirname "$0")/.."
COMMON="--task code --corpus-file $C/cpython.bytes --train-bytes 40000000 --normalize --key-conv 2 --top-k 4 --lr 0.001 --decay 1.0 --d-model 256 --d-ff 512 --n-layers 4 --steps 8000 --val-bytes 2000000 --seed 0 --device $DEV --warmup 500 --cosine --wd 0.1 --chunk 256 --arms mingru_sparse_post --gate"
run() { NAME=$1; shift; [ -f results/$NAME.json ] && { echo "skip $NAME"; return; }; echo "=== $NAME"; uv run python scripts/run_sparsela.py --run-name "$NAME" $COMMON "$@" 2>&1 | grep -v "^  "; }
run slotwin_cpython_T2048_nk1024_s0  --batch 8  --seq-len 2048 --n-key 1024
run slotwin_cpython_T4096_nk1024_s0  --batch 4  --seq-len 4096 --n-key 1024
run slotwin_cpython_T4096_nk16384_s0 --batch 4  --seq-len 4096 --n-key 16384
run slotwin_cpython_T1024_nk1024_s0  --batch 16 --seq-len 1024 --n-key 1024
