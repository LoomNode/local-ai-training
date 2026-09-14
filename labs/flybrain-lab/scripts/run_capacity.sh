#!/usr/bin/env bash
# Run 38: does the memory need to grow with the model? Same 20M kernel setting as run_20m.sh (d=512, 8 layers, seq 2048, 16k steps),
# two arms that scale the memory the way the transformer's retrieval scales:
#   slots16k : one head, 16384 slots (4x)            -> slots scale with the window
#   depth4   : four heads of 4096, after layers 2/4/6/8 (--mem-every 2) -> heads scale with depth
# Baselines already in results/: m20_linux_gate_s0 (1 head, 4096: 0.91), m20_linux_s0 (mingru / transformer 0.78).
# usage: scripts/run_capacity.sh <device> <slots16k|depth4> <seed> [steps]
set -u
DEV=${1:-cuda:0}; ARM=${2:-depth4}; S=${3:-0}; STEPS=${4:-16000}
C=corpora
cd "$(dirname "$0")/.."
COMMON="--task code --corpus-file $C/linux_full.bytes --train-bytes 200000000 --normalize --key-conv 2 --top-k 4 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --steps $STEPS --val-bytes 2000000 --save --seed $S --device $DEV --warmup 1000 --cosine --wd 0.1 --chunk 256 --arms mingru_sparse_post --gate"
case $ARM in
  slots16k) EXTRA="--n-key 16384" ;;
  depth4)   EXTRA="--n-key 4096 --mem-every 2" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
NAME=m20_linux_${ARM}_s$S
[ -f results/$NAME.json ] && { echo "skip $NAME"; exit 0; }
echo "=== $NAME"
uv run python scripts/run_sparsela.py --run-name "$NAME" $COMMON $EXTRA 2>&1 | grep -v "^  "
