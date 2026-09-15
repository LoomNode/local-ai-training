#!/usr/bin/env bash
# Run 38: does the memory need to grow with the model? Same 20M kernel setting as run_20m.sh (d=512, 8 layers, seq 2048, 16k steps),
# two arms that scale the memory the way the transformer's retrieval scales:
#   slots16k : one head, 16384 slots (4x)            -> slots scale with the window
#   depth4   : four heads of 4096, after layers 2/4/6/8 (--mem-every 2) -> heads scale with depth
#   depth4x16k : both (the 2x2 corner)
# Baselines already in results/: m20_linux_gate_s0 (1 head, 4096: 0.91), m20_linux_s0 (mingru / transformer 0.78).
#   depth2   : two heads of 4096 (after layer 4 and the top; --mem-every 4), the missing point of the 1/2/4/8 slope
#   depth8   : eight heads of 4096, one after every layer (--mem-every 1); 17.15M = the transformer's 17.09M
# usage: scripts/run_capacity.sh <device> <slots16k|depth4|depth4x16k|depth8> <seed> [steps] [linux|enwik8]
set -u
DEV=${1:-cuda:0}; ARM=${2:-depth4}; S=${3:-0}; STEPS=${4:-16000}; CORPUS=${5:-linux}
C=corpora
cd "$(dirname "$0")/.."
if [ "$CORPUS" = linux ]; then TASK=code; FILE=$C/linux_full.bytes; TB=200000000; else TASK=wiki; FILE=$C/enwik8.bytes; TB=90000000; fi
COMMON="--task $TASK --corpus-file $FILE --train-bytes $TB --normalize --key-conv 2 --top-k 4 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --steps $STEPS --val-bytes 2000000 --save --seed $S --device $DEV --warmup 1000 --cosine --wd 0.1 --chunk 256 --arms mingru_sparse_post --gate"
case $ARM in
  slots16k) EXTRA="--n-key 16384" ;;
  depth4)   EXTRA="--n-key 4096 --mem-every 2" ;;
  depth4x16k) EXTRA="--n-key 16384 --mem-every 2" ;;
  depth8)   EXTRA="--n-key 4096 --mem-every 1" ;;
  depth2)   EXTRA="--n-key 4096 --mem-every 4" ;;
  depth4x64k) EXTRA="--n-key 65536 --mem-every 2 --chunk 512" ;;  # slots for a 64k window at <=4 writes/slot; chunk 512 halves the per-chunk state copies
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
NAME=m20_${CORPUS}_${ARM}_s$S
[ -f results/$NAME.json ] && { echo "skip $NAME"; exit 0; }
echo "=== $NAME"
uv run python scripts/run_sparsela.py --run-name "$NAME" $COMMON $EXTRA 2>&1 | grep -v "^  "
