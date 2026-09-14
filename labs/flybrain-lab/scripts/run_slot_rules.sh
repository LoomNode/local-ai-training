#!/usr/bin/env bash
# Run 39: do slots need to GROW, and with what? Two rules, each a slope rather than a point. Head arm only (the backbone
# does not depend on slot count; its numbers are in the Run 29 ladder and the tuned/ctx4096 runs).
#   width  : Run 29 ladder setting (T=512, 4 layers, 3000 steps), slots = 4*d and 16*d vs the existing fixed 1024
#   window : tuned d=256 CPython setting (8000 steps, tokens/step = 16384 held equal), windows 1024/2048/4096 x slots 1024/4096/16384
# usage: scripts/run_slot_rules.sh <device> <width|window>
set -u
DEV=${1:-cuda:1}; WHICH=${2:-width}
C=corpora
cd "$(dirname "$0")/.."
run() { NAME=$1; shift; [ -f results/$NAME.json ] && { echo "skip $NAME"; return; }; echo "=== $NAME"; uv run python scripts/run_sparsela.py --run-name "$NAME" "$@" 2>&1 | grep -v "^  "; }
if [ "$WHICH" = width ]; then
  declare -A FILE=([code]=$C/code.bytes [cpp]=$C/cpp.bytes)
  for MULT in 4 16; do for D in 64 128 256 384; do NK=$((MULT*D)); [ $NK -eq 1024 ] && continue  # 1024 is the Run 29 ladder itself
    for CORPUS in code cpp; do for S in 0 1; do
      run slotwidth_${CORPUS}_d${D}_nk${NK}_s$S --task code --corpus-file "${FILE[$CORPUS]}" --device "$DEV" \
        --arms mingru_sparse_post --normalize --key-conv 2 --top-k 4 --n-key $NK --lr 0.001 --decay 1.0 \
        --d-model $D --d-ff $((2*D)) --n-layers 4 --steps 3000 --batch 16 --seq-len 512 --train-bytes 9000000 --val-bytes 1000000 --seed $S
    done; done
  done; done
else
  COMMON="--task code --corpus-file $C/cpython.bytes --train-bytes 40000000 --normalize --key-conv 2 --top-k 4 --lr 0.001 --decay 1.0 --d-model 256 --d-ff 512 --n-layers 4 --steps 8000 --val-bytes 2000000 --seed 0 --device $DEV --warmup 500 --cosine --wd 0.1 --chunk 256"
  # existing cells: T=2048 nk=4096 (tuned_cpython_gate_s0 1.084), T=4096 nk=4096 (ctx4096_cpython_gate_s0 1.042); backbone at T=1024 is new
  run slotwin_cpython_T1024_mingru_s0 $COMMON --batch 16 --seq-len 1024 --arms mingru
  for T in 1024 2048 4096; do B=$((16384/T)); for NK in 1024 4096 16384; do
    [ $T -ne 1024 ] && [ $NK -eq 4096 ] && continue
    run slotwin_cpython_T${T}_nk${NK}_s0 $COMMON --batch $B --seq-len $T --n-key $NK --arms mingru_sparse_post --gate
  done; done
fi
