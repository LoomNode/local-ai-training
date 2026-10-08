#!/usr/bin/env bash
# Shared by scale_gpu1.sh and scale_gpu0.sh: the budget-scaling lock test. At the base's cost (one hour) fewer layers won
# (l4b 0.652, l6 0.657, l4 0.659 against 8 layers at 0.670 / 0.657 with the 2^28 tables), which can be the budget talking: an
# 8-layer model gets 13,560 steps in the hour, a 4-layer one 25,562. So the depth curve again at twice the cost, all three
# with the 2^28 two-way tables (feat_all28), steps = 2 x the quiet session's calibrated budget (calsession_2026-10-07.json):
#   l8x2_mt_r   8 layers of 512   27,120 steps   (2 x allf's 13,560: same model and inputs as all28f)
#   l6x2_mt_r   6 layers of 512   35,344 steps   (2 x l6's 17,672)
#   l4x2_mt_r   4 layers of 512   51,124 steps   (2 x l4b's 25,562)
# If the gap closes or flips with budget, depth is a budget question and the big train picks by its own budget; if 4 layers
# still win, the lock takes the shallow model. No new calibration: the paces are the session's.
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-run4
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
S=/tmp/claude-1000/-games-ailab/f5e9946e-97cc-48fe-9eb3-0b019140c726/scratchpad/tables
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed
C=/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes
MEM=$SP/results/mem
OUT=$SP/results/tables
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357"
LOOK="4,8,12,16,32,64,116,132,202,203,301"
ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
TFI="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32 --fused-channels --mem-feat $MEM/feat_all28 --mem-orders $ALL18"
declare -A FLAGS=(
  [l8x2_mt_r]="$TFI --n-layers 8"
  [l6x2_mt_r]="$TFI --n-layers 6"
  [l4x2_mt_r]="$TFI --n-layers 4"
)
declare -A STEPS=([l8x2_mt_r]=27120 [l6x2_mt_r]=35344 [l4x2_mt_r]=51124)
arm() {  # arm <name>: claim it for this unit (a claim held by a unit that is no longer active is stale: a guard yield lost that run)
  name=$1
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  if [ -f "results/claim_$name" ]; then
    owner=$(cat "results/claim_$name")
    [ "$owner" != "$UNIT" ] && systemctl --user is-active --quiet "$owner" && { echo "$name is running under $owner"; return 3; }
  fi
  echo "$UNIT" > "results/claim_$name"
  steps=${STEPS[$name]}
  echo; echo "=== $name $(date +%T): $steps steps = twice the base's calibrated cost"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON ${FLAGS[$name]} --steps "$steps" --save --val-fixed 128 --eval-every 2000 2>&1 | grep --line-buffered -v "^  "
}
sweep() {  # sweep <names...>: run every arm not done, waiting for one that another unit is running before moving on
  for a in "$@"; do
    while :; do
      arm "$a"; rc=$?
      [ "$rc" = 3 ] || break
      sleep 60
    done
  done
}
score() {  # score <name>: the dev windows, one call per arm so a missing checkpoint stops nothing else
  name=$1; short=${name%_r}
  [ -f "$OUT/k_$short.json" ] && return 0
  "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_$short=results/ckpt/${name}_transformer.pt@$MEM/feat_all28" || echo "scoring $name failed"
}
preflight() {  # preflight: no compute process on this card (and under 2,500 MiB in use, for GPU 0's desktop)
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$apps" = 0 ] && [ "$used" -lt 2500 ] || { echo "pre-flight failed: $apps compute processes, $used MiB in use"; exit 1; }
}
