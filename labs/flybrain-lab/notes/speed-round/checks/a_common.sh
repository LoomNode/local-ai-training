#!/usr/bin/env bash
# Shared by calsession2.sh and a_gpu1.sh: radical bet A (Kyle, 2026-10-08: "let's test all the radical bets next") and the cheap
# form of bet B. How small can the network go, with the combined inputs through the fused channel and the 2^28 two-way tables
# (the 4-layer arm l4b_mt_r, 0.652, is the reference):
#   l2b_mt_r   2 layers of 512
#   l1b_mt_r   1 layer
#   l0b_mt_r   0 layers: byte embedding + the channels, a final norm and the output head; a mixer of the table's answers
#   l4c_mt_r   4 layers with 2^29 two-way tables (feat_all29): the table-size step at 4 layers
#   l4n_mt_r   4 layers, 2^28 tables, plus learned hashed n-gram slots (--ngram-emb, row 2c's channel: 0.019 alone on the backbone)
#              on top of the count table, 2^21 rows x 32 per order (dense updates: --ngram-sparse cannot be compiled, "Cannot access storage
#              of SparseTensorImpl", which is what killed its calibration twice on 2026-10-08): does a learned slot add anything a counted one lacks?
# Budgets: steps = 14,331 x base pace / arm pace from calibration session 2 (cal3_*.json, calsession2.sh), never seconds. a2_gpu1.sh reruns
# the three arms the first pass lost (l1b, l0b: over the 60,000-step cap; l4n: the sparse flag) with CAL (the calibration prefix) and
# MAXSTEPS set per arm.
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-run4
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed
C=/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes
MEM=$SP/results/mem
OUT=$SP/results/tables
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357"
LOOK="4,8,12,16,32,64,116,132,202,203,301"
ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
TFI="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32 --fused-channels"
BASE="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32 --mem-feat $MEM/feat --mem-orders 8,16,32"
declare -A FLAGS=(
  [l2b_mt_r]="$TFI --mem-feat $MEM/feat_all28 --mem-orders $ALL18 --n-layers 2"
  [l1b_mt_r]="$TFI --mem-feat $MEM/feat_all28 --mem-orders $ALL18 --n-layers 1"
  [l0b_mt_r]="$TFI --mem-feat $MEM/feat_all28 --mem-orders $ALL18 --n-layers 0"
  [l4c_mt_r]="$TFI --mem-feat $MEM/feat_all29 --mem-orders $ALL18 --n-layers 4"
  [l4n_mt_r]="$TFI --mem-feat $MEM/feat_all28 --mem-orders $ALL18 --n-layers 4 --ngram-emb 4,8,12,16,32 --ngram-rows 2097152 --ngram-dim 32"
)
declare -A FEAT=([l2b_mt_r]=feat_all28 [l1b_mt_r]=feat_all28 [l0b_mt_r]=feat_all28 [l4c_mt_r]=feat_all29 [l4n_mt_r]=feat_all28)
ARMS="l2b_mt_r l1b_mt_r l0b_mt_r l4c_mt_r l4n_mt_r"
arm() {  # arm <name>: claim it for this unit (a claim held by a unit that is no longer active is stale); steps by the session's ratio
  name=$1
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  if [ -f "results/claim_$name" ]; then
    owner=$(cat "results/claim_$name")
    [ "$owner" != "$UNIT" ] && systemctl --user is-active --quiet "$owner" && { echo "$name is running under $owner"; return 3; }
  fi
  echo "$UNIT" > "results/claim_$name"
  cal=${CAL:-cal3}
  if [ ! -f "results/${cal}_$name.json" ]; then
    echo; echo "=== calibration $name $(date +%T) (NOT from the quiet session)"
    "$PY" scripts/run_sparsela.py --run-name "${cal}_$name" $COMMON ${FLAGS[$name]} --steps 400 --val-fixed 8 --eval-every 100 2>&1 | grep --line-buffered -v "^  " | tail -n 4
  fi
  steps=$("$PY" "$S/budget_ratio.py" "results/${cal}_$name.json" "results/${cal}_base.json" 14331 "${MAXSTEPS:-60000}") || { echo "calibration of $name failed"; rm -f "results/claim_$name"; return 2; }
  echo; echo "=== $name $(date +%T): $steps steps at the base's calibrated cost"
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
  "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_$short=results/ckpt/${name}_transformer.pt@$MEM/${FEAT[$name]}" || echo "scoring $name failed"
}
preflight() {  # preflight: no compute process on this card (and under 2,500 MiB in use, for GPU 0's desktop)
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$apps" = 0 ] && [ "$used" -lt 2500 ] || { echo "pre-flight failed: $apps compute processes, $used MiB in use"; exit 1; }
}
