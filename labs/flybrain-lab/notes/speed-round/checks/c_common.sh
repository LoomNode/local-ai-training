#!/usr/bin/env bash
# Shared by calsession3.sh and c_gpu1.sh: bet C (the stack only where a cheap gate is unsure, --skip-frac; spike/speed-skip 3d9687f, run
# worktree speed-run5) and bet E (PAQ bit-probability tables with a 16-bit slot check, feat_paqbits2; the first build, feat_paqbits, was swamped by collisions) at the base's calibrated cost. Reference: l4b_mt_r 0.652.

#   c30_mt_r    4 layers, 2^28 tables, the stack on the 70 % of positions the gate is unsure about (--skip-frac 0.3)
#   c50_mt_r    the same with half the positions skipped
#   c50l8_mt_r  8 layers with half the positions skipped: the saved compute spent on depth (about a 4-layer step on every position)
#   e_mt_r      4 layers, the 18 inputs plus the PAQ bit-probability tables' two candidates for orders 4, 8, 116, 202 (feat_all28_paq, 26 inputs):
#               alone the tagged bit tables cover a third of the count table's positions (stats_feat_paqbits2.json), so they are tested as an addition
# Budgets: steps = 14,331 x base pace / arm pace from calibration session 3 (cal4_*.json, calsession3.sh), never seconds.
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-run5
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
  [c30_mt_r]="$TFI --mem-feat $MEM/feat_all28 --mem-orders $ALL18 --n-layers 4 --skip-frac 0.3"
  [c50_mt_r]="$TFI --mem-feat $MEM/feat_all28 --mem-orders $ALL18 --n-layers 4 --skip-frac 0.5"
  [c50l8_mt_r]="$TFI --mem-feat $MEM/feat_all28 --mem-orders $ALL18 --n-layers 8 --skip-frac 0.5"
  [e_mt_r]="$TFI --mem-feat $MEM/feat_all28_paq --mem-orders $ALL18,6004,6008,6116,6202,7004,7008,7116,7202 --n-layers 4"
)
declare -A FEAT=([c30_mt_r]=feat_all28 [c50_mt_r]=feat_all28 [c50l8_mt_r]=feat_all28 [e_mt_r]=feat_all28_paq)
arms() {  # the arm list, decided when the queue reaches it: the E arm joins once its features exist
  local a="c30_mt_r c50_mt_r c50l8_mt_r"; [ -f "$MEM/feat_all28_paq/features.json" ] && a="$a e_mt_r"; echo "$a"
}
arm() {  # arm <name>: claim it for this unit (a claim held by a unit that is no longer active is stale); steps by the session's ratio
  name=$1
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  if [ -f "results/claim_$name" ]; then
    owner=$(cat "results/claim_$name")
    [ "$owner" != "$UNIT" ] && systemctl --user is-active --quiet "$owner" && { echo "$name is running under $owner"; return 3; }
  fi
  echo "$UNIT" > "results/claim_$name"
  if [ ! -f "results/cal4_$name.json" ]; then
    echo; echo "=== calibration $name $(date +%T) (NOT from the quiet session)"
    "$PY" scripts/run_sparsela.py --run-name "cal4_$name" $COMMON ${FLAGS[$name]} --steps 400 --val-fixed 8 --eval-every 100 2>&1 | grep --line-buffered -v "^  " | tail -n 4
  fi
  steps=$("$PY" "$S/budget_ratio.py" "results/cal4_$name.json" results/cal4_base.json 14331 "${MAXSTEPS:-60000}") || { echo "calibration of $name failed"; rm -f "results/claim_$name"; return 2; }
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
  SCORE_CODE="$WT" "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_$short=results/ckpt/${name}_transformer.pt@$MEM/${FEAT[$name]}" || echo "scoring $name failed"
}
preflight() {  # preflight: no compute process on this card (and under 2,500 MiB in use, for GPU 0's desktop)
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$apps" = 0 ] && [ "$used" -lt 2500 ] || { echo "pre-flight failed: $apps compute processes, $used MiB in use"; exit 1; }
}
