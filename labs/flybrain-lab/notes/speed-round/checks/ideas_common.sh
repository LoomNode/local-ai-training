#!/usr/bin/env bash
# Shared by ideas_gpu1.sh and ideas_gpu0.sh: the arms of ideas 4 and 3 of 2026-10-07 (Kyle: "4 sounds the most interesting. i like 3 too"), each with
# the combined arm's inputs (feat_all, 18 orders, fused channel, match lookup 8/16/32) at the base arm's seconds (3,307 s): a 400-step
# calibration, then budget.py sets the steps. Reference arms: alleqf_tf_mt_r (equal seconds) and allf_tf_mt_r (equal steps).
#   paq_mt_r      the eleven lookup orders plus seven PAQ-style keys (indirect, sparse, record; feat_paq, paq_keys.py), counters dropped
#   top2_mt_r     the eleven lookup orders from top-two tables, both candidates with counts (feat_top2, top2_tables.py)
#   l4_mt_r       4 layers of 512 with the combined inputs     fewer weights, same table, more steps in the hour
#   l6_mt_r       6 layers of 512 with the combined inputs
#   l4b_mt_r      4 layers of 512 with the 2^28 two-way tables (feat_all28)
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-run4
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
S=/tmp/claude-1000/-games-ailab/f5e9946e-97cc-48fe-9eb3-0b019140c726/scratchpad/tables
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed
C=/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes
MEM=$SP/results/mem
OUT=$SP/results/tables
TARGET=3307
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357"
LOOK="4,8,12,16,32,64,116,132,202,203,301"
ALL18="$LOOK,1001,1002,1003,1004,1005,1006,1007"
TFI="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32 --fused-channels"
declare -A FLAGS=(
  [paq_mt_r]="$TFI --mem-feat $MEM/feat_paq --mem-orders $LOOK,401,402,501,502,503,601,602"
  [top2_mt_r]="$TFI --mem-feat $MEM/feat_top2 --mem-orders $LOOK,5004,5008,5012,5016,5032,5064,5116,5132,5202,5203,5301"
  [l4_mt_r]="$TFI --mem-feat $MEM/feat_all --mem-orders $ALL18 --n-layers 4"
  [l6_mt_r]="$TFI --mem-feat $MEM/feat_all --mem-orders $ALL18 --n-layers 6"
  [l4b_mt_r]="$TFI --mem-feat $MEM/feat_all28 --mem-orders $ALL18 --n-layers 4"
)
declare -A CKPT=([paq_mt_r]=transformer [top2_mt_r]=transformer [l4_mt_r]=transformer [l6_mt_r]=transformer [l4b_mt_r]=transformer)
declare -A FEAT=([paq_mt_r]=feat_paq [top2_mt_r]=feat_top2 [l4_mt_r]=feat_all [l6_mt_r]=feat_all [l4b_mt_r]=feat_all28)
arm() {  # arm <name>: claim it for this unit (a claim held by a unit that is no longer active is stale: a guard yield lost that run),
  name=$1   # calibrate (400 steps), budget to TARGET seconds, train; skipped when the JSON exists or another active unit holds the claim
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  if [ -f "results/claim_$name" ]; then
    owner=$(cat "results/claim_$name")
    [ "$owner" != "$UNIT" ] && systemctl --user is-active --quiet "$owner" && { echo "$name is running under $owner"; return 3; }
  fi
  echo "$UNIT" > "results/claim_$name"
  if [ ! -f "results/cal_$name.json" ]; then
    echo; echo "=== calibration $name $(date +%T)"
    "$PY" scripts/run_sparsela.py --run-name "cal_$name" $COMMON ${FLAGS[$name]} --steps 400 --val-fixed 8 --eval-every 100 2>&1 | grep --line-buffered -v "^  " | tail -n 4
  fi
  steps=$("$PY" budget.py "results/cal_$name.json" "$TARGET" 1.06) || { echo "calibration of $name failed"; rm -f "results/claim_$name"; return 2; }
  echo; echo "=== $name $(date +%T): $steps steps for $TARGET s"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON ${FLAGS[$name]} --steps "$steps" --save --val-fixed 128 --eval-every 2000 2>&1 | grep --line-buffered -v "^  "
}
sweep() {  # sweep <names...>: run every arm not done, waiting for one that another unit is running before moving on; ends with all done or failed
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
  "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_$short=results/ckpt/${name}_${CKPT[$name]}.pt@$MEM/${FEAT[$name]}" || echo "scoring $name failed"
}
preflight() {  # preflight: no compute process on this card (and under 2,500 MiB in use, for GPU 0's desktop)
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$apps" = 0 ] && [ "$used" -lt 2500 ] || { echo "pre-flight failed: $apps compute processes, $used MiB in use"; exit 1; }
}
