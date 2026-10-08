#!/usr/bin/env bash
# Shared by tier1_gpu1.sh and tier1_gpu0.sh (Kyle, 2026-10-07: "queue up tier 1"): the backbone arms that need flags only, each with
# the combined arm's inputs (feat_all, 18 orders, fused channel, match lookup 8/16/32) at the base arm's seconds (3,307 s): a 400-step
# calibration, then budget.py sets the steps. Reference arms: alleqf_tf_mt_r (equal seconds) and allf_tf_mt_r (equal steps).
#   s12_tf_mt_r   transformer 12 x 384 (d_ff 768)        deeper and narrower
#   s6_tf_mt_r    transformer 6 x 768 (d_ff 1536)        shallower and wider
#   w4k_tf_mt_r   transformer 8 x 512, window 4096       longer context, fewer steps
#   hyb_mt_r      Kyle's hybrid (minGRU + sparse-key heads + one local-attention layer), never run with the inputs
#   gla_mt_r      the lab's gated-linear-attention backbone (Run 28), a recurrent backbone with flags only
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-run4
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed
C=/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes
MEM=$SP/results/mem
OUT=$SP/results/tables
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357"
INPUTS="--match-feat 8,16,32 --fused-channels --mem-feat $MEM/feat_all --mem-orders 4,8,12,16,32,64,116,132,202,203,301,1001,1002,1003,1004,1005,1006,1007"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table $INPUTS"
HYB="--arms mingru_sparse_post --gate --mem-every 2 --local-attn 512 --key-convs 2,4 --tf32 --fused --compile --scan doubling --head-amp --sparse-claws $INPUTS"
GLA="--arms gla --tf32 --fused --compile $INPUTS"
declare -A FLAGS=(
  [s12_tf_mt_r]="$TF --d-model 384 --n-layers 12 --d-ff 768"
  [s6_tf_mt_r]="$TF --d-model 768 --n-layers 6 --d-ff 1536"
  [w4k_tf_mt_r]="$TF --seq-len 4096"
  [hyb_mt_r]="$HYB"
  [gla_mt_r]="$GLA"
)
declare -A CKPT=([s12_tf_mt_r]=transformer [s6_tf_mt_r]=transformer [w4k_tf_mt_r]=transformer [hyb_mt_r]=mingru_sparse_post [gla_mt_r]=gla)
arm() {  # arm <name>: claim it for this unit (a claim held by a unit that is no longer active is stale: a guard yield lost that run);
  name=$1   # steps = 14,331 x base pace / arm pace from the quiet calibration session (calsession.sh, cal2_*.json; Kyle, 2026-10-07:
            # "stop using literal time passage as a metric"); an arm the session missed is calibrated here and marked as such in the log
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  if [ -f "results/claim_$name" ]; then
    owner=$(cat "results/claim_$name")
    [ "$owner" != "$UNIT" ] && systemctl --user is-active --quiet "$owner" && { echo "$name is running under $owner"; return 3; }
  fi
  echo "$UNIT" > "results/claim_$name"
  if [ ! -f "results/cal2_$name.json" ]; then
    echo; echo "=== calibration $name $(date +%T) (NOT from the quiet session)"
    "$PY" scripts/run_sparsela.py --run-name "cal2_$name" $COMMON ${FLAGS[$name]} --steps 400 --val-fixed 8 --eval-every 100 2>&1 | grep --line-buffered -v "^  " | tail -n 4
  fi
  steps=$("$PY" "$S/budget_ratio.py" "results/cal2_$name.json" results/cal2_base.json 14331) || { echo "calibration of $name failed"; rm -f "results/claim_$name"; return 2; }
  echo; echo "=== $name $(date +%T): $steps steps at the base's calibrated cost"
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
  "$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "k_$short=results/ckpt/${name}_${CKPT[$name]}.pt@$MEM/feat_all" || echo "scoring $name failed"
}
preflight() {  # preflight: no compute process on this card (and under 2,500 MiB in use, for GPU 0's desktop)
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "$apps" = 0 ] && [ "$used" -lt 2500 ] || { echo "pre-flight failed: $apps compute processes, $used MiB in use"; exit 1; }
}
