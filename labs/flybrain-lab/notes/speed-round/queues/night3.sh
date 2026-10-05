#!/usr/bin/env bash
# Speed round, night queue 3 on GPU 1 (spike/speed-int at 613575d): test 3, local learning, 3,600 training seconds each (notes/SPEED.md).
# Each arm: a 400-step calibration (pace and first-step overhead), then one run whose step count fills 3,600 s, then the
# checkpoint scored on the 1,024 dev windows in fp32. A name whose JSON exists is skipped. A failed arm does not stop the queue.
# usage: night3.sh [units to wait for, in order ...]
# The caller picks the card by UUID: CUDA_VISIBLE_DEVICES hides the other card, so the visible one is cuda:0.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-run
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
C=/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes
cd "$WT"
for u in "$@"; do
  while systemctl --user is-active --quiet "$u"; do sleep 30; done
  sleep 20
done
exec > >(tee -a results/speed_night3.log) 2>&1
echo "night queue 3 start $(date '+%F %T'); load $(cut -d' ' -f1-3 /proc/loadavg)"
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357"
TF="--arms transformer --rope"
ZH="--arms mingru --local-attn 512"
HYB="--arms mingru_sparse_post --gate --mem-every 2 --local-attn 512 --key-convs 2,4"
HYB2="--arms mingru_sparse_post --gate --mem-every 4 --local-attn 512 --key-convs 2,4"
FAST="--tf32 --fused --compile"
EAGER="--tf32 --fused"
preflight() {
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  lim=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=power.limit --format=csv,noheader | tr -d ' ')
  if [ "$apps" != 0 ] || [ "$lim" != "150.00W" ]; then echo "pre-flight failed $(date +%T): $apps compute processes on the card, power limit $lim"; return 1; fi
}
arm() {  # arm <name> <flags...>
  name=$1; shift
  run=speed_${name}_3600_s0; cal=speedcal_${name}
  [ -f "results/$run.json" ] && { echo "skip $run"; return 0; }
  preflight || return 1
  if [ ! -f "results/$cal.json" ]; then
    echo; echo "=== $cal $(date +%T)"
    "$PY" scripts/run_sparsela.py --run-name "$cal" $COMMON --steps 400 --val-fixed 8 --eval-every 100 "$@" 2>&1 | grep --line-buffered -v "^  " | tail -n 8
  fi
  steps=$("$PY" budget.py "results/$cal.json" 3600 1.06) || { echo "calibration of $name failed"; return 2; }
  echo; echo "=== $run $(date +%T): $steps steps; load $(cut -d' ' -f1-3 /proc/loadavg)"
  "$PY" scripts/run_sparsela.py --run-name "$run" $COMMON --steps "$steps" --save --val-fixed 128 --eval-every 1000 "$@" 2>&1 | grep --line-buffered -v "^  "
  ck=$(ls results/ckpt/"$run"_*.pt 2>/dev/null | head -n 1)
  if [ -n "$ck" ]; then
    echo "--- scoring $ck on dev $(date +%T)"
    "$PY" scripts/cross_eval.py --ckpts "$ck" --corpus-file "$C" --offset 202117357 --fixed 1024 --lens 2048 --device cuda:0 --run-name "speed_score_dev_${name}" 2>&1 | tail -n 6 || echo "scoring of $name failed"
  else
    echo "no checkpoint for $run"
  fi
  [ -f "results/$run.json" ] || { echo "$run did not finish"; return 3; }
}
SP="$FAST --scan doubling --head-amp --sparse-claws"   # the hyb_c switches of night queue 1, so each arm differs from it by one flag
SPE="$EAGER --head-amp"                                # eager fallback when a compiled arm fails
arm t3_lb2_c $HYB $SP --local-blocks 2 || arm t3_lb2_e $HYB $SPE --local-blocks 2
arm t3_fzodd_c $HYB $SP --freeze-layers 1,3,5,7 || arm t3_fzodd_e $HYB $SPE --freeze-layers 1,3,5,7
arm t3_mem_c $HYB $SP --train-only readout,embed,heads || arm t3_mem_e $HYB $SPE --train-only readout,embed,heads
echo; echo "night queue 3 done $(date '+%F %T'); load $(cut -d' ' -f1-3 /proc/loadavg)"
