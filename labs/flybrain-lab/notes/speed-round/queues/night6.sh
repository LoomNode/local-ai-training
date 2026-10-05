#!/usr/bin/env bash
# Speed round, night queue 6 on GPU 1 (spike/speed-int at 782e0bc): the bar again under quiet conditions.
# The plain transformer of night queue 1 (tf_c, 12,668 steps) ran while a game used the other card; the same model's pace was
# 13 % better by 04:00 (t2_tf_match_c: 14,320 steps), so the later arms got more steps in their hour than the bar did.
# tf_c2 is the same arm with the step count a quiet hour buys. zh_c2 runs only when the backbone's quiet pace buys over 5 %
# more steps than zh_c got (13,849). A pace check of the hybrid is taken for test 3's reference (hyb_c: 7,783 steps).
# Calibrations are the pace checks night queue 5 leaves behind (speedcal2_*), when they exist.
# usage: night6.sh [units to wait for, in order ...]
# The caller picks the card by UUID: CUDA_VISIBLE_DEVICES hides the other card, so the visible one is cuda:0.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-run3
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
C=/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes
cd "$WT"
for u in "$@"; do
  while systemctl --user is-active --quiet "$u"; do sleep 30; done
  sleep 20
done
exec > >(tee -a results/speed_night6.log) 2>&1
echo "night queue 6 start $(date '+%F %T'); load $(cut -d' ' -f1-3 /proc/loadavg)"
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357"
TF="--arms transformer --rope"
ZH="--arms mingru --local-attn 512"
HYB="--arms mingru_sparse_post --gate --mem-every 2 --local-attn 512 --key-convs 2,4"
FAST="--tf32 --fused --compile"
TARGET=3600; SCORE_EXTRA=""
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
  steps=$("$PY" budget.py "results/$cal.json" "$TARGET" 1.06) || { echo "calibration of $name failed"; return 2; }
  echo; echo "=== $run $(date +%T): $steps steps; load $(cut -d' ' -f1-3 /proc/loadavg)"
  "$PY" scripts/run_sparsela.py --run-name "$run" $COMMON --steps "$steps" --save --val-fixed 128 --eval-every 1000 "$@" 2>&1 | grep --line-buffered -v "^  "
  ck=$(ls results/ckpt/"$run"_*.pt 2>/dev/null | head -n 1)
  if [ -n "$ck" ]; then
    echo "--- scoring $ck on dev $(date +%T)"
    "$PY" scripts/cross_eval.py --ckpts "$ck" --corpus-file "$C" --offset 202117357 --fixed 1024 --lens 2048 --device cuda:0 --run-name "speed_score_dev_${name}" $SCORE_EXTRA 2>&1 | tail -n 6 || echo "scoring of $name failed"
  else
    echo "no checkpoint for $run"
  fi
  [ -f "results/$run.json" ] || { echo "$run did not finish"; return 3; }
}
# the pace checks of night queue 5 double as these arms' calibrations
[ -f results/speedcal2_tf_c.json ] && [ ! -f results/speedcal_tf_c2.json ] && cp results/speedcal2_tf_c.json results/speedcal_tf_c2.json
[ -f results/speedcal2_zh_c.json ] && [ ! -f results/speedcal_zh_c2.json ] && cp results/speedcal2_zh_c.json results/speedcal_zh_c2.json
arm tf_c2 $TF $FAST
zsteps=0
[ -f results/speedcal_zh_c2.json ] && zsteps=$("$PY" budget.py results/speedcal_zh_c2.json 3600 1.06 2>/dev/null || echo 0)
if [ "$zsteps" -gt 14541 ]; then   # 13,849 x 1.05
  arm zh_c2 $ZH $FAST --scan doubling
else
  echo; echo "zh_c2 not run: a quiet hour buys $zsteps steps against the 13,849 zh_c got (rerun only above 14,541)"
fi
if preflight; then
  echo; echo "=== speedcal2_hyb_c $(date +%T)"
  "$PY" scripts/run_sparsela.py --run-name speedcal2_hyb_c $COMMON --steps 400 --val-fixed 8 --eval-every 100 $HYB $FAST --scan doubling --head-amp --sparse-claws 2>&1 | grep --line-buffered -v "^  " | tail -n 3
  "$PY" budget.py results/speedcal2_hyb_c.json 3600 1.06 > /dev/null
fi
echo; echo "night queue 6 done $(date '+%F %T'); load $(cut -d' ' -f1-3 /proc/loadavg)"
