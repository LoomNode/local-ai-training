#!/usr/bin/env bash
# Speed round, GPU 0 night window (22:00-05:00, Kyle's display card): a second equal-seconds point at about two hours.
# The step budgets come from GPU 1's measured long-run paces in night queue 1 (hybrid 3,493.6 s / 7,783 steps = 0.4489 s/step,
# transformer 3,678.2 s / 12,668 steps = 0.2904): 16,000 hybrid steps are 7,182 GPU-1 seconds, which is 24,735 transformer steps.
# Seconds measured on this card are not used (it drives the desktop). The caller picks the card by UUID.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-heads
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
C=/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes
cd "$WT"
exec > >(tee -a results/speed_gpu0.log) 2>&1
echo "GPU 0 queue start $(date '+%F %T')"
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357"
TF="--arms transformer --rope"
HYB="--arms mingru_sparse_post --gate --mem-every 2 --local-attn 512 --key-convs 2,4"
FAST="--tf32 --fused --compile"
run() {  # run <run name> <steps> <flags...>
  name=$1; steps=$2; shift 2
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo; echo "=== $name $(date +%T): $steps steps"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON --steps "$steps" --save --val-fixed 128 --eval-every 1000 "$@" 2>&1 | grep --line-buffered -v "^  "
  ck=$(ls results/ckpt/"$name"_*.pt 2>/dev/null | head -n 1)
  if [ -n "$ck" ]; then
    echo "--- scoring $ck on dev $(date +%T)"
    "$PY" scripts/cross_eval.py --ckpts "$ck" --corpus-file "$C" --offset 202117357 --fixed 1024 --lens 2048 --device cuda:0 --run-name "speed_score_dev_${name}" 2>&1 | tail -n 6 || echo "scoring of $name failed"
  fi
}
run speed_tf_c_2h_s0 24735 $TF $FAST
run speed_hyb_c_2h_s0 16000 $HYB $FAST --scan doubling --head-amp --sparse-claws
echo; echo "GPU 0 queue done $(date '+%F %T')"
