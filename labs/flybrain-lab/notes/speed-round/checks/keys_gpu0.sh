#!/usr/bin/env bash
# "What else repeats": three arms at the transformer-with-both-memories' step count (14,331), GPU 0, guarded:
#   A keys    match + byte table (8,16,32) + template keys (116,132) + word keys (202,203) + line key (301)
#   B orders  match + byte table at orders 4,8,12,16,32,64
#   C base    match + byte table (8,16,32), the round's best arm again with --rope-table (the equal-steps baseline)
# then the three scored on the dev windows. B and C first; A waits for the key features to exist.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-run4
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
S=/tmp/claude-1000/-games-ailab/f5e9946e-97cc-48fe-9eb3-0b019140c726/scratchpad/tables
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed
C=/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes
MEM=$SP/results/mem
OUT=$SP/results/tables
cd "$WT"
exec > >(tee -a "$OUT/keys_gpu0.log") 2>&1
echo "keys arms start $(date '+%F %T')"
while systemctl --user is-active --quiet tables-assoc-gpu0.service; do sleep 30; done
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
[ "$apps" = 0 ] || { echo "pre-flight failed: $apps compute processes on the card"; exit 1; }
cat "$MEM"/feat_orders/*.npy > /dev/null
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357 --val-fixed 128 --eval-every 2000 --steps 14331 --save"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32"
run() {  # run <name> <flags...>
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo; echo "=== $name $(date +%T)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TF "$@" 2>&1 | grep --line-buffered -v "^  "
}
run orders_tf_mt_r --mem-feat "$MEM/feat_orders" --mem-orders 4,8,12,16,32,64
run base_tf_mt_r --mem-feat "$MEM/feat" --mem-orders 8,16,32
while [ ! -f "$MEM/feat_keys/features.json" ]; do sleep 30; done     # the key builder (CPU, about 40 minutes) runs alongside the first two arms
cat "$MEM"/feat_keys/*.npy > /dev/null
run keys_tf_mt_r --mem-feat "$MEM/feat_keys" --mem-orders 8,16,32,116,132,202,203,301
echo; echo "=== scoring $(date +%T)"
"$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" \
  "k_keys_tf_mt=results/ckpt/keys_tf_mt_r_transformer.pt@$MEM/feat_keys" "k_orders_tf_mt=results/ckpt/orders_tf_mt_r_transformer.pt@$MEM/feat_orders" "k_base_tf_mt=results/ckpt/base_tf_mt_r_transformer.pt@$MEM/feat"
echo "keys arms done $(date '+%F %T')"
