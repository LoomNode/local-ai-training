#!/usr/bin/env bash
# Second half of gpu1_queue.sh. The first queue finished alleqf_tf_mt_r at 07:24 and then waited for feat_all28, which was never
# being built: the watcher that was to start assoc_all.py waited on `pgrep -f "key_tables.py --keep-keys"`, which matched the
# watcher's own shell, so GPU 1 sat idle from 07:24 to 08:45. Now: the fused equal-steps rerun (tables-fused-gpu1b) goes first
# while assoc_all.py builds feat_all28 on the CPU; this queue waits for both, then runs all28f_tf_mt_r and scores both arms.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-run4
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed
C=/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes
MEM=$SP/results/mem
OUT=$SP/results/tables
cd "$WT"
exec > >(tee -a "$OUT/gpu1_queue.log") 2>&1
echo "gpu1 queue 2 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
while systemctl --user is-active --quiet tables-fused-gpu1b.service; do sleep 30; done
while [ ! -f "$MEM/feat_all28/features.json" ]; do sleep 30; done
sleep 10
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
[ "$apps" = 0 ] || { echo "pre-flight failed: $apps compute processes on the card"; exit 1; }
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357 --val-fixed 128 --eval-every 2000 --save"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32 --fused-channels"
ORD="4,8,12,16,32,64,116,132,202,203,301,1001,1002,1003,1004,1005,1006,1007"
run() {  # run <name> <flags...>
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo; echo "=== $name $(date +%T)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TF "$@" 2>&1 | grep --line-buffered -v "^  "
}
cat "$MEM"/feat_all28/*.npy > /dev/null
run all28f_tf_mt_r --steps 14331 --mem-feat "$MEM/feat_all28" --mem-orders "$ORD"
echo; echo "=== scoring $(date +%T)"
"$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" \
  "k_alleqf_tf_mt=results/ckpt/alleqf_tf_mt_r_transformer.pt@$MEM/feat_all" "k_all28f_tf_mt=results/ckpt/all28f_tf_mt_r_transformer.pt@$MEM/feat_all28"
echo "gpu1 queue 2 done $(date '+%F %T')"
