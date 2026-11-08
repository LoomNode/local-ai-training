#!/usr/bin/env bash
# "What else repeats", item 5: the counters arm (match + byte table 8,16,32 + the seven per-position counters as pseudo-orders
# 1001-1007), at the same 14,331 steps as the keys/orders/base arms, GPU 0. Waits for the keys queue (tables-keys2-gpu0) to end and
# for the counter features to exist, then starts its own guard (a guard started earlier would see the keys queue's process as foreign).
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
exec > >(tee -a "$OUT/counters_gpu0.log") 2>&1
echo "counters arm start $(date '+%F %T')"
while [ ! -f "$MEM/feat_counters/features.json" ]; do sleep 30; done
while systemctl --user is-active --quiet tables-keys2-gpu0.service; do sleep 30; done
sleep 30
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
[ "$apps" = 0 ] || { echo "pre-flight failed: $apps compute processes on the card"; exit 1; }
systemd-run --user --unit tables-counters-guard /bin/bash "$S/probe_guard.sh" tables-counters-gpu0.service "$OUT/guard.log"
cat "$MEM"/feat_counters/*.npy > /dev/null
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357 --val-fixed 128 --eval-every 2000 --steps 14331 --save"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32"
run() {  # run <name> <flags...>
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo; echo "=== $name $(date +%T)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TF "$@" 2>&1 | grep --line-buffered -v "^  "
}
run counters_tf_mt_r --mem-feat "$MEM/feat_counters" --mem-orders 8,16,32,1001,1002,1003,1004,1005,1006,1007
echo; echo "=== scoring $(date +%T)"
"$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" \
  "k_counters_tf_mt=results/ckpt/counters_tf_mt_r_transformer.pt@$MEM/feat_counters"
echo "counters arm done $(date '+%F %T')"
