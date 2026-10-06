#!/usr/bin/env bash
# "What else repeats", the combined arm: match + byte table at orders 4,8,12,16,32,64 + the template/word/line keys (116,132,202,203,301)
# + the seven counters (1001-1007), at the same 14,331 steps as the other arms, GPU 0. Waits for the counters arm (tables-counters-gpu0)
# to end, then starts its own guard (a guard started earlier would see the counters arm's process as foreign).
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
exec > >(tee -a "$OUT/all_gpu0.log") 2>&1
echo "combined arm start $(date '+%F %T')"
while [ ! -f "$MEM/feat_all/features.json" ]; do sleep 30; done
while systemctl --user is-active --quiet tables-counters-gpu0.service; do sleep 30; done
sleep 30
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
[ "$apps" = 0 ] || { echo "pre-flight failed: $apps compute processes on the card"; exit 1; }
systemd-run --user --unit tables-all-guard /bin/bash "$S/probe_guard.sh" tables-all-gpu0.service "$OUT/guard.log"
cat "$MEM"/feat_all/*.npy > /dev/null
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357 --val-fixed 128 --eval-every 2000 --steps 14331 --save"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32"
run() {  # run <name> <flags...>
  name=$1; shift
  [ -f "results/$name.json" ] && { echo "skip $name"; return 0; }
  echo; echo "=== $name $(date +%T)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $TF "$@" 2>&1 | grep --line-buffered -v "^  "
}
run all_tf_mt_r --mem-feat "$MEM/feat_all" --mem-orders 4,8,12,16,32,64,116,132,202,203,301,1001,1002,1003,1004,1005,1006,1007
echo; echo "=== scoring $(date +%T)"
"$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" \
  "k_all_tf_mt=results/ckpt/all_tf_mt_r_transformer.pt@$MEM/feat_all"
echo "combined arm done $(date '+%F %T')"
