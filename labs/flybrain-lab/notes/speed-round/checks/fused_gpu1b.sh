#!/usr/bin/env bash
# The combined arm with --fused-channels (the channel without its loop over orders), at the base arm's 14,331 steps: if the fused step
# costs what the base step costs, this row is the recipe at equal steps and equal seconds at once. First a 400-step calibration, whose
# pace is compared with the unfused calibration of the equal-seconds queue (cal_all_tf_mt_r). GPU 1 (Kyle: "can't you run on gpu 1?"): the model
# server is stopped for the run and left down (Kyle, 2026-10-07: "just leave the local model server down unless you need it"). The run worktree must be pinned at a commit that has --fused-channels.
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
exec > >(tee -a "$OUT/fused_gpu1b.log") 2>&1
echo "fused combined arm start $(date '+%F %T') at $(git rev-parse --short HEAD)"
grep -q "fused-channels" scripts/run_sparsela.py || { echo "the run worktree has no --fused-channels: pin it first"; exit 1; }
# rerun after the first attempt died of a bus error (a feature file it had mapped was rewritten by a builder); waits for the queue
while systemctl --user is-active --quiet tables-gpu1-queue.service; do sleep 30; done
sleep 10
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
[ "$apps" = 0 ] || { echo "pre-flight failed: $apps compute processes on the card"; exit 1; }
cat "$MEM"/feat_all/*.npy > /dev/null
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32 --fused-channels"
ALL="--mem-feat $MEM/feat_all --mem-orders 4,8,12,16,32,64,116,132,202,203,301,1001,1002,1003,1004,1005,1006,1007"
if [ ! -f results/cal_allf_tf_mt_r.json ]; then
  echo; echo "=== calibration, fused $(date +%T)"
  "$PY" scripts/run_sparsela.py --run-name cal_allf_tf_mt_r $COMMON $TF $ALL --steps 400 --val-fixed 8 --eval-every 100 2>&1 | grep --line-buffered -v "^  " | tail -n 4
fi
for cal in cal_all_tf_mt_r cal_allf_tf_mt_r; do
  [ -f "results/$cal.json" ] && "$PY" budget.py "results/$cal.json" 3307 1.06 2>&1 | sed "s/^/$cal: /" | tail -n 2
done
if [ ! -f results/allf_tf_mt_r.json ]; then
  echo; echo "=== allf_tf_mt_r $(date +%T)"
  "$PY" scripts/run_sparsela.py --run-name allf_tf_mt_r $COMMON $TF $ALL --steps 14331 --save --val-fixed 128 --eval-every 2000 2>&1 | grep --line-buffered -v "^  "
fi
echo; echo "=== scoring $(date +%T)"
"$PY" "$S/score_tables.py" --corpus "$C" --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" \
  "k_allf_tf_mt=results/ckpt/allf_tf_mt_r_transformer.pt@$MEM/feat_all"
echo "fused combined arm done $(date '+%F %T')"
