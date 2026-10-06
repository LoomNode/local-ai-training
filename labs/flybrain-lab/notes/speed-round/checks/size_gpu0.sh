#!/usr/bin/env bash
# Fly idea 1, write what surprised you: the plain transformer's per-position loss over the training bytes (GPU), tables gated by
# it at several sizes (CPU), the two transformer table-fed checkpoints scored on the dev windows with each (GPU). GPU 0, guarded.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
S=/tmp/claude-1000/-games-ailab/f5e9946e-97cc-48fe-9eb3-0b019140c726/scratchpad/tables
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed
CK=/games/ailab/lab/flybrain-lab/results/ckpt
OUT=$SP/results/tables
G=$SP/results/mem/gated
cd /games/ailab/lab/flybrain-lab
mkdir -p "$G"
exec > >(tee -a "$OUT/size_gpu0.log") 2>&1
echo "table size and replacement start $(date '+%F %T')"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
[ "$apps" = 0 ] || { echo "pre-flight failed: $apps compute processes on the card"; exit 1; }
if [ ! -f "$SP/results/mem/loss_tf_r.npy" ]; then
  echo "=== loss pass $(date +%T)"
  "$PY" "$S/loss_pass.py" "$CK/speed_tf_r_3600_s0_transformer.pt" "$SP/results/mem/loss_tf_r.npy" --device cuda:0
fi
echo "=== tables $(date +%T)"
CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 0-5,8-13 env OMP_NUM_THREADS=4 "$PY" "$S/gated_tables.py" "$SP/results/mem/loss_tf_r.npy" "$G" \
  full28:28:0 full29:29:0 keep24:24:0:keep keep25:25:0:keep keep27:27:0:keep
echo "=== scoring $(date +%T)"
jobs=()
for cond in full28 full29 keep24 keep25 keep27; do
  jobs+=("g_tf_tab_$cond=$CK/speed_t2_tf_tab_c_3600_s0_transformer.pt@$G/$cond" "g_tf_mt_$cond=$CK/speed_t2_tf_match_tab_c_3600_s0_transformer.pt@$G/$cond")
done
"$PY" "$S/score_tables.py" --corpus corpora/linux_full.bytes --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "${jobs[@]}"
echo "table size and replacement done $(date '+%F %T')"
