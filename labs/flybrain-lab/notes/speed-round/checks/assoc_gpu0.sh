#!/usr/bin/env bash
# Two-way tables at three sizes and a direct-mapped 2^30 table (CPU builds), scored with the two transformer table-fed checkpoints (GPU 0).
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
S=/tmp/claude-1000/-games-ailab/f5e9946e-97cc-48fe-9eb3-0b019140c726/scratchpad/tables
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed
CK=/games/ailab/lab/flybrain-lab/results/ckpt
OUT=$SP/results/tables
G=$SP/results/mem/gated
cd /games/ailab/lab/flybrain-lab
exec > >(tee -a "$OUT/assoc_gpu0.log") 2>&1
echo "two-way tables start $(date '+%F %T')"
CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 0-5,8-13 env OMP_NUM_THREADS=4 "$PY" "$S/assoc_tables.py" "$G" assoc24:24 assoc25:25 assoc27:27
CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 0-5,8-13 env OMP_NUM_THREADS=4 true
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
[ "$apps" = 0 ] || { echo "pre-flight failed: $apps compute processes on the card"; exit 1; }
jobs=()
for cond in assoc24 assoc25 assoc27 full30 keep29; do
  jobs+=("g_tf_tab_$cond=$CK/speed_t2_tf_tab_c_3600_s0_transformer.pt@$G/$cond" "g_tf_mt_$cond=$CK/speed_t2_tf_match_tab_c_3600_s0_transformer.pt@$G/$cond")
done
"$PY" "$S/score_tables.py" --corpus corpora/linux_full.bytes --offset 202117357 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" "${jobs[@]}"
echo "two-way tables done $(date '+%F %T')"
