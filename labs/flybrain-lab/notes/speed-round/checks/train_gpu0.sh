#!/usr/bin/env bash
# Tables follow-up, the reference for probe 2: the transformer TRAINED on the CPython bytes the table was built from, at the
# corrected bar's step count (15,953), with match lookup + the CPython table (A) and plain (B); then both scored on the probe's
# CPython windows. GPU 0 (idle; Kyle said "use gpu when available"), guarded by probe_guard.sh. Resumable: a finished run's JSON is skipped
# by the runner; a finished score by the scorer.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-run4
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
S=/tmp/claude-1000/-games-ailab/f5e9946e-97cc-48fe-9eb3-0b019140c726/scratchpad/tables
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed
C=/games/ailab/lab/flybrain-lab/corpora/cpython.bytes
FEAT=$SP/results/mem_cpython/feat_train
OUT=$SP/results/tables
cd "$WT"
exec > >(tee -a "$OUT/train_gpu0.log") 2>&1
echo "cpython reference runs start $(date '+%F %T')"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
[ "$apps" = 0 ] || { echo "pre-flight failed: $apps compute processes on the card"; exit 1; }
COMMON="--task code --corpus-file $C --train-bytes 40000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 40000000 --val-fixed 128 --eval-every 2000 --steps 15953 --save"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table"
cat "$FEAT"/*.npy > /dev/null
for spec in "cpy_tf_mt_r|$TF --match-feat 8,16,32 --mem-feat $FEAT --mem-orders 8,16,32" "cpy_tf_r|$TF"; do
  name=${spec%%|*}; flags=${spec#*|}
  [ -f "results/$name.json" ] && { echo "skip $name"; continue; }
  echo; echo "=== $name $(date +%T)"
  "$PY" scripts/run_sparsela.py --run-name "$name" $COMMON $flags 2>&1 | grep --line-buffered -v "^  "
done
echo; echo "=== scoring $(date +%T)"
"$PY" "$S/score_tables.py" --corpus "$C" --offset 40000000 --n 1024 --T 2048 --device cuda:0 --out-dir "$OUT" \
  "p2_cpy_tf_mt_own=results/ckpt/cpy_tf_mt_r_transformer.pt@$SP/results/mem_cpython/own" "p2_cpy_tf_plain=results/ckpt/cpy_tf_r_transformer.pt"
echo "cpython reference runs done $(date '+%F %T')"
