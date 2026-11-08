#!/usr/bin/env bash
# The combined arm, fused channel, 14,331 steps, with the 2^28 two-way tables for all eleven lookup orders (feat_all28, built by
# assoc_all.py in 172 s). GPU 0, guarded (Kyle's card: the guard stops this unit when a game or any other compute process appears),
# started while GPU 1 is busy with the fused equal-steps rerun. gpu1_queue3.sh runs this arm on GPU 1 instead if this unit is stopped
# before it writes its JSON, and scores it either way.
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
exec > >(tee -a "$OUT/all28_gpu0.log") 2>&1
echo "all28 gpu0 start $(date '+%F %T') at $(git rev-parse --short HEAD)"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "$apps" = 0 ] && [ "$used" -lt 2500 ] || { echo "pre-flight failed: $apps compute processes, $used MiB in use"; exit 1; }
systemd-run --user --unit tables-all28-guard /bin/bash "$S/probe_guard.sh" tables-all28-gpu0.service "$OUT/guard.log"
cat "$MEM"/feat_all28/*.npy > /dev/null
COMMON="--task code --corpus-file $C --train-bytes 200000000 --val-bytes 2000000 --normalize --key-conv 2 --top-k 4 --n-key 4096 --lr 0.0006 --decay 1.0 --d-model 512 --d-ff 1024 --n-layers 8 --batch 8 --seq-len 2048 --seed 0 --device cuda:0 --warmup 1000 --cosine --wd 0.1 --chunk 256 --amp --meter --val-offset 202117357 --val-fixed 128 --eval-every 2000 --save"
TF="--arms transformer --rope --tf32 --fused --compile --rope-table --match-feat 8,16,32 --fused-channels"
ORD="4,8,12,16,32,64,116,132,202,203,301,1001,1002,1003,1004,1005,1006,1007"
[ -f results/all28f_tf_mt_r.json ] && { echo "skip all28f_tf_mt_r"; exit 0; }
echo; echo "=== all28f_tf_mt_r $(date +%T)"
"$PY" scripts/run_sparsela.py --run-name all28f_tf_mt_r $COMMON $TF --steps 14331 --mem-feat "$MEM/feat_all28" --mem-orders "$ORD" 2>&1 | grep --line-buffered -v "^  "
echo "all28 gpu0 done $(date '+%F %T')"
