#!/usr/bin/env bash
# Speed round: every saved one-hour and two-hour checkpoint scored once on the TEST yardstick windows (fp32, the 1,024 windows from
# byte 211,576,468), after the last queue. Dev chose nothing here that test could confirm or refute per arm; this is the one read
# of the split the runs never saw. A checkpoint trained with the corpus table gets the table's features. A name whose JSON exists
# is skipped; a failed score does not stop the loop.
# usage: score_test.sh [units to wait for, in order ...]
# The caller picks the card by UUID: CUDA_VISIBLE_DEVICES hides the other card, so the visible one is cuda:0.
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
WT=/games/ailab/lab/flybrain-lab/.worktrees/speed-run3
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
C=/games/ailab/lab/flybrain-lab/corpora/linux_full.bytes
FEAT=/games/ailab/lab/flybrain-lab/.worktrees/speed/results/mem/feat
cd "$WT"
for u in "$@"; do
  while systemctl --user is-active --quiet "$u"; do sleep 30; done
  sleep 20
done
exec > >(tee -a results/speed_score_test.log) 2>&1
echo "test scoring start $(date '+%F %T')"
for ck in results/ckpt/speed_*_s0_*.pt; do
  [ -f "$ck" ] || continue
  name=$(basename "$ck" .pt); name=${name#speed_}; name=${name%_s0_*}   # speed_t2_tf_tab_c_3600_s0_transformer.pt -> t2_tf_tab_c_3600
  out=speed_score_test_$name
  [ -f "results/$out.json" ] && continue
  extra=""
  case "$name" in *_tab_*) extra="--mem-feat $FEAT" ;; esac
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  [ "$apps" = 0 ] || { echo "stopping $(date +%T): $apps compute processes on the card"; break; }
  echo "--- $name $(date +%T)"
  "$PY" scripts/cross_eval.py --ckpts "$ck" --corpus-file "$C" --offset 211576468 --fixed 1024 --lens 2048 --device cuda:0 --run-name "$out" $extra 2>&1 | tail -n 1 || echo "scoring of $name failed"
done
echo "test scoring done $(date '+%F %T')"
