#!/usr/bin/env bash
# The hindsight check's scoring (2026-10-21; a side agent's point relayed by Kyle: the world is a 10-18 snapshot, the sessions are older):
# the headers arm and the spans arm with the as-of memory (table_retrieve.py --asof: a window reads only files created before its session),
# scored by Qwen with the sweep's cache; paired lines to qwen-sweep/sweep.txt. Arms: hdrasof (dedup + file headers, as of), spansasof (the
# plain spans, as of). Each arm is taken under a lock, so two copies of this chain (one a card) share the work. WAIT_FOR=<file> waits first.
# usage: CUDA_VISIBLE_DEVICES=<uuid> [WAIT_FOR=<marker>] talk_qwen_asof_gpu.sh   (log results/tables/qwen-sweep.log; marker qwen-sweep/asof.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; C2=$SP/notes/model2/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; PYQ=/games/ailab/lab/flybrain-lab/.venv-qwen/bin/python
NV=/games/ailab/lab/flybrain-lab/.venv-qwen/lib/python3.12/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
R=$SP/results/tables; REF=$R/qwen/qwen_memory.json; SW=$R/qwen-sweep; CACHE=$SW/cache/scores.jsonl
MODEL=/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
cd "$SP"; exec >> "$R/qwen-sweep.log" 2>&1
echo "the as-of arms: start $(date '+%F %T') on $CUDA_VISIBLE_DEVICES (waiting for ${WAIT_FOR:-nothing})"
[ -n "${WAIT_FOR:-}" ] && until [ -f "$WAIT_FOR" ]; do sleep 300; done
idle() { while :; do local apps used; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$CUDA_VISIBLE_DEVICES"); used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ "${apps:-0}" -eq 0 ] && [ "${used:-99999}" -lt 2500 ] && return 0; sleep 60; done; }
score() {   # name  label  choices.json  qwen_memory.py args...   (under a lock; skipped if another copy holds it or the result exists)
  local NAME=$1 OUT=$SW/$1 LABEL=$2 CH=$3; shift 3
  grep -qx "$NAME=$LABEL" "$SW/arms.txt" 2>/dev/null || echo "$NAME=$LABEL" >> "$SW/arms.txt"
  if [ ! -f "$OUT/qwen_memory.json" ]; then
    exec 9>"$OUT.lock"; flock -n 9 || { echo "== $NAME is being scored by another copy: skipped here"; exec 9>&-; return 0; }
    idle; echo "== score $NAME $(date +%T)"
    ( umask 077; "$PYQ" "$C2/qwen_memory.py" --model "$MODEL" --out "$OUT" --conditions memory --choices "$CH" --compare "$REF" --label "$LABEL" --cache "$CACHE" "$@" ) || { echo "score $NAME failed"; exec 9>&-; return 1; }
    exec 9>&-
  fi
  tail -2 "$OUT/qwen_memory.txt" | sed "s/^/$NAME: /" | tee -a "$SW/sweep.txt"
}
score hdrasof "file headers, as of the session" "$SW/hdrasof/choices.json" --headers
score spansasof "the spans, as of the session" "$SW/spansasof/choices.json"
[ -f "$SW/hdrasof/qwen_memory.json" ] && [ -f "$SW/spansasof/qwen_memory.json" ] && { touch "$SW/asof.done"; echo "as-of done $(date '+%F %T')"; }
