#!/usr/bin/env bash
# The three follow-ups of the tables' arm (Kyle, 2026-10-20: "Queue up those 3"), one chain on GPU 1 behind the CPU choosers:
#   1 the breadth vote      results/tables/qwen-breadth  table_retrieve.py --mode breadth (the span count with the tables' hashes, the keys as vocabularies)
#   2 the dedup chooser     results/tables/qwen-dedup    table_retrieve.py --mode spans --dedup (never two chunks of one text)
#   3 the easy-window gate  results/tables/qwen-gate     qwen_memory.py --conditions first (Qwen on the first half alone), then gate_sweep.py over every run, offline
# Each Qwen arm: smoke (3 windows) then the full paired run against the span run (results/tables/qwen/qwen_memory.json); its lines go to
# <dir>/qwen_memory.txt and talk/leads.txt; the gate's to qwen-gate/gate.txt. A failed arm is logged and the chain goes on.
# usage: CUDA_VISIBLE_DEVICES=<uuid> talk_qwen_arms_gpu.sh   (log results/tables/qwen-arms.log; done file results/tables/qwen-gate/arms.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; C2=$SP/notes/model2/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; PYQ=/games/ailab/lab/flybrain-lab/.venv-qwen/bin/python
NV=/games/ailab/lab/flybrain-lab/.venv-qwen/lib/python3.12/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
R=$SP/results/tables; REF=$R/qwen/qwen_memory.json; TOUT=$R/talk; GOUT=$R/qwen-gate
MODEL=/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
mkdir -p "$GOUT"; cd "$SP"; exec >> "$R/qwen-arms.log" 2>&1
echo "the three arms: start $(date '+%F %T') on $CUDA_VISIBLE_DEVICES"
idle() { while :; do local apps used; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$CUDA_VISIBLE_DEVICES"); used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); [ "$apps" = 0 ] && [ -n "$used" ] && [ "$used" -lt 2500 ] && return 0; sleep 60; done; }
arm() {   # name  label  choices.json
  local OUT=$R/qwen-$1
  until [ -f "$3" ]; do sleep 30; done
  if [ ! -f "$OUT/qwen_memory.json" ]; then
    idle; echo "== $1 smoke $(date +%T)"; ( umask 077; "$PYQ" "$C2/qwen_memory.py" --model "$MODEL" --out "$OUT/smoke" --conditions memory --choices "$3" --compare "$REF" --label "$2" --limit 3 ) || { echo "$1 smoke failed"; return 1; }
    idle; echo "== $1 full $(date +%T)"; ( umask 077; "$PYQ" "$C2/qwen_memory.py" --model "$MODEL" --out "$OUT" --conditions memory --choices "$3" --compare "$REF" --label "$2" ) || { echo "$1 full run failed"; return 1; }
  fi
  tail -2 "$OUT/qwen_memory.txt" | tee -a "$TOUT/leads.txt"
}
arm breadth "the breadth vote" "$R/qwen-breadth/choices.json"
arm dedup "the dedup chooser" "$R/qwen-dedup/choices.json"
if [ ! -f "$GOUT/qwen_memory.json" ]; then
  idle; echo "== the first half alone $(date +%T)"; ( umask 077; "$PYQ" "$C2/qwen_memory.py" --model "$MODEL" --out "$GOUT" --conditions first ) || echo "first-half pass failed"
fi
echo "== gate sweep $(date +%T)"
( umask 077; nice -n 19 taskset -c 0-5,8-13 "$PY" "$C2/gate_sweep.py" --ref "$REF" --first "$GOUT/qwen_memory.json" --choices "$R/qwen-dedup/choices.json" \
  --runs "the tables=$R/qwen-tables/qwen_memory.json" "the breadth vote=$R/qwen-breadth/qwen_memory.json" "the dedup chooser=$R/qwen-dedup/qwen_memory.json" --out "$GOUT/gate.txt" ) \
  && grep "^gate: the spans" "$GOUT/gate.txt" | tee -a "$TOUT/leads.txt"
touch "$GOUT/arms.done"; echo "done $(date '+%F %T')"
