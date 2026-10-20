#!/usr/bin/env bash
# The tables choose what Qwen reads (Kyle, 2026-10-20: "lets try" the table-driven retrieval): qwen_memory.py's memory condition on the
# same windows, the same Qwen, the same budget and prompt, with the chunks table_retrieve.py chose (results/tables/qwen-tables/choices.json)
# instead of the 12-byte-span retriever's; the per-window result paired against the span run (results/tables/qwen/qwen_memory.json).
# usage: CUDA_VISIBLE_DEVICES=<uuid> talk_qwen_tables_gpu.sh   (log results/tables/qwen-tables/qwen.log; lines qwen-tables/qwen_memory.txt + talk/leads.txt; done file qwen-tables/qwen.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; C2=$SP/notes/model2/checks
PYQ=/games/ailab/lab/flybrain-lab/.venv-qwen/bin/python
NV=/games/ailab/lab/flybrain-lab/.venv-qwen/lib/python3.12/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
OUT=$SP/results/tables/qwen-tables; REF=$SP/results/tables/qwen/qwen_memory.json; TOUT=$SP/results/tables/talk
MODEL=/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
mkdir -p "$OUT"; cd "$SP"; exec >> "$OUT/qwen.log" 2>&1
echo "Qwen + the tables' chunks: start $(date '+%F %T') on $CUDA_VISIBLE_DEVICES"
until [ -f "$OUT/choices.json" ]; do sleep 30; done
idle() { while :; do local apps used; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$CUDA_VISIBLE_DEVICES"); used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); [ "$apps" = 0 ] && [ -n "$used" ] && [ "$used" -lt 2500 ] && return 0; sleep 60; done; }
if [ ! -f "$OUT/qwen_memory.json" ]; then
  idle; echo "== smoke $(date +%T)"; ( umask 077; "$PYQ" "$C2/qwen_memory.py" --model "$MODEL" --out "$OUT/smoke" --conditions memory --choices "$OUT/choices.json" --compare "$REF" --limit 3 ) || { echo "smoke failed"; exit 1; }
  idle; echo "== full run $(date +%T)"; ( umask 077; "$PYQ" "$C2/qwen_memory.py" --model "$MODEL" --out "$OUT" --conditions memory --choices "$OUT/choices.json" --compare "$REF" ) || { echo "full run failed"; exit 1; }
fi
tail -2 "$OUT/qwen_memory.txt" | tee -a "$TOUT/leads.txt"
touch "$OUT/qwen.done"; echo "done $(date '+%F %T')"
