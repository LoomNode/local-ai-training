#!/usr/bin/env bash
# The sweep's follow-on (2026-10-21 05:40, after group B: file headers −0.192 on his turns against the spans' −0.115, four texts of 750
# −0.167, a rule separator −0.145, each alone): the three combined, on GPU 1 after the sweep (waits for qwen-sweep/sweep.done). The same
# choose/score as talk_qwen_sweep_gpu.sh (the prompt cache, the arms registered in arms.txt, paired lines to sweep.txt), then sweep_table.py
# over every arm of the sweep plus these into qwen-sweep/table2.md. Arms (the dedup chooser's choices and c750top4's are reused):
#   hdrsep     the dedup chooser + file headers + the rule separator
#   hdr750     four texts of 750 + file headers
#   hdrsep750  four texts of 750 + file headers + the rule separator
# usage: CUDA_VISIBLE_DEVICES=<uuid> talk_qwen_combo_gpu.sh   (log results/tables/qwen-sweep.log; marker qwen-sweep/combo.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; C2=$SP/notes/model2/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; PYQ=/games/ailab/lab/flybrain-lab/.venv-qwen/bin/python
NV=/games/ailab/lab/flybrain-lab/.venv-qwen/lib/python3.12/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
R=$SP/results/tables; REF=$R/qwen/qwen_memory.json; SW=$R/qwen-sweep; CACHE=$SW/cache/scores.jsonl; DEDUP=$R/qwen-dedup/choices.json
MODEL=/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
CPU="nice -n 19 taskset -c 0-5,8-13"
cd "$SP"; exec >> "$R/qwen-sweep.log" 2>&1
echo "the combinations: start $(date '+%F %T') on $CUDA_VISIBLE_DEVICES (waiting for sweep.done)"
until [ -f "$SW/sweep.done" ]; do sleep 300; done
idle() { while :; do local apps used; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$CUDA_VISIBLE_DEVICES"); used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ "${apps:-0}" -eq 0 ] && [ "${used:-99999}" -lt 2500 ] && return 0; sleep 60; done; }
score() {   # name  label  choices.json  qwen_memory.py args...
  local NAME=$1 OUT=$SW/$1 LABEL=$2 CH=$3; shift 3
  grep -qx "$NAME=$LABEL" "$SW/arms.txt" 2>/dev/null || echo "$NAME=$LABEL" >> "$SW/arms.txt"
  if [ ! -f "$OUT/qwen_memory.json" ]; then
    idle; echo "== score $NAME $(date +%T)"
    ( umask 077; "$PYQ" "$C2/qwen_memory.py" --model "$MODEL" --out "$OUT" --conditions memory --choices "$CH" --compare "$REF" --label "$LABEL" --cache "$CACHE" "$@" ) || { echo "score $NAME failed"; return 1; }
  fi
  tail -2 "$OUT/qwen_memory.txt" | sed "s/^/$NAME: /" | tee -a "$SW/sweep.txt"
}
score hdrsep "headers and the rule" "$DEDUP" --headers --sep '\n\n---\n\n'
score hdr750 "four texts of 750 with headers" "$SW/c750top4/choices.json" --chunk 750 --headers
score hdrsep750 "four texts of 750, headers and the rule" "$SW/c750top4/choices.json" --chunk 750 --headers --sep '\n\n---\n\n'
echo "== the table with the combinations $(date +%T)"
ARGS=("the tables=$R/qwen-tables/qwen_memory.json" "the breadth vote=$R/qwen-breadth/qwen_memory.json" "the dedup chooser=$R/qwen-dedup/qwen_memory.json")
while IFS='=' read -r n l; do case "$n" in firstmem-*|spans|dedup200) ;; *) [ -f "$SW/$n/qwen_memory.json" ] && ARGS+=("$l=$SW/$n/qwen_memory.json");; esac; done < "$SW/arms.txt"
( umask 077; $CPU "$PY" "$C2/sweep_table.py" --ref "$REF" --out "$SW/table2.md" "${ARGS[@]}" )
touch "$SW/combo.done"; echo "combinations done $(date '+%F %T')"
