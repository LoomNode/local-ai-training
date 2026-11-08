#!/usr/bin/env bash
# The sweep (Kyle, 2026-10-20: "we should exhaustively test all 3 separately. like you were quick to write things off"): each of the three
# follow-ups on its own grid, one chain on GPU 1, every Qwen prompt cached by content (qwen_memory.py --cache: a prompt any arm scored is
# never scored twice, and a stopped chain resumes where it was). Three groups, in this order:
#   B  the span chooser's own grid: the floor (min hits 6, 10), one chunk only, the best chunk adjacent to the window, the dedup test (100
#      and 500 bytes, the span share), three texts of 1,000 bytes, four of 750, one of 3,000, double the budget (four chunks), a rule
#      separator, file headers
#   C  the gate's new signal: Qwen on the first half GIVEN the memory (the spans' chunks and the dedup chooser's)
#   A  the breadth vote's grid: the families (12 alone; 16; 8; 32+16+12; bytes only; keys only; 12 + the words and lines), the cap (none,
#      4, 64), the vote (one, the family's length, rarity), dedup
# then sweep_table.py (every arm paired against the span run, bootstrap intervals, the oracle over all arms) and gate_sweep2.py.
# Each arm: table_retrieve.py on the CPU (if its choices.json is missing), then qwen_memory.py --conditions memory --cache; its paired
# lines go to results/tables/qwen-sweep/sweep.txt. A failed arm is logged and the chain goes on; an arm with a qwen_memory.json is skipped.
# usage: CUDA_VISIBLE_DEVICES=<uuid> talk_qwen_sweep_gpu.sh   (log results/tables/qwen-sweep.log; markers qwen-sweep/{B,C,A}.done, sweep.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; C2=$SP/notes/model2/checks
PY=/games/ailab/lab/flybrain-lab/.venv/bin/python; PYQ=/games/ailab/lab/flybrain-lab/.venv-qwen/bin/python
NV=/games/ailab/lab/flybrain-lab/.venv-qwen/lib/python3.12/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
R=$SP/results/tables; REF=$R/qwen/qwen_memory.json; SW=$R/qwen-sweep; CACHE=$SW/cache/scores.jsonl; DEDUP=$R/qwen-dedup/choices.json
MODEL=/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
CPU="nice -n 19 taskset -c 0-5,8-13"
mkdir -p "$SW/cache"; cd "$SP"; exec >> "$R/qwen-sweep.log" 2>&1
echo "the sweep: start $(date '+%F %T') on $CUDA_VISIBLE_DEVICES"
idle() { while :; do local apps used; apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null | grep -c "$CUDA_VISIBLE_DEVICES"); used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ "${apps:-0}" -eq 0 ] && [ "${used:-99999}" -lt 4000 ] && return 0; sleep 60; done; }
if [ ! -f "$SW/cache/seeded" ]; then
  echo "== seed $(date +%T)"
  ( umask 077; $CPU "$PY" "$C2/qwen_cache_seed.py" --cache "$CACHE" --model "$MODEL" "$R/qwen=warm" "$R/qwen=memory:$DEDUP:span_choices" "$R/qwen-gate=first" \
      "$R/qwen-tables=memory:$R/qwen-tables/choices.json" "$R/qwen-breadth=memory:$R/qwen-breadth/choices.json" "$R/qwen-dedup=memory:$DEDUP" ) && touch "$SW/cache/seeded"
fi
choose() {   # name  table_retrieve.py args...
  local OUT=$SW/$1; shift
  [ -f "$OUT/choices.json" ] && return 0
  echo "== choose $(basename "$OUT") $(date +%T)"; ( umask 077; $CPU "$PY" "$C2/table_retrieve.py" --out "$OUT" "$@" ) || { echo "choose $(basename "$OUT") failed"; return 1; }
}
score() {   # name  label  choices.json|-  qwen_memory.py args...   (the arm is registered for the table)
  local NAME=$1 OUT=$SW/$1 LABEL=$2 CH=$3; shift 3; local CARGS=(); [ "$CH" != "-" ] && CARGS=(--choices "$CH")
  grep -qx "$NAME=$LABEL" "$SW/arms.txt" 2>/dev/null || echo "$NAME=$LABEL" >> "$SW/arms.txt"
  if [ ! -f "$OUT/qwen_memory.json" ]; then
    idle; echo "== score $NAME $(date +%T)"
    ( umask 077; "$PYQ" "$C2/qwen_memory.py" --model "$MODEL" --out "$OUT" --conditions memory "${CARGS[@]}" --compare "$REF" --label "$LABEL" --cache "$CACHE" "$@" ) || { echo "score $NAME failed"; return 1; }
  fi
  tail -2 "$OUT/qwen_memory.txt" | sed "s/^/$NAME: /" | tee -a "$SW/sweep.txt"
}
# ---- B: the span chooser's own grid ---------------------------------------------------------------------------------------------------------
choose spans --mode spans                                    && score spans "the spans (from the cache)" "$SW/spans/choices.json"
choose dedup200 --mode spans --dedup                         && score dedup200 "the dedup chooser (from the cache)" "$SW/dedup200/choices.json"
choose floor6 --mode spans --dedup --min-hits 6              && score floor6 "the floor at 6 hits" "$SW/floor6/choices.json"
choose floor10 --mode spans --dedup --min-hits 10            && score floor10 "the floor at 10 hits" "$SW/floor10/choices.json"
choose top1 --mode spans --top 1                             && score top1 "one chunk" "$SW/top1/choices.json"
score toplast "the best chunk last" "$DEDUP" --order top-last
choose dedup100 --mode spans --dedup --dedup-len 100         && score dedup100 "dedup at 100 bytes" "$SW/dedup100/choices.json"
choose dedup500 --mode spans --dedup --dedup-len 500         && score dedup500 "dedup at 500 bytes" "$SW/dedup500/choices.json"
choose dedupfrac --mode spans --dedup --dedup-frac 0.5       && score dedupfrac "dedup by span share" "$SW/dedupfrac/choices.json"
choose c1000top3 --mode spans --dedup --chunk 1000 --top 3   && score c1000top3 "three texts of 1,000" "$SW/c1000top3/choices.json" --chunk 1000
choose c750top4 --mode spans --dedup --chunk 750 --top 4     && score c750top4 "four texts of 750" "$SW/c750top4/choices.json" --chunk 750
choose c3000top1 --mode spans --chunk 3000 --top 1           && score c3000top1 "one text of 3,000" "$SW/c3000top1/choices.json" --chunk 3000
choose top4 --mode spans --dedup --top 4 --max-mem 6400      && score top4 "four chunks, double budget" "$SW/top4/choices.json"
score seprule "a rule separator" "$DEDUP" --sep '\n\n---\n\n'
score headers "file headers" "$DEDUP" --headers
touch "$SW/B.done"; echo "== B done $(date '+%F %T')"
# ---- C: the gate's new signal ---------------------------------------------------------------------------------------------------------------
score firstmem-spans "the spans" - --conditions first_mem
score firstmem-dedup "the dedup chooser" "$DEDUP" --conditions first_mem
touch "$SW/C.done"; echo "== C done $(date '+%F %T')"
# ---- A: the breadth vote's grid -------------------------------------------------------------------------------------------------------------
ALL=64,32,16,12,8,4,301,132,116,203,202,108,201,104,200
choose b12 --mode breadth --families 12 --max-chunks 16      && score b12 "breadth at 12" "$SW/b12/choices.json"
choose bbytes --mode breadth --families 64,32,16,12,8,4 --max-chunks 16 && score bbytes "breadth, bytes only" "$SW/bbytes/choices.json"
choose bkeys --mode breadth --families 301,132,116,203,202,108,201,104,200 --max-chunks 16 && score bkeys "breadth, keys only" "$SW/bkeys/choices.json"
choose bdedup --mode breadth --families $ALL --max-chunks 16 --dedup && score bdedup "breadth with dedup" "$SW/bdedup/choices.json"
choose border --mode breadth --families $ALL --max-chunks 16 --vote order --min-hits 36 && score border "breadth by length" "$SW/border/choices.json"
choose brarity --mode breadth --families $ALL --max-chunks 16 --vote rarity --min-hits 1 && score brarity "breadth by rarity" "$SW/brarity/choices.json"
choose bcap0 --mode breadth --families $ALL --max-chunks 0   && score bcap0 "breadth, no cap" "$SW/bcap0/choices.json"
choose bcap4 --mode breadth --families $ALL --max-chunks 4   && score bcap4 "breadth, cap 4" "$SW/bcap4/choices.json"
choose bcap64 --mode breadth --families $ALL --max-chunks 64 && score bcap64 "breadth, cap 64" "$SW/bcap64/choices.json"
choose b16 --mode breadth --families 16 --max-chunks 16      && score b16 "breadth at 16" "$SW/b16/choices.json"
choose b8 --mode breadth --families 8 --max-chunks 16        && score b8 "breadth at 8" "$SW/b8/choices.json"
choose b321612 --mode breadth --families 32,16,12 --max-chunks 16 && score b321612 "breadth at 32, 16 and 12" "$SW/b321612/choices.json"
choose b12words --mode breadth --families 12,301,203,202 --max-chunks 16 && score b12words "breadth at 12 with the words and lines" "$SW/b12words/choices.json"
touch "$SW/A.done"; echo "== A done $(date '+%F %T')"
# ---- the table and the gate -----------------------------------------------------------------------------------------------------------------
echo "== the table $(date +%T)"
ARGS=("the tables=$R/qwen-tables/qwen_memory.json" "the breadth vote=$R/qwen-breadth/qwen_memory.json" "the dedup chooser=$R/qwen-dedup/qwen_memory.json")
while IFS='=' read -r n l; do case "$n" in firstmem-*|spans|dedup200) ;; *) ARGS+=("$l=$SW/$n/qwen_memory.json");; esac; done < "$SW/arms.txt"
( umask 077; $CPU "$PY" "$C2/sweep_table.py" --ref "$REF" --out "$SW/table.md" "${ARGS[@]}" )
echo "== the gate $(date +%T)"
( umask 077; $CPU "$PY" "$C2/gate_sweep2.py" --ref "$REF" --first "$R/qwen-gate/qwen_memory.json" --window-choices "$SW/spans/choices.json" --out "$SW/gate2.txt" \
    --runs "the spans=$SW/spans/qwen_memory.json:$SW/spans/choices.json:$SW/firstmem-spans/qwen_memory.json:$SW/top1/qwen_memory.json" \
           "the dedup chooser=$SW/dedup200/qwen_memory.json:$SW/dedup200/choices.json:$SW/firstmem-dedup/qwen_memory.json:$SW/top1/qwen_memory.json" \
           "the breadth vote=$R/qwen-breadth/qwen_memory.json:$R/qwen-breadth/choices.json" "the tables=$R/qwen-tables/qwen_memory.json:$R/qwen-tables/choices.json" )
touch "$SW/sweep.done"; echo "done $(date '+%F %T')"
