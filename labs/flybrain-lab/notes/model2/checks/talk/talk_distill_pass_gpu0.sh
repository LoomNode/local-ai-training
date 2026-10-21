#!/usr/bin/env bash
# The teacher pass on GPU 0 (Kyle, 2026-10-20: "it just saves time to use both cards. gpu 0 is idle most of the time"; the tender
# gpu0_tend.sh stops this unit when a game takes the card and relaunches it when the card is free; the pass resumes by window).
# P0 the bench (both teachers on the first MB, once), then P1 the pass with the chosen teacher into /vault/ailab/flybrain-mem/distill/teacher.
# usage: CUDA_VISIBLE_DEVICES=<GPU 0 uuid> talk_distill_pass_gpu0.sh   (log results/tables/distill/pass.log; done file .../teacher/pass.done)
set -u -o pipefail
: "${CUDA_VISIBLE_DEVICES:?}"
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; C2=$SP/notes/model2/checks; PYQ=/games/ailab/lab/flybrain-lab/.venv-qwen/bin/python; PY=/games/ailab/lab/flybrain-lab/.venv/bin/python
NV=/games/ailab/lab/flybrain-lab/.venv-qwen/lib/python3.12/site-packages/nvidia; export LD_LIBRARY_PATH=$NV/cuda_runtime/lib:$NV/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
C=/games/ailab/corpora; VM=/vault/ailab/flybrain-mem/distill; OUT=$SP/results/tables/distill
M27=/vault/ailab/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf; MOE=/vault/ailab/models/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf
mkdir -p "$OUT" "$VM/teacher"; cd "$SP"; exec >> "$OUT/pass.log" 2>&1; umask 077
log() { echo "[$(date '+%F %T')] $*"; }
free_gb() { df --output=avail -BG "$1" | tail -1 | tr -dc '0-9'; }
echo "$(hostname) $$ $(date '+%F %T')" > "$VM/teacher/pass.owner"
log "the pass on $CUDA_VISIBLE_DEVICES: start; /vault free $(free_gb /vault) GB"
[ "$(free_gb /vault)" -ge 20 ] || { log "under 20 GB free on /vault: stop"; exit 1; }
if [ ! -s "$OUT/bench.txt" ]; then
  for M in "$M27" "$MOE"; do
    log "bench $(basename "$M")"
    "$PYQ" "$C2/distill_teacher.py" --slice "$C/distill.bytes" --model "$M" --bench 1 2>&1 | grep "^bench:\|windows," | sed "s|^|$(basename "$M"): |" | tee -a "$OUT/bench.txt" || log "bench $(basename "$M") failed"
  done
fi
TEACHER=$M27
KBS=$(grep "^$(basename "$M27"): bench:" "$OUT/bench.txt" | tail -1 | sed 's/.*"kb_per_s": \([0-9.]*\).*/\1/')
if [ -n "${KBS:-}" ] && "$PY" -c "import sys; sys.exit(0 if float('$KBS') < 0.64 else 1)"; then TEACHER=$MOE; log "the 27B reads $KBS KB/s: 32 MB past 14 h, the MoE teaches"; else log "the 27B reads ${KBS:-?} KB/s: it teaches"; fi
if [ ! -f "$VM/teacher/teacher.json" ]; then
  log "the pass: $(basename "$TEACHER") (resumes from progress.json if there)"
  "$PYQ" "$C2/distill_teacher.py" --slice "$C/distill.bytes" --out "$VM/teacher" --model "$TEACHER" || { log "the pass stopped (rc $?)"; exit 1; }
fi
[ -e "$C/distill-teacher" ] || ln -s "$VM/teacher" "$C/distill-teacher"
touch "$VM/teacher/pass.done"; log "the pass is done: $(grep -o '"kb_per_s": [0-9.]*\|"dropped_mass_mean": [0-9.]*\|"cover_mean": [0-9.]*\|"fallback_rate": [0-9.]*' "$VM/teacher/teacher.json" | tr '\n' ' ')"
