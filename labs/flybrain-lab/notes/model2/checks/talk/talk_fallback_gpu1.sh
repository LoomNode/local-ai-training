#!/usr/bin/env bash
# A talk source's reader on GPU 1 as the fallback for GPU 0's slot (Kyle, 2026-10-17: "i think everything should prioritise using gpu 1"):
# waits for a trigger file (the reader ahead of it on GPU 1), then, unless the source's score line exists, waits for GPU 0's job to end
# (its slot, <name>-orch, is stopped so it cannot restart the job on GPU 0; a run in progress is left to finish) and runs talk_readers.sh
# for the source on GPU 1, resuming from the job's newest checkpoint if it left one.
# usage: systemd-run --user --unit <unit> /bin/bash talk_fallback_gpu1.sh <trigger file> <gpu0 job name> <src>
set -u
TRIGGER=${1:?usage: talk_fallback_gpu1.sh <trigger file> <gpu0 job name> <src>}; JOB=${2:?}; SRC=${3:?}
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; OUT=$SP/results/tables/talk; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
G1=GPU-51164b62-fd30-253a-5738-55844d46af9c
exec >> "$OUT/fallback_$SRC.log" 2>&1
echo "fallback for $SRC start $(date '+%F %T'): after $TRIGGER, unless GPU 0's $JOB delivers it"
until [ -f "$TRIGGER" ]; do sleep 120; done
grep -q "^$SRC:" "$OUT/scores.txt" 2>/dev/null && { echo "$(date '+%F %T') $SRC scored by GPU 0: nothing to do"; exit 0; }
echo "$(date '+%F %T') trigger present: stopping the GPU 0 slot $JOB-orch (no further restarts on GPU 0)"
systemctl --user stop "$JOB-orch.service" 2>/dev/null
while systemctl --user is-active --quiet "$JOB.service"; do sleep 60; done   # a run in progress on GPU 0 finishes or is stopped for a game
systemctl --user stop "$JOB-guard.service" 2>/dev/null
grep -q "^$SRC:" "$OUT/scores.txt" 2>/dev/null && { echo "$(date '+%F %T') $SRC scored by GPU 0: nothing to do"; exit 0; }
echo "$(date '+%F %T') GPU 0's job ended without the score: running $SRC on GPU 1"
CUDA_VISIBLE_DEVICES=$G1 UNIT=$JOB-gpu1.service exec /bin/bash "$S/talk_readers.sh" "$SRC"
