#!/usr/bin/env bash
# A slot on GPU 0 for one job, by the card's rules (Kyle, 2026-10-14: "Super nice that I can just use it whenever I want and it just
# recovers"): waits for a trigger file (what the job must come after), parks the cooling queue's filler (touch big/STOP_big8; big8-gpu0
# stopped right after its next checkpoint save, so nothing is lost), waits for the card to be idle for three minutes, runs the job
# as the unit <name> under the game guard (probe_guard.sh stops it for a game; the loop restarts it when the card is idle again: the
# job must skip what it has already done), until the done file exists; then releases the filler (rm STOP_big8). Generalises
# qwen_orchestrate.sh. No GPU of its own; run it as its own unit:
#   systemd-run --user --unit <name>-orch /bin/bash gpu0_slot.sh <name> <trigger file> <done file> <script> [args...]
# log: results/tables/use/<name>_slot.log. The script runs with CUDA_VISIBLE_DEVICES=<GPU 0 uuid> and UNIT=<name>.service.
set -u
NAME=${1:?usage: gpu0_slot.sh <name> <trigger file> <done file> <script> [args...]}; TRIGGER=${2:?}; DONE=${3:?}; shift 3
SCRIPT=${1:?a script to run}; shift
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SP=/games/ailab/lab/flybrain-lab/.worktrees/speed; BIG=$SP/results/tables/big; U=$SP/results/tables/use
G0=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
mkdir -p "$U"
exec >> "$U/${NAME}_slot.log" 2>&1
echo "slot $NAME start $(date '+%F %T'): waiting for $TRIGGER; done when $DONE exists"
until [ -f "$TRIGGER" ]; do sleep 120; done
[ -f "$DONE" ] && { echo "$(date '+%F %T') already done"; exit 0; }
echo "$(date '+%F %T') trigger present: parking the filler"
touch "$BIG/STOP_big8"
if systemctl --user is-active --quiet big8-gpu0.service; then
  n=$(grep -c "^saved big8_transformer" "$BIG/big8.log")
  until [ "$(grep -c "^saved big8_transformer" "$BIG/big8.log")" -gt "$n" ] || ! systemctl --user is-active --quiet big8-gpu0.service; do sleep 10; done
  sleep 3; systemctl --user stop big8-gpu0.service; echo "$(date '+%F %T') big8-gpu0 stopped after $(grep '^saved big8_transformer' "$BIG/big8.log" | tail -n 1)"
fi
until [ -f "$DONE" ]; do
  idle=0
  while [ "$idle" -lt 6 ]; do
    sleep 30
    n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$G0")
    used=$(nvidia-smi -i "$G0" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "$n" = 0 ] && [ "$used" -lt 2500 ]; then idle=$((idle + 1)); else idle=0; fi
  done
  echo "$(date '+%F %T') card idle: starting $NAME"
  systemctl --user reset-failed "$NAME.service" 2>/dev/null
  systemd-run --user --unit "$NAME" --setenv CUDA_VISIBLE_DEVICES="$G0" --setenv UNIT="$NAME.service" /bin/bash "$SCRIPT" "$@" 2>&1 | tail -n 1
  sleep 5
  systemd-run --user --unit "$NAME-guard" /bin/bash "$S/probe_guard.sh" "$NAME.service" "$SP/results/tables/guard.log" 2>&1 | tail -n 1
  while systemctl --user is-active --quiet "$NAME.service"; do sleep 60; done
  echo "$(date '+%F %T') $NAME ended ($(tail -n 1 "$SP/results/tables/guard.log"))"
  sleep 60
done
rm -f "$BIG/STOP_big8"
echo "$(date '+%F %T') $NAME done; the filler is released"
