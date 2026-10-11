#!/usr/bin/env bash
# The supervisor of the big train's GPU 0 copy (Kyle, 2026-10-11: 8 layers on his gaming card, 12 on GPU 1; "tough to say 12 or 8" so both):
# waits for the card to be idle for three minutes, starts the run unit (bigtrain.sh, which resumes from its latest checkpoint) and its
# guard (probe_guard.sh stops the run unit when a game or another compute process appears), waits for the run unit to end, and goes
# round again, until the stop file exists. It never touches the GPU itself, so the guard does not stop it. At most 2,000 steps (about
# 10 minutes at 8 layers) are lost per game session. Also prunes nothing: bigscore.sh scores and prunes the checkpoints.
# usage: NAME=big8 LAYERS=8 CUDA_VISIBLE_DEVICES=<GPU 0 uuid> bigsuper_gpu0.sh ; stop with: touch results/tables/big/STOP_big8
set -u
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the UUID of GPU 0}"; : "${NAME:?}"; : "${LAYERS:?}"   # no apostrophe inside ${...:?...}: bash pairs it
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scale_common.sh"
RUN_UNIT=$NAME-gpu0.service
STOP=$OUT/big/STOP_$NAME
mkdir -p "$OUT/big"
exec >> "$OUT/big/${NAME}_super.log" 2>&1
echo "supervisor for $NAME start $(date '+%F %T')"
while [ ! -f "$STOP" ]; do
  idle=0
  while [ "$idle" -lt 6 ] && [ ! -f "$STOP" ]; do
    sleep 30
    n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
    used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "$n" = 0 ] && [ "$used" -lt 2500 ]; then idle=$((idle + 1)); else idle=0; fi
  done
  [ -f "$STOP" ] && break
  echo "$(date '+%F %T') card idle for three minutes: starting $RUN_UNIT"
  systemctl --user reset-failed "$RUN_UNIT" 2>/dev/null
  systemd-run --user --unit "${RUN_UNIT%.service}" --setenv CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" --setenv NAME="$NAME" --setenv LAYERS="$LAYERS" --setenv UNIT="$RUN_UNIT" /bin/bash "$S/bigtrain.sh" 2>&1 | tail -n 1
  sleep 5
  systemd-run --user --unit "${RUN_UNIT%.service}-guard" /bin/bash "$S/probe_guard.sh" "$RUN_UNIT" "$OUT/guard.log" 2>&1 | tail -n 1
  while systemctl --user is-active --quiet "$RUN_UNIT"; do sleep 60; done
  echo "$(date '+%F %T') $RUN_UNIT ended ($(tail -n 1 "$OUT/guard.log"))"
  sleep 60
done
echo "supervisor for $NAME stopped $(date '+%F %T') (stop file)"
