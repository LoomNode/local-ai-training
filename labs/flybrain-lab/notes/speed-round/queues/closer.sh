#!/usr/bin/env bash
# Starts the model server again when the GPU 1 queue is empty: none of the named units has been active for 15 minutes in a
# row and the card lists no compute process. Runs detached, so the server comes back even if no session is watching.
# usage: closer.sh <unit> [more units ...]
set -u
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one GPU UUID}"
LOG=/games/ailab/lab/flybrain-lab/.worktrees/speed-heads/results/speed_closer.log
quiet=0
while [ "$quiet" -lt 15 ]; do
  busy=0
  for u in "$@"; do systemctl --user is-active --quiet "$u" && busy=1; done
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$CUDA_VISIBLE_DEVICES")
  if [ "$busy" = 1 ] || [ "$apps" != 0 ]; then quiet=0; else quiet=$((quiet + 1)); fi
  sleep 60
done
echo "$(date '+%F %T') queue empty for 15 minutes: starting the model server" >> "$LOG"
systemctl --user start qwen-server.service && echo "$(date '+%F %T') model server started" >> "$LOG"
