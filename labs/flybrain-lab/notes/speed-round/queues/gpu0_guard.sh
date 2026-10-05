#!/usr/bin/env bash
# Keeps a probe queue on GPU 0 polite: stops the queue unit when any compute process that is not ours appears on GPU 0
# (Kyle's card), or when the 22:00-05:00 window closes. "Ours" = the processes of the queue unit and of the training unit.
# usage: gpu0_guard.sh <queue unit> <log file>
set -u
UNIT=$1; LOG=$2
GPU0=GPU-cdb4714c-560c-9a81-6420-f07eb59b5a5d
WT=/games/ailab/lab/flybrain-lab/.worktrees/probes
pids_of() { cg=$(systemctl --user show -p ControlGroup --value "$1" 2>/dev/null); [ -n "$cg" ] && cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null; }
stop_queue() { echo "$(date +%T) $1" >> "$LOG"; systemctl --user stop "$UNIT"; rmdir "$WT"/results/*.lock 2>/dev/null; }
while systemctl --user is-active --quiet "$UNIT"; do
  ours=" $( { pids_of "$UNIT"; pids_of flybrain-spine-s0.service; pids_of qwen-server.service; } | tr '\n' ' ') "  # the model server is a background service, not Kyle at the card
  for pid in $(nvidia-smi -i "$GPU0" --query-compute-apps=pid --format=csv,noheader); do
    case "$ours" in *" $pid "*) ;; *) stop_queue "yielding GPU 0 to pid $pid ($(ps -o comm= -p "$pid"))"; exit 0 ;; esac
  done
  h=$(date +%H%M)
  if [ "$h" -ge 0500 ] && [ "$h" -lt 2200 ]; then stop_queue "window closed"; exit 0; fi
  sleep 60
done
echo "$(date +%T) queue ended on its own" >> "$LOG"
